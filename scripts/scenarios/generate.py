"""Headless scenario renderer + annotator. Run under Blender:

  blender -b --python scripts/scenarios/generate.py -- \
      --scenario depletion --out ~/datasets/store-scenarios [options]

Options (after the ``--``):
  --scenario NAME   one of depletion|spills|people|misplaced|multiview|all
  --out DIR         dataset root (default ~/datasets/store-scenarios)
  --samples N       Cycles samples (default 16)
  --device cpu|gpu  render device (default cpu; GPU is opt-in, see RUNBOOK)
  --smoke           tiny set (few frames/cameras) for the committed fixture + tests
  --planogram PATH  planogram.json (default: repo scripts/planogram.json)

Writes RGB frames, optional 16-bit depth (mm), per-image annotation JSON, a COCO
export, scenario-level ground_truth.json, camera intrinsics/extrinsics and a
manifest, in the layout documented in docs/scenario-format.md. It never touches the
repo's scenes/ or the running Blender GUI — build a *separate* headless process.
"""
from __future__ import annotations

import glob
import os
import sys
import time

import bpy
import numpy as np
from mathutils import Matrix

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # scripts/
sys.path.insert(0, _HERE)                     # scripts/scenarios/

import catalog_geom as geom           # noqa: E402
import annotate                        # noqa: E402
import dataset as ds                   # noqa: E402
import scenario_defs as sd             # noqa: E402
import build_scene                     # noqa: E402


def _args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []

    def opt(name, default=None):
        return argv[argv.index(name) + 1] if name in argv else default
    return dict(
        scenario=opt("--scenario", "all"),
        out=os.path.expanduser(opt("--out", "~/datasets/store-scenarios")),
        samples=int(opt("--samples", "16")),
        device=opt("--device", "cpu"),
        smoke="--smoke" in argv,
        res=opt("--res", None),
        planogram=opt("--planogram", os.path.join(os.path.dirname(_HERE), "planogram.json")),
    )


def setup_render(samples, device):
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    if device == "gpu":
        try:
            sc.cycles.device = 'GPU'
            sc.cycles.denoiser = 'OPTIX'
            prefs = bpy.context.preferences.addons["cycles"].preferences
            prefs.compute_device_type = 'OPTIX'
            prefs.get_devices()
            for d in prefs.devices:
                d.use = d.type != 'CPU'
        except Exception as e:
            print("GPU setup failed, falling back to CPU:", e)
            device = "cpu"
    if device == "cpu":
        sc.cycles.device = 'CPU'
        sc.cycles.denoiser = 'OPENIMAGEDENOISE'
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = 'NONE'
        prefs.get_devices()
        for d in prefs.devices:
            d.use = d.type == 'CPU'
    sc.view_settings.view_transform = 'AgX'
    sc.render.image_settings.file_format = 'PNG'
    sc.render.image_settings.color_mode = 'RGB'

    vl = sc.view_layers[0]
    vl.use_pass_z = True
    vl.use_pass_object_index = True
    sc.use_nodes = True
    nt = sc.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    rl = nt.nodes.new("CompositorNodeRLayers")
    comp = nt.nodes.new("CompositorNodeComposite")
    nt.links.new(rl.outputs["Image"], comp.inputs["Image"])
    fo = nt.nodes.new("CompositorNodeOutputFile")
    fo.format.file_format = 'OPEN_EXR'
    fo.format.color_depth = '32'
    fo.file_slots.clear()
    fo.file_slots.new("depth")
    fo.file_slots.new("index")
    nt.links.new(rl.outputs["Depth"], fo.inputs["depth"])
    nt.links.new(rl.outputs["IndexOB"], fo.inputs["index"])
    return fo


def _load_exr_channel0(path):
    img = bpy.data.images.load(path)
    w, h = img.size
    arr = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)[..., 0]
    arr = arr[::-1]          # Blender pixels are bottom-up; flip to top-down
    bpy.data.images.remove(img)
    return arr


def _write_depth_png(depth_m, path):
    """16-bit grayscale PNG, value = depth in millimetres (0 = no hit)."""
    sc = bpy.context.scene
    h, w = depth_m.shape
    mm = depth_m * 1000.0
    mm[(mm <= 0) | (mm > 65000)] = 0.0      # clamp background sentinel (1e10) to 0
    norm = (mm / 65535.0).astype(np.float32)
    img = bpy.data.images.new("depth_tmp", width=w, height=h, float_buffer=True)
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    rgba[..., 0] = rgba[..., 1] = rgba[..., 2] = norm[::-1]   # back to bottom-up
    rgba[..., 3] = 1.0
    img.pixels.foreach_set(rgba.ravel())      # foreach_set + update commits the buffer
    img.update()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ims = sc.render.image_settings
    prev = (ims.file_format, ims.color_depth, ims.color_mode,
            sc.view_settings.view_transform)
    ims.file_format = 'PNG'
    ims.color_depth = '16'
    ims.color_mode = 'BW'
    # 'Raw' bypasses the AgX view transform so the stored 16-bit value is a linear
    # passthrough of norm*65535 (i.e. exactly the depth in millimetres).
    sc.view_settings.view_transform = 'Raw'
    img.save_render(path, scene=sc)
    (ims.file_format, ims.color_depth, ims.color_mode,
     sc.view_settings.view_transform) = prev
    bpy.data.images.remove(img)


def render_scenario(scn, plan, root, fo, samples):
    sc = bpy.context.scene
    objects, cam_objs = build_scene.build(scn, plan)
    p = ds.paths(root, scn.name)
    exr_tmp = os.path.join(p["base"], "_exr")
    os.makedirs(exr_tmp, exist_ok=True)
    coco_images = []
    manifest_images = []
    t0 = time.time()
    for cp in scn.cameras:
        ds.write_camera(root, scn.name, cp.to_json())
        sc.camera = cam_objs[cp.name]
        sc.render.resolution_x = cp.width
        sc.render.resolution_y = cp.height
        sc.render.resolution_percentage = 100
        for frame in range(scn.n_frames):
            build_scene.apply_frame(scn, objects, frame)
            sc.frame_set(frame + 1)
            for f in glob.glob(os.path.join(exr_tmp, "*.exr")):
                os.remove(f)
            fo.base_path = exr_tmp
            rgb_path = os.path.join(p["base"], ds.rel_frame(cp.name, frame))
            os.makedirs(os.path.dirname(rgb_path), exist_ok=True)
            sc.render.filepath = rgb_path
            bpy.ops.render.render(write_still=True)

            idx_files = glob.glob(os.path.join(exr_tmp, "index*.exr"))
            index_map = _load_exr_channel0(idx_files[0]) if idx_files else None
            ann = annotate.image_annotation(scn, cp, frame)
            if index_map is not None:
                ann = annotate.refine_with_index(ann, index_map)
            ann["file_name"] = ds.rel_frame(cp.name, frame)
            ds.write_annotation(root, scn.name, cp.name, frame, ann)

            depth_rel = None
            if scn.with_depth:
                dep_files = glob.glob(os.path.join(exr_tmp, "depth*.exr"))
                if dep_files:
                    depth_m = _load_exr_channel0(dep_files[0])
                    depth_rel = ds.rel_depth(cp.name, frame)
                    _write_depth_png(depth_m, os.path.join(p["base"], depth_rel))

            img_rec = dict(file_name=ds.rel_frame(cp.name, frame), width=cp.width,
                           height=cp.height, frame=frame, camera=cp.name,
                           annotations=ann["annotations"])
            coco_images.append(img_rec)
            manifest_images.append(dict(file=ds.rel_frame(cp.name, frame),
                                        annotation=ds.rel_annotation(cp.name, frame),
                                        depth=depth_rel, width=cp.width, height=cp.height,
                                        frame=frame, camera=cp.name))
    # scenario-level outputs
    coco = annotate.build_coco(coco_images, annotate.categories_in(scn),
                               info=dict(scenario=scn.name, kind=scn.kind, seed=scn.seed))
    ds.write_coco(root, scn.name, coco)
    ds.write_ground_truth(root, scn.name, scn.ground_truth)
    man = scn.manifest()
    man["images"] = manifest_images
    man["samples"] = samples
    ds.write_manifest(root, scn.name, man)
    for f in glob.glob(os.path.join(exr_tmp, "*.exr")):
        os.remove(f)
    try:
        os.rmdir(exr_tmp)
    except OSError:
        pass
    dt = round(time.time() - t0, 1)
    print(f"[{scn.name}] {len(manifest_images)} images in {dt}s -> {p['base']}")


def smoke_overrides(kind):
    """Tiny parameters for the committed fixture / tests."""
    return {
        "depletion": dict(n_frames=4, fps=2, n_remove=3, n_restock=1, n_shoppers=1),
        "spills": dict(n_frames=4, fps=2),
        "people": dict(n_frames=4, fps=2, n_people=3),
        "misplaced": dict(),
        "multiview": dict(n_views=3),
    }[kind]


def main():
    a = _args()
    plan = geom.load_planogram(a["planogram"])
    fo = setup_render(a["samples"], a["device"])
    kinds = list(sd.REGISTRY) if a["scenario"] == "all" else [a["scenario"]]
    res = None
    if a["res"]:
        res = tuple(int(v) for v in a["res"].lower().split("x"))
    for kind in kinds:
        params = smoke_overrides(kind) if a["smoke"] else {}
        scn = sd.REGISTRY[kind](plan, **params)
        if a["smoke"]:
            scn.name = scn.name + "_smoke"
        if res:
            for cp in scn.cameras:
                cp.width, cp.height = res
        render_scenario(scn, plan, a["out"], fo, a["samples"])
    print("DONE")


if __name__ == "__main__":
    main()
