"""Export the supermarket to USD for Isaac Sim.
  blender -b scenes/supermarket.blend --python scripts/export_usd.py -- [--out export/supermarket.usdc]
Z-up, metres, instanced products (linked duplicates -> USD instances), custom properties
(sku, category, brand, price, slot...) written as `userProperties:*` on each prim.

Verify the result offline (no Blender, no GPU) with scripts/validate_usd.py.
"""
import bpy, sys, os, time, shutil, tempfile

ROOT = "/home/manishsaini/workspaces/blender-supermarket"   # same convention as build_supermarket.py
ASSETS_TEXTURES = os.path.join(ROOT, "assets", "textures")

args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
out = args[args.index("--out") + 1] if "--out" in args else os.path.join(ROOT, "export", "supermarket.usdc")
os.makedirs(os.path.dirname(out), exist_ok=True)


def _stage_textures_for_export():
    """Make every image's *backing file* basename equal its datablock *name*.

    Blender's USD exporter copies each image's backing file into `textures/` but
    authors the reference as `@./textures/<image datablock name>@`. When the two
    differ, the reference dangles and the texture is missing once the stage loads
    in Isaac. The product/sign PNGs are fine (there name == basename already), but
    the two Poly Haven materials (`floor_tiles_08`, `white_plaster_02`) are imported
    by the MCP Poly Haven tool from *temp* files (`tmp*.jpg`) while their datablocks
    are named `floor_tiles_08_Diffuse.jpg` etc. — so the floor and all five walls
    lose their textures on export (see scripts/validate_usd.py, which catches this).

    Repoint each affected image at a file named exactly `<image.name>`, preferring
    the canonical, versioned copy in `assets/textures/` and otherwise staging a copy
    of the current backing file. In-memory only — the .blend on disk is untouched,
    and this runs in a headless export process, never in the live GUI session.
    """
    staging = os.path.join(tempfile.gettempdir(), "usd_export_tex_stage")
    os.makedirs(staging, exist_ok=True)
    fixed = 0
    for img in bpy.data.images:
        if img.source != 'FILE':
            continue
        name = img.name
        # Only handle names that are plain filenames (have an extension, no path sep).
        if os.sep in name or "/" in name or "." not in name:
            continue
        src = bpy.path.abspath(img.filepath) if img.filepath else ""
        if src and os.path.basename(src) == name and os.path.exists(src):
            continue  # already consistent
        canonical = os.path.join(ASSETS_TEXTURES, name)
        chosen = canonical if os.path.exists(canonical) else (src if src and os.path.exists(src) else "")
        if not chosen:
            print(f"[export] WARN: no backing file to stage for image {name!r}; "
                  f"its USD texture path may not resolve")
            continue
        dst = os.path.join(staging, name)
        try:
            if os.path.abspath(chosen) != os.path.abspath(dst):
                shutil.copy2(chosen, dst)
            img.filepath = dst
            fixed += 1
        except OSError as exc:
            print(f"[export] WARN: could not stage texture for {name!r}: {exc}")
    print(f"[export] staged {fixed} texture(s) so authored USD paths resolve")


try:
    _stage_textures_for_export()
except Exception as exc:   # never let the texture-staging guard break the export
    print(f"[export] WARN: texture staging skipped ({exc!r})")

t = time.time()
kw = dict(filepath=out, export_animation=False, export_uvmaps=True, export_normals=True, export_materials=True,
          use_instancing=True, generate_preview_surface=True, export_textures=True, export_textures_mode='NEW', overwrite_textures=True,
          relative_paths=True, root_prim_path="/Supermarket", export_custom_properties=True, custom_properties_namespace="userProperties",
          export_lights=True, export_cameras=True, export_meshes=True, convert_world_material=False, evaluation_mode='RENDER')
try:
    bpy.ops.wm.usd_export(**kw)
except TypeError as e:
    print("retrying with a reduced argument set:", e)
    bpy.ops.wm.usd_export(filepath=out, export_materials=True, export_textures=True, use_instancing=True,
                          relative_paths=True, root_prim_path="/Supermarket")
print(f"USD written: {out} ({os.path.getsize(out) / 1e6:.1f} MB) in {round(time.time() - t, 1)} s")
print("Verify offline (no GPU):  python3 scripts/validate_usd.py --store", out)
