"""Render still shots from the saved scene. Usage:
  blender -b scenes/supermarket.blend --python scripts/render_shots.py -- [--samples N] [--cams Cam_A,Cam_B] [--out renders] [--eevee] [--res 1920x1080]
"""
import bpy, sys, os, time
args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
def opt(name, default):
    return args[args.index(name) + 1] if name in args else default
_ROOT = os.environ.get("SUPERMARKET_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
samples = int(opt("--samples", "128")); out = opt("--out", os.path.join(_ROOT, "renders"))
cams = opt("--cams", "Cam_Entrance,Cam_Aisle,Cam_Shelf,Cam_Checkout,Cam_Produce,Cam_Overview").split(",")
res = opt("--res", "1920x1080").split("x")
sc = bpy.context.scene
sc.render.resolution_x, sc.render.resolution_y = int(res[0]), int(res[1]); sc.render.resolution_percentage = 100
if "--eevee" in args:
    sc.render.engine = 'BLENDER_EEVEE_NEXT'; sc.eevee.taa_render_samples = samples
elif "--cpu" in args:
    # CPU Cycles fallback for evidence renders when the GPU is held (e.g. a live
    # Isaac/robot stack). OptiX denoiser is GPU-only, so use OpenImageDenoise.
    sc.render.engine = 'CYCLES'; sc.cycles.samples = samples; sc.cycles.device = 'CPU'
    sc.cycles.use_denoising = True; sc.cycles.denoiser = 'OPENIMAGEDENOISE'
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = 'NONE'; prefs.get_devices()
    for d in prefs.devices: d.use = d.type == 'CPU'
else:
    sc.render.engine = 'CYCLES'; sc.cycles.samples = samples; sc.cycles.device = 'GPU'
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = 'OPTIX'; prefs.get_devices()
    for d in prefs.devices: d.use = d.type != 'CPU'
sc.render.image_settings.file_format = 'PNG'
os.makedirs(out, exist_ok=True)
hide_for_topdown = [o for o in bpy.data.objects if o.name in ("Ceiling",) or o.name.startswith(("LEDPanel_", "aisle_sign_", "sign_", "LEDLight_"))]
for c in cams:
    ob = bpy.data.objects.get(c)
    if not ob: print("no camera", c); continue
    for o in hide_for_topdown: o.hide_render = (c == "Cam_TopDown")
    if c == "Cam_TopDown":
        sc.view_settings.exposure += 0.8; sc.render.resolution_x, sc.render.resolution_y = int(res[0]), int(int(res[0]) * 0.86)
    sc.camera = ob; t = time.time()
    sc.render.filepath = os.path.join(out, f"{c}.png")
    bpy.ops.render.render(write_still=True)
    print(f"rendered {c} in {round(time.time() - t, 1)} s -> {sc.render.filepath}")
    if c == "Cam_TopDown":
        sc.view_settings.exposure -= 0.8; sc.render.resolution_x, sc.render.resolution_y = int(res[0]), int(res[1])
