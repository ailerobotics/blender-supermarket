"""Render the Cam_Walk fly-through to an MP4 with EEVEE.
  blender -b scenes/supermarket.blend --python scripts/render_walkthrough.py -- [--res 1280x720] [--samples 32] [--out renders/walkthrough.mp4]
"""
import bpy, sys, os, time
args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
def opt(n, d): return args[args.index(n) + 1] if n in args else d
res = opt("--res", "1280x720").split("x"); samples = int(opt("--samples", "32"))
out = opt("--out", "/home/manishsaini/workspaces/blender-supermarket/renders/walkthrough.mp4")
sc = bpy.context.scene
sc.camera = bpy.data.objects["Cam_Walk"]
sc.render.engine = 'BLENDER_EEVEE_NEXT'; sc.eevee.taa_render_samples = samples
try:
    sc.eevee.shadow_pool_size = '512'; sc.eevee.shadow_ray_count = 1; sc.eevee.shadow_step_count = 2
    sc.eevee.use_raytracing = True
except Exception as e: print("eevee opts:", e)
# 63 shadow-casting area lights overflow EEVEE's shadow buffer: give every second panel a shadow-less light copy
lights = sorted([o for o in bpy.data.objects if o.type == 'LIGHT' and o.name.startswith("LEDLight_")], key=lambda o: o.name)
if lights:
    soft = lights[0].data.copy(); soft.name = "LED_panel_light_noshadow"; soft.use_shadow = False
    for i, o in enumerate(lights):
        if i % 2: o.data = soft
def _progress(scene):
    if scene.frame_current % 30 == 0: print(f"frame {scene.frame_current}/{scene.frame_end} {round(time.time() - t, 1)} s", flush=True)
bpy.app.handlers.render_post.append(_progress)
sc.render.resolution_x, sc.render.resolution_y = int(res[0]), int(res[1]); sc.render.resolution_percentage = 100
sc.render.image_settings.file_format = 'FFMPEG'; sc.render.ffmpeg.format = 'MPEG4'; sc.render.ffmpeg.codec = 'H264'
sc.render.ffmpeg.constant_rate_factor = 'HIGH'; sc.render.ffmpeg.gopsize = 24
sc.render.filepath = out
t = time.time()
bpy.ops.render.render(animation=True)
print(f"walkthrough rendered: {out} frames {sc.frame_start}-{sc.frame_end} in {round(time.time() - t, 1)} s")
