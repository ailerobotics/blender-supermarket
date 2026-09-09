"""Export the supermarket to USD for Isaac Sim.
  blender -b scenes/supermarket.blend --python scripts/export_usd.py -- [--out export/supermarket.usdc]
Z-up, metres, instanced products (linked duplicates -> USD instances), custom properties
(sku, category, brand, price, slot...) written as `userProperties:*` on each prim.
"""
import bpy, sys, os, time
args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
out = args[args.index("--out") + 1] if "--out" in args else "/home/manishsaini/workspaces/blender-supermarket/export/supermarket.usdc"
os.makedirs(os.path.dirname(out), exist_ok=True)
t = time.time()
kw = dict(filepath=out, export_animation=False, export_uvmaps=True, export_normals=True, export_materials=True,
          use_instancing=True, generate_preview_surface=True, export_textures=True, export_textures_mode='NEW', overwrite_textures=True,
          relative_paths=True, root_prim_path="/Supermarket", export_custom_properties=True, custom_properties_namespace="userProperties",
          export_lights=True, export_cameras=True, export_meshes=True, convert_world_material=False, evaluation_mode='RENDER')
try:
    bpy.ops.wm.usd_export(**kw)
except TypeError as e:
    print("retrying with a reduced argument set:", e)
    bpy.ops.wm.usd_export(filepath=out, export_materials=True, export_textures=True, use_instancing=True, root_prim_path="/Supermarket")
print(f"USD written: {out} ({os.path.getsize(out) / 1e6:.1f} MB) in {round(time.time() - t, 1)} s")
