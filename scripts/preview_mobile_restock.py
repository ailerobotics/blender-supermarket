"""CPU Blender static collision-geometry preview, NOT an Isaac physics run.

Run headless with --factory-startup. Only writes the caller's scratch PNG.
Robot links are the actual generated URDF collision meshes/primitives at home FK;
the cart is imported from the generated USD. No simulated motion is claimed.
"""
import argparse
import json
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

sys.path.insert(0,str(Path(__file__).resolve().parent))
from twin.urdf_probe import parse_urdf, link_world_transforms, resolve_package_uri, rpy_to_matrix

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--manifest',type=Path,required=True)
p.add_argument('--package-root',action='append',required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
repo=Path(__file__).resolve().parents[1]
if any(repo/d in a.output.resolve().parents for d in ('scenes','export','renders')):
    raise ValueError('preview output must be scratch')
m=json.loads(a.manifest.read_text())
robot=parse_urdf(m['mobile_urdf'])
transforms=link_world_transforms(robot,m['transport_positions'])
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

def material(name,color,metallic=0.):
    mat=bpy.data.materials.new(name)
    mat.diffuse_color=(*color,1)
    mat.use_nodes=True
    node=mat.node_tree.nodes.get('Principled BSDF')
    node.inputs['Base Color'].default_value=(*color,1)
    node.inputs['Metallic'].default_value=metallic
    node.inputs['Roughness'].default_value=.36
    return mat
blue=material('UR collision geometry',(.16,.40,.60),.35)
base=material('Mobile platform',(.17,.21,.27),.4)
for name,link in robot.links.items():
    for geometry in link.collisions:
        scale=(1,1,1)
        if geometry.kind=='box':
            bpy.ops.mesh.primitive_cube_add(size=1)
            scale=geometry.box_size
        elif geometry.kind=='cylinder':
            bpy.ops.mesh.primitive_cylinder_add(vertices=32,radius=geometry.radius,depth=geometry.length)
        elif geometry.kind=='sphere':
            bpy.ops.mesh.primitive_uv_sphere_add(segments=24,ring_count=12,radius=geometry.radius)
        elif geometry.kind=='mesh':
            path,_=resolve_package_uri(geometry.mesh_filename,a.package_root)
            if not path:
                raise ValueError('unresolved collision mesh: '+geometry.mesh_filename)
            if Path(path).suffix.lower()=='.stl':
                bpy.ops.wm.stl_import(filepath=path)
            elif Path(path).suffix.lower()=='.obj':
                bpy.ops.wm.obj_import(filepath=path,forward_axis='Y',up_axis='Z')
            else:
                raise ValueError('unsupported collision mesh: '+path)
            scale=geometry.mesh_scale
        else:
            raise ValueError('unsupported geometry')
        ob=bpy.context.object
        origin=np.eye(4);origin[:3,:3]=rpy_to_matrix(geometry.origin_rpy)
        origin[:3,3]=geometry.origin_xyz
        pose=transforms[name]@origin
        pose[2,3]+=m['spec']['base_z']
        ob.matrix_world=Matrix(pose.tolist())@Matrix.Diagonal((*scale,1))
        ob.name=name+'_collision'
        ob.data.materials.clear()
        ob.data.materials.append(blue if name.startswith('ur_') else base)
before=set(bpy.data.objects)
bpy.ops.wm.usd_import(filepath=m['cart_usd'])
new=set(bpy.data.objects)-before
for ob in new:
    if ob.parent not in new:
        ob.location.x-=m['spec']['cart_hitch_x']-m['spec']['hitch_x']
floor=material('Floor',(.68,.70,.72))
bpy.ops.mesh.primitive_plane_add(size=200,location=(0,0,-.015))
bpy.context.object.data.materials.append(floor)
for pos,energy,size in (((1,-3,5),1500,5),((-4,2,4),1100,4)):
    bpy.ops.object.light_add(type='AREA',location=pos)
    ob=bpy.context.object;ob.data.energy=energy;ob.data.shape='DISK';ob.data.size=size
    ob.rotation_euler=(Vector((-.5,0,.8))-ob.location).to_track_quat('-Z','Y').to_euler()
bpy.ops.object.camera_add(location=(3.5,-5,3.0))
cam=bpy.context.object
cam.rotation_euler=(Vector((-.6,0,.8))-cam.location).to_track_quat('-Z','Y').to_euler()
cam.data.type='ORTHO';cam.data.ortho_scale=4.6
scene=bpy.context.scene;scene.camera=cam
scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=24
scene.cycles.use_denoising=True;scene.cycles.denoiser='OPENIMAGEDENOISE'
prefs=bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type='NONE'
scene.render.threads_mode='FIXED';scene.render.threads=4
scene.render.resolution_x=1200;scene.render.resolution_y=760;scene.render.resolution_percentage=100
scene.world.color=(.35,.35,.35)
scene.render.image_settings.file_format='PNG'
a.output.parent.mkdir(parents=True,exist_ok=True)
scene.render.filepath=str(a.output.resolve())
bpy.ops.render.render(write_still=True)
