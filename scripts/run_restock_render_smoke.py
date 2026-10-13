#!/usr/bin/env python3
"""Bounded native RTX/RGB-D smoke, separate from the autonomous mission.

One static cube, floor and physical camera mount. No robot, ROS node or planner.
Parent holds the GPU guard while monitoring its exact task-owned child session.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from twin.restock.guard import available_ram_mb,gpu_slot
from twin.restock.station import OwnedProcess


def write(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def runtime_environment(root):
    root=Path(root).resolve()
    lib=root/'usr/lib/x86_64-linux-gnu'
    icd=root/'usr/share/vulkan/icd.d/nvidia_icd.json'
    egl=root/'usr/share/glvnd/egl_vendor.d/10_nvidia.json'
    if any(not p.is_file() for p in (lib/'libcuda.so.1',lib/'libnvidia-ml.so.1',
                                    root/'usr/bin/nvidia-smi',icd,egl)):
        raise ValueError('complete verified task-local NVIDIA runtime required')
    env=dict(os.environ)
    for key in ('LD_PRELOAD','LD_AUDIT','LD_DEBUG','LD_DEBUG_OUTPUT','DISPLAY','WAYLAND_DISPLAY',
                'VK_INSTANCE_LAYERS','VK_LOADER_LAYERS_ENABLE','VK_LOADER_LAYERS_ALLOW',
                'VK_ADD_DRIVER_FILES','__NV_PRIME_RENDER_OFFLOAD','__NV_PRIME_RENDER_OFFLOAD_PROVIDER',
                '__VK_LAYER_NV_optimus','__NV_SMOOTH_MOTION','__EGL_EXTERNAL_PLATFORM_CONFIG_FILENAMES'):
        env.pop(key,None)
    env.update(LD_LIBRARY_PATH=str(lib)+':'+str(lib/'vdpau'),
        PATH=str(root/'usr/bin')+os.pathsep+env.get('PATH',os.defpath),
        VK_DRIVER_FILES=str(icd),VK_ICD_FILENAMES=str(icd),VK_LOADER_LAYERS_DISABLE='~implicit~',
        __EGL_VENDOR_LIBRARY_FILENAMES=str(egl),
        __EGL_EXTERNAL_PLATFORM_CONFIG_DIRS=str(root/'usr/share/egl/egl_external_platform.d'),
        __GLX_VENDOR_LIBRARY_NAME='nvidia',ROS_DOMAIN_ID='88',ROS_LOCALHOST_ONLY='1',
        ROS_AUTOMATIC_DISCOVERY_RANGE='LOCALHOST',ROS_STATIC_PEERS='',
        OMNI_KIT_ACCEPT_EULA='YES')
    return env


def services():
    result={}
    for scope in ([],['--user']):
        command=['systemctl',*scope,'list-units','--type=service','--state=running',
                 '--no-legend','--plain','--no-pager']
        rows=subprocess.run(command,capture_output=True,text=True,timeout=10,check=True).stdout
        names=[line.split()[0] for line in rows.splitlines() if line.strip()]
        values=subprocess.run(['systemctl',*scope,'show',*names,
            '--property=Id,ActiveState,SubState,MainPID'],capture_output=True,text=True,
            timeout=10,check=True).stdout
        result['user' if scope else 'system']=sorted(values.strip().split('\n\n'))
    return result


def sample():
    text=subprocess.run(['nvidia-smi','--query-gpu=memory.total,memory.used,memory.free',
                         '--format=csv,noheader,nounits'],capture_output=True,text=True,
                        timeout=10,check=True).stdout.strip()
    total,used,free=[int(v.strip()) for v in text.split(',')]
    return dict(wall_time=time.time(),gpu_total_mib=total,gpu_used_mib=used,
                gpu_free_mib=free,ram_available_mib=available_ram_mb())


def worker(output,ffmpeg):
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'width':640,'height':360,
        'renderer':'RayTracedLighting','anti_aliasing':0,'multi_gpu':False})
    try:
        import numpy as np
        import omni.usd
        from pxr import UsdGeom,UsdLux,Gf
        from isaacsim.core.api import World
        from twin.restock.native_cameras import CameraRig
        from twin.restock_camera.geometry import look_at
        from twin.restock.recording import Recorder
        world=World(physics_dt=1/60,rendering_dt=1/60,stage_units_in_meters=1.)
        stage=omni.usd.get_context().get_stage()
        UsdGeom.Xform.Define(stage,'/World')
        floor=UsdGeom.Cube.Define(stage,'/World/Floor')
        floor.CreateSizeAttr(1.);floor.AddScaleOp().Set(Gf.Vec3d(4,4,.02))
        floor.AddTranslateOp().Set(Gf.Vec3d(0,0,-.5))
        floor.CreateDisplayColorAttr([(0.25,0.3,0.35)])
        cube=UsdGeom.Cube.Define(stage,'/World/StaticTestCube')
        cube.CreateSizeAttr(.4);cube.AddTranslateOp().Set(Gf.Vec3d(0,0,.2))
        cube.CreateDisplayColorAttr([(0.8,0.15,0.05)])
        light=UsdLux.DomeLight.Define(stage,'/World/Light')
        light.CreateIntensityAttr(1000.)
        optical=look_at((1.5,-1.5,1.2),(0,0,.2))
        mount=np.eye(4);mount[:3,3]=[1.5,-1.5,1.18]
        spec=dict(id='smoke',parent='map',parent_from_optical=optical.tolist(),
            intrinsics=dict(width=640,height=360,fx=500.,fy=500.,cx=320.,cy=180.,near=.05,far=10.),
            collision_solids=[dict(parent_from_solid=mount.tolist(),size_m=[.04,.04,.03])])
        rig=CameraRig(stage=stage,specs=[spec],parent_paths={'map':'/World'})
        world.reset()
        camera=rig.cameras['smoke']
        for _ in range(90): world.step(render=True)
        recorder=Recorder(output/'frames',fps=30)
        last=None
        for _ in range(180):
            world.step(render=True)
            frame=camera.get_current_frame(clone=True)
            identity=frame.get('rendering_frame')
            if identity is None or identity==last: continue
            rgb=np.asarray(frame.get('rgb'))
            depth=np.asarray(frame.get('distance_to_image_plane'),float)
            if rgb.shape!=(360,640,4) or depth.shape!=(360,640):
                raise ValueError('native RGB-D shape mismatch')
            if np.std(rgb[:,:,:3])<2 or np.count_nonzero(np.isfinite(depth)&(depth>0))<1000:
                raise ValueError('native RGB-D is empty or uniform')
            recorder.add(rgb[:,:,:3],render_id=str(identity),simulation_time=float(frame['rendering_time']))
            if not recorder.frames[:-1]:
                np.save(output/'depth.npy',depth)
                write(output/'camera.json',dict(spec=spec,intrinsics=camera.get_intrinsics_matrix().tolist(),
                    rendering_frame=int(identity),rendering_time=float(frame['rendering_time']),
                    finite_depth_pixels=int(np.count_nonzero(np.isfinite(depth)&(depth>0)))))
            last=identity
            if len(recorder.frames)>=30: break
        if len(recorder.frames)<30: raise RuntimeError('insufficient real rendered frames')
        video=recorder.encode(output/'native-render-smoke.mp4',ffmpeg=ffmpeg)
        stage.GetRootLayer().Export(str(output/'smoke.usda'))
        libraries=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                          if any(name in line for name in ('libcuda','libnvidia','libGLX_nvidia'))})
        write(output/'native-result.json',dict(status='passed',evidence='static native RTX RGB-D smoke only',
            robot_motion=False,autonomous_mission=False,video=video,nvidia_libraries=libraries))
    finally:
        app.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--ffmpeg',required=True,type=Path)
    parser.add_argument('--isaac-python',type=Path)
    parser.add_argument('--nvidia-runtime',type=Path)
    parser.add_argument('--heavy-threshold-mb',type=int,choices=(1500,2500),default=1500)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker:
        worker(args.output,args.ffmpeg);return
    if not args.isaac_python or not args.nvidia_runtime:
        parser.error('--isaac-python and --nvidia-runtime required')
    environment=runtime_environment(args.nvidia_runtime)
    os.environ.clear();os.environ.update(environment)
    args.output.mkdir(parents=True,exist_ok=False)
    record=dict(status='failed',code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                telemetry=[],runtime_root=str(args.nvidia_runtime),wall_limit_s=300)
    child=None
    try:
        with gpu_slot(heavy_threshold_mb=args.heavy_threshold_mb) as decision:
            if decision.free_mb<8192: raise RuntimeError('need at least 8 GiB free VRAM')
            record['guard']=asdict(decision)
            record['services_before']=services()
            command=[str(args.isaac_python),str(Path(__file__).resolve()),'--worker',
                     '--output',str(args.output),'--ffmpeg',str(args.ffmpeg)]
            child=OwnedProcess(command,args.output/'isaac.log',env=environment)
            try:
                child.start()
                record['pid']=child.process.pid
                started=time.monotonic()
                while child.alive:
                    value=sample();record['telemetry'].append(value)
                    write(args.output/'run.json',record)
                    if value['ram_available_mib']<4096 or value['gpu_free_mib']<1024:
                        raise RuntimeError('task resource floor reached')
                    if time.monotonic()-started>300: raise TimeoutError('native smoke exceeded 300s')
                    time.sleep(1.)
                record['exit_code']=child.process.returncode
                if child.process.returncode: raise RuntimeError('native child failed; inspect isaac.log')
                record['status']='passed'
            finally:
                if child: child.close()
                record['services_after']=services()
                record['services_unchanged']=record['services_before']==record['services_after']
                record['after']=sample()
    except Exception as exc:
        record['reason']=str(exc)
        raise
    finally:
        write(args.output/'run.json',record)


if __name__=='__main__': main()
