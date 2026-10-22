#!/usr/bin/env python3
"""Bounded native RTX/RGB-D smoke, separate from the autonomous mission.

Default: one static cube, floor and physical camera mount. The optional full-store
view includes static robot/cart CAD. Neither mode runs robot motion or a planner.
Parent holds the GPU guard while monitoring its exact task-owned child session.
"""
import argparse
from dataclasses import asdict
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from twin.restock.guard import available_ram_mb,gpu_slot
from twin.restock.station import OwnedProcess


def write(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def renderer_metadata():
    """Read effective native settings; do not infer them from requested config."""
    import carb.dictionary
    import carb.settings
    import cv2
    settings=carb.settings.get_settings()
    tree=settings.get_settings_dictionary('/rtx')
    return dict(opencv_version=cv2.__version__,opencv_module=cv2.__file__,
        opencv_build=cv2.getBuildInformation(),
        effective_rtx=carb.dictionary.get_dictionary().get_dict_copy(tree),
        note='Actual settings snapshot; absent keys are not inferred defaults.')


def runtime_environment(root,output):
    root=Path(root).resolve()
    lib=root/'usr/lib/x86_64-linux-gnu'
    icd=root/'usr/share/vulkan/icd.d/nvidia_icd.json'
    egl=root/'usr/share/glvnd/egl_vendor.d/10_nvidia.json'
    if any(not p.is_file() for p in (lib/'libcuda.so.1',lib/'libnvidia-ml.so.1',
                                    root/'usr/bin/nvidia-smi',icd,egl)):
        raise ValueError('complete verified task-local NVIDIA runtime required')
    # Kit's CUDA interop plugin requests the unversioned NVML development name.
    # Distribution runtime packages supply only .so.1; without this task-local
    # alias the loader can fall back to a mismatched system development symlink.
    aliases=Path(output)/'runtime-links'
    aliases.mkdir()
    for name in ('libnvidia-ml','libcuda'):
        (aliases/(name+'.so')).symlink_to(lib/(name+'.so.1'))
    env=dict(os.environ)
    for key in ('LD_PRELOAD','LD_AUDIT','LD_DEBUG','LD_DEBUG_OUTPUT','DISPLAY','WAYLAND_DISPLAY',
                'VK_INSTANCE_LAYERS','VK_LOADER_LAYERS_ENABLE','VK_LOADER_LAYERS_ALLOW',
                'VK_ADD_DRIVER_FILES','__NV_PRIME_RENDER_OFFLOAD','__NV_PRIME_RENDER_OFFLOAD_PROVIDER',
                '__VK_LAYER_NV_optimus','__NV_SMOOTH_MOTION','__EGL_EXTERNAL_PLATFORM_CONFIG_FILENAMES'):
        env.pop(key,None)
    env.update(LD_LIBRARY_PATH=str(aliases)+':'+str(lib)+':'+str(lib/'vdpau'),
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


def worker(output,ffmpeg,runtime_config=None,view='source',assembly_only=False,coupling_only=False,towing_only=False,humans_only=False):
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'width':640,'height':360,
        'renderer':'RayTracedLighting','anti_aliasing':0,'multi_gpu':False})
    try:
        write(output/'renderer-start.json',renderer_metadata())
        if assembly_only:
            from twin.restock.assembly_check import run
            config=json.loads(Path(runtime_config).read_text())
            result=run(app,config,output)
            write(output/'native-result.json',dict(status='passed',assembly=result,
                evidence='stationary native assembly diagnostic, not mission or video'))
            return
        if coupling_only:
            from twin.restock.coupling_check import run
            config=json.loads(Path(runtime_config).read_text())
            result=run(app,config,output)
            write(output/'native-result.json',dict(status=result['status'],coupling=result,
                evidence='native cart coupling diagnostic, not mission or video'))
            return
        if towing_only:
            from twin.restock.towing_check import run
            config=json.loads(Path(runtime_config).read_text())
            result=run(app,config,output,ffmpeg=ffmpeg)
            write(output/'native-result.json',dict(status=result['status'],towing=result,
                evidence='native straight-towing diagnostic with VFR recording, not an autonomous mission'))
            return
        if humans_only:
            from twin.humans.gpu_bench import run as humans_run
            config=json.loads(Path(runtime_config).read_text())
            result=humans_run(app,config,output,ffmpeg=ffmpeg)
            write(output/'native-result.json',dict(status=result['status'],humans=result,
                evidence='native RTX ground-truth human-safety bench (scripted people crossing the '
                         'robot, ground-truth projection into the real candidate camera); not a '
                         'neural-detector measurement and not an autonomous mission'))
            return
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
        light=UsdLux.DomeLight.Define(stage,'/World/Light')
        light.CreateIntensityAttr(1000.)
        required_codes=set()
        if runtime_config:
            from twin.restock.view_scene import build
            config=json.loads(Path(runtime_config).read_text())
            camera,spec,required_codes=build(stage,config,output,view=view)
            if view=='source':
                required_codes.add(json.loads(Path(config['scenario']).read_text())['job']['sku'])
        else:
            floor=UsdGeom.Cube.Define(stage,'/World/Floor')
            floor.CreateSizeAttr(1.);floor.AddScaleOp().Set(Gf.Vec3d(4,4,.02))
            floor.AddTranslateOp().Set(Gf.Vec3d(0,0,-.5))
            floor.CreateDisplayColorAttr([(0.25,0.3,0.35)])
            cube=UsdGeom.Cube.Define(stage,'/World/StaticTestCube')
            cube.CreateSizeAttr(.4);cube.AddTranslateOp().Set(Gf.Vec3d(0,0,.2))
            cube.CreateDisplayColorAttr([(0.8,0.15,0.05)])
            optical=look_at((1.5,-1.5,1.2),(0,0,.2))
            mount=np.eye(4);mount[:3,3]=[1.5,-1.5,1.18]
            spec=dict(id='smoke',parent='map',parent_from_optical=optical.tolist(),
                intrinsics=dict(width=640,height=360,fx=500.,fy=500.,cx=320.,cy=180.,near=.05,far=10.),
                collision_solids=[dict(parent_from_solid=mount.tolist(),size_m=[.04,.04,.03])])
            rig=CameraRig(stage=stage,specs=[spec],parent_paths={'map':'/World'})
            camera=rig.cameras['smoke']
        world.reset()
        for _ in range(90): world.step(render=True)
        write(output/'renderer-capture.json',renderer_metadata())
        recorder=Recorder(output/'frames',fps=60)
        last=None
        last_native_time=None
        observed_frames=[]
        decoded_codes=set()
        # Decode with the PRODUCTION FixedTileQR (one per camera, stateful across frames),
        # not a full-frame cv2 pass: the mission's support binding requires each marker read
        # by this exact tiled decoder with no geometry fallback, and a full-frame cv2 pass
        # masked a production-only failure (data/sim-restock-camera-v2/shelf-tile-seam-root-cause.md).
        marker_camera_id=str(spec.get('id') or view)
        qr_decoder=None
        if required_codes:
            from twin.restock_camera.qr import FixedTileQR
            from twin.restock_camera.sensed import FrameEvidence
            qr_decoder=FixedTileQR(marker_camera_id)
        target_frames=6 if runtime_config else 60
        width,height=spec['intrinsics']['width'],spec['intrinsics']['height']
        (output/'depth-frames').mkdir()
        for _ in range(180):
            step_started=time.monotonic()
            world.step(render=True)
            step_finished=time.monotonic()
            frame=camera.get_current_frame(clone=True)
            clone_finished=time.monotonic()
            identity=frame.get('native_rendering_frame',frame.get('rendering_frame'))
            if identity is None or identity==last: continue
            native_time=Fraction(identity['referenceTimeNumerator'],identity['referenceTimeDenominator'])
            if last_native_time is not None and native_time<=last_native_time:
                raise ValueError('native camera identity moved backwards')
            rgb=np.asarray(frame.get('rgb'))
            depth=np.asarray(frame.get('distance_to_image_plane'),float)
            if rgb.shape!=(height,width,4) or depth.shape!=(height,width):
                raise ValueError('native RGB-D shape mismatch')
            if np.std(rgb[:,:,:3])<2 or np.count_nonzero(np.isfinite(depth)&(depth>0))<1000:
                raise ValueError('native RGB-D is empty or uniform')
            recorder.add(rgb[:,:,:3],render_id=str(identity),simulation_time=float(frame['rendering_time']))
            rgb_written=time.monotonic()
            depth_path=output/'depth-frames'/f'{len(recorder.frames)-1:06d}.npz'
            segmentation=frame.get('instance_id_segmentation',{})
            ids=np.asarray(segmentation.get('data'))
            paths=segmentation.get('info',{}).get('idToLabels',{})
            if ids.shape!=depth.shape or ids.dtype!=np.uint32 or not paths:
                raise ValueError('same-acquisition instance IDs/path mapping unavailable')
            np.savez_compressed(depth_path,depth=depth,instance_ids=ids)
            depth_written=time.monotonic()
            codes=[]
            if qr_decoder is not None:
                # Synthetic-but-valid freshness: a static optical bench is parked, with
                # strictly increasing frame_id (capture index) and acquisition stamp, so
                # the decoder's replay refusal holds while we decode the real pixels.
                stamp=float(frame['rendering_time'])
                evidence=FrameEvidence(marker_camera_id,len(recorder.frames),stamp,stamp,'bench','bench')
                qr=qr_decoder.decode(rgb[:,:,:3],evidence,now=stamp,parked=True)
                codes=[d.text for d in qr.detections]
                decoded_codes.update(codes)
            decode_finished=time.monotonic()
            observed_frames.append(dict(rendering_frame=identity,rendering_time=float(frame['rendering_time']),
                acquired_steady=frame.get('acquired_steady'),session_id=frame.get('session_id'),
                steady_domain=frame.get('steady_domain'),decoded_codes=list(codes),
                instance_id_to_path=paths,
                wall_durations_s=dict(world_step=step_finished-step_started,
                    clone=clone_finished-step_finished,rgb_validation_and_write=rgb_written-clone_finished,
                    depth_segmentation_write=depth_written-rgb_written,qr_decode=decode_finished-depth_written),
                rgb_sha256=recorder.frames[-1]['sha256'],depth_file=str(depth_path.relative_to(output)),
                depth_sha256=hashlib.sha256(depth_path.read_bytes()).hexdigest(),
                rgb_depth_identity_model='same Camera render-product acquisition callback and cloned frame'))
            write(output/'observed-frames.json',observed_frames)
            if not recorder.frames[:-1]:
                np.save(output/'depth.npy',depth)
                write(output/'camera.json',dict(spec=spec,intrinsics=camera.get_intrinsics_matrix().tolist(),
                    rendering_frame=identity,rendering_time=float(frame['rendering_time']),
                    finite_depth_pixels=int(np.count_nonzero(np.isfinite(depth)&(depth>0)))))
            last=identity
            last_native_time=native_time
            if len(recorder.frames)>=target_frames: break
        if len(recorder.frames)<target_frames: raise RuntimeError('insufficient real rendered frames')
        video=recorder.encode(output/'native-render-smoke.mp4',ffmpeg=ffmpeg)
        stage.GetRootLayer().Export(str(output/'smoke.usda'))
        libraries=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                          if any(name in line for name in ('libcuda','libnvidia','libGLX_nvidia'))})
        missing=sorted(required_codes-decoded_codes)
        write(output/'native-result.json',dict(status='failed' if missing else 'passed',
            reason='required physical marker decode missing: '+','.join(missing) if missing else '',
            evidence='static native RTX RGB-D optical bench only',required_codes=sorted(required_codes),
            decoded_codes=sorted(decoded_codes),robot_motion=False,autonomous_mission=False,
            video=video,nvidia_libraries=libraries))
    except BaseException as exc:
        # Kit shutdown may terminate the interpreter with zero even while Python
        # unwinds an exception. Persist the failure before entering app.close().
        write(output/'native-result.json',dict(status='failed',reason=str(exc),
            traceback=traceback.format_exc()))
        raise
    finally:
        app.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--ffmpeg',required=True,type=Path)
    parser.add_argument('--isaac-python',type=Path)
    parser.add_argument('--nvidia-runtime',type=Path)
    parser.add_argument('--runtime-config',type=Path,help='optional full-store static candidate camera bench')
    parser.add_argument('--view',choices=('source','cart','shelf'),default='source')
    parser.add_argument('--assembly-only',action='store_true',help='stationary native CAD/physics/hold diagnostic; no planner or video')
    parser.add_argument('--coupling-only',action='store_true',help='native dock+latch+hold diagnostic; no planner, towing or video')
    parser.add_argument('--towing-only',action='store_true',help='native straight forward-towing diagnostic with VFR recording; no planner or steering')
    parser.add_argument('--humans-only',action='store_true',help='native ground-truth human-safety bench: scripted people cross the robot, GT detection+reaction on the real camera; no planner, mission or neural detector')
    parser.add_argument('--heavy-threshold-mb',type=int,choices=(1500,2500),default=1500)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if sum(bool(x) for x in (args.assembly_only,args.coupling_only,args.towing_only,args.humans_only))>1:
        parser.error('choose at most one of --assembly-only / --coupling-only / --towing-only / --humans-only')
    if (args.assembly_only or args.coupling_only or args.towing_only or args.humans_only) and not args.runtime_config:
        parser.error('--assembly-only/--coupling-only/--towing-only/--humans-only requires --runtime-config')
    if args.worker:
        worker(args.output,args.ffmpeg,args.runtime_config,args.view,args.assembly_only,args.coupling_only,args.towing_only,args.humans_only);return
    if not args.isaac_python or not args.nvidia_runtime:
        parser.error('--isaac-python and --nvidia-runtime required')
    args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    environment=runtime_environment(args.nvidia_runtime,args.output)
    if args.assembly_only or args.coupling_only or args.towing_only:
        from twin.restock.native_environment import ros_environment
        environment=ros_environment(environment,args.isaac_python)
    # The humans bench runs the ground-truth provider in-process (no ROS node), so the
    # lane ROS domain is moot; the GPU guard's isolation() pins every session to
    # ROS_DOMAIN_ID=88 and must not be weakened, so we leave the domain at 88.
    os.environ.clear();os.environ.update(environment)
    # The humans bench renders a multi-second people crossing (many frames), so it
    # needs a longer wall cap than the static smoke -- still bounded well under the
    # 15-minute lane limit firstmate set for one guarded GPU session.
    wall_limit_s=870 if args.humans_only else 300
    record=dict(status='failed',code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                telemetry=[],runtime_root=str(args.nvidia_runtime),wall_limit_s=wall_limit_s)
    child=None
    try:
        with gpu_slot(heavy_threshold_mb=args.heavy_threshold_mb) as decision:
            if decision.free_mb<8192: raise RuntimeError('need at least 8 GiB free VRAM')
            record['guard']=asdict(decision)
            record['services_before']=services()
            command=[str(args.isaac_python),str(Path(__file__).resolve()),'--worker',
                     '--output',str(args.output),'--ffmpeg',str(args.ffmpeg)]
            if args.runtime_config:
                command.extend(['--runtime-config',str(args.runtime_config),'--view',args.view])
            if args.assembly_only:
                command.append('--assembly-only')
            if args.coupling_only:
                command.append('--coupling-only')
            if args.towing_only:
                command.append('--towing-only')
            if args.humans_only:
                command.append('--humans-only')
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
                    if time.monotonic()-started>wall_limit_s: raise TimeoutError(f'native run exceeded {wall_limit_s}s')
                    time.sleep(1.)
                record['exit_code']=child.process.returncode
                if child.process.returncode: raise RuntimeError('native child failed; inspect isaac.log')
                native=json.loads((args.output/'native-result.json').read_text())
                if args.assembly_only:
                    complete=(native.get('assembly',{}).get('settled') is True
                              and native.get('assembly',{}).get('steps',0)>=120)
                elif args.coupling_only:
                    c=native.get('coupling',{})
                    complete=(c.get('settled') is True and c.get('coupled') is True
                              and c.get('hold_steps',0)>=120)
                elif args.towing_only:
                    t=native.get('towing',{})
                    complete=(t.get('towed') is True and t.get('no_wheel_lift') is True
                              and t.get('video',{}).get('frames',0)>=10)
                elif args.humans_only:
                    h=native.get('humans',{})
                    s=h.get('summary',{})
                    complete=(h.get('status')=='passed' and h.get('frames',0)>=2
                              and s.get('slow_seen') is True and s.get('stop_seen') is True
                              and s.get('arm_paused') is True and s.get('base_resumed') is True)
                else:
                    complete=native.get('video',{}).get('frames',0)>=(6 if args.runtime_config else 60)
                if native.get('status')!='passed' or not complete:
                    raise RuntimeError('native result incomplete: '+native.get('reason','missing frames'))
                # Decode the whole recorded video, not merely its container header.
                if not (args.assembly_only or args.coupling_only or args.towing_only or args.humans_only):
                    subprocess.run([str(args.ffmpeg),'-v','error','-nostdin','-i',
                        str(args.output/'native-render-smoke.mp4'),'-f','null','-'],
                        check=True,capture_output=True,timeout=30)
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
