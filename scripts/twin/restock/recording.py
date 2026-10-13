"""Persist actual rendered RGB frames with timestamps, then encode CPU H.264.

The caller supplies camera render IDs and simulation timestamps. No synthetic
frames, repeats or interpolation are generated to fill missing sensor output.
"""
import hashlib
import json
import math
import re
from pathlib import Path
import subprocess


class Recorder:
    def __init__(self, directory, *, fps=30):
        if type(fps) is not int or fps <= 0:
            raise ValueError('positive integer recording FPS required')
        self.directory=Path(directory)
        self.directory.mkdir(parents=True,exist_ok=False)
        self.fps,self.frames=fps,[]
        self.ids=set()

    def add(self, rgb, *, render_id, simulation_time):
        import numpy as np
        from PIL import Image
        if not isinstance(render_id,str) or not render_id or render_id in self.ids:
            raise ValueError('new actual camera render ID required')
        if not math.isfinite(simulation_time) or simulation_time < 0:
            raise ValueError('invalid frame clock')
        if self.frames and simulation_time<=self.frames[-1]['simulation_time']:
            raise ValueError('acquisition timestamps must strictly increase')
        rgb=np.asarray(rgb)
        if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or any(v%2 for v in rgb.shape[:2]):
            raise ValueError('even-sized uint8 HxWx3 rendered RGB required')
        if self.frames and list(rgb.shape)!=self.frames[0]['shape']:
            raise ValueError('recording resolution changed')
        path=self.directory/f'{len(self.frames):06d}.png'
        Image.fromarray(rgb).save(path)
        frame=dict(render_id=render_id,simulation_time=simulation_time,shape=list(rgb.shape),
                   file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        self.frames.append(frame);self.ids.add(render_id)
        # Save incrementally so an interrupted run retains inspectable evidence.
        (self.directory/'frames.json').write_text(json.dumps(dict(nominal_fps=self.fps,
            timing='variable acquisition intervals',frames=self.frames),indent=2)+'\n')

    @classmethod
    def from_manifest(cls, path):
        """Read preserved captures without modifying their files or timestamps."""
        path=Path(path)
        data=json.loads(path.read_text())
        result=object.__new__(cls)
        result.directory=path.parent
        result.fps=data.get('nominal_fps',data.get('fps',30))
        result.frames=data['frames']
        result.ids={f['render_id'] for f in result.frames}
        return result

    def encode(self, output, *, ffmpeg):
        output=Path(output)
        if not self.frames:
            raise ValueError('need captured frames')
        if output.exists():
            raise ValueError('refusing to overwrite recording')
        if type(self.fps) is not int or self.fps<=0:
            raise ValueError('positive nominal FPS required for final display duration')
        stamps=[f['simulation_time'] for f in self.frames]
        if (any(not math.isfinite(s) or s<0 for s in stamps)
                or any(b<=a for a,b in zip(stamps,stamps[1:]))
                or len({f['render_id'] for f in self.frames})!=len(self.frames)
                or any(not isinstance(f['render_id'],str) or not f['render_id'] for f in self.frames)):
            raise ValueError('unique captures and strictly increasing acquisition timestamps required')
        pts=[round((s-stamps[0])*1000000) for s in stamps]
        if any(b<=a for a,b in zip(pts,pts[1:])):
            raise ValueError('captures cannot be distinguished at microsecond video timebase')
        lines=['ffconcat version 1.0']
        for i,frame in enumerate(self.frames):
            if not re.fullmatch(r'\d{6}\.png',frame['file']):
                raise ValueError('invalid captured filename')
            path=(self.directory/frame['file']).resolve()
            if hashlib.sha256(path.read_bytes()).hexdigest()!=frame['sha256']:
                raise ValueError('captured frame hash mismatch')
            if '\n' in str(path) or '\r' in str(path):
                raise ValueError('newline in capture path')
            quoted=str(path).replace("'","'\\''")
            lines.extend(["file '"+quoted+"'",'option framerate 1000000'])
            if i+1<len(pts):
                lines.append(f'duration {(pts[i+1]-pts[i])/1000000:.6f}')
        manifest=output.with_suffix('.ffconcat')
        with manifest.open('x') as stream:
            stream.write('\n'.join(lines)+'\n')
        subprocess.run([str(ffmpeg),'-nostdin','-n','-loglevel','error','-f','concat','-safe','0',
            '-i',str(manifest),'-an','-c:v','libx264','-pix_fmt','yuv420p','-bf','0',
            '-fps_mode','passthrough','-enc_time_base','1:1000000',
            '-video_track_timescale','1000000','-crf','20','-movflags','+faststart',str(output)],
            check=True,timeout=120)
        if not output.is_file() or output.stat().st_size==0:
            raise RuntimeError('encoder produced no video')
        decoded=subprocess.run([str(ffmpeg),'-nostdin','-v','info','-i',str(output),
            '-vf','showinfo','-an','-fps_mode','passthrough','-f','null','-'],
            capture_output=True,text=True,check=True,timeout=120)
        basis=re.search(r'config in time_base:\s*(\d+)/(\d+)',decoded.stderr)
        if basis is None:
            raise RuntimeError('decoder did not report presentation timebase')
        unit=int(basis[1])/int(basis[2])
        times=[int(s)*unit for s in re.findall(r'\bn:\s*\d+\s+pts:\s*(-?\d+)\s+pts_time:',decoded.stderr)]
        if len(times)!=len(pts) or any(abs(t-p/1000000)>2e-6 for t,p in zip(times,pts)):
            raise RuntimeError('decoded video frame count or acquisition intervals differ')
        result=dict(file=str(output),sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                    frames=len(self.frames),nominal_fps=self.fps,codec='H.264',timing='VFR',
                    presentation_timebase_seconds=1e-6,decoded_presentation_times=times,
                    last_frame_duration='one video timebase tick; no later acquisition inferred',
                    first_simulation_time=self.frames[0]['simulation_time'],
                    last_simulation_time=self.frames[-1]['simulation_time'])
        output.with_suffix('.timing.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
