"""Persist actual rendered RGB frames with timestamps, then encode CPU H.264.

The caller supplies camera render IDs and simulation timestamps. No synthetic
frames, repeats or interpolation are generated to fill missing sensor output.
"""
import hashlib
import json
import math
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
        if self.frames and not math.isclose(simulation_time-self.frames[-1]['simulation_time'],
                                            1/self.fps,abs_tol=1e-4):
            raise ValueError('missing/irregular render cadence; refusing fabricated CFR frames')
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
        (self.directory/'frames.json').write_text(json.dumps(dict(fps=self.fps,frames=self.frames),indent=2)+'\n')

    def encode(self, output, *, ffmpeg):
        output=Path(output)
        if len(self.frames)<2:
            raise ValueError('need at least two captured frames')
        if output.exists():
            raise ValueError('refusing to overwrite recording')
        subprocess.run([str(ffmpeg),'-nostdin','-n','-loglevel','error','-framerate',str(self.fps),
            '-i',str(self.directory/'%06d.png'),'-an','-c:v','libx264','-pix_fmt','yuv420p',
            '-crf','20','-movflags','+faststart',str(output)],check=True,timeout=120)
        if not output.is_file() or output.stat().st_size==0:
            raise RuntimeError('encoder produced no video')
        return dict(file=str(output),sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                    frames=len(self.frames),fps=self.fps,codec='H.264',
                    first_simulation_time=self.frames[0]['simulation_time'],
                    last_simulation_time=self.frames[-1]['simulation_time'])
