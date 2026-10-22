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
    def __init__(self, directory, *, fps=30, caption=None):
        if type(fps) is not int or fps <= 0:
            raise ValueError('positive integer recording FPS required')
        self.directory=Path(directory)
        self.directory.mkdir(parents=True,exist_ok=False)
        self.fps,self.frames=fps,[]
        self.ids=set()
        # Optional run caption (a start-phase shortcut bench run): persisted in frames.json so any
        # later encode -- including a re-encode from the manifest -- burns it into the walkthrough.
        # None => frames.json and the encode are byte-identical.
        self.caption=caption

    def capture(self, frame):
        """Record a newly acquired camera frame; skip only an unchanged cache.

        Physics ticks can outnumber sensor callbacks. Never turn a cached sensor
        frame into a new acquisition by assigning the current physics time.
        """
        import numpy as np
        identity=frame.get('native_rendering_frame',frame['rendering_frame'])
        render_id=json.dumps(identity,sort_keys=True,separators=(',',':'))
        stamp=float(frame['rendering_time'])
        if self.frames and self.frames[-1]['render_id']==render_id:
            if stamp!=self.frames[-1]['simulation_time']:
                raise ValueError('cached camera identity changed acquisition timestamp')
            return False
        rgb=np.asarray(frame['rgb'])
        # The render product can return an empty 1-D buffer on the first ticks before the
        # camera has produced a real RGBA image. Skip until a real frame arrives rather
        # than crash on the HxWx3 slice; this never fabricates or repeats a frame.
        if rgb.ndim!=3 or rgb.shape[2]<3:
            return False
        self.add(rgb[:,:,:3],render_id=render_id,simulation_time=stamp)
        return True

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
        manifest=dict(nominal_fps=self.fps,timing='variable acquisition intervals',frames=self.frames)
        if self.caption:
            manifest['caption']=self.caption
        (self.directory/'frames.json').write_text(json.dumps(manifest,indent=2)+'\n')

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
        result.caption=data.get('caption')
        return result

    def encode(self, output, *, ffmpeg, caption=None):
        """Encode the captures to a VFR H.264 ``output``. ``caption`` (default: the recorder's own) is
        burned in as a full-length top banner + set as the title/comment metadata and written beside
        the video as ``<output>.caption.srt``; if the burn-in filter is unavailable the video is still
        encoded uncaptioned (metadata + sidecar kept) and the timing record says so."""
        output=Path(output)
        caption=caption if caption is not None else getattr(self,'caption',None)
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
        command=[str(ffmpeg),'-nostdin','-n','-loglevel','error','-f','concat','-safe','0',
            '-i',str(manifest),'-an','-c:v','libx264','-pix_fmt','yuv420p','-bf','0',
            '-fps_mode','passthrough','-enc_time_base','1:1000000',
            '-video_track_timescale','1000000','-crf','20','-movflags','+faststart',str(output)]
        burned=caption_error=None
        if caption:
            command[-1:-1]=['-metadata',f'title={caption}','-metadata',f'comment={caption}']
            srt=output.with_suffix('.caption.srt')
            srt.write_text('1\n00:00:00,000 --> 99:59:59,000\n'+' '.join(str(caption).split())+'\n')
            burned=False
            if re.fullmatch(r'[\w./ -]+',str(srt.resolve())):
                filt=(f"subtitles=filename='{srt.resolve()}':force_style='Alignment=6,Fontsize=11,"
                      "BorderStyle=3,OutlineColour=&H80000000,PrimaryColour=&H0000FFFF,MarginV=6'")
                try:
                    subprocess.run(command[:-1]+['-vf',filt,command[-1]],check=True,timeout=300,
                                   capture_output=True,text=True)
                    burned=True
                except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as exc:
                    caption_error=(getattr(exc,'stderr','') or str(exc))[-400:]
                    output.unlink(missing_ok=True)
            else:
                caption_error='caption sidecar path not filter-safe; metadata-only caption'
        if not burned:
            subprocess.run(command,check=True,timeout=120)
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
        if caption:
            result.update(caption=caption,caption_burned=bool(burned),caption_error=caption_error)
        output.with_suffix('.timing.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
