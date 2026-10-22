"""Compile ONE continuous demo video out of per-stage segments drawn from DIFFERENT runs.

The per-run demo cut (:mod:`twin.demo.walkthrough_cut`) tells the story of a single run. This
module tells the seven-stage story by taking each stage from whichever run performed it best and
stitching the pieces into one film whose decision/reason-tree panel (the demo-004/005 panel,
:mod:`twin.demo.panel`) runs CONTINUOUSLY across the seams: stage numbers climb 1..7, the mission
clock is cumulative, earlier stages read ``done`` and later ones ``idle`` no matter which source a
frame came from.

CPU only, no GPU, no simulator import; every source run dir is read-only. Two source kinds, both
reduced to the SAME per-frame panel ``entry`` (``{tree, header, chips}``) that ``panel.render_panel``
draws, plus a scene image:

* ``real``   -- a ``run_mobile_restock.py`` run dir (``mission-frames/`` + ``bt_events.jsonl`` +
  ``nav-trace.jsonl``). The panel entry is rebuilt per frame by :mod:`twin.demo.bt_timeline`; the
  scene is the recorded 960x720 overhead frame, under a freshly drawn caption band.
* ``choreo`` -- a choreographed demo dir (``decision_log.json`` + the composited ``walkthrough.mp4``;
  the raw scene-frames are not kept). The panel entry is the log's own per-frame payload; the scene
  is the left region cropped out of the composited mp4 (its caption band is already baked in).

Honesty: the panel ladder is rewritten only to reflect GLOBAL progress (a stage shown earlier in the
compile reads done, later reads idle); the active stage's own node, the measured chips and the scene
pixels are passed through from the source untouched. Which run supplies which stage is recorded in the
emitted stage-to-source table so the provenance of every second is auditable.
"""
from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import bt_timeline as BT
from . import panel as PANEL
from . import run_evidence as E
from . import walkthrough_cut as W

OUT_W, OUT_H = 1920, 1080
PANEL_W = 480
REGION_W = OUT_W - PANEL_W           # 1440: the scene region left of the panel
FPS = 15
XFADE_FRAMES = 8                     # dissolve length at a source boundary

# Scene-band palette (matches walkthrough_cut's band).
BAND_BG = (12, 16, 24)
FG = (235, 238, 242)
ACCENT = (90, 170, 255)
AMBER = (255, 190, 60)


# ---------------- spec ----------------
@dataclass
class Segment:
    """One stage of the compile: a contiguous frame range of one source run.

    ``kind`` is ``real`` | ``choreo`` (auto-detected from the run dir when omitted). ``frame_start`` /
    ``frame_end`` are inclusive indices into that source's frame sequence (mission-frames for real,
    decision_log frames for choreo). ``min_seconds`` holds the last frame so a sub-second stage still
    earns a legible beat. ``labels`` are honest on-screen provenance tags for the caption band.
    """
    stage: str
    source: str                       # run dir name (under mission_root) or absolute path
    frame_start: int
    frame_end: int
    kind: str | None = None
    min_seconds: float = 0.0
    caption: str | None = None
    labels: list = field(default_factory=list)


@dataclass
class CompileSpec:
    mission_root: Path
    segments: list
    fps: int = FPS
    xfade_frames: int = XFADE_FRAMES
    title: str = 'demo-006'

    @classmethod
    def from_json(cls, path):
        data = json.loads(Path(path).read_text())
        root = Path(data['mission_root'])
        segs = [Segment(**s) for s in data['segments']]
        return cls(mission_root=root, segments=segs, fps=int(data.get('fps', FPS)),
                   xfade_frames=int(data.get('xfade_frames', XFADE_FRAMES)),
                   title=str(data.get('title', 'demo-006')))


def _run_dir(spec, source):
    p = Path(source)
    return p if p.is_absolute() else spec.mission_root / source


def detect_kind(run_dir):
    run_dir = Path(run_dir)
    if (run_dir / 'decision_log.json').is_file() and not (run_dir / 'mission-frames' / 'frames.json').is_file():
        return 'choreo'
    return 'real'


def load_run_lenient(run_dir):
    """Load a real run dir, or a CARVED single-stage segment that has mission-frames + bt_events +
    nav-trace but NO mission.json (a helper's extracted clip). A full run dir goes through
    ``E.load_run``; a carved segment is assembled into an equivalent ``E.Run`` with an empty mission
    event log (its panel comes from the bt_events it does carry)."""
    run_dir = Path(run_dir).resolve()
    if (run_dir / 'mission.json').is_file():
        return E.load_run(run_dir)
    trace = E.read_trace(run_dir / 'nav-trace.jsonl')
    offset, source = 0.0, 'none (assumed 0)'
    worlds = [r['world_t'] - r['t'] for r in trace if isinstance(r.get('world_t'), (int, float))]
    if worlds:
        worlds.sort()
        offset, source = worlds[len(worlds) // 2], 'nav-trace world_t'
    frames = []
    manifest = run_dir / 'mission-frames' / 'frames.json'
    if manifest.is_file():
        data = json.loads(manifest.read_text())
        for i, f in enumerate(data.get('frames', [])):
            st = float(f['simulation_time'])
            frames.append(E.Frame(i, manifest.parent / f['file'], st, st - offset))
    return E.Run(run_dir=run_dir, name=run_dir.name, mission={'state': 'running', 'reason': '', 'events': []},
                 events=[], trace=trace, frames=frames,
                 bt_events=E._read_jsonl(run_dir / 'bt_events.jsonl'),
                 clock_offset=float(offset), clock_source=source, scenario=E._find_scenario(run_dir), spec={})


# ---------------- continuity: rewrite one panel tree for global progress ----------------
def rewrite_tree(tree, stage, *, final=False):
    """Make the seven-stage ladder read GLOBALLY: every stage before ``stage`` is done, every stage
    after it idle, and ``stage`` itself is at least running (so the active stage is never blank even
    when the source run recorded no step ticks for it). The active stage's node is otherwise passed
    through. Mutates and returns ``tree``."""
    order = {s: i for i, s in enumerate(E.STAGES)}
    k = order[stage]
    by_name = {n.get('name', ''): n for n in tree.get('children', [])}
    for s, j in order.items():
        node = by_name.get(f'stage:{s}')
        if node is None:
            continue
        if j < k:
            node['status'] = 'success'
            node['feedback'] = 'done'
            for st in node.get('children', []):
                st['status'] = 'success'
        elif j > k:
            node['status'] = 'idle'
            node['feedback'] = ''
            for st in node.get('children', []):
                st['status'] = 'idle'
                st['feedback'] = ''
        else:
            if node.get('status') in (None, 'idle', 'shortcut'):
                node['status'] = 'running'
                node['feedback'] = 'running'
                steps = node.get('children', [])
                if steps and all(st.get('status') == 'idle' for st in steps):
                    steps[0]['status'] = 'running'
    tree['status'] = 'success' if final else 'running'
    return tree


def rewrite_entry(entry, stage, *, global_frame, clock, final=False):
    """Rewrite a source panel entry for its place in the compile: continuous ladder, cumulative clock,
    global frame counter, the mission read as running until the final frame."""
    rewrite_tree(entry.setdefault('tree', {}), stage, final=final)
    hdr = entry.setdefault('header', {})
    hdr['stage_title'] = BT.STAGE_TITLES.get(stage, stage)
    hdr['clock'] = clock
    hdr['frame'] = global_frame
    hdr['mission_state'] = 'complete' if final else 'running'
    return entry


# ---------------- scene compositing ----------------
def composite_frame(scene_img, panel_img):
    """Scene scaled into the left region (letterboxed to aspect), panel pasted right -> OUT_W x OUT_H.
    Inlined from twin.demo.compositor to avoid importing the GPU-side Recorder."""
    from PIL import Image
    canvas = Image.new('RGB', (OUT_W, OUT_H), PANEL.BG)
    sw, sh = scene_img.size
    scale = min(REGION_W / sw, OUT_H / sh)
    nw, nh = max(2, int(sw * scale)), max(2, int(sh * scale))
    scene = scene_img.resize((nw, nh), Image.LANCZOS)
    canvas.paste(scene, ((REGION_W - nw) // 2, (OUT_H - nh) // 2))
    if panel_img.size != (PANEL_W, OUT_H):
        panel_img = panel_img.resize((PANEL_W, OUT_H), Image.LANCZOS)
    canvas.paste(panel_img, (REGION_W, 0))
    return canvas


def draw_band(scene_img, stage, *, clock, coupled, labels):
    """Draw the caption band (stage idx/7 + title + caption + any provenance labels) and a sim clock
    onto a 960x720 recorded scene, matching the demo cut's band so real and choreo scenes read alike."""
    from PIL import Image, ImageDraw
    w, h = scene_img.size
    canvas = scene_img.convert('RGB').copy()
    band_h = 118
    band = Image.new('RGBA', (w, band_h), BAND_BG + (205,))
    d = ImageDraw.Draw(band)
    idx = E.STAGES.index(stage) + 1
    d.text((18, 10), f'{idx}/7', font=W.font(26, True), fill=ACCENT + (255,))
    d.text((80, 10), E.TITLES[stage].upper(), font=W.font(26, True), fill=FG + (255,))
    caption = E.CAPTIONS[stage]
    y = W._text_block(d, (20, 48), caption, W.font(17), FG + (255,), w - 40, 3)
    if labels:
        text = '; '.join(str(x) for x in labels)
        W._text_block(d, (20, y + 2), text, W.font(14, True), AMBER + (255,), w - 40, 2)
    base = canvas.crop((0, h - band_h, w, h)).convert('RGBA')
    canvas.paste(Image.alpha_composite(base, band).convert('RGB'), (0, h - band_h))
    d2 = ImageDraw.Draw(canvas)
    clk = f'sim t = {clock:6.1f} s   ·   {"coupled" if coupled else "uncoupled"}'
    cw = d2.textlength(clk, font=W.font(15))
    d2.rounded_rectangle((w - 26 - cw, 10, w - 10, 38), radius=6, fill=BAND_BG)
    d2.text((w - 18 - cw, 15), clk, font=W.font(15), fill=FG)
    return canvas


# ---------------- choreo frame extraction (from the composited mp4) ----------------
def extract_mp4_frames(mp4, frame_start, frame_end, out_dir, *, ffmpeg):
    """Decode composited frames [frame_start, frame_end] of ``mp4`` to PNGs in ``out_dir`` (0-indexed,
    inclusive). Returns the list of PNG paths in order."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = frame_end - frame_start + 1
    cmd = [ffmpeg, '-nostdin', '-loglevel', 'error', '-i', str(mp4),
           '-vf', f"select='between(n\\,{frame_start}\\,{frame_end})'", '-vsync', '0',
           '-frames:v', str(n), str(out_dir / 'f_%06d.png')]
    subprocess.run(cmd, check=True)
    return sorted(out_dir.glob('f_*.png'))


# ---------------- per-segment frame generator ----------------
class Compiler:
    def __init__(self, spec, *, ffmpeg, tmp_dir):
        self.spec = spec
        self.ffmpeg = ffmpeg
        self.tmp = Path(tmp_dir)
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.global_frame = 0
        self.table = []                         # stage-to-source rows

    def _plan_real(self, seg, run_dir):
        """Resolve a real segment to (render_one, count). render_one(i) -> (scene_960, entry, coupled)."""
        from PIL import Image
        run = load_run_lenient(run_dir)
        timeline = BT.BtTimeline.from_run(run)
        model = E.cart_model(run)
        step_reasons = {}
        BT.seed_reasons(timeline, run, model, step_reasons)
        frames = run.frames[seg.frame_start:seg.frame_end + 1]
        if not frames:
            raise ValueError(f'{run_dir.name}: no frames in [{seg.frame_start},{seg.frame_end}]')
        hold = max(0, round(seg.min_seconds * self.spec.fps) - len(frames))
        seq = list(frames) + [frames[-1]] * hold

        def render_one(i):
            fr = seq[i]
            entry = BT.panel_entry(timeline, run, model, t=fr.t, header_stage=seg.stage,
                                   frame_index=0, step_reasons=step_reasons)
            row = run.row_at(fr.t)
            return Image.open(fr.file).convert('RGB'), entry, bool(row and row['coupled'])
        return render_one, len(seq), hold, 'real'

    def _plan_choreo(self, seg, run_dir):
        """Resolve a choreo segment to (render_one, count). Scene cropped from the composited mp4."""
        from PIL import Image
        log = json.loads((run_dir / 'decision_log.json').read_text())['frames']
        mp4 = run_dir / 'walkthrough.mp4'
        seg_tmp = self.tmp / f'{run_dir.name}_{seg.stage}_{seg.frame_start}_{seg.frame_end}'
        pngs = extract_mp4_frames(mp4, seg.frame_start, seg.frame_end, seg_tmp, ffmpeg=self.ffmpeg)
        entries = log[seg.frame_start:seg.frame_end + 1]
        if len(pngs) != len(entries):
            raise ValueError(f'{run_dir.name}: {len(pngs)} decoded frames vs {len(entries)} log entries '
                             f'for [{seg.frame_start},{seg.frame_end}]')
        hold = max(0, round(seg.min_seconds * self.spec.fps) - len(entries))
        pairs = list(zip(pngs, entries)) + [None] * hold

        def render_one(i):
            png, logentry = pairs[i] if pairs[i] is not None else pairs[len(pairs) - hold - 1]
            entry = json.loads(json.dumps(logentry['panel']))     # deep copy (rewritten in place)
            with Image.open(png) as im:
                scene = im.convert('RGB').crop((0, 0, REGION_W, OUT_H))
            coupled = any(v == 'yes' for (lbl, v) in entry.get('chips', []) if lbl == 'coupled')
            return scene, entry, coupled
        return render_one, len(pairs), hold, 'choreo'

    def segment_arrays(self, seg, *, is_last_segment):
        """Yield finished OUT_WxOUT_H uint8 frames for one segment, rewriting each panel for continuity."""
        run_dir = _run_dir(self.spec, seg.source)
        kind = seg.kind or detect_kind(run_dir)
        render_one, count, hold, kind = (self._plan_choreo(seg, run_dir) if kind == 'choreo'
                                         else self._plan_real(seg, run_dir))
        for i in range(count):
            scene, entry, coupled = render_one(i)
            self.global_frame += 1
            clock = self.global_frame / self.spec.fps
            final = is_last_segment and i == count - 1
            rewrite_entry(entry, seg.stage, global_frame=self.global_frame, clock=clock, final=final)
            if kind == 'real':
                scene = draw_band(scene, seg.stage, clock=clock, coupled=coupled, labels=seg.labels)
            panel_img = PANEL.render_panel(entry, size=(PANEL_W, OUT_H))
            yield np.asarray(composite_frame(scene, panel_img), dtype=np.uint8)
        self.table.append({'stage': seg.stage, 'source': run_dir.name, 'kind': kind,
                           'frame_start': seg.frame_start, 'frame_end': seg.frame_end,
                           'held_frames': hold, 'emitted': count, 'labels': list(seg.labels)})


# ---------------- crossfade-aware streaming encode ----------------
def _blend(a, b, alpha):
    return (a.astype(np.float32) * (1 - alpha) + b.astype(np.float32) * alpha).astype(np.uint8)


def render(spec, out, *, ffmpeg, tmp_dir):
    """Encode the compile to ``out`` (refuses to overwrite) and write ``out.with_suffix('.table.json')``
    (the stage-to-source provenance table). Returns a summary dict."""
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'refusing to overwrite {out}')
    out.parent.mkdir(parents=True, exist_ok=True)
    comp = Compiler(spec, ffmpeg=ffmpeg, tmp_dir=tmp_dir)
    cmd = [ffmpeg, '-nostdin', '-n', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', f'{OUT_W}x{OUT_H}', '-r', str(spec.fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
           '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    emitted = 0
    N = max(0, spec.xfade_frames)
    prev_tail = []                 # held last N frames of the previous segment
    prev_source = None
    try:
        for si, seg in enumerate(spec.segments):
            is_last = si == len(spec.segments) - 1
            source = str(seg.source)
            xfade = N > 0 and prev_source is not None and source != prev_source
            gen = comp.segment_arrays(seg, is_last_segment=is_last)
            ring = []
            if xfade:
                head = []
                for _ in range(N):
                    try:
                        head.append(next(gen))
                    except StopIteration:
                        break
                m = min(len(prev_tail), len(head))
                for i in range(m):
                    alpha = (i + 1) / (m + 1)
                    proc.stdin.write(_blend(prev_tail[i], head[i], alpha).tobytes())
                    emitted += 1
                for i in range(m, len(head)):          # head longer than the held tail: pass through
                    proc.stdin.write(head[i].tobytes())
                    emitted += 1
                prev_tail = []
            else:
                for f in prev_tail:
                    proc.stdin.write(f.tobytes())
                    emitted += 1
                prev_tail = []
            for f in gen:
                ring.append(f)
                if len(ring) > N:
                    proc.stdin.write(ring.pop(0).tobytes())
                    emitted += 1
            prev_tail = ring
            prev_source = source
        for f in prev_tail:
            proc.stdin.write(f.tobytes())
            emitted += 1
        proc.stdin.close()
        if proc.wait(timeout=1200) != 0:
            raise RuntimeError(f'ffmpeg exited {proc.returncode}')
    except BaseException:
        proc.kill()
        raise
    if not out.is_file() or out.stat().st_size == 0:
        raise RuntimeError('encoder produced no video')
    summary = {'video': str(out), 'size': [OUT_W, OUT_H], 'fps': spec.fps, 'output_frames': emitted,
               'duration_s': round(emitted / spec.fps, 2), 'xfade_frames': N,
               'segments': comp.table}
    out.with_suffix('.table.json').write_text(json.dumps(summary, indent=1, default=str) + '\n')
    return summary
