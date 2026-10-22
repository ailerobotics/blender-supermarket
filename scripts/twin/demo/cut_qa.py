"""Frame-exact CLEAN cut ranges for the compiled demo video (sim-video-cut-qa-v1).

The compile (:mod:`twin.demo.stitch`) takes each stage from whichever source performed it best.
This module decides, frame by frame, how much of each candidate source+stage is SAFE to show:
it scans the poses (``nav-trace.jsonl`` / ``bt_events.jsonl`` for real runs, the per-frame
``reads`` + panel tree of ``decision_log.json`` for choreographed demos) and the pixels (PNG frames,
or the composited ``walkthrough.mp4`` for choreo) for visual faults, merges the faults a human
verified by eye (``visual_faults.json``: the ones no recorded pose can show, e.g. carton TILT --
the traces carry carton yaw only), and cuts:

* an ENDING fault (tilt, mid-way unlatch, position jump, carton floating/off the cart, whiteout
  wall clip, black frame) at frame ``f`` ends the usable range ``margin`` frames before it
  (0.5 s of OUTPUT video = 8 frames at 15 fps -- each source frame is one output frame); a fault
  window at the very start of a stage moves the start past it instead;
* a STALL (no visible motion for longer than 2 s of output video) is not wrong, only dead air:
  its interior is spliced out, keeping 0.5 s on each side, and the splice is verified seamless
  (mean |delta| between the two joined frames below ``SPLICE_MAX``).

CPU only; every source dir is read-only. Output schema ``cut_ranges/v1`` (see ``build``).
"""
from __future__ import annotations

import bisect
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

FPS_OUT = 15
MARGIN_S = 0.5
MARGIN_FRAMES = math.ceil(MARGIN_S * FPS_OUT)        # 8
STALL_S = 2.0
STALL_FRAMES = int(STALL_S * FPS_OUT)                # 30: a static run LONGER than this is a stall
STATIC_DIFF = 1.0           # mean |delta| (0..255 grey, 240x180) to the stall's FIRST frame below which the
                            # view is still static (cumulative, so a slow lift is motion, not a stall)
SPLICE_MAX = 1.5            # a stall splice is seamless when the joined frames differ by less than this
JUMP_RATIO, JUMP_ABS = 4.0, 6.0   # image jump: delta > JUMP_RATIO x local median AND > JUMP_ABS
WHITEOUT, BLACK = 238.0, 15.0     # frame mean grey (camera inside a wall / dead frame)
BASE_VMAX, CART_VMAX, CARTON_VMAX = 2.0, 2.0, 3.0     # m/s: faster than this between frames = a jump
JUMP_MIN_M = 0.15
YAW_WMAX_DEG, YAW_MIN_DEG = 120.0, 25.0
ON_CART_TOL_M = 0.10        # carton riding the cart may not drift more than this relative to it
HELD_TOL_M = 0.12           # choreo: gripper pad within this of the carton centre = held
RIDE_TOL_M = 0.03           # choreo: carton shifting less than this per frame in the cart frame = riding it

ENDING = ('carton_tilt', 'unlatch', 'position_jump', 'carton_floating', 'carton_off_cart', 'whiteout',
          'black_frame', 'image_jump', 'clip', 'story')
# A TERMINAL fault leaves the scene wrong for the rest of the stage (the carton stays tilted, the robot
# drives on alone after the unlatch): nothing after it in that stage may be shown.
TERMINAL = ('carton_tilt', 'unlatch')
STAGES = ('collect', 'couple', 'outbound', 'park', 'transfer', 'verify', 'return')


# ---------------- sources ----------------
def _jsonl(path):
    out = []
    path = Path(path)
    if path.is_file():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass                      # torn last line of a killed run
    return out


def detect_kind(src):
    src = Path(src)
    if (src / 'mission-frames' / 'frames.json').is_file():
        return 'real'
    if (src / 'decision_log.json').is_file():
        return 'choreo'
    return None


class Source:
    """Uniform per-frame view of a source: ``times``, ``stage``, ``pose`` rows, an image loader."""

    def __init__(self, path):
        self.path = Path(path)
        self.name = self.path.name
        self.kind = detect_kind(self.path)
        if self.kind == 'real':
            self._load_real()
        elif self.kind == 'choreo':
            self._load_choreo()
        else:
            raise FileNotFoundError(f'{self.path}: no mission-frames/frames.json nor decision_log.json')

    # real: run_mobile_restock.py run dir (or an extract_stage_segment carve of one)
    def _load_real(self):
        data = json.loads((self.path / 'mission-frames' / 'frames.json').read_text())
        self.frames_meta = data.get('frames', [])
        self.files = [self.path / 'mission-frames' / f['file'] for f in self.frames_meta]
        trace = [r for r in _jsonl(self.path / 'nav-trace.jsonl') if 't' in r and 'x' in r]
        trace.sort(key=lambda r: r['t'])
        worlds = sorted(r['world_t'] - r['t'] for r in trace if isinstance(r.get('world_t'), (int, float)))
        self.clock_offset = worlds[len(worlds) // 2] if worlds else 0.0
        self.times = [float(f['simulation_time']) - self.clock_offset for f in self.frames_meta]
        self.trace = trace
        self._tt = [r['t'] for r in trace]
        self.bt = _jsonl(self.path / 'bt_events.jsonl')
        mission = self.path / 'mission.json'
        self.mission = json.loads(mission.read_text()) if mission.is_file() else {}
        first = next((r.get('action') for r in trace if r.get('action') in STAGES), None)
        self.stage = []
        self.rows = []
        for t in self.times:
            row = self.row_at(t)
            self.rows.append(row)
            act = row.get('action') if row else None
            self.stage.append(act if act in STAGES else first)

    def row_at(self, t):
        if not self.trace:
            return None
        i = bisect.bisect_right(self._tt, t) - 1
        return self.trace[max(0, min(i, len(self.trace) - 1))]

    def frame_at(self, t):
        """First frame whose mission time is >= t (the first frame that can SHOW an event at t)."""
        return min(bisect.bisect_left(self.times, t), len(self.times) - 1)

    # choreo: a choreographed demo dir (decision_log.json + composited walkthrough.mp4)
    def _load_choreo(self):
        data = json.loads((self.path / 'decision_log.json').read_text())
        self.log = data['frames']
        self.times = [i / FPS_OUT for i in range(len(self.log))]
        self.clock_offset = 0.0
        self.mission = {}
        self.stage = []
        last = None
        for f in self.log:
            st = None
            for node in f['panel']['tree'].get('children', []):
                if node.get('status') == 'running':
                    st = node['name'].split(':', 1)[1]
                    break
            st = st or last
            self.stage.append(st)
            last = st
        self.video = self.path / 'walkthrough.mp4'

    def __len__(self):
        return len(self.times)

    def stage_spans(self):
        """{stage: (first, last)} frame span, in source order (a stage's frames are contiguous)."""
        spans = {}
        for i, st in enumerate(self.stage):
            if st in STAGES:
                lo, hi = spans.get(st, (i, i))
                spans[st] = (min(lo, i), max(hi, i))
        return spans

    # ---------------- pixels ----------------
    def grey_thumbs(self, size=(240, 180)):
        """Per-frame 240x180 float32 grey thumbnails of the SCENE (choreo: left 1440 px of the mp4)."""
        import cv2
        if self.kind == 'real':
            def load(p):
                im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                return cv2.resize(im, size, interpolation=cv2.INTER_AREA).astype(np.float32)
            with ThreadPoolExecutor(16) as ex:
                return list(ex.map(load, self.files))
        cap = cv2.VideoCapture(str(self.video))
        out = []
        while len(out) < len(self.log):
            ok, im = cap.read()
            if not ok:
                break
            scene = im[:, :1440] if im.shape[1] >= 1440 else im
            g = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
            out.append(cv2.resize(g, size, interpolation=cv2.INTER_AREA).astype(np.float32))
        cap.release()
        return out


# ---------------- detectors ----------------
def _fault(kind, frame, end=None, detail='', channel='pose'):
    return {'kind': kind, 'frame': int(frame), 'end': int(frame if end is None else end),
            'detail': detail, 'channel': channel}


def _real_poses(src):
    """Per frame: base (x, y, yaw_deg), cart (x, y, yaw_deg), carton (x, y, z, yaw_deg), coupled."""
    out = []
    for row in src.rows:
        if row is None:
            out.append(None)
            continue
        cart = row.get('cart')
        carton = row.get('carton')
        out.append({'base': (row['x'], row['y'], row.get('yaw_deg', 0.0)),
                    'cart': tuple(cart[:3]) if cart else None,
                    'carton': tuple(carton[:4]) if carton else None,
                    'coupled': bool(row.get('coupled'))})
    return out


def _choreo_poses(src):
    out = []
    for f in src.log:
        r = f.get('reads') or {}
        bxy, cxyz = r.get('base_xy'), r.get('carton_xyz')
        pad = r.get('pad_xyz')
        held = None
        if cxyz and pad:
            held = math.dist(cxyz, pad) < HELD_TOL_M     # the suction pad sits on the carton top when held
        out.append({'base': (bxy[0], bxy[1], math.degrees(r.get('base_yaw', 0.0))) if bxy else None,
                    'cart': ((r['hitch_xy'][0], r['hitch_xy'][1], math.degrees(r.get('cart_yaw', 0.0)))
                             if r.get('hitch_xy') else None),
                    'carton': (cxyz[0], cxyz[1], cxyz[2], 0.0) if cxyz else None,
                    'held': held})
    return out


def _dyaw(a, b):
    return abs((b - a + 180.0) % 360.0 - 180.0)


def pose_faults(src):
    """Position jumps (base / cart / carton), mid-way unlatch, carton off the cart / floating."""
    poses = _real_poses(src) if src.kind == 'real' else _choreo_poses(src)
    faults = []
    for i in range(1, len(poses)):
        a, b = poses[i - 1], poses[i]
        if not a or not b:
            continue
        dt = max(src.times[i] - src.times[i - 1], 1.0 / 30.0)
        for key, vmax in (('base', BASE_VMAX), ('cart', CART_VMAX), ('carton', CARTON_VMAX)):
            pa, pb = a.get(key), b.get(key)
            if not pa or not pb:
                continue
            d = math.dist(pa[:2] if key != 'carton' else pa[:3], pb[:2] if key != 'carton' else pb[:3])
            if d > max(JUMP_MIN_M, vmax * dt):
                faults.append(_fault('position_jump', i, detail=f'{key} moved {d:.2f} m in one frame '
                                                                f'({dt:.2f} s): teleport'))
            if key != 'carton':
                dy = _dyaw(pa[2], pb[2])
                if dy > max(YAW_MIN_DEG, YAW_WMAX_DEG * dt):
                    faults.append(_fault('position_jump', i, detail=f'{key} yaw jumped {dy:.0f} deg in one frame'))
    # mid-way unlatch: coupling drops after being held (real), or the panel runs an uncouple step (choreo)
    if src.kind == 'real':
        held = False
        for i, p in enumerate(poses):
            if p is None:
                continue
            if p['coupled']:
                held = True
            elif held:
                faults.append(_fault('unlatch', i, detail='measured coupling drops: the robot unlatches '
                                                          'the rollwagen and drives on alone'))
                break
        for ev in src.bt:
            d = ev.get('data') or {}
            if ev.get('type') == 'stage_step' and 'uncouple' in str(d.get('label', '')):
                t = float(d.get('sim_time', ev.get('t', 0.0)))
                faults.append(_fault('unlatch', src.frame_at(t), detail=f"bt step '{d.get('label')}' at "
                                                                        f'mission t={t:.2f} s'))
    else:
        for i, f in enumerate(src.log):
            hit = False
            for node in f['panel']['tree'].get('children', []):
                for st in node.get('children', []):
                    if 'uncouple' in st.get('name', '') and st.get('status') in ('running', 'success'):
                        hit = True
            if hit:
                faults.append(_fault('unlatch', i, detail='panel runs the uncouple step'))
                break
    # scripted transfer (mission transfer_mode=scripted): the carton follows a scripted path while the
    # arm holds still, so ANY carton motion in the transfer is a carton floating free of the gripper
    if src.kind == 'real' and src.mission.get('transfer_mode') == 'scripted':
        idx = [i for i, st in enumerate(src.stage) if st == 'transfer' and poses[i] and poses[i].get('carton')]
        if idx:
            rest, moving, start = poses[idx[0]]['carton'], None, None
            for i in idx[1:]:
                c = poses[i]['carton']
                if start is None and math.dist(c[:3], rest[:3]) > 0.005:
                    start = i
                if start is not None:
                    prev = poses[i - 1]['carton'] if poses[i - 1] else c
                    if math.dist(c[:3], prev[:3]) > 0.002:
                        moving = i
            if start is not None:
                faults.append(_fault('carton_floating', start, end=moving or start,
                                     detail='SCRIPTED transfer (mission transfer_mode=scripted): the carton slides '
                                            'out of the cage and swings to the target on a scripted path while the '
                                            'arm stays still -- floating, not carried'))
    # carton riding the cart (real couple/outbound/park): may not slide off / lift off it
    if src.kind == 'real':
        ref = None
        for i, p in enumerate(poses):
            st = src.stage[i]
            if not p or not p.get('cart') or not p.get('carton') or st not in ('couple', 'outbound', 'park'):
                ref = None if st not in ('couple', 'outbound', 'park') else ref
                continue
            cx, cy, cyaw = p['cart']
            rel = _to_frame(p['carton'][:2], (cx, cy), cyaw)
            if ref is None:
                ref = (rel, p['carton'][2])
                continue
            drift = math.dist(rel, ref[0])
            dz = abs(p['carton'][2] - ref[1])
            if drift > ON_CART_TOL_M or dz > 0.05:
                faults.append(_fault('carton_off_cart', i, detail=f'carton slid {drift:.2f} m / lifted {dz:.2f} m '
                                                                  'relative to the rollwagen'))
                break
    else:
        # choreo: the carton may only MOVE while held by the pad (or riding the cart); motion with the
        # pad away and the carton not riding the hitch frame = floating
        for i in range(1, len(poses)):
            a, b = poses[i - 1], poses[i]
            if not a or not b or not a.get('carton') or not b.get('carton'):
                continue
            moved = math.dist(a['carton'][:3], b['carton'][:3])
            if moved < 0.004 or b.get('held') or a.get('held'):
                continue
            if a.get('cart') and b.get('cart'):
                ra = _to_frame(a['carton'][:2], a['cart'][:2], a['cart'][2])
                rb = _to_frame(b['carton'][:2], b['cart'][:2], b['cart'][2])
                if math.dist(ra, rb) < RIDE_TOL_M and abs(a['carton'][2] - b['carton'][2]) < 0.01:
                    continue                  # rides the cart
            faults.append(_fault('carton_floating', i, detail=f'carton moved {moved * 100:.1f} cm with the '
                                                              'gripper pad away and not riding the cart'))
    return faults


def _to_frame(xy, origin, yaw_deg):
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    dx, dy = xy[0] - origin[0], xy[1] - origin[1]
    return (c * dx + s * dy, -s * dx + c * dy)


def image_faults(src, thumbs=None):
    """Stalls, image jumps (teleport / hard cut), whiteout (camera inside a wall) and black frames."""
    thumbs = src.grey_thumbs() if thumbs is None else thumbs
    n = len(thumbs)
    faults, stalls = [], []
    if n == 0:
        return faults, stalls, np.zeros(0)
    diff = np.array([0.0] + [float(np.abs(thumbs[i] - thumbs[i - 1]).mean()) for i in range(1, n)])
    means = np.array([float(t.mean()) for t in thumbs])
    for i in range(n):
        if means[i] > WHITEOUT:
            faults.append(_fault('whiteout', i, detail=f'frame mean {means[i]:.0f}: camera inside geometry '
                                                       '(wall clip)', channel='image'))
        elif means[i] < BLACK:
            faults.append(_fault('black_frame', i, detail=f'frame mean {means[i]:.0f}', channel='image'))
    for i in range(1, n):
        lo, hi = max(1, i - 4), min(n, i + 5)
        local = np.median(np.concatenate([diff[lo:i], diff[i + 1:hi]])) if hi - lo > 1 else 0.0
        if diff[i] > JUMP_ABS and diff[i] > JUMP_RATIO * max(local, 0.25):
            faults.append(_fault('image_jump', i, detail=f'mean |delta| {diff[i]:.1f} vs local {local:.1f}: '
                                                         'hard visual jump', channel='image'))
    s = 0
    for i in range(1, n + 1):
        if i < n and float(np.abs(thumbs[i] - thumbs[s]).mean()) < STATIC_DIFF:
            continue
        if (i - 1) - s + 1 > STALL_FRAMES:
            stalls.append((s, i - 1))
        s = i
    return _merge_runs(faults), stalls, diff


def _merge_runs(faults):
    """Collapse consecutive same-kind frames into one windowed fault."""
    faults = sorted(faults, key=lambda f: (f['kind'], f['frame']))
    out = []
    for f in faults:
        if out and out[-1]['kind'] == f['kind'] and f['frame'] <= out[-1]['end'] + 1 \
                and out[-1]['channel'] == f['channel']:
            out[-1]['end'] = max(out[-1]['end'], f['end'])
        else:
            out.append(dict(f))
    return sorted(out, key=lambda f: f['frame'])


# ---------------- ranges ----------------
def stall_splices(stalls, thumbs, span):
    """Drop each stall's interior (keep MARGIN_FRAMES at both ends); verify the join is seamless."""
    drops = []
    for a, b in stalls:
        a, b = max(a, span[0]), min(b, span[1])
        if b - a + 1 <= STALL_FRAMES:
            continue
        keep_a, keep_b = a + MARGIN_FRAMES - 1, b - MARGIN_FRAMES + 1
        if thumbs is not None:
            while keep_b - keep_a > 1 and float(np.abs(thumbs[keep_a] - thumbs[keep_b]).mean()) > SPLICE_MAX:
                keep_b -= 1                   # pull the join closer until it is seamless
            join = float(np.abs(thumbs[keep_a] - thumbs[keep_b]).mean())
        else:
            join = None
        if keep_b - keep_a > 1:
            drops.append({'start': keep_a + 1, 'end': keep_b - 1, 'stall': [a, b],
                          'join_mean_abs_delta': None if join is None else round(join, 3)})
    return drops


def stage_ranges(span, faults, drops, *, min_frames=FPS_OUT):
    """Clean keep intervals inside ``span`` given ending faults and stall drops.

    Every ending fault window [f, e] (dilated by MARGIN_FRAMES each side) is excluded. The PRIMARY
    keep is the first clean interval of at least ``min_frames`` (1 s); a fault window at the stage start
    pushes it past the fault. Stall drops split the interval into seamless splices."""
    lo, hi = span
    bad = []
    for f in faults:
        if f['kind'] in ENDING and f['end'] >= lo and f['frame'] <= hi:
            bad.append((f['frame'] - MARGIN_FRAMES, f['end'] + MARGIN_FRAMES, f))
    bad.sort(key=lambda x: x[0])
    clean, cur = [], lo
    for a, b, _ in bad:
        if a > cur:
            clean.append([cur, min(a - 1, hi)])
        cur = max(cur, b + 1)
        if cur > hi:
            break
    if cur <= hi:
        clean.append([cur, hi])
    clean = [c for c in clean if c[1] >= c[0]]
    term = min((f['frame'] for _, _, f in bad if f['kind'] in TERMINAL), default=None)
    if term is not None:              # nothing after a terminal fault is usable
        clean = [c for c in clean if c[1] < term]
    primary = next((c for c in clean if c[1] - c[0] + 1 >= min(min_frames, hi - lo + 1)), None)
    if primary is None and term is not None and clean:
        primary = clean[0]            # short, but the only honest footage before the terminal fault
    keep = []
    if primary:
        pieces = [list(primary)]
        for d in drops:
            nxt = []
            for p in pieces:
                if d['start'] > p[0] and d['end'] < p[1]:
                    nxt += [[p[0], d['start'] - 1], [d['end'] + 1, p[1]]]
                else:
                    nxt.append(p)
            pieces = nxt
        keep = pieces
    return primary, keep, clean, bad


def scan(path, visual=None, stages=None):
    """Full scan of one source -> its ``sources[name]`` record (see ``build``)."""
    src = Source(path)
    thumbs = src.grey_thumbs()
    imgf, stalls, diff = image_faults(src, thumbs)
    faults = _merge_runs(pose_faults(src)) + imgf
    for v in (visual or []):
        faults.append({'kind': v['kind'], 'frame': int(v['frame']), 'end': int(v.get('end', v['frame'])),
                       'detail': v.get('detail', ''), 'channel': 'visual', 'evidence': v.get('evidence')})
    faults.sort(key=lambda f: f['frame'])
    rec = {'kind': src.kind, 'n_frames': len(src), 'clock_offset_s': round(src.clock_offset, 4),
           'stages': {}}
    for st, span in src.stage_spans().items():
        if stages and st not in stages:
            continue
        in_span = [f for f in faults if f['end'] >= span[0] and f['frame'] <= span[1]]
        drops = stall_splices(stalls, thumbs, span)
        primary, keep, clean, bad = stage_ranges(span, in_span, drops)
        first = min((b[2] for b in bad), key=lambda f: f['frame'], default=None)
        cuts = []
        if primary:
            if primary[0] > span[0]:
                cuts.append({'start': span[0], 'end': primary[0] - 1,
                             'reason': _why(bad, span[0], primary[0] - 1, lead=True)})
            for d in drops:
                if primary[0] < d['start'] and d['end'] < primary[1]:
                    cuts.append({'start': d['start'], 'end': d['end'],
                                 'reason': f"stall #{d['stall'][0]}-{d['stall'][1]} "
                                           f"({(d['stall'][1] - d['stall'][0] + 1) / FPS_OUT:.1f} s of output with "
                                           f"no visible motion) spliced to 2x{MARGIN_S} s; join mean|delta|="
                                           f"{d['join_mean_abs_delta']}"})
            if primary[1] < span[1]:
                cuts.append({'start': primary[1] + 1, 'end': span[1],
                             'reason': _why(bad, primary[1] + 1, span[1], lead=False)})
        else:
            cuts.append({'start': span[0], 'end': span[1],
                         'reason': 'no clean stretch of >= 1 s: ' + _why(bad, span[0], span[1], lead=False)})
        rec['stages'][st] = {
            'stage_span': list(span),
            'usable': primary is not None,
            'keep': keep,
            'primary': primary,
            'first_fault': first,
            'cuts': cuts,
            'faults': in_span,
            'stalls': [list(s) for s in stalls if s[1] >= span[0] and s[0] <= span[1]],
            'sim_t': [round(src.times[span[0]], 2), round(src.times[span[1]], 2)],
        }
    return rec


def _why(bad, a, b, *, lead):
    hits = [f for lo, hi, f in bad if hi >= a - MARGIN_FRAMES and lo <= b + MARGIN_FRAMES]
    if not hits:
        return 'outside the primary clean range'
    f = hits[-1] if lead else hits[0]
    where = f"#{f['frame']}" + (f"-{f['end']}" if f['end'] != f['frame'] else '')
    edge = 'after' if lead else 'before'
    return (f"{f['kind']} at {where} ({f['channel']}: {f['detail']}); range ends/starts {MARGIN_FRAMES} frames "
            f"({MARGIN_S} s of output) {edge} it")


def build(mission_root, sources, visual_faults=None, notes=None):
    """``cut_ranges/v1``: {sources: {name: {kind, n_frames, stages: {stage: {...}}}}}."""
    visual_faults = visual_faults or {}
    out = {'schema': 'cut_ranges/v1', 'mission_root': str(mission_root), 'fps_out': FPS_OUT,
           'margin_s': MARGIN_S, 'margin_frames': MARGIN_FRAMES, 'stall_s': STALL_S,
           'frame_index': '0-based index into <source>/mission-frames/frames.json (real) or '
                          '<source>/decision_log.json frames (choreo; = walkthrough.mp4 frame)',
           'keep_semantics': 'keep = ordered inclusive [start, end] frame pieces of ONE source+stage; '
                             'consecutive pieces are seamless splices (only stall dead-air removed); '
                             'primary = the single contiguous clean range they come from',
           'sources': {}}
    for name in sources:
        path = Path(mission_root) / name
        if detect_kind(path) is None:
            out['sources'][name] = {'missing': True}
            continue
        out['sources'][name] = scan(path, visual_faults.get(name))
        if notes and name in notes:
            out['sources'][name]['note'] = notes[name]
    return out
