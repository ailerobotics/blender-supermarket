"""Automated pass/fail check of a mission run dir before its demo video is sent (CPU only).

Checks (each PASS / WARN / FAIL / UNVERIFIED):
  * stages_in_order   -- all seven stages appear in the event log, in mission order. A start-phase
                         shortcut stage counts as present but is a WARN (synthesized, not performed).
  * mission_succeeded -- mission.json state == 'complete' with no degraded stage.
  * stages_on_video   -- every performed stage has walkthrough frames inside its trace window.
  * carton_on_cart    -- the carton rests on the rollwagen deck through couple/outbound/park.
  * carton_on_bay     -- after the transfer, the carton rests on the transfer-bay top.
  * demo_video        -- a walkthrough video exists in the run dir (informational).

Carton evidence, best first: the recorded carton + cart poses (nav-trace ``carton``/``cart`` fields,
written when the run set RESTOCK_TRACE_POSES=1), cross-checked for a frozen (stale) pose read; else
frame analysis -- the overhead chase frame is sampled inside the projected carton footprint (cart: the
deck-rest position given the measured base pose + hitch geometry; bay: the static bay top) and the
fraction of carton-coloured pixels decides it. Frame analysis is labelled as such in the table.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import run_evidence as E

PASS, WARN, FAIL, UNVERIFIED = 'PASS', 'WARN', 'FAIL', 'UNVERIFIED'

# Frame-analysis tuning (overhead chase camera, fill light riding with the camera). The carton reads
# warm-white (R>=240, R-B 3..40); the floor (~229), the cart shelf (grey), the neutral-white bay/robot
# (R-B<=1) and the saturated-yellow floor lines (R-B>60) all fail it. See data report for the numbers.
CARTON_MIN_R = 240
CARTON_MIN_B = 200
CARTON_RB = (3, 40)
FRAME_SHOWS_CARTON = 0.40       # carton-coloured fraction of the (shrunk) projected footprint
STAGE_FRAMES_PASS = 0.85        # fraction of in-view frames that must show the carton
POSE_ROWS_PASS = 0.98           # fraction of traced rows that must be on the support
Z_TOL = 0.05                    # carton centre height tolerance above its support (m)


@dataclass
class Check:
    name: str
    result: str
    evidence: str
    details: dict = field(default_factory=dict)


# ---------------- frame analysis ----------------
def _carton_fraction(image, polygon):
    """Fraction of carton-coloured pixels inside the pixel polygon, or None when it is out of view."""
    import numpy as np
    from PIL import Image, ImageDraw
    w, h = image.size
    us = [p[0] for p in polygon]
    vs = [p[1] for p in polygon]
    if min(us) < 0 or min(vs) < 0 or max(us) >= w or max(vs) >= h:
        return None
    x0, y0 = int(math.floor(min(us))), int(math.floor(min(vs)))
    x1, y1 = int(math.ceil(max(us))) + 1, int(math.ceil(max(vs))) + 1
    crop = np.asarray(image.crop((x0, y0, x1, y1)).convert('RGB')).astype(int)
    mask_img = Image.new('L', (x1 - x0, y1 - y0), 0)
    ImageDraw.Draw(mask_img).polygon([(u - x0, v - y0) for u, v in polygon], fill=1)
    mask = np.asarray(mask_img).astype(bool)
    if mask.sum() < 20:
        return None
    r, g, b = crop[..., 0], crop[..., 1], crop[..., 2]
    rb = r - b
    carton = (r >= CARTON_MIN_R) & (b >= CARTON_MIN_B) & (rb >= CARTON_RB[0]) & (rb <= CARTON_RB[1])
    return float(carton[mask].mean())


def _frames_in(run, lo, hi):
    return [f for f in run.frames if lo <= f.t <= hi]


def carton_fractions_on_cart(run, model, camera, frames):
    from PIL import Image
    out = []
    for f in frames:
        row = run.row_at(f.t)
        if row is None or not f.file.is_file():
            continue
        x, y, z, yaw, _ = E.carton_pose(run, model, f.t)
        top = z + model.carton_size[2] / 2
        poly = camera.project((row['x'], row['y'], row['yaw']),
                              E.rect_corners((x, y), yaw, model.carton_size[:2], top, shrink=0.7))
        with Image.open(f.file) as im:
            cam = camera if im.size == (camera.width, camera.height) else camera.for_size(*im.size)
            if cam is not camera:
                poly = cam.project((row['x'], row['y'], row['yaw']),
                                   E.rect_corners((x, y), yaw, model.carton_size[:2], top, shrink=0.7))
            frac = _carton_fraction(im, poly)
        if frac is not None:
            out.append((f, frac))
    return out


def carton_fractions_on_bay(run, model, camera, frames):
    from PIL import Image
    out = []
    top = model.bay_top_z + model.carton_size[2]      # carton top when resting on the bay
    for f in frames:
        row = run.row_at(f.t)
        if row is None or not f.file.is_file():
            continue
        corners = E.rect_corners(model.bay_center, 0.0, model.bay_size, top, shrink=0.6)
        with Image.open(f.file) as im:
            cam = camera if im.size == (camera.width, camera.height) else camera.for_size(*im.size)
            frac = _carton_fraction(im, cam.project((row['x'], row['y'], row['yaw']), corners))
        if frac is not None:
            out.append((f, frac))
    return out


# ---------------- pose analysis ----------------
def _on_cart(row, model):
    cart, carton = row.get('cart'), row.get('carton')
    if not (isinstance(cart, list) and isinstance(carton, list) and len(cart) >= 3 and len(carton) >= 3):
        return None
    cyaw = math.radians(float(cart[2]))
    dx, dy = float(carton[0]) - float(cart[0]), float(carton[1]) - float(cart[1])
    lx, ly = math.cos(cyaw) * dx + math.sin(cyaw) * dy, -math.sin(cyaw) * dx + math.cos(cyaw) * dy
    z_ok = abs(float(carton[2]) - (model.support_z + model.carton_size[2] / 2)) <= Z_TOL
    return abs(lx) <= model.cart_half[0] and abs(ly) <= model.cart_half[1] and z_ok


def _on_bay(carton, model):
    hx, hy = model.bay_size[0] / 2, model.bay_size[1] / 2
    return (abs(float(carton[0]) - model.bay_center[0]) <= hx
            and abs(float(carton[1]) - model.bay_center[1]) <= hy
            and abs(float(carton[2]) - (model.bay_top_z + model.carton_size[2] / 2)) <= Z_TOL)


def _pose_trace_frozen(rows):
    """True when the cart moved > 0.5 m across ``rows`` but the recorded carton did not (< 0.1 m):
    a stale USD read, not evidence that the carton rode along."""
    pairs = [(r['cart'], r['carton']) for r in rows
             if isinstance(r.get('cart'), list) and isinstance(r.get('carton'), list)]
    if len(pairs) < 2:
        return False
    (c0, k0), (c1, k1) = pairs[0], pairs[-1]
    cart_move = math.hypot(c1[0] - c0[0], c1[1] - c0[1])
    carton_move = math.hypot(k1[0] - k0[0], k1[1] - k0[1])
    return cart_move > 0.5 and carton_move < 0.1


# ---------------- the checks ----------------
def check_stages(run):
    status = E.stage_status(run)
    order = E.stage_order(run)
    expected = [s for s in E.STAGES if s in order]
    in_order = order == expected
    missing = [s for s in E.STAGES if status[s] in ('missing', 'started', 'failed', 'unavailable')]
    shortcut = [s for s in E.STAGES if status[s] == 'shortcut']
    parts = []
    for s in E.STAGES:
        tag = {'succeeded': 'ok', 'shortcut': 'SHORTCUT', 'missing': 'missing', 'started': 'unfinished',
               'failed': 'FAILED', 'unavailable': 'UNAVAILABLE'}[status[s]]
        parts.append(f'{s}={tag}')
    evidence = ' '.join(parts) + ('' if in_order else f' | OUT OF ORDER: {" > ".join(order)}')
    if missing or not in_order:
        result = FAIL
    elif shortcut:
        result = WARN
    else:
        result = PASS
    return Check('stages_in_order', result, evidence, {'status': status, 'order': order})


def check_success(run):
    state = run.mission.get('state')
    degraded = run.mission.get('degraded') or []
    if state == 'complete' and not degraded:
        return Check('mission_succeeded', PASS, 'state=complete')
    why = E.short_reason(run.mission.get('reason'), 90)
    extra = f' degraded={degraded}' if degraded else ''
    return Check('mission_succeeded', FAIL, f'state={state}{extra}: {why}')


def check_stage_frames(run):
    status = E.stage_status(run)
    windows = E.stage_windows(run)
    by_stage = E.frames_by_stage(run)
    counts, short = {}, []
    for s in E.STAGES:
        if status[s] not in ('succeeded', 'failed', 'started') or s not in windows:
            continue
        lo, hi = windows[s]
        n = len(by_stage[s])
        counts[s] = n
        if n == 0:
            short.append(f'{s} ({hi - lo:.2f}s)')
    if not run.frames:
        return Check('stages_on_video', FAIL, 'no mission-frames recorded')
    evidence = ' '.join(f'{s}={n}' for s, n in counts.items()) + f' frames (clock offset '\
        f'{run.clock_offset:+.2f}s from {run.clock_source})'
    if short:
        return Check('stages_on_video', WARN, evidence + f' | no frame inside: {", ".join(short)} '
                     '(shown as a caption card)', {'counts': counts})
    return Check('stages_on_video', PASS, evidence, {'counts': counts})


def check_carton_on_cart(run, model=None, camera=None):
    model = model or E.cart_model(run)
    camera = camera or E.ChaseCamera()
    windows = E.stage_windows(run)
    status = E.stage_status(run)
    stages = [s for s in ('couple', 'outbound', 'park') if s in windows and status[s] != 'shortcut']
    synthesized = [s for s in ('couple', 'outbound', 'park') if status[s] == 'shortcut']
    if not stages and synthesized:
        # A start-phase bench (e.g. start_at=transfer) never towed: say so, and still check the carton
        # is on the deck where the run begins (its first second of footage).
        first = [f for f in run.frames if run.frames and f.t <= run.frames[0].t + 1.0]
        fr = carton_fractions_on_cart(run, model, camera, first) if not run.has_pose_trace else []
        seen = f'; carton on deck at the start {sum(v >= FRAME_SHOWS_CARTON for _, v in fr)}/{len(fr)} frames' \
            if fr else ''
        return Check('carton_on_cart', UNVERIFIED, f'{"/".join(synthesized)} synthesized (start-phase '
                     f'shortcut): no tow recorded to check{seen}')
    if not stages:
        return Check('carton_on_cart', FAIL, 'no couple/outbound/park in the trace')
    if run.has_pose_trace:
        per, frozen, bad_total, n_total = [], False, 0, 0
        for s in stages:
            lo, hi = windows[s]
            rows = [r for r in run.trace if lo <= r['t'] <= hi]
            verdicts = [v for v in (_on_cart(r, model) for r in rows) if v is not None]
            if s == 'outbound' and _pose_trace_frozen(rows):
                frozen = True
            n_total += len(verdicts)
            bad_total += sum(1 for v in verdicts if not v)
            per.append(f'{s} {sum(verdicts)}/{len(verdicts)}')
        if frozen:
            return Check('carton_on_cart', UNVERIFIED, 'pose trace: carton pose did not move with the towed '
                         'cart (stale read) -- ' + ', '.join(per))
        if n_total == 0:
            return Check('carton_on_cart', UNVERIFIED, 'pose trace present but no carton/cart rows in stages')
        ok = (n_total - bad_total) / n_total >= POSE_ROWS_PASS
        return Check('carton_on_cart', PASS if ok else FAIL,
                     'pose trace rows on deck: ' + ', '.join(per))
    per, all_ok, n_any, details = [], True, 0, {}
    by_stage = E.frames_by_stage(run)
    for s in stages:
        frames = by_stage[s]
        if not frames:
            # Too short to own a capture (park can latch in one tick): judge the first frame after it.
            hi = windows[s][1]
            frames = [f for f in run.frames if hi < f.t <= hi + 1.0][:1]
        fr = carton_fractions_on_cart(run, model, camera, frames)
        if not fr:
            per.append(f'{s} n/a (no in-view frame)')
            continue
        n_any += len(fr)
        shown = sum(1 for _, v in fr if v >= FRAME_SHOWS_CARTON)
        share = shown / len(fr)
        median = sorted(v for _, v in fr)[len(fr) // 2]
        details[s] = {'frames': len(fr), 'shown': shown, 'median_fraction': round(median, 3),
                      'worst': [(f.index, round(v, 3)) for f, v in sorted(fr, key=lambda p: p[1])[:3]]}
        per.append(f'{s} {shown}/{len(fr)}')
        all_ok = all_ok and share >= STAGE_FRAMES_PASS
        if s == 'park' and status.get('park') == 'succeeded' and fr[-1][1] < FRAME_SHOWS_CARTON:
            all_ok = False
    if n_any == 0:
        return Check('carton_on_cart', UNVERIFIED, 'no pose trace and no in-view frame to analyse')
    return Check('carton_on_cart', PASS if all_ok else FAIL,
                 'frame analysis (no pose trace) frames showing carton on deck: ' + ', '.join(per), details)


def check_carton_on_bay(run, model=None, camera=None):
    model = model or E.cart_model(run)
    camera = camera or E.ChaseCamera()
    t_done = E.success_time(run, 'transfer')
    if t_done is None:
        return Check('carton_on_bay', FAIL, 'transfer never succeeded: no placement to check')
    if run.has_pose_trace:
        rows = [r for r in run.trace if r['t'] > t_done and isinstance(r.get('carton'), list)]
        if not rows:
            return Check('carton_on_bay', UNVERIFIED, 'pose trace has no carton row after the transfer')
        on = [_on_bay(r['carton'], model) for r in rows]
        ok = sum(on) / len(on) >= POSE_ROWS_PASS and on[-1]
        last = rows[-1]['carton']
        return Check('carton_on_bay', PASS if ok else FAIL,
                     f'pose trace rows on bay top {sum(on)}/{len(on)}; last carton '
                     f'({last[0]:.2f},{last[1]:.2f},{last[2]:.2f}) vs bay {model.bay_center} top {model.bay_top_z}')
    fr = carton_fractions_on_bay(run, model, camera, [f for f in run.frames if f.t > t_done])
    if len(fr) < 3:
        return Check('carton_on_bay', UNVERIFIED, f'no pose trace; only {len(fr)} in-view frame(s) of the bay')
    shown = sum(1 for _, v in fr if v >= FRAME_SHOWS_CARTON)
    ok = shown / len(fr) >= STAGE_FRAMES_PASS
    return Check('carton_on_bay', PASS if ok else FAIL,
                 f'frame analysis (no pose trace): bay top shows the carton in {shown}/{len(fr)} frames')


def check_video(run):
    for name in ('walkthrough-demo.mp4', 'walkthrough.mp4'):
        path = run.run_dir / name
        if path.is_file() and path.stat().st_size > 1000:
            return Check('demo_video', PASS, f'{name} {path.stat().st_size / 1e6:.1f} MB')
    return Check('demo_video', WARN, 'no walkthrough video in the run dir (build it with make_walkthrough_cut.py)')


def run_checks(run):
    model = E.cart_model(run)
    camera = E.ChaseCamera()
    return [check_stages(run), check_success(run), check_stage_frames(run),
            check_carton_on_cart(run, model, camera), check_carton_on_bay(run, model, camera),
            check_video(run)]


def overall(checks):
    gating = [c for c in checks if c.name != 'demo_video']
    if any(c.result in (FAIL, UNVERIFIED) for c in gating):
        return FAIL
    return WARN if any(c.result == WARN for c in checks) else PASS


def format_table(run, checks):
    width = max(len(c.name) for c in checks)
    lines = [f'run {run.name}: {E.result_line(run)}',
             f'{"check".ljust(width)}  result      evidence',
             f'{"-" * width}  ----------  --------']
    for c in checks:
        lines.append(f'{c.name.ljust(width)}  {c.result.ljust(10)}  {c.evidence}')
    labels = E.workaround_labels(run)
    if labels:
        lines.append('labels: ' + '; '.join((f'{l.stage}: ' if l.stage else '') + l.text for l in labels))
    lines.append(f'{"OVERALL".ljust(width)}  {overall(checks)}')
    return '\n'.join(lines)
