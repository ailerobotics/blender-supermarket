"""Read one mission run dir (CPU only) into the facts the demo cut and the run checker share.

A run dir is what ``run_mobile_restock.py`` leaves behind: ``mission.json`` (state + the inventory
event log), ``nav-trace.jsonl`` (one row per control tick: mission time ``t``, the active ``action``,
the measured base pose, ``coupled``/``parked``), ``mission-frames/frames.json`` (the overhead
walkthrough_chase captures, stamped in Isaac stage time) and ``bt_events.jsonl``. Nothing here
writes into the run dir.

Clocks: trace ``t`` is the mission clock (tick*dt); frame ``simulation_time`` is Isaac's stage time.
Their offset is constant within a run (0.05 s cold, 0.55 s after a start-phase shortcut settle) and
is taken, in order of preference, from the trace's own ``world_t`` (RESTOCK_TRACE_POSES=1 runs), the
first ``CLOCK_DIAG`` line of the sibling ``<run>.console.log``, or 0.
"""
from __future__ import annotations

import bisect
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

STAGES = ('collect', 'couple', 'outbound', 'park', 'transfer', 'verify', 'return')

TITLES = {'collect': 'Collect', 'couple': 'Couple', 'outbound': 'Outbound tow',
          'park': 'Park & uncouple', 'transfer': 'Transfer', 'verify': 'Verify', 'return': 'Return'}

CAPTIONS = {
    'collect': 'Pick the carton from the backstock and load it onto the rollwagen',
    'couple': 'Drive up to the rollwagen and latch the tow hitch',
    'outbound': 'Coupled tow from the backroom dock to the transfer bay',
    'park': 'Stop at the transfer bay parking spot',
    'transfer': 'Arm moves the carton from the rollwagen onto the bay top',
    'verify': 'Confirm the carton is on the bay; commit the stock',
    'return': 'Return to the dock',
}

# The evidence key each performed stage must carry (mirrors twin.restock.mission._validate).
_SUCCESS_KEYS = {'collect': 'loaded', 'couple': 'coupled', 'outbound': 'arrived', 'park': 'stationary',
                 'transfer': 'released', 'return': 'home'}


@dataclass(frozen=True)
class Label:
    """One honest on-screen workaround/provenance label (e.g. verify_mode=ground_truth)."""
    stage: str | None   # the stage it qualifies, or None for the whole run
    key: str
    value: str

    @property
    def text(self):
        return f'{self.key}={self.value}'


@dataclass
class Frame:
    index: int
    file: Path
    sim_time: float      # Isaac stage time as recorded
    t: float             # mission clock (sim_time - clock offset)


@dataclass
class Run:
    run_dir: Path
    name: str
    mission: dict
    events: list
    trace: list                      # compact rows (see _compact)
    frames: list                     # [Frame]
    bt_events: list
    clock_offset: float
    clock_source: str
    scenario: dict | None = None
    spec: dict = field(default_factory=dict)
    caption: str | None = None       # frames.json run caption (a start-phase shortcut bench run)
    _times: list = field(default_factory=list, repr=False)

    # ---------------- trace lookups ----------------
    def row_at(self, t):
        """The trace row at (or just before) mission time ``t``; None for an empty trace."""
        if not self.trace:
            return None
        if not self._times:
            self._times = [r['t'] for r in self.trace]
        i = bisect.bisect_right(self._times, t) - 1
        return self.trace[max(0, min(i, len(self.trace) - 1))]

    def action_at(self, t):
        row = self.row_at(t)
        return row['action'] if row else None

    @property
    def has_pose_trace(self):
        return any('carton' in r and r['carton'] is not None for r in self.trace)


def _compact(row):
    out = {'t': float(row['t']), 'action': row.get('action'), 'x': float(row['x']), 'y': float(row['y']),
           'yaw': math.radians(float(row.get('yaw_deg', 0.0))), 'lin': float(row.get('lin', 0.0) or 0.0),
           'coupled': bool(row.get('coupled')), 'parked': bool(row.get('parked'))}
    vel = row.get('vel') or {}
    out['rw'] = float(vel.get('rw', 0.0) or 0.0)
    for key in ('cart', 'carton', 'world_t'):    # RESTOCK_TRACE_POSES=1 extras
        if key in row:
            out[key] = row[key]
    return out


def read_trace(path):
    rows = []
    path = Path(path)
    if not path.is_file():
        return rows
    with path.open() as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue                      # a torn last line from a killed run
            if 't' in row and 'x' in row and 'y' in row:
                rows.append(_compact(row))
    rows.sort(key=lambda r: r['t'])
    return rows


def _read_jsonl(path):
    out = []
    path = Path(path)
    if path.is_file():
        for line in path.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def console_clock_offset(console_log):
    """world_current_time - stage_time from the first CLOCK_DIAG line, or None."""
    console_log = Path(console_log) if console_log else None
    if console_log is None or not console_log.is_file():
        return None
    pattern = re.compile(r'CLOCK_DIAG tick=\d+ stage_time=([-\d.]+) world_current_time=([-\d.]+)')
    with console_log.open(errors='replace') as stream:
        for line in stream:
            m = pattern.search(line)
            if m:
                return float(m[2]) - float(m[1])
    return None


def _find_console_log(run_dir):
    parent = run_dir.parent
    for name in (f'{run_dir.name}.console.log', f'{run_dir.name}-console.log'):
        if (parent / name).is_file():
            return parent / name
    return None


def _find_scenario(run_dir):
    for path in (run_dir / 'mission-scenario.json', run_dir.parent / 'mission-scenario.json'):
        if path.is_file():
            try:
                return json.loads(path.read_text())
            except ValueError:
                return None
    return None


def load_run(run_dir, *, console_log=None, scenario=None):
    run_dir = Path(run_dir).resolve()
    mission_path = run_dir / 'mission.json'
    if not mission_path.is_file():
        raise FileNotFoundError(f'{mission_path} missing: not a mission run dir')
    mission = json.loads(mission_path.read_text())
    trace = read_trace(run_dir / 'nav-trace.jsonl')
    # Clock offset between frame stage time and trace mission time.
    offset, source = None, 'none (assumed 0)'
    worlds = [r['world_t'] - r['t'] for r in trace if isinstance(r.get('world_t'), (int, float))]
    if worlds:
        worlds.sort()
        offset, source = worlds[len(worlds) // 2], 'nav-trace world_t'
    if offset is None:
        log = Path(console_log) if console_log else _find_console_log(run_dir)
        offset = console_clock_offset(log)
        if offset is not None:
            source = f'CLOCK_DIAG in {log.name}'
    if offset is None:
        offset = 0.0
    frames, caption = [], None
    manifest = run_dir / 'mission-frames' / 'frames.json'
    if manifest.is_file():
        data = json.loads(manifest.read_text())
        caption = data.get('caption') if isinstance(data.get('caption'), str) else None
        for i, f in enumerate(data.get('frames', [])):
            st = float(f['simulation_time'])
            frames.append(Frame(i, manifest.parent / f['file'], st, st - offset))
    if scenario is None:
        scenario = _find_scenario(run_dir)
    elif not isinstance(scenario, dict):
        scenario = json.loads(Path(scenario).read_text())
    spec = {}
    assembly = run_dir / 'assembly.json'
    if assembly.is_file():
        try:
            spec = json.loads(assembly.read_text())['composition']['assets']['spec']
        except (ValueError, KeyError, TypeError):
            spec = {}
    return Run(run_dir=run_dir, name=run_dir.name, mission=mission, events=list(mission.get('events') or []),
               trace=trace, frames=frames, bt_events=_read_jsonl(run_dir / 'bt_events.jsonl'),
               clock_offset=float(offset), clock_source=source, scenario=scenario, spec=spec,
               caption=caption)


# ---------------- stages ----------------
def event_stage(state):
    """Map an inventory event state ('couple', 'couple_succeeded', 'collect_shortcut', ...) to its stage."""
    if not isinstance(state, str):
        return None
    for stage in STAGES:
        if state == stage or state.startswith(stage + '_'):
            return stage
    return None


def stage_status(run):
    """Per stage: 'succeeded' | 'shortcut' | 'unavailable' | 'failed' | 'started' | 'missing'.

    Driven by the durable event log (mission.json events) -- the same record the stock commit uses."""
    status = {s: 'missing' for s in STAGES}
    current = None
    for ev in run.events:
        state = ev.get('state')
        if state == 'failed':
            if current is not None and status[current] == 'started':
                status[current] = 'failed'
            continue
        stage = event_stage(state)
        if stage is None:
            continue
        if state.endswith('_succeeded'):
            status[stage] = 'succeeded'
        elif state.endswith('_shortcut'):
            status[stage] = 'shortcut'
        elif state.endswith('_unavailable'):
            status[stage] = 'unavailable'
        elif state == stage and status[stage] == 'missing':
            status[stage] = 'started'
        current = stage
    return status


def stage_order(run):
    """Stages in order of first appearance in the event log."""
    seen = []
    for ev in run.events:
        stage = event_stage(ev.get('state'))
        if stage is not None and stage not in seen:
            seen.append(stage)
    return seen


def success_time(run, stage):
    """Mission time of the stage's measured success evidence, or None."""
    for ev in run.events:
        if ev.get('state') == stage + '_succeeded':
            t = (ev.get('evidence') or {}).get('simulation_time')
            if isinstance(t, (int, float)) and math.isfinite(t):
                return float(t)
    return None


def stage_windows(run):
    """{stage: (t_first, t_last)} from the trace's per-tick ``action`` (mission time)."""
    windows = {}
    for row in run.trace:
        stage = row['action']
        if stage not in STAGES:
            continue
        lo, hi = windows.get(stage, (row['t'], row['t']))
        windows[stage] = (min(lo, row['t']), max(hi, row['t']))
    return windows


def frames_by_stage(run):
    """{stage: [Frame]}: each recorded frame belongs to the stage active (per the trace) at its time.
    Frames before the first trace row belong to the first traced stage."""
    out = {s: [] for s in STAGES}
    first = next((r['action'] for r in run.trace if r['action'] in STAGES), None)
    if first is None:
        return out
    for f in run.frames:
        stage = run.action_at(f.t)
        out[stage if stage in STAGES else first].append(f)
    return out


def uncouple_time(run):
    """Mission time the measured coupling first drops after having been held, or None."""
    held = False
    for row in run.trace:
        if row['coupled']:
            held = True
        elif held:
            return row['t']
    return None


def couple_time(run):
    for row in run.trace:
        if row['coupled']:
            return row['t']
    return None


# ---------------- workaround / provenance labels ----------------
_LABEL_KEY = re.compile(r'(_mode$|^mode$|workaround|^start_at$)')
_HONEST_SOURCES = {'sensor', 'test_fixture'}


def _walk(obj, stage, out, depth=0):
    if depth > 6:
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(key, str) and _LABEL_KEY.search(key) and isinstance(value, (str, int, float, bool)):
                out.append(Label(stage, key, str(value)))
            elif key == 'source' and isinstance(value, str) and value not in _HONEST_SOURCES \
                    and value != 'start_phase_shortcut':
                out.append(Label(stage, 'source', value))
            elif isinstance(key, str) and 'workaround' in key and isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, str):
                        out.append(Label(stage, key, item))
                    elif isinstance(item, dict):
                        name = item.get('name') or item.get('key') or key
                        val = item.get('value', item.get('mode', True))
                        out.append(Label(item.get('stage', stage), str(name), str(val)))
            else:
                _walk(value, stage, out, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _walk(item, stage, out, depth + 1)


def workaround_labels(run):
    """Every workaround/provenance label the lanes record in mission.json, de-duplicated, in order.

    Picks up: the start-phase shortcut (start_at + the synthesized stages), any key ending ``_mode``
    or containing ``workaround`` (top level or in a stage's evidence), a non-sensor evidence
    ``source`` (e.g. ground_truth), and degraded (unavailable) stages."""
    labels = []
    shortcut = run.mission.get('shortcut')
    skip = {'events', 'shortcut', 'walkthrough_caption'}
    if isinstance(shortcut, dict) and shortcut.get('start_at'):
        skip.add('start_at')                 # the per-stage labels below already say it
        for stage in shortcut.get('skipped') or []:
            labels.append(Label(stage, 'start_at', f"{shortcut['start_at']} ({stage} synthesized, not performed)"))
    for caption in (run.mission.get('walkthrough_caption'), run.caption):
        if isinstance(caption, str) and caption.strip():
            labels.append(Label(None, 'caption', caption.strip()))
    top = {k: v for k, v in run.mission.items() if k not in skip}
    _walk(top, None, labels)
    for ev in run.events:
        _walk(ev.get('evidence') or {}, event_stage(ev.get('state')), labels)
    for stage in run.mission.get('degraded') or []:
        labels.append(Label(stage, 'degraded', 'capability unavailable'))
    seen, out = set(), []
    for label in labels:
        k = (label.stage, label.key, label.value)
        if k not in seen:
            seen.add(k)
            out.append(label)
    return out


def result_line(run):
    state = run.mission.get('state', 'unknown')
    if state == 'complete':
        return 'MISSION COMPLETE'
    status = stage_status(run)
    failed = next((s for s in STAGES if status[s] == 'failed'), None)
    where = f' at {failed.upper()}' if failed else ''
    return f'MISSION {str(state).upper()}{where}'


def short_reason(reason, limit=110):
    reason = str(reason or '').strip()
    reason = re.sub(r'\s*\[[^\]]*\.py:\d+[^\]]*\]', '', reason)   # drop traceback breadcrumbs
    return reason if len(reason) <= limit else reason[:limit - 1] + '…'


# ---------------- geometry: the overhead chase camera + the cart/carton model ----------------
@dataclass(frozen=True)
class ChaseCamera:
    """The ``walkthrough_chase`` camera (twin.restock.sensing_sources): parented to ur_base_link,
    2.6 m above it, identity orientation (looks straight down; image right = base +x, image up =
    base +y), fx=fy=430 at 960x720 (scaled with RESTOCK_WALKTHROUGH_WIDTH/HEIGHT)."""
    width: int = 960
    height: int = 720
    fx: float = 430.0
    fy: float = 430.0
    cx: float = 480.0
    cy: float = 360.0
    camera_z: float = 0.5 + 2.6     # arm_mount (ur_base_link) world height + the 2.6 m mount

    def for_size(self, width, height):
        sx, sy = width / 960.0, height / 720.0
        return ChaseCamera(width, height, 430.0 * sx, 430.0 * sy, 480.0 * sx, 360.0 * sy, self.camera_z)

    def project(self, base_pose, points):
        """World points [(x,y,z)] -> pixels [(u,v)] for the base pose (x, y, yaw)."""
        bx, by, yaw = base_pose
        c, s = math.cos(yaw), math.sin(yaw)
        out = []
        for x, y, z in points:
            dx, dy = x - bx, y - by
            lx, ly = c * dx + s * dy, -s * dx + c * dy
            depth = max(1e-3, self.camera_z - z)
            out.append((self.cx + self.fx * lx / depth, self.cy - self.fy * ly / depth))
        return out


@dataclass(frozen=True)
class CartModel:
    hitch_offset: float = -1.27            # cart centre in the base frame when coupled straight
    load_center: tuple = (0.18, 0.0)       # carton centre in the cart body frame
    carton_yaw_in_cart: float = -math.pi / 2
    carton_size: tuple = (0.6, 0.4, 0.16)
    support_z: float = 0.9325              # cart stock-support top (world z)
    cart_half: tuple = (0.40, 0.34)        # rollwagen footprint half-extents (0.8 x 0.68)
    initial_cart: tuple = (-5.73, 15.25, 0.0)
    bay_center: tuple = (-2.0, 16.42)
    bay_size: tuple = (0.7, 0.5)
    bay_top_z: float = 0.73


def cart_model(run):
    """The cart/carton/bay geometry for this run: scenario + assembly spec, defaults otherwise."""
    kw = {}
    spec = run.spec or {}
    if isinstance(spec.get('hitch_x'), (int, float)) and isinstance(spec.get('cart_hitch_x'), (int, float)):
        kw['hitch_offset'] = float(spec['hitch_x']) - float(spec['cart_hitch_x'])
    if isinstance(spec.get('cart_length'), (int, float)) and isinstance(spec.get('cart_width'), (int, float)):
        kw['cart_half'] = (float(spec['cart_length']) / 2, float(spec['cart_width']) / 2)
    sc = run.scenario or {}
    staging = sc.get('initial_staging') or {}
    unit = sc.get('handling_unit') or {}
    if isinstance(staging.get('cart_load_center_body_xy'), list):
        kw['load_center'] = tuple(float(v) for v in staging['cart_load_center_body_xy'][:2])
    if isinstance(staging.get('cart_support_z_world'), (int, float)):
        kw['support_z'] = float(staging['cart_support_z_world'])
    if isinstance(staging.get('cart_body_pose'), list) and len(staging['cart_body_pose']) >= 3:
        kw['initial_cart'] = tuple(float(v) for v in staging['cart_body_pose'][:3])
        yaws = staging.get('unit_yaws_world') or {}
        if isinstance(yaws.get('cart'), (int, float)):
            kw['carton_yaw_in_cart'] = float(yaws['cart']) - kw['initial_cart'][2]
    if isinstance(unit.get('dimensions_m'), list) and len(unit['dimensions_m']) == 3:
        kw['carton_size'] = tuple(float(v) for v in unit['dimensions_m'])
    bay = sc.get('transfer_bay') or {}
    if isinstance(bay.get('center_xy'), list):
        kw['bay_center'] = tuple(float(v) for v in bay['center_xy'][:2])
    if isinstance(bay.get('dimensions_m'), list):
        kw['bay_size'] = tuple(float(v) for v in bay['dimensions_m'][:2])
    if isinstance(bay.get('top_z'), (int, float)):
        kw['bay_top_z'] = float(bay['top_z'])
    return CartModel(**kw)


def compose(pose, local):
    x, y, yaw = pose
    lx, ly = local[0], local[1]
    c, s = math.cos(yaw), math.sin(yaw)
    return (x + c * lx - s * ly, y + s * lx + c * ly)


def rect_corners(center_xy, yaw, size_xy, z, shrink=1.0):
    hx, hy = size_xy[0] / 2 * shrink, size_xy[1] / 2 * shrink
    return [(*compose((center_xy[0], center_xy[1], yaw), (sx * hx, sy * hy)), z)
            for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))]


def estimated_cart_pose(run, model, t):
    """Cart (x, y, yaw) at mission time t WITHOUT a pose trace: parked at its initial pose until the
    measured coupling latches, then rigidly behind the base on the hitch (straight-tow assumption),
    then left where it was when the coupling drops. Recorded ``cart`` trace fields win when present."""
    row = run.row_at(t)
    if row is None:
        return model.initial_cart
    if isinstance(row.get('cart'), list) and len(row['cart']) >= 3:
        x, y, yaw_deg = row['cart'][:3]
        return (float(x), float(y), math.radians(float(yaw_deg)))
    t_couple, t_uncouple = couple_time(run), uncouple_time(run)
    if t_couple is None or t < t_couple:
        return model.initial_cart
    t_ref = t if (t_uncouple is None or t < t_uncouple) else t_uncouple - 1e-6
    ref = run.row_at(t_ref)
    cx, cy = compose((ref['x'], ref['y'], ref['yaw']), (model.hitch_offset, 0.0))
    return (cx, cy, ref['yaw'])


def carton_pose(run, model, t):
    """(x, y, z, yaw, source): the recorded carton pose when traced, else the deck-rest estimate."""
    row = run.row_at(t)
    if row is not None and isinstance(row.get('carton'), list) and len(row['carton']) >= 3:
        c = row['carton']
        yaw = math.radians(float(c[3])) if len(c) > 3 else 0.0
        return (float(c[0]), float(c[1]), float(c[2]), yaw, 'trace')
    cart = estimated_cart_pose(run, model, t)
    x, y = compose(cart, model.load_center)
    return (x, y, model.support_z + model.carton_size[2] / 2, cart[2] + model.carton_yaw_in_cart, 'estimate')
