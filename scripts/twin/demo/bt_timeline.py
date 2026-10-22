"""Reconstruct the per-frame decision/reason tree for a REAL autonomous run from its
recorded evidence -- no Isaac, no choreography timeline, run dir read-only.

The choreographed demo (:mod:`twin.demo.decision_log`) drives the demo-004 panel from live
Isaac prim reads. This module drives the SAME panel from what the autonomous mission already
wrote to its run dir:

* ``bt_events.jsonl`` / ``bt_snapshot.json`` -- the behaviour-tree driver's own ticks
  (``stage_begin`` / ``stage_step`` / ``stage_succeeded`` / ``stage_failed`` / ``stage_shortcut``).
  Replaying them to a mission time gives every node's status + the authored step labels, so the
  panel tree is exactly what the mission ticked (not a hand-authored structure).
* the recorded poses (``nav-trace.jsonl`` base pose + the ``RESTOCK_TRACE_POSES=1`` cart/carton
  fields, via :mod:`twin.demo.run_evidence`) and ``mission.json`` -- the reason value for each
  decision, MEASURED from the recorded scene: the couple hitch gap, the base distance to the
  transfer bay, the nav lane progress, the placed-carton delta from the slot target. These are the
  same quantities the host prints to the console (``FINE_DOCK_DIAG`` hitch_mm, ``NAV_ARRIVAL_DIAG``
  pose/goal, the transfer target) -- reproduced here read-only from the time-indexed trace so every
  panel value is honest and aligned to the frame it is drawn on.

Everything is pure (CPU, no GPU, no simulator import) so it is tested on synthetic run dirs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import choreography as C
from . import measures as M
from . import run_evidence as E

STAGES = E.STAGES  # ('collect','couple','outbound','park','transfer','verify','return')

# Robot/cart geometry (asset properties, not the choreographed layout): the robot hitch point sits
# HITCH_BEHIND m behind base_link along -heading; the cart drawbar dock sits DOCK_AHEAD m ahead of the
# cart centre along +cart-heading. Reused from the demo constants (same physical robot + rollwagen);
# the couple hitch gap they yield matches the host's FINE_DOCK_DIAG hitch_mm within a few mm.
HITCH_BEHIND = C.HITCH_BEHIND
DOCK_AHEAD = C.DOCK_AHEAD

# How the panel renders a replayed stage status (panel.STATUS_RGB knows idle/running/success/failure).
# A start-phase shortcut stage was SYNTHESIZED, never actuated, so it is drawn idle (muted, i.e. "not
# performed") with an explicit 'synthesized' feedback tag -- never green, which would imply it ran.
_PANEL_STATUS = {'idle': 'idle', 'running': 'running', 'success': 'success',
                 'failure': 'failure', 'shortcut': 'idle'}
_STAGE_FEEDBACK = {'idle': '', 'running': 'running', 'success': 'done',
                   'failure': 'FAILED', 'shortcut': 'synthesized (shortcut)'}

# Panel header title per stage: 1..7 in mission order (matches the scene band's "idx/7").
STAGE_TITLES = {s: f'{i + 1} · {E.TITLES[s]}' for i, s in enumerate(STAGES)}

# Fallback step plans (label, kind) for a stage the event log never began (a shortcut-synthesized
# stage): the authored mission plan, so the tree still shows the (idle) decomposition.
try:
    from twin.restock.bt import stage_phases as _SP

    def _fallback_steps(stage):
        plan = _SP.RETURN_UNCOUPLE_PLAN if stage == 'return' else _SP.PLANS.get(stage, [])
        return [(label, kind) for label, kind in plan]
except Exception:                                    # py_trees-free envs still import this module
    def _fallback_steps(stage):
        return []


def _short_fault(reason):
    """A compact one-line fault for a failed step (traceback breadcrumbs stripped). A no-progress
    nav fault keeps the planner's last reason (e.g. 'no base progress · planning_stop')."""
    import re
    text = E.short_reason(reason, 999)
    last = re.search(r"last reason '([^']+)'", text)
    if 'no base progress' in text and last:
        return f'no base progress · {last.group(1)}'
    for sep in (';', ' [', ' ('):                    # keep the head clause, drop trailing detail
        if sep in text:
            text = text.split(sep, 1)[0]
    return text[:58] + ('…' if len(text) > 58 else '')


@dataclass
class BtTimeline:
    """Replayable behaviour-tree history for one run, built from its recorded ``bt_events``."""
    events: list                       # sorted [{'t','type','stage','step','label','kind','steps','reason'}]
    labels: dict                       # stage -> [(label, kind)] (recorded, else fallback)
    shortcut_stages: set
    stage_end: dict                    # stage -> last event mission time
    step_windows: dict                 # (stage, idx) -> (t_begin, t_eval) of the last attempt
    mission_state: str
    mission_reason: str
    stage_fail_reason: dict = field(default_factory=dict)   # stage -> short recorded failure reason

    @property
    def last_event_t(self):
        return self.events[-1]['t'] if self.events else 0.0

    def return_failure_reason(self):
        """A short 'why return stopped' for the failed return step (the host's last planner reason)."""
        import re
        m = re.search(r"last reason '([^']+)'", self.mission_reason or '')
        return f"no base progress · {m.group(1)}" if m else 'no base progress'

    # ---------------- construction ----------------
    @classmethod
    def from_run(cls, run):
        raw = [e for e in run.bt_events if isinstance(e, dict) and 't' in e and 'data' in e]
        raw.sort(key=lambda e: e['t'])
        labels, shortcut, stage_end, fail_reason = {}, set(), {}, {}
        kinds = {}                                   # (stage, idx) -> kind from stage_step
        for e in raw:
            d = e['data'] or {}
            stage = d.get('stage')
            if stage is None:
                continue
            stage_end[stage] = e['t']
            if e['type'] == 'stage_failed' and d.get('reason'):
                fail_reason[stage] = _short_fault(d['reason'])
            if e['type'] == 'stage_shortcut':
                shortcut.add(stage)
            if e['type'] == 'stage_begin' and isinstance(d.get('steps'), list):
                labels[stage] = [(str(s), None) for s in d['steps']]   # kinds filled from stage_step
            if e['type'] == 'stage_step' and isinstance(d.get('step'), int):
                kinds[(stage, d['step'])] = d.get('kind')
        for stage in STAGES:
            if stage not in labels:
                labels[stage] = _fallback_steps(stage)
            labels[stage] = [(lab, kinds.get((stage, i)) or kind or 'step')
                             for i, (lab, kind) in enumerate(labels[stage])]
        step_windows = cls._step_windows(raw, labels, stage_end)
        return cls(events=raw, labels=labels, shortcut_stages=shortcut, stage_end=stage_end,
                   step_windows=step_windows, mission_state=str(run.mission.get('state', 'unknown')),
                   mission_reason=str(run.mission.get('reason') or ''), stage_fail_reason=fail_reason)

    @staticmethod
    def _step_windows(raw, labels, stage_end):
        """Per (stage, idx): (t_begin, t_eval) of the step's LAST attempt. ``t_eval`` is just before
        the step completed (its own window end), so a frozen reason reflects the decision outcome."""
        begins = {}                                  # (stage, idx) -> last begin time
        order = {}                                   # stage -> [(idx, t)] in tick order (last attempt)
        last_attempt = {}
        for e in raw:
            if e['type'] != 'stage_step':
                continue
            d = e['data']
            stage, idx, att = d.get('stage'), d.get('step'), d.get('attempt', 1)
            if stage is None or not isinstance(idx, int):
                continue
            if last_attempt.get(stage) != att:
                last_attempt[stage] = att
                order[stage] = []
            begins[(stage, idx)] = e['t']
            order[stage].append((idx, e['t']))
        windows = {}
        for stage, seq in order.items():
            for k, (idx, t) in enumerate(seq):
                nxt = seq[k + 1][1] if k + 1 < len(seq) else stage_end.get(stage, t)
                windows[(stage, idx)] = (t, max(t, nxt - 1e-3))
        return windows

    # ---------------- replay ----------------
    def state_at(self, t):
        """Replay every event with ``ev.t <= t``; return ``{stage: {'status', 'steps':[{'status'}],
        'active_step'}}`` -- the node statuses the live tree would show at mission time ``t``."""
        n = {s: len(self.labels[s]) for s in STAGES}
        status = {s: 'idle' for s in STAGES}
        steps = {s: ['idle'] * n[s] for s in STAGES}
        active = None
        for e in self.events:
            if e['t'] > t:
                break
            d = e['data'] or {}
            stage = d.get('stage')
            typ = e['type']
            if stage is None or stage not in status:
                continue
            if typ == 'stage_shortcut':
                status[stage] = 'shortcut'
            elif typ == 'stage_begin':
                status[stage] = 'running'
                steps[stage] = ['idle'] * n[stage]
                active = (stage, None)
            elif typ == 'stage_step':
                i = d.get('step')
                if isinstance(i, int):
                    for j in range(min(i, n[stage])):
                        steps[stage][j] = 'success'
                    if i < n[stage]:
                        steps[stage][i] = 'running'
                    status[stage] = 'running'
                    active = (stage, i)
            elif typ == 'stage_succeeded':
                status[stage] = 'success'
                steps[stage] = ['success'] * n[stage]
                active = None
            elif typ == 'stage_failed':
                status[stage] = 'failure'
                if active and active[0] == stage and isinstance(active[1], int) and active[1] < n[stage]:
                    steps[stage][active[1]] = 'failure'
                else:
                    for j in range(n[stage]):
                        if steps[stage][j] in ('idle', 'running'):
                            steps[stage][j] = 'failure'
                            break
        out = {}
        for s in STAGES:
            ai = active[1] if (active and active[0] == s) else None
            out[s] = {'status': status[s], 'steps': [{'status': st} for st in steps[s]],
                      'active_step': ai}
        return out

    def stage_title(self, stage):
        return STAGE_TITLES.get(stage, stage or '')


# ---------------- reasons (measured from the recorded scene) ----------------
def build_reads(run, model, t):
    """The recorded scene reads at mission time ``t`` the reason/chip formatters consume."""
    row = run.row_at(t)
    bx, by, byaw = (row['x'], row['y'], row['yaw']) if row else (0.0, 0.0, 0.0)
    cart = E.estimated_cart_pose(run, model, t)
    cx, cy, cz, cyaw, src = E.carton_pose(run, model, t)
    hitch = (bx - HITCH_BEHIND * math.cos(byaw), by - HITCH_BEHIND * math.sin(byaw))
    dock = (cart[0] + DOCK_AHEAD * math.cos(cart[2]), cart[1] + DOCK_AHEAD * math.sin(cart[2]))
    return {'_t': t, 'base_xy': (bx, by), 'base_yaw': byaw, 'cart_yaw': cart[2],
            'hitch_xy': hitch, 'dock_xy': dock, 'carton_xyz': (cx, cy, cz), 'carton_src': src,
            'lin': float(row['lin']) if row else 0.0, 'coupled': bool(row and row['coupled']),
            'articulation_rad': (cart[2] - byaw + math.pi) % (2 * math.pi) - math.pi,
            'probe': (row or {}).get('probe') or {}}


def bay_target(model):
    """The transfer slot target (the scripted ground-truth place target): bay centre + the carton
    half-height above the bay top (the host logs this as the TRANSFER target)."""
    return (model.bay_center[0], model.bay_center[1], model.bay_top_z + model.carton_size[2] / 2.0)


def nav_progress(run, stage, t):
    """Fraction (0..1) along the recorded base path of ``stage`` reached by time ``t`` -- the lane
    progress, measured from the trace itself (no authored centreline)."""
    rows = [r for r in run.trace if r['action'] == stage]
    if len(rows) < 2:
        return None
    cum, upto, prev = 0.0, 0.0, rows[0]
    for r in rows[1:]:
        d = math.hypot(r['x'] - prev['x'], r['y'] - prev['y'])
        cum += d
        if r['t'] <= t:
            upto = cum
        prev = r
    return (upto / cum) if cum > 1e-6 else None


def compute_reason(timeline, run, model, stage, idx, reads):
    """Measured reason string for ``stage``'s step ``idx`` from this frame's recorded reads."""
    if stage in timeline.shortcut_stages:
        return 'start-phase shortcut — synthesized, not performed'
    bay = bay_target(model)
    if stage == 'couple':
        return M.hitch_gap(reads['hitch_xy'], reads['dock_xy'], reads['base_yaw'], reads['cart_yaw'])['reason']
    if stage == 'outbound':
        d = M.base_to_pose(reads['base_xy'], reads['base_yaw'], model.bay_center, math.pi, label='transfer bay')
        frac = nav_progress(run, stage, reads['_t'])
        prog = f'tow {frac * 100:.0f}% · ' if frac is not None else ''
        return prog + d['reason']
    if stage == 'park':
        d = M.base_to_pose(reads['base_xy'], reads['base_yaw'], model.bay_center, math.pi, label='bay')
        moving = reads['lin'] >= 0.02
        return f"{'settling' if moving else 'stationary'} {reads['lin'] * 1000:.0f} mm/s · {d['reason']}"
    if stage == 'transfer':
        if idx == 0:
            return f'slot target ({bay[0]:.2f}, {bay[1]:.2f}, {bay[2]:.2f}) m · scripted GT'
        return M.placement_error(reads['carton_xyz'], bay)['reason']
    if stage == 'verify':
        return M.placement_error(reads['carton_xyz'], bay)['reason'] + ' · qty 1'
    if stage == 'return':
        labels = timeline.labels.get('return', [])
        label = labels[idx][0] if idx < len(labels) else ''
        if timeline.mission_state == 'failed' and ('park' in label or idx == 0):
            return timeline.return_failure_reason()
        return f'return: {label}' if label else 'return'
    return ''


def compute_chips(reads):
    """Always-on 'measured from scene' chips (label, value) for the panel footer."""
    bx, by = reads['base_xy']
    chips = [('base x,y', f'{bx:.2f}, {by:.2f} m'),
             ('base yaw', f"{math.degrees(reads['base_yaw']):.0f}°"),
             ('coupled', 'yes' if reads['coupled'] else 'no'),
             ('articulation', f"{math.degrees(reads['articulation_rad']):+.0f}°"),
             ('carton z', f"{reads['carton_xyz'][2]:.2f} m"),
             ('pose source', reads['carton_src'])]
    return chips


# ---------------- panel entry assembly ----------------
def seed_reasons(timeline, run, model, step_reasons):
    """Freeze each performed step's reason at its own window end, so completed steps keep showing what
    drove them even when no recorded frame fell inside the (often sub-second) step. Steps with no
    stage_step tick of their own (a stage that succeeded before ticking a trailing leaf) are evaluated
    at the stage end; shortcut (synthesized) stages get a uniform synthesized tag."""
    for (stage, idx), (_t0, t_eval) in timeline.step_windows.items():
        reads = build_reads(run, model, t_eval)
        step_reasons[(stage, idx)] = compute_reason(timeline, run, model, stage, idx, reads)
    for stage in STAGES:
        if stage in timeline.shortcut_stages or stage not in timeline.stage_end:
            continue
        reads = build_reads(run, model, timeline.stage_end[stage])
        for i in range(len(timeline.labels.get(stage, []))):
            step_reasons.setdefault((stage, i), compute_reason(timeline, run, model, stage, i, reads))
    for stage in timeline.shortcut_stages:
        for i in range(len(timeline.labels.get(stage, []))):
            step_reasons.setdefault((stage, i), 'start-phase shortcut — synthesized')


def panel_entry(timeline, run, model, *, t, header_stage, frame_index, step_reasons,
                mission_state=None):
    """Assemble one demo-004 panel entry (tree / header / chips) for mission time ``t``.

    ``step_reasons`` is accumulated across frames: the active step's reason is recomputed live here
    (so a running decision shows the current measured value); completed steps keep their frozen value.
    """
    states = timeline.state_at(t)
    reads = build_reads(run, model, t)
    reads['_t'] = t
    for stage, st in states.items():
        ai = st['active_step']
        if ai is not None:
            step_reasons[(stage, ai)] = compute_reason(timeline, run, model, stage, ai, reads)
    stage_nodes = []
    for stage in STAGES:
        st = states[stage]
        step_nodes = []
        for i, (label, kind) in enumerate(timeline.labels[stage]):
            sstatus = st['steps'][i]['status'] if i < len(st['steps']) else 'idle'
            # Only a started step (not idle) shows its measured reason, so the panel never leaks a
            # future decision's value onto a step that has not run yet; a FAILED step shows the
            # recorded fault, not its last measured value.
            if sstatus == 'failure':
                reason = timeline.stage_fail_reason.get(stage) or step_reasons.get((stage, i), '')
            elif sstatus != 'idle':
                reason = step_reasons.get((stage, i), '')
            else:
                reason = ''
            step_nodes.append({'name': f'{stage}:{label}', 'status': sstatus,
                               'feedback': reason, 'kind': kind or 'step', 'children': []})
        stage_nodes.append({'name': f'stage:{stage}', 'status': _PANEL_STATUS.get(st['status'], 'idle'),
                            'feedback': _STAGE_FEEDBACK.get(st['status'], ''), 'kind': 'StageRetry',
                            'children': step_nodes})
    tree = {'name': 'restock mission', 'status': _roll_up([s['status'] for s in stage_nodes]),
            'feedback': '', 'kind': 'Sequence', 'children': stage_nodes}
    # Live mission state: 'running' until the terminal event, then the recorded final state.
    live_state = mission_state or (timeline.mission_state if t >= timeline.last_event_t else 'running')
    header = {'stage_title': timeline.stage_title(header_stage), 'clock': t, 'frame': frame_index,
              'mission_state': live_state}
    return {'tree': tree, 'header': header, 'chips': compute_chips(reads)}


def _roll_up(child_statuses):
    if 'failure' in child_statuses:
        return 'failure'
    if 'running' in child_statuses:
        return 'running'
    if child_statuses and all(s == 'success' for s in child_statuses):
        return 'success'
    if any(s == 'success' for s in child_statuses):
        return 'running'
    return 'idle'
