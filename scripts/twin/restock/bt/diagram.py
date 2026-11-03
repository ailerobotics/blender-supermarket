"""Render the restock mission behaviour tree as a diagram **generated from the code**.

This is a *pure* module (no ``py_trees``, no simulator): it reconstructs the authored tree
structure from the same single-source-of-truth modules the live tree is built from --
:mod:`twin.restock.bt.stage_phases` (the per-stage ``(label, kind)`` plans) and
:mod:`twin.restock.bt.events` (the event contract the reactive guards and the retry decorator
react to) -- plus a small metadata registry (:data:`IMPL` / :data:`GUARD_EVENTS`) naming the
``file:function`` that implements each fixed node. Because the stage subtrees are built from
``PLANS``, *adding a stage or a step there makes it appear here automatically* (that is the
add-a-stage recipe in ``docs/behaviour_tree.md``).

Three products, all from one :func:`build_model`:

* :func:`render_svg`  -- a standalone, static SVG of the whole tree (node type + stage + impl +
  the events each node reacts to), coloured by the run status when a run alignment is supplied.
* :func:`render_html` -- a self-contained HTML page: the same tree as an interactive DOM, the
  coldwin16 walkthrough video embedded, and every stage/step node clickable to seek the video to
  the moment that node was active in the run.
* :func:`align_to_run` -- replay a run's ``bt_events.jsonl`` (via :mod:`twin.demo.bt_timeline`) and
  its ``mission-frames/frames.json`` to colour each node by the sim-time range it was active and to
  map that range to a video timecode.

A parity test (``test_restock_bt_diagram.py``) asserts, when ``py_trees`` is importable, that this
reconstruction matches the *real* serialised tree node-for-node, so the diagram cannot silently
drift from the code it documents.
"""
from __future__ import annotations

import bisect
import html
import json
from pathlib import Path

from . import events as _events
from . import stage_phases as _sp

# --------------------------------------------------------------------------------------------------
# Node metadata registry -- the file:function that implements each FIXED node (the stage subtrees are
# generated from PLANS below). Keyed by a stable role slug. ``cls`` is the py_trees class name the
# live view serialises (so the parity test can match it); ``category`` is the brief's
# sequence/selector/action/condition/decorator classification.
# --------------------------------------------------------------------------------------------------
SEQUENCE, DECORATOR, ACTION, CONDITION = 'sequence', 'decorator', 'action', 'condition'

IMPL = {
    'restock_root':  ('tree.py', 'build_tree'),
    'event_pump':    ('leaves.py', 'EventPump'),
    'reactive':      ('tree.py', 'build_reactive'),
    'health_guard':  ('leaves.py', 'HealthGuard'),
    'human_safety':  ('leaves.py', 'HumanSafety'),
    'obstacle_avoid': ('leaves.py', 'ObstacleAvoid'),
    'baseline_guard': ('leaves.py', 'BaselineGuard'),
    'mission':       ('tree.py', 'build_mission'),
    'stage_retry':   ('leaves.py', 'StageRetry'),
    'stage_steps':   ('tree.py', 'build_stage'),
    'phase_gate':    ('leaves.py', 'PhaseGate'),
    'finalize':      ('leaves.py', 'Finalize'),
}

# The events / guard triggers each reactive leaf reacts to. Derived strings are cross-checked against
# events.EVENT_TYPES by the unit test so a renamed event cannot rot this table silently.
GUARD_EVENTS = {
    'health_guard':   ('stuck', 'low_battery'),
    'human_safety':   ('human_detected', 'human_cleared'),
    'obstacle_avoid': ('dynamic_obstacle', 'obstacle_cleared', 'blocked_aisle'),
    'baseline_guard': ('baseline_expired',),
}

# What each reactive leaf does when its event fires (for the diagram's "handled by" column).
GUARD_ACTION = {
    'health_guard':   'brake to safe park + honest mission.fail (FAILS -> tears tree down)',
    'human_safety':   'slow-zone: slow; stop-zone: safety hold + pause arm; resume when clear',
    'obstacle_avoid': 'safety hold + request nav re-plan; resume when clear',
    'baseline_guard': 'hold + re-commission the station baseline via the stations provider',
}


def _n(name, role, cls, category, *, stage='', kind='', reacts=(), memory=None, children=None):
    impl_file, impl_fn = IMPL.get(role, ('', ''))
    impl = f'scripts/twin/restock/bt/{impl_file}:{impl_fn}' if impl_file else ''
    return {
        'name': name, 'role': role, 'cls': cls, 'category': category, 'stage': stage,
        'kind': kind, 'reacts': list(reacts), 'memory': memory, 'impl': impl,
        'children': list(children or []),
    }


def _stage_steps(stage, *, plans, timeline, return_uncouple):
    """The ordered ``(label, kind)`` steps of a stage. A run alignment's recorded labels win (they
    carry the exact run's mode tags, e.g. ``[ground truth]``); otherwise the authored PLANS, with the
    uncouple-at-parking return plan selected when requested."""
    if timeline is not None and stage in getattr(timeline, 'labels', {}):
        return list(timeline.labels[stage])
    if stage == 'return' and return_uncouple:
        return [(lbl, k) for lbl, k in _sp.RETURN_UNCOUPLE_PLAN]
    return [(lbl, k) for lbl, k in plans.get(stage, [])]


def build_model(*, plans=None, stage_order=None, timeline=None, return_uncouple=False):
    """Build the full behaviour-tree model (a nested dict of annotated nodes) from the code.

    * ``plans`` -- the stage -> ``[(label, kind)]`` plans (defaults to ``stage_phases.PLANS``; a test
      passes an extended copy to register a dummy stage -- the add-a-stage recipe).
    * ``stage_order`` -- the stages the mission drives, in order (defaults to ``plans`` order).
    * ``timeline`` -- an optional :class:`twin.demo.bt_timeline.BtTimeline`; when given, each stage's
      recorded step labels (with their run-specific mode tags) and the uncouple-return choice are
      taken from the run, so the diagram matches that run exactly.
    * ``return_uncouple`` -- without a timeline, select the uncouple-at-parking return plan.
    """
    plans = dict(_sp.PLANS) if plans is None else dict(plans)
    if timeline is not None and getattr(timeline, 'shortcut_stages', None) is not None:
        return_uncouple = 'return' in getattr(timeline, 'labels', {}) and \
            any(lbl == 'uncouple' for lbl, _ in timeline.labels.get('return', []))
    stages = list(stage_order) if stage_order is not None else list(plans.keys())

    # Reactive subtree: the four guards, in tick order (tree.build_reactive).
    reactive = _n('reactive', 'reactive', 'Sequence', SEQUENCE, memory=False, children=[
        _n('health_guard', 'health_guard', 'HealthGuard', CONDITION,
           reacts=GUARD_EVENTS['health_guard']),
        _n('human_safety', 'human_safety', 'HumanSafety', ACTION,
           reacts=GUARD_EVENTS['human_safety']),
        _n('obstacle_avoid', 'obstacle_avoid', 'ObstacleAvoid', ACTION,
           reacts=GUARD_EVENTS['obstacle_avoid']),
        _n('baseline_guard', 'baseline_guard', 'BaselineGuard', ACTION,
           reacts=GUARD_EVENTS['baseline_guard']),
    ])

    # Mission subtree: one StageRetry subtree per active stage, then finalize.
    stage_nodes = []
    for stage in stages:
        steps = _stage_steps(stage, plans=plans, timeline=timeline, return_uncouple=return_uncouple)
        gates = [_n(f'{stage}:{label}', 'phase_gate', 'PhaseGate', CONDITION,
                    stage=stage, kind=kind) for label, kind in steps]
        steps_seq = _n(f'{stage} steps', 'stage_steps', 'Sequence', SEQUENCE,
                       stage=stage, memory=True, children=gates)
        stage_nodes.append(_n(f'stage:{stage}', 'stage_retry', 'StageRetry', DECORATOR,
                              stage=stage, reacts=sorted(_events.STAGE_FAILURE_TYPES),
                              children=[steps_seq]))
    stage_nodes.append(_n('finalize', 'finalize', 'Finalize', ACTION))
    mission = _n('mission', 'mission', 'Sequence', SEQUENCE, memory=True, children=stage_nodes)

    root = _n('restock_root', 'restock_root', 'Sequence', SEQUENCE, memory=False, children=[
        _n('event_pump', 'event_pump', 'EventPump', ACTION, reacts=sorted(_events.EVENT_TYPES)),
        reactive,
        mission,
    ])
    return root


def iter_nodes(node, depth=0):
    """Depth-first (name, node, depth) walk in tick order."""
    yield node, depth
    for child in node['children']:
        yield from iter_nodes(child, depth + 1)


# --------------------------------------------------------------------------------------------------
# Run alignment: colour each node by the sim-time range it was active, mapped to a video timecode.
# --------------------------------------------------------------------------------------------------
class _RunShim:
    """The minimal duck-typed ``run`` ``twin.demo.bt_timeline.BtTimeline.from_run`` reads."""
    def __init__(self, bt_events, mission):
        self.bt_events = bt_events
        self.mission = mission


def _read_jsonl(path):
    out = []
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _frame_sim_times(run_dir):
    """Sorted list of per-frame simulation_time for the run's walkthrough frames (the video the brief
    links to is these frames at a fixed fps). Empty if the run has no frames manifest."""
    man_path = Path(run_dir) / 'mission-frames' / 'frames.json'
    if not man_path.exists():
        return []
    man = json.loads(man_path.read_text())
    frames = man['frames'] if isinstance(man, dict) and 'frames' in man else man
    frames = sorted(frames, key=lambda f: f['file'])
    return [float(f['simulation_time']) for f in frames]


def _sim_to_video(sim_t, frame_times, fps):
    """Video time (seconds) of the first frame whose simulation_time >= ``sim_t`` (the whole-run mp4
    plays those frames at ``fps``). ``None`` when there are no frames."""
    if not frame_times:
        return None
    i = bisect.bisect_left(frame_times, sim_t)
    i = min(max(i, 0), len(frame_times) - 1)
    return i / float(fps)


def align_to_run(run_dir, *, fps=20.0):
    """Replay ``run_dir`` and return an alignment the renderers colour the tree with:

    ``{'stage': {<stage>: {status, sim_start, sim_end, video_start, video_end}},
       'step':  {(<stage>, <idx>): {status, sim_start, sim_end, video_start, video_end}},
       'fired': {<event_type>: count},
       'mission_state', 'mission_reason', 'sim_end', 'video_duration', 'fps'}``

    Statuses come from :mod:`twin.demo.bt_timeline` (the canonical BT-event replay); the sim windows
    come from the recorded ``sim_time`` on each event, mapped to a video timecode via ``frames.json``.
    """
    from twin.demo import bt_timeline as B

    run_dir = Path(run_dir)
    bt_events = _read_jsonl(run_dir / 'bt_events.jsonl')
    mission = {}
    mj = run_dir / 'mission.json'
    if mj.exists():
        try:
            mission = json.loads(mj.read_text())
        except json.JSONDecodeError:
            mission = {}
    timeline = B.BtTimeline.from_run(_RunShim(bt_events, mission))
    frame_times = _frame_sim_times(run_dir)

    # Per-stage and per-step sim windows, straight from the recorded sim_time on each event.
    stage_begin, stage_end, stage_status = {}, {}, {}
    step_begin, step_end, step_status = {}, {}, {}
    fired = {}
    for e in sorted(bt_events, key=lambda e: e.get('t', 0.0)):
        d = e.get('data') or {}
        stage = d.get('stage')
        typ = e.get('type')
        st = d.get('sim_time')
        if typ in _events.EVENT_TYPES:
            fired[typ] = fired.get(typ, 0) + 1
        if stage is None or st is None:
            continue
        st = float(st)
        if typ == 'stage_begin':
            stage_begin[stage] = st
            stage_status[stage] = 'running'
        elif typ == 'stage_step':
            idx = d.get('step')
            if isinstance(idx, int):
                step_begin.setdefault((stage, idx), st)
                step_end[(stage, idx)] = st
                step_status[(stage, idx)] = 'success'
                # a step begins where the previous one ended
                if idx > 0 and (stage, idx - 1) in step_begin:
                    step_end[(stage, idx - 1)] = st
        elif typ == 'stage_succeeded':
            stage_end[stage] = st
            stage_status[stage] = 'success'
        elif typ in ('stage_failed', 'stage_degraded'):
            stage_end[stage] = st
            stage_status[stage] = 'failure' if typ == 'stage_failed' else 'degraded'
        elif typ == 'stage_shortcut':
            stage_status[stage] = 'shortcut'

    # Close the final step window of each stage at the stage end.
    for (stage, idx) in list(step_begin):
        if (stage, idx) not in step_end or step_end[(stage, idx)] <= step_begin[(stage, idx)]:
            step_end[(stage, idx)] = stage_end.get(stage, step_begin[(stage, idx)])
        # mark the last-observed step of a failed stage as failure
        if stage_status.get(stage) == 'failure':
            last_idx = max(i for (s, i) in step_begin if s == stage)
            if idx == last_idx:
                step_status[(stage, idx)] = 'failure'

    def _window(begin, end, key):
        b = begin.get(key)
        en = end.get(key, b)
        return {
            'sim_start': b, 'sim_end': en,
            'video_start': _sim_to_video(b, frame_times, fps) if b is not None else None,
            'video_end': _sim_to_video(en, frame_times, fps) if en is not None else None,
        }

    stage_align = {}
    for stage in stage_begin:
        w = _window(stage_begin, stage_end, stage)
        w['status'] = stage_status.get(stage, 'idle')
        stage_align[stage] = w
    step_align = {}
    for key in step_begin:
        w = _window(step_begin, step_end, key)
        w['status'] = step_status.get(key, 'success')
        step_align[key] = w

    sim_end = max([*stage_end.values(), *(step_end.values() or [0.0])], default=0.0)
    return {
        'stage': stage_align, 'step': step_align, 'fired': fired,
        'mission_state': timeline.mission_state, 'mission_reason': timeline.mission_reason,
        'sim_end': sim_end, 'fps': fps,
        'video_duration': (len(frame_times) / fps) if frame_times else None,
        'n_frames': len(frame_times),
    }


def annotate_model(model, align):
    """Attach, in place, an ``active`` dict to every node from a run alignment (see
    :func:`align_to_run`). Stage and step nodes get their window + status; reactive guards get
    ``armed`` / ``fired`` from the run's event log; infra nodes (event_pump, the composites) get
    ``active`` for the whole run. Returns ``model``."""
    fired = align.get('fired', {})
    for node, _ in iter_nodes(model):
        role = node['role']
        if role == 'stage_retry':
            w = align['stage'].get(node['stage'], {})
            node['active'] = dict(w, scope='stage')
        elif role == 'stage_steps':
            # the steps sequence spans its children's windows
            kids = [align['step'].get((node['stage'], i)) for i in range(len(node['children']))]
            kids = [k for k in kids if k]
            if kids:
                node['active'] = {
                    'status': align['stage'].get(node['stage'], {}).get('status', 'idle'),
                    'sim_start': min(k['sim_start'] for k in kids if k['sim_start'] is not None),
                    'sim_end': max(k['sim_end'] for k in kids if k['sim_end'] is not None),
                    'video_start': min((k['video_start'] for k in kids
                                        if k['video_start'] is not None), default=None),
                    'video_end': max((k['video_end'] for k in kids
                                      if k['video_end'] is not None), default=None),
                    'scope': 'stage'}
            else:
                node['active'] = {'status': 'idle', 'scope': 'stage'}
        elif role == 'phase_gate':
            # recover the index from the position in the parent; filled by the caller via index map
            node.setdefault('active', {'status': 'idle', 'scope': 'step'})
        elif role in GUARD_EVENTS:
            hits = sum(fired.get(ev, 0) for ev in GUARD_EVENTS[role])
            node['active'] = {'status': 'fired' if hits else 'armed', 'fired': hits, 'scope': 'guard'}
        else:  # event_pump / root / reactive / mission / finalize -- ran throughout
            node['active'] = {'status': 'active', 'scope': 'infra'}
    # fill phase-gate windows now that we can index them within each stage
    for node, _ in iter_nodes(model):
        if node['role'] == 'stage_steps':
            for i, gate in enumerate(node['children']):
                w = align['step'].get((node['stage'], i))
                if w:
                    gate['active'] = dict(w, scope='step')
                else:
                    gate['active'] = {'status': 'idle', 'scope': 'step'}
    return model


# --------------------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------------------
STATUS_COLOR = {
    'success': '#2ea043', 'running': '#d29922', 'failure': '#da3633', 'degraded': '#bb8009',
    'shortcut': '#6e7681', 'armed': '#484f58', 'fired': '#1f6feb', 'active': '#238636',
    'idle': '#30363d',
}
CATEGORY_FILL = {
    SEQUENCE: '#15213a', DECORATOR: '#2d2140', ACTION: '#10281b', CONDITION: '#2a2410',
}


def _status_of(node):
    return (node.get('active') or {}).get('status', 'idle')


def _fmt_t(v):
    if v is None:
        return '—'
    m, s = divmod(float(v), 60.0)
    return f'{int(m):d}:{s:04.1f}'


def _reacts_label(node):
    r = node.get('reacts') or []
    if not r:
        return ''
    if node['role'] == 'event_pump':
        return f'drains all {len(r)} event types'
    if node['role'] == 'stage_retry':
        return 'retries: ' + ', '.join(r)
    return 'on: ' + ', '.join(r)


def render_svg(model, *, align=None, title='restock mission behaviour tree'):
    """A standalone, static SVG of the whole tree. Each row: a status dot, the node name, its
    py_trees class, its implementing ``file:function``, and the events it reacts to. When ``align``
    is present each stage/step row also shows the sim window + the video timecode it maps to."""
    rows = list(iter_nodes(model))
    row_h, indent, pad_top = 24, 24, 96
    width = 1400
    height = pad_top + len(rows) * row_h + 24
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}" font-family="system-ui,sans-serif" font-size="13">']
    out.append(f'<rect width="{width}" height="{height}" fill="#0d1117"/>')
    out.append(f'<text x="18" y="30" fill="#e6edf3" font-size="18" font-weight="700">'
               f'{html.escape(title)}</text>')
    sub = 'generated from twin.restock.bt (stage_phases.PLANS + events)'
    if align:
        sub += (f' · aligned to run: mission {align["mission_state"]}'
                f' · sim {align["sim_end"]:.0f}s · video {_fmt_t(align.get("video_duration"))}')
    out.append(f'<text x="18" y="50" fill="#8b949e" font-size="12">{html.escape(sub)}</text>')
    # legend
    lx = 18
    for label, key in [('sequence', SEQUENCE), ('decorator', DECORATOR), ('action', ACTION),
                       ('condition', CONDITION)]:
        out.append(f'<rect x="{lx}" y="62" width="14" height="14" rx="3" '
                   f'fill="{CATEGORY_FILL[key]}" stroke="#30363d"/>')
        out.append(f'<text x="{lx + 20}" y="73" fill="#8b949e" font-size="11">{label}</text>')
        lx += 95
    for label, key in [('success', 'success'), ('running', 'running'), ('failure', 'failure'),
                       ('armed/idle', 'armed'), ('ran', 'active')]:
        out.append(f'<circle cx="{lx + 6}" cy="69" r="6" fill="{STATUS_COLOR[key]}"/>')
        out.append(f'<text x="{lx + 16}" y="73" fill="#8b949e" font-size="11">{label}</text>')
        lx += 92

    y = pad_top
    for node, depth in rows:
        x = 18 + depth * indent
        color = STATUS_COLOR.get(_status_of(node), '#30363d')
        fill = CATEGORY_FILL.get(node['category'], '#161b22')
        box_w = width - x - 18
        out.append(f'<rect x="{x}" y="{y}" width="{box_w}" height="{row_h - 4}" rx="4" '
                   f'fill="{fill}" stroke="#30363d"/>')
        out.append(f'<circle cx="{x + 12}" cy="{y + (row_h - 4) / 2}" r="5" fill="{color}"/>')
        name = html.escape(node['name'])
        out.append(f'<text x="{x + 26}" y="{y + 14}" fill="#e6edf3" font-weight="600">{name}</text>')
        meta = f'{node["cls"]} · {node["category"]}'
        if node.get('memory') is not None:
            meta += f' · memory={str(node["memory"]).lower()}'
        out.append(f'<text x="{x + 240}" y="{y + 14}" fill="#8b949e" font-size="11">'
                   f'{html.escape(meta)}</text>')
        if node['impl']:
            out.append(f'<text x="{x + 470}" y="{y + 14}" fill="#58a6ff" font-size="10">'
                       f'{html.escape(node["impl"].split("/")[-1])}</text>')
        react = _reacts_label(node)
        act = node.get('active') or {}
        detail = react
        if act.get('scope') in ('stage', 'step') and act.get('video_start') is not None:
            detail = (f'video {_fmt_t(act["video_start"])}–{_fmt_t(act.get("video_end"))}'
                      f'  (sim {act.get("sim_start"):.1f}–{float(act.get("sim_end") or 0):.1f}s)')
            if react:
                detail += '  ·  ' + react
        if len(detail) > 96:
            detail = detail[:95] + '…'
        if detail:
            out.append(f'<text x="{x + 700}" y="{y + 14}" fill="#8b949e" font-size="10">'
                       f'{html.escape(detail)}</text>')
        y += row_h
    out.append('</svg>')
    return '\n'.join(out)


def _node_html(node, align_present):
    status = _status_of(node)
    act = node.get('active') or {}
    color = STATUS_COLOR.get(status, '#30363d')
    cls_attr = f'node cat-{node["category"]}'
    seek = ''
    vs = act.get('video_start')
    if align_present and vs is not None:
        seek = f' data-seek="{vs:.2f}" role="button" tabindex="0"'
        cls_attr += ' seekable'
    meta_bits = [f'<span class=cls>{html.escape(node["cls"])}</span>',
                 f'<span class=cat>{node["category"]}</span>']
    if node.get('memory') is not None:
        meta_bits.append(f'<span class=mem>memory={str(node["memory"]).lower()}</span>')
    if node['impl']:
        meta_bits.append(f'<span class=impl>{html.escape(node["impl"])}</span>')
    react = _reacts_label(node)
    if react:
        meta_bits.append(f'<span class=react>{html.escape(react)}</span>')
    if act.get('scope') == 'guard':
        meta_bits.append(f'<span class=win>{"fired ×%d" % act.get("fired", 0) if act.get("fired") else "armed · never triggered"}</span>')
    if act.get('scope') in ('stage', 'step') and vs is not None:
        meta_bits.append(f'<span class=win>▶ video {_fmt_t(vs)}–{_fmt_t(act.get("video_end"))} '
                         f'· sim {act.get("sim_start"):.1f}–{float(act.get("sim_end") or 0):.1f}s</span>')
    dot = f'<span class=dot style="background:{color}"></span>'
    row = (f'<div class="row"{seek}>{dot}<span class=nm>{html.escape(node["name"])}</span>'
           f'<span class=meta>{" ".join(meta_bits)}</span></div>')
    kids = ''.join(_node_html(c, align_present) for c in node['children'])
    return f'<div class="{cls_attr}">{row}{kids}</div>'


def render_html(model, *, align=None, video_url='', title='restock mission behaviour tree',
                video_label='', svg_filename=None):
    """A self-contained HTML page: the tree as an interactive, coloured DOM; the walkthrough video
    embedded (``video_url``); every seekable node jumps the video to the moment it was active."""
    align_present = align is not None
    tree_html = _node_html(model, align_present)
    sub = 'Generated from the code (twin.restock.bt). Click any coloured stage/step node to jump the video to when it ran.'
    state_badge = ''
    if align:
        reason = f' — {html.escape(align["mission_reason"])}' if align.get('mission_reason') else ''
        state_badge = (f'<span class="badge s-{align["mission_state"]}">mission '
                       f'{html.escape(align["mission_state"])}{reason}</span>'
                       f'<span class=badge>sim {align["sim_end"]:.0f}s · '
                       f'{align.get("n_frames", 0)} frames @ {align["fps"]:.0f} fps</span>')
    video_block = ''
    if video_url:
        cap = html.escape(video_label or video_url)
        video_block = (f'<video id=vid src="{html.escape(video_url)}" controls preload=metadata '
                       f'playsinline></video><div class=vcap id=vcap>{cap}</div>')
    svg_link = (f'<a class=dl href="{html.escape(svg_filename)}" download>download SVG</a>'
                if svg_filename else '')
    return _HTML_TEMPLATE.format(
        title=html.escape(title), sub=sub, state_badge=state_badge,
        video_block=video_block, tree=tree_html, svg_link=svg_link)


_HTML_TEMPLATE = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
 :root{{color-scheme:dark}}
 body{{font:14px/1.45 system-ui,sans-serif;margin:0;background:#0d1117;color:#c9d1d9}}
 header{{padding:12px 16px;background:#161b22;border-bottom:1px solid #30363d;position:sticky;top:0;z-index:5}}
 h1{{font-size:16px;margin:0 0 4px}} .sub{{color:#8b949e;font-size:12px;max-width:70ch}}
 .badges{{margin-top:6px;display:flex;flex-wrap:wrap;gap:6px}}
 .badge{{padding:2px 8px;border-radius:10px;font-size:12px;background:#21262d;border:1px solid #30363d}}
 .badge.s-complete{{background:#1f6f3f;border-color:#2ea043;color:#fff}}
 .badge.s-failed{{background:#5a2a2a;border-color:#da3633;color:#fff}}
 .wrap{{display:flex;flex-wrap:wrap;gap:16px;padding:14px 16px;max-width:1500px;margin:0 auto;align-items:flex-start}}
 .col-video{{flex:1 1 420px;position:sticky;top:96px}}
 .col-tree{{flex:2 1 620px;min-width:340px}}
 video{{width:100%;border-radius:8px;background:#000}}
 .vcap{{color:#8b949e;font-size:12px;margin:6px 2px}}
 .dl{{color:#7cc4ff;font-size:12px}}
 .node{{margin-left:16px;border-left:2px solid #21262d;padding-left:8px}}
 .node:first-child{{margin-left:0;border-left:none;padding-left:0}}
 .row{{display:flex;flex-wrap:wrap;align-items:baseline;gap:8px;padding:3px 6px;border-radius:5px;margin:2px 0}}
 .cat-sequence>.row{{background:#15213a}} .cat-decorator>.row{{background:#2d2140}}
 .cat-action>.row{{background:#10281b}} .cat-condition>.row{{background:#2a2410}}
 .row.has>.row{{}}
 .seekable>.row{{cursor:pointer}} .seekable>.row:hover{{outline:1px solid #58a6ff}}
 .dot{{width:9px;height:9px;border-radius:50%;flex:0 0 auto;align-self:center}}
 .nm{{font-weight:600;color:#e6edf3}}
 .meta{{color:#8b949e;font-size:11px;display:flex;flex-wrap:wrap;gap:8px}}
 .cls{{color:#d2a8ff}} .cat{{color:#8b949e}} .mem{{color:#8b949e}}
 .impl{{color:#58a6ff;font-family:ui-monospace,monospace}}
 .react{{color:#f0a868}} .win{{color:#7ee787}}
 h2{{font-size:13px;color:#8b949e;margin:4px 0 8px;text-transform:uppercase;letter-spacing:.04em}}
</style></head><body>
<header>
 <h1>{title}</h1>
 <div class=sub>{sub}</div>
 <div class=badges>{state_badge}</div>
</header>
<div class=wrap>
 <div class=col-video>
  <h2>coldwin16 walkthrough</h2>
  {video_block}
  <div style="margin-top:8px">{svg_link}</div>
 </div>
 <div class=col-tree>
  <h2>Behaviour tree</h2>
  {tree}
 </div>
</div>
<script>
 var vid=document.getElementById('vid'), vcap=document.getElementById('vcap');
 function seek(t){{ if(!vid) return; vid.currentTime=t; vid.play&&vid.play().catch(function(){{}});
   if(vcap) vcap.textContent='seeked to '+t.toFixed(1)+'s'; }}
 document.querySelectorAll('.seekable>.row[data-seek]').forEach(function(r){{
   var t=parseFloat(r.getAttribute('data-seek'));
   r.addEventListener('click',function(){{seek(t);}});
   r.addEventListener('keydown',function(e){{if(e.key==='Enter'||e.key===' '){{e.preventDefault();seek(t);}}}});
 }});
</script>
</body></html>"""
