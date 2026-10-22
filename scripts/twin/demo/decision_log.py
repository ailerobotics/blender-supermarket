"""Turn per-frame scene READS into the decision-log entry that drives the panel.

Pure (no Isaac): the Isaac driver reads prim world poses into a ``reads`` dict each
frame; this module computes the reason string for the active node and the always-on
"measured from scene" chips from those reads (via :mod:`twin.demo.measures`), and
assembles the per-frame panel entry (annotated tree + header + chips). Keeping value
extraction here makes it CPU-testable against synthetic reads.
"""
from __future__ import annotations

import math

from . import choreography as C
from . import measures as M
from . import tree_model

# Static geometry the reasons compare against (the real mission poses/lane centrelines).
GEOM = {
    'pickup': C.PICKUP, 'place_approach': C.PLACE_APPROACH, 'lane_out_start': C.LANE_OUT_START,
    'shelf': C.SHELF, 'cart_parking': C.CART_PARKING, 'home': C.HOME, 'backaway': (-7.6, 16.0, 0.0),
    'outbound_wp': [(x, y) for x, y, _ in C.OUTBOUND_WAYPOINTS],
    'return_wp': [(x, y) for x, y, _ in C.RETURN_WAYPOINTS],
    'robot_radius': 0.65, 'articulation_tol': math.radians(12), 'slot_depth': 0.47,
    'cage_shelf_z': C.CAGE_SHELF_Z,
}


def compute_reason(stage, step, reads, geom=GEOM):
    """Reason string (value driving the decision) for ``stage:step`` from this frame's reads."""
    base_xy = reads['base_xy']
    base_yaw = reads['base_yaw']

    def to(name):
        p = geom[name]
        return M.base_to_pose(base_xy, base_yaw, (p[0], p[1]), p[2], label=name.replace('_', ' '))['reason']

    if stage == 'collect':
        if step == 0:
            return to('pickup')
        if step == 1:
            cx, cy, _ = reads['carton_xyz']
            return f'carton seen at ({cx:.2f}, {cy:.2f}) · 0.60×0.40 m'
        if step == 2:
            return M.carton_grasp_gap(reads['pad_xyz'], reads['carton_xyz'])['reason']
        if step == 3:
            return to('place_approach')
        if step == 4:
            # Empty cage: free height from the deck surface up to the top rail (carton not placed yet).
            deck_z = reads['cage_target_xyz'][2] - C.CARTON_HALF_H
            r = M.cage_clearance(deck_z, reads['cage_top_z'])
            return f"empty deck→rail {r['clearance_m'] * 100:.0f} cm"
        if step == 5:
            return M.placement_error(reads['carton_xyz'], reads['cage_target_xyz'])['reason']
        if step == 6:
            # Placed: clearance from the loaded carton's TOP up to the rail (< empty deck clearance).
            r = M.cage_clearance(reads['carton_top_z'], reads['cage_top_z'])
            return f"carton top→rail {r['clearance_m'] * 100:.0f} cm"
    if stage == 'couple':
        if step == 0:
            return to('lane_out_start')
        return M.hitch_gap(reads['hitch_xy'], reads['dock_xy'], base_yaw, reads['cart_yaw'])['reason']
    if stage == 'outbound':
        prog = M.lane_progress(base_xy, geom['outbound_wp'], lane_id='outbound')['reason']
        clr = M.forward_clearance(base_xy, base_yaw, reads.get('obstacles', []),
                                  robot_radius=geom['robot_radius'], brake_m=reads['brake_m'])['reason']
        return f'{prog} · {clr}'
    if stage == 'park':
        art = M.articulation_reason(reads['articulation_rad'], geom['articulation_tol'])['reason']
        return f"{to('shelf')} · {art}"
    if stage == 'transfer':
        if step == 0:
            return M.slot_free_depth(reads['carton_xyz'][1], geom['shelf'][1], geom['slot_depth'])['reason']
        return M.placement_error(reads['carton_xyz'], reads['shelf_target_xyz'])['reason']
    if stage == 'verify':
        return M.placement_error(reads['carton_xyz'], reads['shelf_target_xyz'])['reason'] + ' · qty 1'
    if stage == 'return':
        if step == 0:
            prog = M.lane_progress(base_xy, geom['return_wp'], lane_id='return')['reason']
            clr = M.forward_clearance(base_xy, base_yaw, reads.get('obstacles', []),
                                      robot_radius=geom['robot_radius'], brake_m=reads['brake_m'])['reason']
            return f'{prog} · {clr}'
        if step == 1:
            return to('cart_parking')
        if step == 2:
            return M.hitch_gap(reads['hitch_xy'], reads['dock_xy'], base_yaw, reads['cart_yaw'])['reason'] + ' · released'
        if step == 3:
            return to('backaway')
        if step == 4:
            return to('home')
    return ''


def compute_chips(reads, frame, geom=GEOM):
    """Always-on measured chips (label, value) for the panel footer."""
    base_xy, base_yaw = reads['base_xy'], reads['base_yaw']
    chips = [('base x,y', f"{base_xy[0]:.2f}, {base_xy[1]:.2f} m"),
             ('base yaw', f"{math.degrees(base_yaw):.0f}°"),
             ('coupled', 'yes' if frame['coupled'] else 'no'),
             ('articulation', f"{math.degrees(reads['articulation_rad']):+.0f}°")]
    clr = M.forward_clearance(base_xy, base_yaw, reads.get('obstacles', []),
                              robot_radius=geom['robot_radius'], brake_m=reads['brake_m'])['clearance_m']
    chips.append(('obstacle ahead', 'clear' if clr == math.inf else f'{clr:.1f} m'))
    cz = reads['carton_xyz'][2]
    chips.append(('carton z', f'{cz:.2f} m'))
    return chips


_STAGE_WORD = {tree_model.SUCCESS: 'done', tree_model.RUNNING: 'running',
               tree_model.FAILURE: 'FAILED', tree_model.IDLE: ''}


def build_entry(frame, step_reasons, chips, *, mission_state='running'):
    """Assemble one panel entry from a choreography frame + accumulated step reasons + chips.

    ``step_reasons`` maps ``'{stage}:{idx}' -> reason`` (the driver accumulates the active
    step's measured reason so completed steps keep showing what drove them).
    """
    states = {}
    for stage, steps in tree_model.build_structure():
        ns = frame['node_states'].get(stage, {})
        step_states = ns.get('steps', [])
        out_steps = []
        for i, _ in enumerate(steps):
            ss = step_states[i] if i < len(step_states) else {'status': tree_model.IDLE}
            out_steps.append({'status': ss.get('status', tree_model.IDLE),
                              'reason': step_reasons.get(f'{stage}:{i}', '')})
        states[stage] = {'status': ns.get('status', tree_model.IDLE),
                         'steps': out_steps, 'feedback': _STAGE_WORD.get(ns.get('status', tree_model.IDLE), '')}
    tree = tree_model.annotate(states)
    header = {'stage_title': tree_model.STAGE_TITLES.get(frame['stage'], frame['stage']),
              'clock': frame['t'], 'frame': frame['frame'], 'mission_state': mission_state}
    return {'tree': tree, 'header': header, 'chips': chips}
