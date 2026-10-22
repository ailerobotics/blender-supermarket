"""The scripted (choreographed) seven-stage timeline for the restock demo.

PURE math -- no Isaac. ``build_timeline(fps)`` precomputes a list of per-frame ``Frame``
dicts giving, for each capture frame: the base pose, the cart pose (a simple kinematic
trailer while coupled), the carton world pose + mode, the arm phase, the active
stage/step and the per-stage/step node statuses the panel shows. The Isaac driver SETS
these poses each frame and (separately) READS the resulting prim poses back to compute
the panel's reason values, so the panel is measured from the scene, not from these
scripted targets.

Geometry comes from the real mission config/scenario (mission poses, PR154 tow-lane
centrelines, cart-cage height, shelf slot). See ``data/sim-demo-video-v1/report.md``.
"""
from __future__ import annotations

import math

from . import tree_model

TWO_PI = 2 * math.pi


def _wrap(a):
    return (a + math.pi) % TWO_PI - math.pi


# ---- Geometry constants (map frame, metres/radians), from mission-scenario.json +
# ---- docs/examples/tow-lanes-mission-fragment.json. These seed the SCRIPTED motion;
# ---- the panel values are read from the live stage, not from these.
# demo-005 start layout: COLLECT is a stationary front pick (no driving) and COUPLE is a
# short straight reverse onto the cart's front drawbar. The rollwagen's open loading face AND
# its drawbar are BOTH on the cart +X front (``mobile_assets.py``: "front (+X) open for
# loading"; cart hitch at chassis-local (0.65, 0, ...)); the robot hitch is on the robot REAR
# (base-link (-0.62, 0, ...)). So the only pose where BOTH a front-face pick and a
# straight-reverse latch hold is: robot facing the tow direction (south) with the cart
# directly BEHIND it (same yaw), the cart's open front toward the robot's rear. The arm
# reaches into that open front to pick/seat the carton on the cart's upper shelf, and a short
# straight reverse seats the rear hitch on the front drawbar. Facing the tow direction from
# the start also removes demo-004's reorientation loop -- OUTBOUND is now one gentle arc.
SOUTH = -math.pi / 2
PICKUP = (-2.3, 12.75, SOUTH)             # spawn = pick pose; robot is stationary through COLLECT
START = PICKUP
PLACE_APPROACH = PICKUP                    # no drive to the cart; the arm loads it in place
LANE_OUT_START = (-2.3, 13.0, SOUTH)       # dock pose, a 0.25 m straight reverse (north) from PICKUP
CART_STAGED = (-2.3, 14.27, SOUTH)         # cart directly behind, same yaw (1.27 m dock separation)
SHELF = (-3.05, 6.0, math.pi)              # unchanged: TRANSFER/VERIFY identical to demo-004
CART_PARKING = (-9.0, 16.0, math.pi)       # unchanged: RETURN identical to demo-004
HOME = (-5.73, 16.0, 0.0)                  # unchanged

# OUTBOUND: tow the coupled train from the dock, THROUGH THE BACKROOM DOOR, into the aisle
# to the shelf -- no reorientation loop (the robot already faces the tow direction) and no
# in-place spin. The sales/backroom dividing wall is at y=11.125 (store_environment.py:
# Wall_back_{west,east}); the ONLY gap is the service portal `sales_backroom_portal` centred
# at x=-3.875 with a 2.5 m clear opening (x in [-5.125, -2.625], jambs just outside). The
# earlier demo-005 arc hugged x~-1.7 the whole descent, so it crossed the wall line at
# x~-2.06 -- i.e. straight THROUGH the solid east wall, ~0.6 m past the door edge. This path
# dips west to the portal column (base holds x~-3.70 so BOTH the base disk and the trailed
# cart cross the opening near its centre -- base at x=-3.70, cart at x=-3.58), holds the
# column south of the wall long enough for the cart to clear it aligned, then returns east and
# runs straight down to rejoin demo-005's proven westward pull-through into the shelf. The
# TRANSFER pose (SHELF) and the shelf approach arc are unchanged. Invariants (proved on CPU by
# tests/test_door_path.py): y strictly decreases (no doubling back), no reverse, min turn
# radius ~0.93 m (== demo-005's), peak cart articulation 53.2 deg (< the 55 deg budget and
# below demo-005's 55.2 deg), and every base/cart footprint along the path clears the wall and
# both jambs (>= 0.12 m) with zero intersection.
OUTBOUND_WAYPOINTS = [LANE_OUT_START,
    (-3.15, 12.162, -1.95), (-3.70, 11.55, -1.68), (-3.70, 11.125, -1.60),
    (-3.70, 10.0, -1.57), (-2.68, 9.30, -1.45), (-1.66, 9.10, -1.33),
    (-1.66, 8.20, -1.44), (-1.70, 7.0, -1.60), (-1.78, 6.55, -1.80),
    (-2.05, 6.15, -2.3), (-2.60, 6.02, -2.85), SHELF]

# PR154 return tow lane (base centreline): shelf -> cart parking spot.
RETURN_WAYPOINTS = [SHELF,
    (-4.54, 7.67, 1.456), (-3.88, 9.78, 1.571), (-3.88, 11.99, 1.571),
    (-3.88, 14.2, 1.623), (-5.54, 15.65, 3.039), (-7.74, 15.88, 3.039), CART_PARKING]

# Carton world poses (x, y, z, yaw) for the fixed modes.
P_SUPPORT = (-2.3, 13.95, 1.00, SOUTH)         # carton as-loaded on the cart's upper shelf (grasp seals; ~0.14 m proud of the front)
CAGE_SHELF_Z = 0.9325                           # cart_support_z_world (scenario)
CAGE_LOAD_BODY_XY = (0.17, 0.0)                 # COLLECT seats the carton ~0.15 m deeper so it rests fully on the shelf
CARTON_HALF_H = 0.08                            # carton height 0.16 / 2
P_SHELF = (-3.05, 4.826, 0.73 + CARTON_HALF_H, 0.0)   # slot G3_back_s2 surface_z + half

# Trailer geometry while coupled (approximate, for the shot).
HITCH_BEHIND = 0.60      # robot hitch point behind base_link along -heading
CART_TRAIL = 0.75        # cart centre behind the hitch point along -cart_heading
DOCK_AHEAD = 0.73        # cart drawbar dock ahead of the cart centre along +cart_heading;
                         # CART_TRAIL - DOCK_AHEAD = 0.02 m => a ~20 mm latched hitch gap
MAX_ARTICULATION = math.radians(65)

# Arm phases: STOW = transport/stow joint preset; TRACK = IK the gripper onto the carton
# (the arm visibly grasps/carries/places it). Resolved in the Isaac driver.
STOW, TRACK = 'stow', 'track'

# Global stretch so the finished clip lands in the ~2-4 min band the brief asks for
# (the raw segment plan is ~68 s of action). Applied uniformly in build_timeline.
DURATION_SCALE = 2.0


def _ease(u):
    """Smoothstep ease-in/out on u in [0,1]."""
    u = max(0.0, min(1.0, u))
    return u * u * (3 - 2 * u)


def _lerp_pose(a, b, u):
    u = _ease(u)
    dyaw = _wrap(b[2] - a[2])
    return (a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, _wrap(a[2] + dyaw * u))


def _resample_path(waypoints, u):
    """Position+heading at fraction u in [0,1] along a polyline by arc length."""
    u = _ease(u)
    segs = [(waypoints[i], waypoints[i + 1]) for i in range(len(waypoints) - 1)]
    lens = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in segs]
    total = sum(lens) or 1.0
    target = u * total
    acc = 0.0
    for (a, b), L in zip(segs, lens):
        if acc + L >= target or (a, b) is segs[-1]:
            t = 0.0 if L == 0 else (target - acc) / L
            t = max(0.0, min(1.0, t))
            dyaw = _wrap(b[2] - a[2])
            return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, _wrap(a[2] + dyaw * t))
        acc += L
    return waypoints[-1]


def _lerp_xyz_arc(a, b, u, lift=0.0):
    """Lerp a carton world pose (x,y,z,yaw) with an optional raised arc on z."""
    u = _ease(u)
    dyaw = _wrap(b[3] - a[3])
    z = a[2] + (b[2] - a[2]) * u + lift * math.sin(math.pi * u)
    return (a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, z, _wrap(a[3] + dyaw * u))


# ---- The scripted segment plan. Each segment: (stage, step_index, kind, duration_s, spec).
# kind in {'line','path','dwell'}; spec carries the motion + the carton/arm/event hints.
def _plan():
    seg = []

    def add(stage, step, duration, *, base=None, path=None, carton='hold', arm=STOW,
            event=None):
        seg.append(dict(stage=stage, step=step, duration=duration, base=base, path=path,
                        carton=carton, arm=arm, event=event))

    # 1 COLLECT -------------------------------------------------------------
    add('collect', 0, 3.0, base=(START, PICKUP), carton='support', arm=STOW)      # drive→pick
    add('collect', 1, 2.0, base=(PICKUP, PICKUP), carton='support', arm=TRACK)  # perceive carton
    add('collect', 2, 3.0, base=(PICKUP, PICKUP), carton=('support', 'carried'), arm=TRACK)  # grasp·lift·tuck
    add('collect', 3, 3.0, base=(PICKUP, PLACE_APPROACH), carton='carried', arm=TRACK)  # drive→rollwagen
    add('collect', 4, 1.5, base=(PLACE_APPROACH, PLACE_APPROACH), carton='carried', arm=TRACK)  # perceive deck
    add('collect', 5, 3.0, base=(PLACE_APPROACH, PLACE_APPROACH), carton=('carried', 'cage'), arm=TRACK)  # place
    add('collect', 6, 1.5, base=(PLACE_APPROACH, PLACE_APPROACH), carton='cage', arm=STOW)  # verify placed

    # 2 COUPLE --------------------------------------------------------------
    add('couple', 0, 3.0, base=(PLACE_APPROACH, LANE_OUT_START), carton='cage', arm=STOW)  # drive→cart
    add('couple', 1, 1.5, base=(LANE_OUT_START, LANE_OUT_START), carton='cage', arm=STOW)  # dock drawbar
    add('couple', 2, 1.5, base=(LANE_OUT_START, LANE_OUT_START), carton='cage', arm=STOW,
        event='couple')                                                            # attach

    # 3 OUTBOUND ------------------------------------------------------------
    add('outbound', 0, 14.0, path=OUTBOUND_WAYPOINTS, carton='cage', arm=STOW)     # tow→shelf

    # 4 PARK ----------------------------------------------------------------
    add('park', 0, 2.0, base=(SHELF, SHELF), carton='cage', arm=STOW)              # stationarize

    # 5 TRANSFER ------------------------------------------------------------
    add('transfer', 0, 1.5, base=(SHELF, SHELF), carton='cage', arm=TRACK)   # perceive slot
    add('transfer', 1, 3.5, base=(SHELF, SHELF), carton=('cage', 'shelf'), arm=TRACK)  # place on shelf
    add('transfer', 2, 1.5, base=(SHELF, SHELF), carton='shelf', arm=STOW)         # verify placed

    # 6 VERIFY --------------------------------------------------------------
    add('verify', 0, 2.0, base=(SHELF, SHELF), carton='shelf', arm=STOW)           # observe inventory

    # 7 RETURN (uncouple at parking spot) -----------------------------------
    add('return', 0, 12.0, path=RETURN_WAYPOINTS, carton='shelf', arm=STOW)        # tow→cart park
    add('return', 1, 1.5, base=(CART_PARKING, CART_PARKING), carton='shelf', arm=STOW)  # brake cart
    add('return', 2, 1.5, base=(CART_PARKING, CART_PARKING), carton='shelf', arm=STOW,
        event='uncouple')                                                          # uncouple
    add('return', 3, 2.5, base=(CART_PARKING, (-7.6, 16.0, 0.0)), carton='shelf', arm=STOW)  # back away
    add('return', 4, 3.0, base=((-7.6, 16.0, 0.0), HOME), carton='shelf', arm=STOW)  # drive→home
    return seg


def _carton_pose(mode, u, base, cart):
    """World (x,y,z,yaw) of the carton for a given mode at fraction u in a segment."""
    def carried():
        # Hover the grasped carton toward the cart (behind the south-facing base, i.e. -forward)
        # so the grasp->lift->seat motion stays on the cart side and within the arm's reach.
        c, s = math.cos(base[2]), math.sin(base[2])
        return (base[0] - c * 1.25, base[1] - s * 1.25, 1.08, base[2])

    def cage():
        c, s = math.cos(cart[2]), math.sin(cart[2])
        ox = c * CAGE_LOAD_BODY_XY[0] - s * CAGE_LOAD_BODY_XY[1]
        oy = s * CAGE_LOAD_BODY_XY[0] + c * CAGE_LOAD_BODY_XY[1]
        return (cart[0] + ox, cart[1] + oy, CAGE_SHELF_Z + CARTON_HALF_H, cart[2])

    fixed = {'support': lambda: P_SUPPORT, 'carried': carried, 'cage': cage,
             'shelf': lambda: P_SHELF}
    if isinstance(mode, tuple):
        a = fixed[mode[0]]()
        b = fixed[mode[1]]()
        return _lerp_xyz_arc(a, b, u, lift=0.25)
    if mode == 'hold':
        mode = 'cage'
    return fixed[mode]()


def _statuses(plan, seg_index, step_frac):
    """Per-stage/step node statuses at the active segment (pure time ordering)."""
    active = plan[seg_index]
    states = {}
    for stage, steps in tree_model.build_structure():
        states[stage] = {'status': tree_model.IDLE,
                         'steps': [{'status': tree_model.IDLE} for _ in steps]}
    # Everything in a segment strictly before the active one is SUCCESS.
    for j in range(seg_index):
        s = plan[j]
        states[s['stage']]['steps'][s['step']]['status'] = tree_model.SUCCESS
    # The active step is RUNNING (or SUCCESS at the very end of its segment).
    a_stage, a_step = active['stage'], active['step']
    states[a_stage]['steps'][a_step]['status'] = (
        tree_model.SUCCESS if step_frac >= 0.999 else tree_model.RUNNING)
    # Stage roll-up.
    for stage, st in states.items():
        child = [x['status'] for x in st['steps']]
        st['status'] = tree_model._roll_up(child)
    return states, a_stage, a_step


def build_timeline(fps=10):
    """Precompute the per-frame timeline. Returns (frames, meta).

    Each frame is a dict: frame, t, stage, step, base, cart, coupled, carton,
    carton_mode, arm, node_states. Pure; the Isaac driver consumes it.
    """
    if not (isinstance(fps, int) and fps > 0):
        raise ValueError('fps must be a positive int')
    plan = _plan()
    dt = 1.0 / fps
    frames = []
    # Trailer state carried across coupled frames.
    coupled = False
    cart = CART_STAGED
    cart_yaw = CART_STAGED[2]
    prev_base = None
    t = 0.0
    frame_idx = 0
    for seg_index, seg in enumerate(plan):
        n = max(1, int(round(seg['duration'] * DURATION_SCALE * fps)))
        for k in range(n):
            u = (k + 1) / n  # end-exclusive-ish so last frame of a seg reaches target
            # Base pose.
            if seg['path'] is not None:
                base = _resample_path(seg['path'], u)
            else:
                a, b = seg['base']
                base = _lerp_pose(a, b, u)
            moving = prev_base is None or math.hypot(base[0] - prev_base[0], base[1] - prev_base[1]) > 0.01
            # Coupling events fire at the start of their segment.
            if seg['event'] == 'couple' and not coupled:
                coupled = True
            # Cart pose: trailer while coupled, else static at last pose.
            if coupled:
                hx = base[0] - HITCH_BEHIND * math.cos(base[2])
                hy = base[1] - HITCH_BEHIND * math.sin(base[2])
                if moving and (hx, hy) != (cart[0], cart[1]):
                    # Towing: the cart centre points toward the hitch point (trailer).
                    cart_yaw = math.atan2(hy - cart[1], hx - cart[0])
                else:
                    # Dwell (park / transfer): let the cart settle straight behind the base so
                    # a parked train reads aligned rather than frozen mid-turn.
                    cart_yaw = _wrap(cart_yaw + 0.25 * _wrap(base[2] - cart_yaw))
                art = _wrap(cart_yaw - base[2])  # clamp to the mechanical articulation limit
                if abs(art) > MAX_ARTICULATION:
                    cart_yaw = _wrap(base[2] + math.copysign(MAX_ARTICULATION, art))
                cx = hx - CART_TRAIL * math.cos(cart_yaw)
                cy = hy - CART_TRAIL * math.sin(cart_yaw)
                cart = (cx, cy, cart_yaw)
            if seg['event'] == 'uncouple':
                coupled = False
            prev_base = base
            carton = _carton_pose(seg['carton'], u, base, cart)
            mode = seg['carton'][1] if isinstance(seg['carton'], tuple) else seg['carton']
            node_states, a_stage, a_step = _statuses(plan, seg_index, u)
            frames.append(dict(frame=frame_idx, t=round(t, 4), stage=a_stage, step=a_step,
                               base=base, cart=cart, coupled=coupled, carton=carton,
                               carton_mode=mode, arm=seg['arm'], node_states=node_states))
            t += dt
            frame_idx += 1
    # Final frame: mark the whole tree SUCCESS.
    done = {stage: {'status': tree_model.SUCCESS,
                    'steps': [{'status': tree_model.SUCCESS} for _ in steps]}
            for stage, steps in tree_model.build_structure()}
    frames[-1]['node_states'] = done
    meta = {'fps': fps, 'frames': len(frames), 'duration_s': round(len(frames) * dt, 2)}
    return frames, meta
