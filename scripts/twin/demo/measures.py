"""Pure geometry that turns raw scene reads into a (value, reason) for a decision node.

Every function here takes plain numbers (read from the live USD prim world transforms by
:mod:`twin.demo.scene_reader`) and returns a small dict with the numeric value(s) and a
short human ``reason`` string for the panel. No typed scene constants: the inputs are the
measured poses; only formatting lives here. These are the SAME quantities the autonomous
navigation/manip gates compute (base-to-pose tolerance, forward braking-zone clearance,
hitch gap, lane remaining, slot free space), reproduced read-only for the panel.
"""
from __future__ import annotations

import math


def wrap_pi(a):
    """Wrap an angle to [-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def _deg(a):
    return math.degrees(a)


def base_to_pose(base_xy, base_yaw, target_xy, target_yaw, *, label):
    """Planar distance + yaw error from the base to a named target pose."""
    dx, dy = target_xy[0] - base_xy[0], target_xy[1] - base_xy[1]
    dist = math.hypot(dx, dy)
    yaw_err = abs(wrap_pi(target_yaw - base_yaw))
    return {'dist_m': dist, 'yaw_err_rad': yaw_err,
            'reason': f'base {dist:.2f} m / {_deg(yaw_err):.0f}° to {label}'}


def hitch_gap(robot_hitch_xy, cart_dock_xy, robot_yaw, cart_yaw):
    """Gap (m) and yaw misalignment (rad) between the robot hitch point and the cart
    drawbar dock -- the couple-stage latch quality."""
    gap = math.hypot(cart_dock_xy[0] - robot_hitch_xy[0], cart_dock_xy[1] - robot_hitch_xy[1])
    yaw_err = abs(wrap_pi(cart_yaw - robot_yaw))
    return {'gap_m': gap, 'yaw_err_rad': yaw_err,
            'reason': f'hitch gap {gap * 1000:.0f} mm / yaw {_deg(yaw_err):.1f}°'}


def polyline_length(points):
    """Total length of a polyline through ``points`` (list of (x, y))."""
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))


def _project_point_segment(p, a, b):
    """Return (foot_point, t, dist) projecting p onto segment a-b (t clamped to [0,1])."""
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    t = 0.0 if denom == 0 else ((px - ax) * dx + (py - ay) * dy) / denom
    t = max(0.0, min(1.0, t))
    fx, fy = ax + t * dx, ay + t * dy
    return (fx, fy), t, math.hypot(px - fx, py - fy)


def lane_progress(base_xy, waypoints, *, lane_id):
    """Progress along an authored tow-lane centreline from the base's nearest point.

    ``waypoints`` is the ordered polyline (list of (x, y)); returns fraction done,
    metres done, metres remaining and the total lane length."""
    total = polyline_length(waypoints)
    cum = 0.0
    best = None  # (dist_to_lane, metres_done)
    for i in range(len(waypoints) - 1):
        a, b = waypoints[i], waypoints[i + 1]
        seg_len = math.hypot(b[0] - a[0], b[1] - a[1])
        _, t, d = _project_point_segment(base_xy, a, b)
        done = cum + t * seg_len
        if best is None or d < best[0]:
            best = (d, done)
        cum += seg_len
    done_m = best[1] if best else 0.0
    remaining = max(0.0, total - done_m)
    frac = 0.0 if total == 0 else done_m / total
    return {'lane_id': lane_id, 'frac': frac, 'done_m': done_m, 'remaining_m': remaining,
            'total_m': total,
            'reason': f'{lane_id}: {frac * 100:.0f}% · {remaining:.1f} m left'}


def forward_clearance(base_xy, base_yaw, obstacles, *, robot_radius, brake_m, lookahead=2.5):
    """Distance to the nearest obstacle AHEAD in the forward braking corridor (the
    direction-aware braking-zone quantity). An obstacle counts only if its centre is ahead
    of the base, within the forward corridor (lateral offset <= robot_radius + its radius)
    and within the look-ahead window. Returns the along-heading gap to the obstacle SURFACE
    (``inf`` when the corridor is clear) -- a positive 'X m ahead' reading, never a
    footprint-overlap negative. ``obstacles`` is a list of (x, y, radius)."""
    hx, hy = math.cos(base_yaw), math.sin(base_yaw)
    window = max(brake_m, lookahead)
    best = math.inf
    for ox, oy, orad in obstacles:
        along = (ox - base_xy[0]) * hx + (oy - base_xy[1]) * hy  # projection onto heading ray
        if along <= 0:
            continue  # behind or abeam the base
        perp = abs((ox - base_xy[0]) * (-hy) + (oy - base_xy[1]) * hx)
        if perp > robot_radius + orad:
            continue  # outside the forward corridor
        gap = along - orad  # distance ahead to the obstacle surface
        if gap <= 0 or gap > window:
            continue  # abeam/overlapping (not ahead), or beyond the look-ahead
        best = min(best, gap)
    txt = 'clear' if best == math.inf else f'{best:.1f} m ahead'
    return {'clearance_m': best, 'brake_m': brake_m,
            'reason': f'fwd zone {txt}'}


def cage_clearance(carton_top_z, cage_top_z):
    """Vertical clearance (m) between the loaded carton's top and the cage TOP RAIL -- the
    rollwagen cage headroom. Positive whenever the carton fits under the rail."""
    clearance = cage_top_z - carton_top_z
    return {'clearance_m': clearance,
            'reason': f'cage clearance {clearance * 100:.0f} cm under rail'}


def slot_free_depth(carton_front_y, slot_front_y, slot_depth_m, *, facing_sign=-1.0):
    """Free depth (m) remaining in the shelf slot ahead of the placed carton front face.

    ``facing_sign`` is the outward +Y/-Y sign of the slot mouth; for G3_back the slot
    opens toward +Y so a carton pushed back leaves depth toward the mouth."""
    used = abs(carton_front_y - slot_front_y)
    free = max(0.0, slot_depth_m - used)
    return {'slot_free_m': free,
            'reason': f'slot free {free * 100:.0f} cm of {slot_depth_m * 100:.0f} cm'}


def placement_error(placed_xyz, target_xyz):
    """3-D error (mm) of the placed carton centroid vs the slot target -- the verify gate."""
    err = math.sqrt(sum((p - t) ** 2 for p, t in zip(placed_xyz, target_xyz)))
    return {'error_mm': err * 1000.0,
            'reason': f'placed Δ {err * 1000:.0f} mm from target'}


def articulation_reason(angle_rad, tol_rad):
    """Format the coupled articulation (base↔cart yaw) angle against its tolerance."""
    ok = abs(angle_rad) <= tol_rad
    return {'angle_rad': angle_rad, 'tol_rad': tol_rad, 'within': ok,
            'reason': f'articulation {_deg(angle_rad):+.0f}° (tol ±{_deg(tol_rad):.0f}°)'}


def carton_grasp_gap(pad_xyz, carton_xyz):
    """Suction-pad to carton-centre gap (mm) -- a coarse grasp residual (see
    ``box_surface_gap`` for the true contact gap to the carton surface)."""
    gap = math.sqrt(sum((a - b) ** 2 for a, b in zip(pad_xyz, carton_xyz)))
    return {'gap_mm': gap * 1000.0,
            'reason': f'contact gap {gap * 1000:.0f} mm at grasp'}


def box_surface_gap(point, center, yaw, dims):
    """Distance (m) from ``point`` to the surface of an axis-aligned-in-its-own-yaw box
    (``center``, z-rotation ``yaw``, full ``dims`` = (dx, dy, dz)); 0 when the point is
    inside the box. The true suction contact gap at grasp (the gripper penetrating the
    carton envelope reads 0, i.e. sealed)."""
    c, s = math.cos(-yaw), math.sin(-yaw)
    dx, dy, dz = point[0] - center[0], point[1] - center[1], point[2] - center[2]
    lx = c * dx - s * dy          # into the box's local frame
    ly = s * dx + c * dy
    hx, hy, hz = dims[0] / 2.0, dims[1] / 2.0, dims[2] / 2.0
    ox = max(abs(lx) - hx, 0.0)
    oy = max(abs(ly) - hy, 0.0)
    oz = max(abs(dz) - hz, 0.0)
    return math.sqrt(ox * ox + oy * oy + oz * oz)
