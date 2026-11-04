"""Least-extended collect place-on-cart approach selector (sim-place-descent-residual-v2, brief part 2).

coldwin31..41 tripped the collect set-down CARTON_TILT because the loaded arm is FAR EXTENDED over the cart
deck: the welded carton's gravity droop pitches it off level (coldwin39: reach 1.086 m -> droop 2.02 deg,
validated by ``carton_level.droop_induced_rpy`` at K=2850) AND, when ``preplace_above_place`` raised the
standoff, the IK DRIFTED the carton ~93 mm toward the +x mouth so the welded pad struck ``post_2`` and
levered it to 4.94 deg (coldwin41). The droop and the post clearance are COUPLED: moving the carton away
from the mouth posts (deeper on the deck, toward -x) costs reach -> more droop; parking the base closer to
the cart buys reach back.

This module searches the two decoupling levers on CPU and reports the combination with the SMALLEST reach
that keeps the drooped carton clear of the mouth posts and the predicted executed tilt low:

  * ``load_center_dx`` -- shift the place load-centre along the cart +x axis (``+`` toward the robot/deeper
    from the mouth); the external run-config ``cart_placement.load_center_body_xy`` key, inside the deck
    support plane + the cart balance rule.
  * ``base_closer_m`` -- park the base closer to the cart; the external run-config ``place_approach`` base
    pose moved toward the cart (base +x faces the cart, so a closer park reduces the base<-cart x offset).

For each candidate it solves the level place IK, measures the reach, the K-fitted droop tilt, the executed
(drooped) per-body clearance sweep (``place_clearance.sweep_executed``, SEAL-anchored) and the carton
load-centre drift, and returns the least-extended feasible pick. The droop tilt is the VALIDATED quantity
(reproduces coldwin39's 2.02); the absolute pad-gap carries the documented ~84 mm CPU-vs-sim offset, so the
gate reads "does the executed pose strike a mouth post" + the drift, not a sub-cm certificate. The
closed-loop backstop (``carton_level.CartonLevelServo``) covers the residual tilt below the bound.

Pure numpy + the Isaac-free URDF FK + usd-core (cart geometry); no Isaac/ROS/GPU.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from twin.restock_manipulation import carton_level as cl
from twin.restock_manipulation import place_clearance as pc
from twin.restock_manipulation.cad_tool import ARM_JOINTS

# The mouth posts the welded pad can strike at the collect set-down (the non-support bodies nearest the
# open +x face). The sweep also checks every other body; these are the ones the set-down actually risks.
MOUTH_POSTS = ('post_2', 'post_3')
TCP_LINK = 'gripper_pad_tcp'


@dataclass(frozen=True)
class Candidate:
    """One evaluated (load_center_dx, base_closer_m) approach."""
    load_center_dx: float
    base_closer_m: float
    reach_m: float
    droop_tilt_deg: float
    min_mouth_gap_mm: float
    load_center_drift_mm: float
    feasible: bool
    per_body_mm: dict


def _level_limits():
    big = 2 * math.pi
    return {'ur_elbow_joint': (0.1, math.pi), 'ur_wrist_1_joint': (-big, 0.65),
            'ur_wrist_2_joint': (0.0, big)}


def evaluate(robot, geom, bodies, anchor_config, *, load_center_xy=(0.18, 0.0), support_z=0.9325,
             yaw_cart_rel=-math.pi / 2, up_in_tool=None, gravity_comp=None, cal_offset_m=None,
             load_center_dx=0.0, base_closer_m=0.0, tilt_target_deg=1.5, drift_max_m=0.050):
    """Evaluate one candidate. ``anchor_config`` is a config whose carton is KNOWN to sit at ``load_center_xy``
    (the base<-cart anchor, e.g. coldwin39's place-end). ``load_center_dx`` shifts the place load-centre by
    that much along cart +x (toward the robot); ``base_closer_m`` parks the base that much closer to the cart
    (reduces the base<-cart x offset). Returns a :class:`Candidate`."""
    gc = gravity_comp if gravity_comp is not None else cl.GravityCompensation().validate()
    up = up_in_tool if up_in_tool is not None else cl.carton_up_in_tool(
        cl.tool_rotation(robot, anchor_config, TCP_LINK))
    T_base_cart = pc.base_from_cart_anchor(robot, anchor_config, geom, load_center_xy=load_center_xy,
                                           support_z=support_z, yaw_cart_rel=yaw_cart_rel)
    # base-closer: base +x faces the cart, so a closer park reduces the base<-cart x translation.
    T_bc = T_base_cart.copy()
    T_bc[0, 3] -= float(base_closer_m)
    if cal_offset_m is None:
        # SEAL anchor: the anchor config's level post_2 gap (so a measured flush contact reads ~0).
        cal_offset_m = pc.sweep_executed([anchor_config], robot, geom, bodies, T_base_cart,
                                         gravity_comp=None)[1].get('post_2', 0.0)
    # Perturb the KNOWN-GOOD anchor TCP (its carton sits exactly at load_center_xy, zero drift) rather than
    # reconstruct a TCP from the grip offset (which carries a systematic ~0.15 m branch error): shift it
    # toward the robot by load_center_dx along the cart +x axis, and -base_closer along the base +x axis
    # (base +x faces the cart, so parking closer moves the cart-fixed target that much toward the base).
    cart_x_axis = np.asarray(T_base_cart, float)[:3, 0]        # cart +x expressed in the base frame
    base_x_axis = np.array([1.0, 0.0, 0.0])
    anchor_tcp = cl.tool_position(robot, anchor_config, TCP_LINK)
    tcp = anchor_tcp + float(load_center_dx) * cart_x_axis - float(base_closer_m) * base_x_axis
    try:
        sol = cl.level_place_ik(robot, tuple(tcp), up, anchor_config, tool_link=TCP_LINK,
                                limits=_level_limits(), gravity_comp=None)
    except ValueError:
        return Candidate(load_center_dx, base_closer_m, float('nan'), float('nan'), float('-inf'),
                         float('nan'), False, {})
    lc = (load_center_xy[0] + float(load_center_dx), load_center_xy[1])
    cfg = [float(sol[j]) for j in ARM_JOINTS]
    reach = cl.reach_xy(robot, cfg, tool_link=TCP_LINK)
    _, _, tilt = cl.droop_induced_rpy(robot, cfg, gc, tool_link=TCP_LINK)
    _, per_body = pc.sweep_executed([cfg], robot, geom, bodies, T_bc, gravity_comp=gc,
                                    cal_offset_m=cal_offset_m)
    mouth = min(per_body[p] for p in MOUTH_POSTS if p in per_body)
    drift_xy = pc.load_center_in_cart(robot, cfg, geom, T_bc)
    drift = math.hypot(drift_xy[0] - lc[0], drift_xy[1] - lc[1])
    # Feasibility keys on the VALIDATED quantities only: the K-fitted droop tilt (reproduces coldwin39's
    # measured 2.02) and the carton DRIFT off the intended load-centre (the coldwin41 regression was a
    # ~93 mm mouth-ward drift). The absolute mouth-post gap carries the documented ~84 mm CPU-vs-sim
    # geometry offset, so it is REPORTED (min_mouth_gap_mm, per_body_mm) for corroboration but is NOT the
    # gate -- the GPU SEAL_DIAG is the real pad-clearance check.
    feasible = (tilt <= tilt_target_deg and drift <= drift_max_m)
    return Candidate(load_center_dx, base_closer_m, reach, tilt, mouth * 1000.0, drift * 1000.0, feasible,
                     {k: round(v * 1000.0, 1) for k, v in sorted(per_body.items())})


def select_least_extended(robot, geom, bodies, anchor_config, *, load_center_xy=(0.18, 0.0),
                          support_z=0.9325, yaw_cart_rel=-math.pi / 2, gravity_comp=None,
                          dx_grid=(0.0, 0.05, 0.10, 0.15), closer_grid=(0.0, 0.10, 0.15, 0.20, 0.25),
                          tilt_target_deg=1.5, drift_max_m=0.050):
    """Grid-search the two levers and return ``(best, all_candidates)``. ``best`` is the feasible candidate
    (predicted droop tilt <= ``tilt_target_deg`` AND carton drift <= ``drift_max_m``) with the SMALLEST
    base move then reach -- the least-disruptive least-extended approach; ``None`` when none is feasible
    (the caller leans harder on the closed-loop backstop). All candidates are returned sorted by reach for
    the report."""
    up = cl.carton_up_in_tool(cl.tool_rotation(robot, anchor_config, TCP_LINK))
    gc = gravity_comp if gravity_comp is not None else cl.GravityCompensation().validate()
    T_base_cart = pc.base_from_cart_anchor(robot, anchor_config, geom, load_center_xy=load_center_xy,
                                           support_z=support_z, yaw_cart_rel=yaw_cart_rel)
    cal = pc.sweep_executed([anchor_config], robot, geom, bodies, T_base_cart,
                            gravity_comp=None)[1].get('post_2', 0.0)
    cands = []
    for dx in dx_grid:
        for closer in closer_grid:
            cands.append(evaluate(robot, geom, bodies, anchor_config, load_center_xy=load_center_xy,
                                  support_z=support_z, yaw_cart_rel=yaw_cart_rel, up_in_tool=up,
                                  gravity_comp=gc, cal_offset_m=cal, load_center_dx=dx,
                                  base_closer_m=closer, tilt_target_deg=tilt_target_deg,
                                  drift_max_m=drift_max_m))
    cands.sort(key=lambda c: (float('inf') if math.isnan(c.reach_m) else c.reach_m))
    feasible = [c for c in cands if c.feasible]
    best = min(feasible, key=lambda c: (c.base_closer_m, c.reach_m)) if feasible else None
    return best, cands
