"""CPU characterisation of the collect PRE-PLACE carried-carton swing (sim-preplace-sweep-tilt-v1).

coldwin50/51/52 each tripped CARTON_TILT on the carried preplace APPROACH to the above-support
standoff (executed pitch 7.64 / 3.28 / 8.44 deg, base rock-steady) while the gate predicted <1 deg.
This bench characterises that swing from the runs' OWN commanded trajectories (the honest data that
IS logged -- the per-tick executed joint / carton attitude is NOT, so this cannot replay the exact
8 deg; that is the GPU gap coldwin53 + the new CARTON_TICK log close) and isolates the cause.

Verdict reproduced here (pure/CPU, numpy + the Isaac-free urdf_probe FK + carton_level):

  1. The COMMANDED preplace path is level: ``predict_path_tilt`` on each run's commanded trajectory is
     ~1.7 deg -- NOT the executed 3-8. So the commanded path is not the cause.
  2. STATIC SAG ruled out: the quasi-static droop model (``GravityCompensation`` K=2850) evaluated per
     waypoint along the sweep is ~constant ~1.7 deg and never approaches 3-8; the model error
     (executed - droop) is a flat 2-7 deg with NO wrist_2 at which the static geometry peaks high.
  3. TRAJECTORY SPEED ruled out: executed tilt is ANTI-correlated with the commanded peak joint speed
     (cw52 is slower-or-equal to cw50 yet tilts MORE).
  4. LEVELLER ruled out: its authority (<=0.18 deg) is dwarfed by the 3-8 deg swing (and cw50 ran with
     the leveller OFF and still swung).
  5. DRIVE PROFILE is the lever: the carried phases that stay level (lift/tuck/transit) run the SOFT
     profile (kp 3e5 / eff 400); only the STIFF preplace (kp 1e6 / eff 2000, unique to pan/preplace/
     place) swings. A reduced per-joint PD+saturation model TRACKS the slow commanded sweep for BOTH
     profiles (so the swing is not a linear position-control lag) -- the swing is a multi-body /
     finite-solver ringing the stiff drive propagates at the Isaac physics step with 5x the effort
     authority; the SOFT profile sits a step milder (lower stiffness-to-timestep margin, lower effort)
     and is the one under which the carried phases demonstrably stayed level.

The FIX (phase_drive.carried_soft_phases=['preplace']) runs the carried approach on the SOFT profile.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from twin.urdf_probe import parse_urdf, link_world_transforms
from twin.restock_manipulation import carton_level as cl
from twin.restock_manipulation.cad_tool import ARM_JOINTS
from twin.restock.physics_setup import phase_drive_gains, resolve_stiff_phases, _PHASE_DRIVE_STIFF_PHASES

# The welded handling unit (single source of truth -- restock_scenario handling_unit.dimensions_m).
CARTON_MASS_KG = 0.45
ISAAC_PHYSICS_DT = 1.0 / 60.0   # the mission physics step the drive is resolved at

_ARM_CHAIN_LINKS = ('ur_shoulder_link', 'ur_upper_arm_link', 'ur_forearm_link',
                    'ur_wrist_1_link', 'ur_wrist_2_link', 'ur_wrist_3_link')


# ----------------------------------------------------------------------- inertia

def reflected_inertia(robot, config, *, carton_mass=CARTON_MASS_KG, tool_link='ur_tool0',
                      root='ur_base_link'):
    """Locked-downstream inertia (kg*m^2) about each arm joint axis at ``config`` -- the inertia the
    joint drive must accelerate when the arm is held rigid downstream (point-mass + parallel-axis of
    every downstream link COM plus the welded carton at the tool). A diagonal approximation (no
    off-diagonal coupling), enough to size the drive dynamics. Pure/CPU; ARM-order dict."""
    cfg = cl._as_joint_map(config) if hasattr(cl, '_as_joint_map') else dict(zip(ARM_JOINTS, config))
    jmap = {j.name: j for j in robot.joints}
    world = link_world_transforms(robot, cfg, root=root)
    masses, coms = {}, {}
    for name, link in robot.links.items():
        inert = getattr(link, 'inertial', None)
        if inert is not None and getattr(inert, 'mass', 0) and name in world:
            T = np.asarray(world[name], float)
            masses[name] = float(inert.mass)
            coms[name] = T[:3, :3] @ np.asarray(inert.origin_xyz, float) + T[:3, 3]
    extra = []
    if carton_mass and tool_link in world:
        extra.append((float(carton_mass), np.asarray(world[tool_link], float)[:3, 3]))
    out = {}
    for idx, jn in enumerate(ARM_JOINTS):
        joint = jmap.get(jn)
        if joint is None or joint.child not in world:
            out[jn] = 0.0
            continue
        Tc = np.asarray(world[joint.child], float)
        origin, axis = Tc[:3, 3], Tc[:3, :3] @ np.asarray(joint.axis, float)
        axis = axis / (np.linalg.norm(axis) or 1.0)
        I = 0.0
        for ln in _ARM_CHAIN_LINKS[idx:]:
            if ln in masses:
                r = coms[ln] - origin
                r_perp = r - np.dot(r, axis) * axis
                I += masses[ln] * float(np.dot(r_perp, r_perp))
        for m, p in extra:
            r = p - origin
            r_perp = r - np.dot(r, axis) * axis
            I += m * float(np.dot(r_perp, r_perp))
        out[jn] = I
    return out


# --------------------------------------------------------------- drive dynamics

def drive_profile_contrast(soft, stiff, inertia):
    """Compare the SOFT and STIFF drive profiles on a joint of the given reflected ``inertia`` at the
    Isaac physics step. Returns the natural frequency, damping ratio and per-step phase advance
    ``dt*wn`` (a ringing indicator: >1 means the implicit solver must resolve a full oscillation inside
    one step) for each, plus the effort-authority ratio. A linear PD at these gains is over-damped, so
    the swing is NOT a linear overshoot -- but the STIFF profile sits deep in the numerically-stiff
    regime (dt*wn far over 1, 5x the effort ceiling), where a finite-iteration solver rings; the SOFT
    profile is a step milder. Pure/CPU."""
    def one(g):
        kp, kd, eff = float(g['kp']), float(g['kd']), float(g['eff'])
        wn = math.sqrt(kp / inertia) if inertia > 0 else float('inf')
        zeta = kd / (2 * math.sqrt(kp * inertia)) if inertia > 0 else float('inf')
        return {'kp': kp, 'kd': kd, 'eff': eff, 'wn': wn, 'zeta': zeta,
                'dt_wn': ISAAC_PHYSICS_DT * wn}
    s, st = one(soft), one(stiff)
    return {'soft': s, 'stiff': st,
            'effort_ratio': st['eff'] / s['eff'],
            'wn_ratio': st['wn'] / s['wn'] if s['wn'] else float('inf'),
            'zeta_ratio': st['zeta'] / s['zeta'] if s['zeta'] else float('inf')}


def simulate_tracking(times, ref_q, inertia, g, *, grav=0.0, dt=1.0 / 480.0):
    """Stable (semi-implicit, implicit-damping) per-joint PD + effort-saturation sim of a single joint
    tracking the piecewise-linear reference ``ref_q`` at ``times`` under drive gains ``g`` and reflected
    ``inertia``, with a constant gravity disturbance ``grav`` (N*m the joint must resist). Returns the
    peak |executed - commanded| tracking error (rad). Deliberately resolved finely (dt 1/480) so the
    reduced linear model is STABLE -- it shows a slow commanded sweep is tracked by both profiles, i.e.
    the swing is not a linear lag. Pure/CPU."""
    kp, kd, eff = float(g['kp']), float(g['kd']), float(g['eff'])
    T = np.asarray(times, float)
    Q = np.asarray(ref_q, float)
    th = float(Q[0])
    thd = 0.0
    peak = 0.0
    for t in np.arange(0.0, float(T[-1]), dt):
        c = float(np.interp(t, T, Q))
        cd = (float(np.interp(t + dt, T, Q)) - float(np.interp(t - dt, T, Q))) / (2 * dt)
        tau_un = kp * (c - th) + kd * (cd - thd) - grav
        if abs(tau_un) < eff:
            thd = (thd + dt / inertia * (kp * (c - th) + kd * cd - grav)) / (1 + dt * kd / inertia)
        else:
            thd = thd + dt * math.copysign(eff, tau_un) / inertia
        th = th + dt * thd
        peak = max(peak, abs(th - c))
    return peak


# ------------------------------------------------------------- carton geometry

def commanded_path_tilt(robot, waypoints, up_in_tool, *, tool_link='ur_tool0'):
    """Max carton tilt (deg) of the COMMANDED waypoint path (``predict_path_tilt``). Pure/CPU."""
    r, p, t = cl.predict_path_tilt(robot, [list(w) for w in waypoints], up_in_tool, tool_link=tool_link)
    return max(r, p)


def per_waypoint_droop(robot, waypoints, up_in_tool, gc, *, tool_link='ur_tool0'):
    """For each waypoint: (wrist_2 rad, quasi-static droop tilt deg, |gravity torque sum| N*m). The droop
    tilt is the carton attitude the K=2850 model predicts the finite servo leaves under the loaded
    gravity torque at that config -- the STATIC-SAG hypothesis. Pure/CPU."""
    rows = []
    for w in waypoints:
        cfg = list(w)
        roll, pitch, tilt = cl.droop_induced_rpy(robot, cfg, gc, tool_link=tool_link,
                                                 up_in_tool=up_in_tool)
        tau = cl.arm_gravity_torques(robot, cfg, carton_mass=gc.carton_mass, tool_link=tool_link)
        rows.append((float(w[4]), max(abs(roll), abs(pitch)), sum(abs(v) for v in tau.values())))
    return rows


# ----------------------------------------------------------------- the report

def characterise_run(robot, run_name, run_fix, up_in_tool, gc, phase_drive, *, tool_link='ur_tool0'):
    """One run's characterisation row. ``run_fix`` = {'t', 'waypoints', 'trip'}. Pure/CPU."""
    wps = run_fix['waypoints']
    times = run_fix['t']
    trip = run_fix['trip']
    cmd_tilt = commanded_path_tilt(robot, wps, up_in_tool, tool_link=tool_link)
    droop = per_waypoint_droop(robot, wps, up_in_tool, gc, tool_link=tool_link)
    droop_max = max(d[1] for d in droop)
    W = np.asarray(wps, float)
    dt = np.diff(np.asarray(times, float))
    peak_v = float(np.max(np.abs(np.diff(W, axis=0)) / dt[:, None])) if len(times) > 1 else 0.0
    w2_sweep = float(W[-1, 4] - W[0, 4])
    # reduced per-joint tracking of the full sweep (soft vs stiff), reconstructed to tool tilt.
    inertia = reflected_inertia(robot, list(W.mean(axis=0)))
    soft_g = phase_drive_gains('tuck', phase_drive)      # the SOFT profile (tuck is never stiff)
    stiff_g = phase_drive_gains('place', phase_drive)    # the STIFF profile
    track = {}
    for label, g in (('soft', soft_g), ('stiff', stiff_g)):
        # worst single-joint tracking error over the arm (each joint's own reflected inertia + grav)
        grav0 = cl.arm_gravity_torques(robot, list(W.mean(axis=0)), carton_mass=gc.carton_mass,
                                       tool_link=tool_link)
        worst = 0.0
        for ji, jn in enumerate(ARM_JOINTS):
            e = simulate_tracking(times, W[:, ji], max(inertia[jn], 1e-4), g, grav=grav0[jn])
            worst = max(worst, e)
        track[label] = math.degrees(worst)
    return {
        'run': run_name,
        'commanded_tilt_deg': round(cmd_tilt, 2),
        'droop_max_deg': round(droop_max, 2),
        'executed_peak_deg': trip['peak'],
        'model_error_deg': round(trip['peak'] - droop_max, 2),
        'peak_cmd_v_rad_s': round(peak_v, 4),
        'w2_sweep_rad': round(w2_sweep, 3),
        'reduced_track_soft_deg': round(track['soft'], 3),
        'reduced_track_stiff_deg': round(track['stiff'], 3),
        'droop_vs_w2': [(round(a, 3), round(b, 2)) for a, b, _ in droop],
    }


def run(urdf_path, fixture_path, *, weld_config=None):
    """Characterise all three runs + the drive-profile contrast + the fix selection. Pure/CPU."""
    robot = parse_urdf(urdf_path)
    fix = json.loads(open(fixture_path).read())
    phase_drive = fix['phase_drive']
    gc = cl.GravityCompensation(carton_mass=CARTON_MASS_KG).validate()
    # Capture the level weld up-axis at the carry pose of cw52 (carton level there: transit 0.03 deg).
    if weld_config is None:
        weld_config = fix['runs']['coldwin52']['waypoints'][0]
    up_in_tool = cl.carton_up_in_tool(cl.tool_rotation(robot, list(weld_config)))
    rows = [characterise_run(robot, name, rf, up_in_tool, gc, phase_drive)
            for name, rf in sorted(fix['runs'].items())]
    contrast = drive_profile_contrast(
        phase_drive['soft'], phase_drive['stiff'],
        reflected_inertia(robot, list(np.asarray(fix['runs']['coldwin52']['waypoints'], float).mean(axis=0)))['ur_wrist_2_joint'])
    # the FIX: carried_soft_phases routes preplace to the SOFT profile
    fixed_cfg = dict(phase_drive, carried_soft_phases=['preplace'])
    fix_select = {
        'baseline_preplace': phase_drive_gains('preplace', phase_drive),
        'fixed_preplace': phase_drive_gains('preplace', fixed_cfg),
        'fixed_place_stays_stiff': phase_drive_gains('place', fixed_cfg),
        'stiff_phases_baseline': sorted(resolve_stiff_phases(phase_drive)),
        'stiff_phases_fixed': sorted(resolve_stiff_phases(fixed_cfg)),
    }
    return {'rows': rows, 'contrast': contrast, 'fix': fix_select}


def _fmt(report):
    out = []
    out.append('PREPLACE SWEEP-TILT CHARACTERISATION (sim-preplace-sweep-tilt-v1)')
    out.append('run        cmd_tilt  droop_max  executed  model_err  peak_v   w2_sweep  '
               'reduced_soft reduced_stiff')
    for r in report['rows']:
        out.append('%-10s %7.2f %9.2f %9.2f %10.2f %8.4f %8.3f %12.3f %12.3f'
                   % (r['run'], r['commanded_tilt_deg'], r['droop_max_deg'], r['executed_peak_deg'],
                      r['model_error_deg'], r['peak_cmd_v_rad_s'], r['w2_sweep_rad'],
                      r['reduced_track_soft_deg'], r['reduced_track_stiff_deg']))
    c = report['contrast']
    out.append('')
    out.append('DRIVE CONTRAST @ wrist_2 (dt=1/60): soft wn=%.0f zeta=%.2f dt*wn=%.2f | '
               'stiff wn=%.0f zeta=%.2f dt*wn=%.2f | effort x%.1f  wn x%.2f  zeta x%.2f'
               % (c['soft']['wn'], c['soft']['zeta'], c['soft']['dt_wn'],
                  c['stiff']['wn'], c['stiff']['zeta'], c['stiff']['dt_wn'],
                  c['effort_ratio'], c['wn_ratio'], c['zeta_ratio']))
    f = report['fix']
    out.append('FIX: stiff_phases %s -> %s ; preplace gains %s -> %s ; place stays %s'
               % (f['stiff_phases_baseline'], f['stiff_phases_fixed'],
                  f['baseline_preplace'], f['fixed_preplace'], f['fixed_place_stays_stiff']))
    return '\n'.join(out)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--urdf', default=os.path.join(os.path.dirname(__file__), '..', 'tests',
                                                    'fixtures', 'collect_stow_arm.urdf'))
    ap.add_argument('--fixture', default=os.path.join(os.path.dirname(__file__), '..', 'tests',
                                                      'fixtures', 'preplace_sweep_commanded.json'))
    args = ap.parse_args(argv)
    print(_fmt(run(args.urdf, args.fixture)))


if __name__ == '__main__':
    main()
