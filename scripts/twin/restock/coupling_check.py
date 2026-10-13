"""Bounded native cart-coupling validation; never an autonomous mission.

Settles the uncoupled robot+cart at the measured docking pose, requests a physical
hitch through the same controller contract the mission uses (fresh, stopped, aligned
observation only), then holds the coupled train and proves the latch keeps the two
hitch points pinned with both bodies stationary. No navigation, planner, sensing or
wheel motion is exercised; drive targets stay zero throughout.
"""
import json
from pathlib import Path

import numpy as np

from twin.mobile import DockObservation
from .physics_setup import assemble
from .isaac_port import body_measurement
from .settle_metrics import settle_report


def dock_observation(assembly, *, age=0.0):
    """Build a DockObservation from measured proprioception, as the mission port does."""
    r, rz, rv, rw = body_measurement(assembly.robot_body)
    c, cz, cv, cw = body_measurement(assembly.cart_body)
    frames = assembly.mobile.spec.manifest()
    return DockObservation(r, c, rz + frames['robot_hitch_local'][2],
        cz + frames['cart_hitch_local'][2], float(np.linalg.norm(rv)), float(np.linalg.norm(cv)),
        float(rw[2]), float(cw[2]), age=age)


def _row(assembly, index, phase, coupling=None):
    row = dict(step=index, phase=phase, simulation_time=float(assembly.world.current_time))
    for name, body in (('robot', assembly.robot_body), ('cart', assembly.cart_body)):
        pose, z, linear, angular = body_measurement(body)
        row[name] = dict(pose_xyz_yaw=[pose.x, pose.y, z, pose.yaw],
            linear_velocity=linear.tolist(), angular_velocity=angular.tolist())
    if coupling is not None:
        row['coupling'] = coupling
    return row


def run(app, config, output, *, settle_steps=150, hold_steps=120):
    output = Path(output)
    scenario = json.loads(Path(config['scenario']).read_text())
    assembly = assemble(app=app, config=config, scenario=scenario, output=output / 'assembly')
    history = []
    try:
        # 1) Settle the UNCOUPLED pair at the authored docking pose.
        for index in range(settle_steps):
            assembly.hold()
            assembly.world.step(render=True)
            history.append(_row(assembly, index + 1, 'pre_couple'))
        pre_settled, pre_settle = settle_report(history[-31:])
        if not pre_settled:
            raise RuntimeError('robot/cart did not settle before docking')

        # 2) Request the physical latch through the controller contract. The Coupler
        #    refuses unless the observation is fresh, stopped and aligned at the hitch.
        dock = dock_observation(assembly)
        coupling_before = assembly.mobile.coupling_state(dock)
        if coupling_before['joint_present']:
            raise RuntimeError('stage already coupled before dock request')
        state = assembly.mobile.attach(dock)  # authors the real hitch joint
        attach_record = dict(requested=True, post_attach_state=state,
            dock_alignment=dict(hitch_error_m=state['hitch_error_m'],
                robot_hitch_z=dock.robot_hitch_z, cart_hitch_z=dock.cart_hitch_z,
                robot_speed=dock.robot_speed, cart_speed=dock.cart_speed,
                robot_yaw=dock.robot.yaw, cart_yaw=dock.cart.yaw))

        # 3) Hold the COUPLED train and prove the latch holds every step.
        healthy_steps = 0
        max_residual = 0.0
        for index in range(hold_steps):
            assembly.hold()
            assembly.world.step(render=True)
            obs = dock_observation(assembly)
            cstate = assembly.mobile.coupling_state(obs)
            max_residual = max(max_residual, cstate['hitch_error_m'])
            row = _row(assembly, settle_steps + index + 1, 'coupled',
                       coupling=dict(healthy=cstate['healthy'], joint_present=cstate['joint_present'],
                                     hitch_error_m=cstate['hitch_error_m'],
                                     articulation_rad=float(obs.robot.yaw - obs.cart.yaw)))
            history.append(row)
            if not cstate['healthy']:
                raise RuntimeError(f'coupling lost at hold step {index + 1}: '
                                   f"residual {cstate['hitch_error_m']:.4f} m")
            healthy_steps += 1
        settled, settle = settle_report(history[-31:])
        if not settled:
            raise RuntimeError('coupled train translated above 5mm/s or 5mrad/s over the settle window')
        articulations = [abs(r['coupling']['articulation_rad']) for r in history if 'coupling' in r]
        return dict(status='passed', settled=True, coupled=True,
            steps=len(history), settle_steps=settle_steps, hold_steps=healthy_steps,
            pre_couple_settle=pre_settle, coupled_settle=settle, attach=attach_record,
            max_hitch_residual_m=float(max_residual),
            max_articulation_rad=float(max(articulations)),
            evidence='native physical hitch created from a fresh stopped aligned observation and '
                     'held for the full window; latch health gated on measured hitch residual and '
                     'articulation, stationarity on measured displacement; drive targets zero throughout',
            autonomous_mission=False, towing_tested=False, planner_started=False)
    finally:
        assembly.hold()
        (output / 'coupling-feedback.json').write_text(json.dumps(history, indent=2) + '\n')
