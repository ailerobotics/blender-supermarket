"""Bounded stationary native assembly diagnostic; never an autonomous mission."""
import json
from pathlib import Path

import numpy as np

from .physics_setup import assemble
from .isaac_port import body_measurement
from .settle_metrics import settle_report


def run(app, config, output):
    output=Path(output)
    scenario=json.loads(Path(config['scenario']).read_text())
    diagnostic=None
    if 'cart_diagnostic_variant' in config:
        from .cart_diagnostic import CartDiagnostic
        diagnostic=CartDiagnostic(output,config['cart_diagnostic_variant'])
    assembly=assemble(app=app,config=config,scenario=scenario,output=output/'assembly',
                      diagnostic_prepare=diagnostic.prepare if diagnostic else None)
    history=[]
    initial={}
    arm_initial=assembly.command_gate.measured().copy()
    for name,body in (('robot',assembly.robot_body),('cart',assembly.cart_body)):
        pose,z,_,_=body_measurement(body)
        initial[name]=np.array([pose.x,pose.y,z,pose.yaw])
    if diagnostic:
        diagnostic.initialize(assembly)
        (output/'measured-initial.json').write_text(json.dumps(dict(
            simulation_time=float(assembly.world.current_time),
            gate_baselines={name:values.tolist() for name,values in initial.items()},
            native=diagnostic.snapshot()),indent=2))
    try:
        for index in range(120):
            if diagnostic:
                diagnostic.phase=f'step-{index+1}'
            assembly.hold()
            assembly.world.step(render=True)
            row=dict(step=index+1,simulation_time=float(assembly.world.current_time))
            if diagnostic:
                row['native_diagnostic']=diagnostic.snapshot()
            for name,body in (('robot',assembly.robot_body),('cart',assembly.cart_body)):
                pose,z,linear,angular=body_measurement(body)
                state=np.array([pose.x,pose.y,z,pose.yaw])
                position,quaternion=body.get_world_pose()
                tilt=2*np.arcsin(min(1.,float(np.linalg.norm(np.asarray(quaternion)[1:3]))))
                row[name]=dict(pose_xyz_yaw=state.tolist(),linear_velocity=linear.tolist(),
                    angular_velocity=angular.tolist(),tilt_rad=float(tilt))
                if (np.linalg.norm(state[:2]-initial[name][:2])>.05 or abs(state[2]-initial[name][2])>.08
                        or tilt>.1 or np.linalg.norm(linear)>1. or np.linalg.norm(angular)>1.):
                    history.append(row)
                    raise RuntimeError(name+' exceeded stationary assembly envelope')
            row['arm_max_position_error']=float(np.max(np.abs(assembly.command_gate.measured()-arm_initial)))
            history.append(row)
            if row['arm_max_position_error']>.15:
                raise RuntimeError('arm exceeded stationary hold envelope')
        # Settle criterion is measured MOTION over the final 0.5 s window, not the
        # TGS-reported instantaneous velocity (a documented solver bias that does not
        # integrate to displacement). See settle_metrics.settle_report. Gross
        # divergence is already aborted per step by the envelope check above.
        if len(history)<31:
            raise RuntimeError('insufficient settle window')
        settled,settle=settle_report(history[-31:])  # 30 intervals, 0.5 s
        if not settled:
            raise RuntimeError('stationary base/cart translated above 5mm/s or 5mrad/s over the settle window')
        return dict(status='passed',steps=len(history),settled=True,settle=settle,
            evidence='two seconds stationary native physics; commanded arm hold and zero wheel '
                     'targets; settling gated on net measured pose displacement over the final 0.5 s '
                     'because the TGS-reported instantaneous velocity carries a solver bias that does '
                     'not integrate to motion',
            reported_velocity_is_solver_biased=True,
            autonomous_mission=False,coupling_tested=False,planner_started=False)
    finally:
        assembly.hold()
        (output/'stationary-feedback.json').write_text(json.dumps(history,indent=2)+'\n')
        if diagnostic:
            diagnostic.save()
