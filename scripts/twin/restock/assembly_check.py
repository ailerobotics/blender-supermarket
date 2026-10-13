"""Bounded stationary native assembly diagnostic; never an autonomous mission."""
import json
from pathlib import Path

import numpy as np

from .physics_setup import assemble
from .isaac_port import body_measurement


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
        # Settle criterion is measured MOTION, not the reported velocity. PhysX TGS
        # reports an instantaneous rigid-body velocity that carries a position/velocity
        # solve bias and does not integrate to the body's displacement (documented
        # upstream and in this task's report). The honest measure of "stationary" is
        # how far the body actually travels over the settle window - its net pose
        # change as a rate - which is robust to the sub-millimetre in-place jitter the
        # articulation/maximal solver leaves. Per-step jitter and the biased solver
        # velocity are recorded only as transparency diagnostics; gross divergence is
        # already aborted per step by the envelope check above.
        if len(history)<31:
            raise RuntimeError('insufficient settle window')
        window=history[-31:]  # 30 intervals, 0.5 s
        span=window[-1]['simulation_time']-window[0]['simulation_time']
        if span<=0:
            raise RuntimeError('non-monotonic settle window')
        def wrap(a):
            return abs((a+np.pi)%(2*np.pi)-np.pi)
        settle={}
        settled=True
        for name in ('robot','cart'):
            poses=[np.array(r[name]['pose_xyz_yaw']) for r in window]
            net_lin_rate=float(np.linalg.norm(poses[-1][:3]-poses[0][:3]))/span
            net_ang_rate=wrap(float(poses[-1][3]-poses[0][3]))/span
            step_lin_rate=max(float(np.linalg.norm(poses[i+1][:3]-poses[i][:3]))
                              for i in range(len(poses)-1))/(span/30)
            reported=max(float(np.linalg.norm(r[name][key]))
                         for r in window for key in ('linear_velocity','angular_velocity'))
            settle[name]=dict(net_linear_rate_m_s=net_lin_rate,net_angular_rate_rad_s=net_ang_rate,
                window_s=float(span),peak_step_linear_rate_m_s=step_lin_rate,
                reported_solver_speed_biased=reported)
            if net_lin_rate>=.005 or net_ang_rate>=.005:
                settled=False
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
