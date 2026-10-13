"""Bounded stationary native assembly diagnostic; never an autonomous mission."""
import json
from pathlib import Path

import numpy as np

from .physics_setup import assemble
from .isaac_port import body_measurement


def run(app, config, output):
    output=Path(output)
    scenario=json.loads(Path(config['scenario']).read_text())
    assembly=assemble(app=app,config=config,scenario=scenario,output=output/'assembly')
    history=[]
    initial={}
    arm_initial=assembly.command_gate.measured().copy()
    for name,body in (('robot',assembly.robot_body),('cart',assembly.cart_body)):
        pose,z,_,_=body_measurement(body)
        initial[name]=np.array([pose.x,pose.y,z,pose.yaw])
    try:
        for index in range(120):
            assembly.hold()
            assembly.world.step(render=True)
            row=dict(step=index+1,simulation_time=float(assembly.world.current_time))
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
        stable=all(max(np.linalg.norm(row[name][key]) for name in ('robot','cart')
                       for key in ('linear_velocity','angular_velocity'))<.005 for row in history[-10:])
        if not stable:
            raise RuntimeError('stationary base/cart did not settle below5mm/s and5mrad/s')
        return dict(status='passed',steps=len(history),settled=True,
            evidence='two seconds stationary native physics; commanded arm hold and zero wheel targets',
            autonomous_mission=False,coupling_tested=False,planner_started=False)
    finally:
        assembly.hold()
        (output/'stationary-feedback.json').write_text(json.dumps(history,indent=2)+'\n')
