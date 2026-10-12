#!/usr/bin/env python3
"""Isaac-Python to host-Jazzy fixture schema smoke; no GPU, DDS or app boot."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.twin.restock_manipulation.types import Observation, Sensors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ros-python',required=True)
    args = parser.parse_args()
    observation = Observation('cpu-contract-fixture','test_fixture',1.,'juice_orange',
                              'rollwagen',(.4,0.,.385,0.,0.,0.,1.),1.)
    sensors = Sensors(1.,(observation,),None,True,False,1.,1.,(0.,)*6,('rollwagen',),1.)
    child = subprocess.run([args.ros_python,'-m','scripts.twin.restock_manipulation.ros_host',
                            '--contract-smoke'],cwd=ROOT,input=json.dumps(asdict(sensors)),
                            text=True,capture_output=True,timeout=20,check=True)
    response = json.loads(child.stdout)
    assert response['result']['phase'] == 'contract_checked'
    assert response['result']['status'] == 'running'
    assert response['result']['evidence'][0]['source'] == 'test_fixture'
    assert response['result']['evidence'][0]['retail_quantity'] is None
    assert response['result']['controls'] == [{'kind':'gripper','enabled':False}]
    assert 'rclpy' not in sys.modules and 'curobo' not in sys.modules
    print(json.dumps({'status':'contract_passed','isaac_python':sys.version.split()[0],
                      **response,'gpu_execution':False,'dds_execution':False},indent=2))


if __name__ == '__main__':
    main()
