"""Offline proof for the fm459 backroom transfer BAY (reach + camera view).

The bay's shelf registration rig (camera + support + markers + lamp + solids) is the store shelf rig
rigidly transformed by T(x,y)->(-x-5.05,-y+21.25) (180deg about z) into the backroom. A rigid transform
PRESERVES the base->surface and camera->surface relative geometry, so the KNOWN-WORKING store arm reach
and camera view transfer exactly. This proves that:
  1. REACH -- the base(park)->bay-surface relative pose EQUALS the base(approach)->store-shelf relative
     pose (so the arm reaches the bay exactly as it reaches the store shelf), and the surface sits inside
     the arm workspace box.
  2. CAMERA VIEW -- the repositioned shelf camera's optical->surface relative pose EQUALS the store
     camera's (same calibrated view), and the surface projects inside the image within near/far.
CPU only; no Isaac/ROS. Exit 0 iff both hold.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

STORE_CAM = '/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mission/camera-config-mast.json'
BAY_CAM = '/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mission/camera-config-bay.json'
HOST = '/home/manishsaini/datasets/twin-robot-in-store-boot-v1/host-cad/host.json'
STORE_BASE = (-3.05, 6.0, math.pi)     # store shelf_approach (known-working base park)
BAY_BASE = (-2.0, 15.25, 0.0)          # backroom bay park (mission shelf_approach)


def _base_mat(p):
    c, s = math.cos(p[2]), math.sin(p[2])
    return np.array([[c, -s, 0, p[0]], [s, c, 0, p[1]], [0, 0, 1, 0], [0, 0, 0, 1]], float)


def _rig(cam_path):
    d = json.loads(Path(cam_path).read_text())
    sh = d['supports']['shelf']
    cam = np.array(d['cameras']['shelf']['parent_from_optical'], float)
    nc = sh['nominal_center']
    surf = np.array([nc[0], nc[1], nc[2], 1.0])
    intr = d['cameras']['shelf']['intrinsics']
    return cam, surf, intr


def main():
    ok = True
    store_cam, store_surf, _ = _rig(STORE_CAM)
    bay_cam, bay_surf, intr = _rig(BAY_CAM)
    # 1a. REACH: base->surface relative pose identical store vs bay
    rel_store = np.linalg.inv(_base_mat(STORE_BASE)) @ store_surf
    rel_bay = np.linalg.inv(_base_mat(BAY_BASE)) @ bay_surf
    reach_match = float(np.linalg.norm(rel_store[:3] - rel_bay[:3]))
    host = json.loads(Path(HOST).read_text())
    wmin, wmax = host['workspace_min'], host['workspace_max']
    in_ws = all(wmin[i] <= rel_bay[i] <= wmax[i] for i in range(3))
    reach_ok = reach_match < 1e-6 and in_ws
    print(f"REACH base->surface rel: store={np.round(rel_store[:3],3).tolist()} bay={np.round(rel_bay[:3],3).tolist()} "
          f"match={reach_match:.2e} in_workspace={in_ws} -> {'PASS' if reach_ok else 'FAIL'}", flush=True)
    ok = ok and reach_ok
    # 1b. CAMERA VIEW: optical->surface relative identical store vs bay (same calibrated view)
    vs_store = np.linalg.inv(store_cam) @ store_surf
    vs_bay = np.linalg.inv(bay_cam) @ bay_surf
    view_match = float(np.linalg.norm(vs_store[:3] - vs_bay[:3]))
    # project the bay surface into the bay camera; z=depth (optical +z forward)
    x, y, z = vs_bay[0], vs_bay[1], vs_bay[2]
    near, far = intr['near'], intr['far']
    u = intr['fx'] * (x / z) + intr['cx'] if z > 1e-6 else -1
    v = intr['fy'] * (y / z) + intr['cy'] if z > 1e-6 else -1
    in_img = 0 <= u <= intr['width'] and 0 <= v <= intr['height']
    in_depth = near <= z <= far
    view_ok = view_match < 1e-6 and in_img and in_depth
    print(f"CAMERA optical->surface rel: match={view_match:.2e} depth_z={z:.3f} (near={near},far={far}) "
          f"px=({u:.0f},{v:.0f}) in_image={in_img} in_depth={in_depth} -> {'PASS' if view_ok else 'FAIL'}", flush=True)
    ok = ok and view_ok
    print('BACKROOM_BAY_PROOF ' + ('PASS' if ok else 'FAIL'), flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
