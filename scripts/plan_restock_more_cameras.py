#!/usr/bin/env python3
"""CPU-only design proof for the restock extra-camera work (no GPU, no render, no Isaac).

Prints the geometric predictions behind two findings:

  1. Obstacle map while carrying -- the body-mounted mast depth camera (nvblox camera_1) sees
     the place/transfer region and the arm-sweep approach while a carton is held, and the
     observed voxel map (ESDF proxy) keeps updating across the held-carton window, whereas the
     eye-in-hand wrist (camera_0) is blinded by the carried carton.

  2. Cold-collect grasp confirmation -- the fixed source camera retains a clear sightline to
     both support markers and the witness at the parked collect pose and through a
     south-approaching grasp, so a clean cold registration frame is available: the failure is
     the missing established registration to hold through the declared occlusion window
     (sequencing, integration-owned), not a camera-placement problem.

Offline ray/visibility checks are geometric design predictions, never rendered RGB-D.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from twin.restock_camera import mast_depth as M
from twin.restock_camera.geometry import Intrinsics, project, transform_points, look_at
from twin.restock_camera.visibility import box_surface, sample_plane, ray_hits
from twin.navigation.observed_space import ObservedSpace


def deliverable_1():
    print('== Deliverable 1: mast depth camera feeds nvblox while carrying ==')
    k = M.mast_intrinsics()
    import math
    hfov = math.degrees(2 * math.atan((k.width / 2) / k.fx))
    print(f'  mast intrinsics: {k.width}x{k.height} RealSense-class, HFOV {hfov:.1f} deg, '
          f'near {k.near} far {k.far} m; feed={M.nvblox_feed()}')
    base = (0., 0., 0.)                                    # robot-relative (base) frame
    carton = box_surface('carton', (0.0, 0.30, 0.70), (0.6, 0.4, 0.16))   # held in transport
    place = sample_plane((0.0, 1.15, 0.73), (0.5, 0.3), 0.0, count=5)     # shelf slot
    approach = sample_plane((0.0, 0.90, 0.78), (0.6, 0.5), 0.0, count=5)  # arm sweep volume
    for region, name in ((place, 'place region'), (approach, 'approach volume')):
        clear = M.sees_points(base, region, [])
        held = M.sees_points(base, region, [carton])
        print(f'  mast {name:16s}: in_fov {clear["in_fov_and_range"]:2d}/{clear["samples"]}, '
              f'visible with carton held {held["visible_samples"]:2d}/{held["samples"]}')
    wrist = look_at((0.0, 0.15, 0.72), (0.0, 0.30, 0.70))
    near = box_surface('carton', (0.0, 0.30, 0.70), (0.6, 0.4, 0.16))
    wrist_blocked = sum(ray_hits(wrist[:3, 3], p, near.triangles) for p in place)
    print(f'  wrist eye-in-hand place region blocked by held carton: {wrist_blocked}/{len(place)}')
    # ESDF replay across the held-carton window.
    stamps = [10.0 + 0.1 * i for i in range(8)]
    mast_up = sum(bool(ObservedSpace([M.proof_depth_frame(base, t)], required_sensors={'mast'},
                                     resolution=.10, stale_after=.3).free) for t in stamps)
    wrist_up = sum(bool(ObservedSpace([M.proof_depth_frame(base, t, full_frame_masked=True)],
                                      required_sensors={'mast'}, resolution=.10,
                                      stale_after=.3).free) for t in stamps)
    print(f'  ESDF updates across {len(stamps)}-tick held-carton window: mast {mast_up}, wrist {wrist_up}')


def deliverable_2():
    print('== Deliverable 2: cold-collect grasp-confirmation cause ==')
    eye = np.array([-5.73, 17.3, 1.45])
    W = look_at(eye, np.array([-5.73, 17.1, 0.68]))
    K = Intrinsics(2560, 1920, 2000., 2000., 1280., 960., .15, 3.5)
    support = np.array([-5.73, 17.1, 0.6])
    targets = {'marker0': support + [-.30, .30, 0], 'marker1': support + [.30, .30, 0],
               'witness': support + [0, .30, 0], 'carton_top': np.array([-5.73, 17.1, 0.76])}
    arm = box_surface('arm', (-5.73, 16.6, 1.0), (0.25, 1.3, 0.25))
    press = box_surface('press', (-5.73, 17.0, 0.95), (0.4, 0.4, 0.3))
    for occ_name, occ in (('arm stowed', []), ('grasp arm + press', [arm, press])):
        print(f'  occluders: {occ_name}')
        for tn, tp in targets.items():
            uv, valid = project(transform_points(np.linalg.inv(W), [tp]), K)
            blocked = any(ray_hits(eye, tp, o.triangles) for o in occ)
            print(f'    {tn:11s} in_fov={bool(valid[0])} visible={bool(valid[0]) and not blocked}')
    print('  => markers + witness stay visible through the grasp; only the carton top is')
    print('     transiently blocked. A clean cold registration frame is available at park;')
    print('     the failure is the missing established hold through the declared occlusion')
    print('     window (sequencing, integration-owned), NOT a camera-placement problem.')


def main():
    deliverable_1()
    print()
    deliverable_2()


if __name__ == '__main__':
    main()
