#!/usr/bin/env python3
"""CPU proof for sim-robust-movegroup-collision-meshes-v1 (no GPU, no Isaac).

Reads the resolved URDF the host ships to move_group and reports, per mesh, whether MoveIt's
``resource_retriever`` would LOAD it -- before the fix (the file verbatim, bare absolute paths)
and after (``moveit_mesh_uris`` on, bare paths rewritten to ``file://`` URIs). When the real
``resource_retriever`` is importable (ROS sourced) it ALSO cross-checks every mesh against it,
so the model and the real loader are proven to agree.

Usage:
  python3 scripts/check_movegroup_mesh_uris.py \
      [--urdf ~/datasets/twin-robot-in-store-boot-v1/host-cad/mobile.urdf]

Exit 0 when: OFF loads zero link meshes (the documented defect) AND ON loads every mesh (and,
if the real retriever is present, agrees mesh-for-mesh). Exit 1 otherwise.
"""
import argparse
import os
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # scripts/ -> import twin.*

from twin.restock_host.mesh_uri import (loadable_meshes, normalize_mesh_uris,  # noqa: E402
                                        retriever_loadable)

DEFAULT_URDF = os.path.expanduser(
    '~/datasets/twin-robot-in-store-boot-v1/host-cad/mobile.urdf')


def _all_meshes(text):
    return [m.get('filename') for m in ET.fromstring(text).iter('mesh')]


def _real_retriever():
    try:
        import resource_retriever as rr
    except Exception:
        return None

    def load(filename):
        try:
            handle = rr.get(filename)
            data = handle.read() if hasattr(handle, 'read') else bytes(handle)
            return True, len(data)
        except Exception as exc:                       # noqa: BLE001 -- report any retriever error
            return False, f'{type(exc).__name__}: {exc}'
    return load


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--urdf', default=DEFAULT_URDF, help='resolved URDF move_group loads')
    args = ap.parse_args(argv)

    text = open(args.urdf).read()
    meshes = _all_meshes(text)
    fixed = normalize_mesh_uris(text)
    off_col, off_vis = loadable_meshes(text, role='collision'), loadable_meshes(text, role='visual')
    on_col, on_vis = loadable_meshes(fixed, role='collision'), loadable_meshes(fixed, role='visual')

    print(f'URDF: {args.urdf}')
    print(f'meshes: {len(meshes)} total')
    print(f'OFF (shipped, bare absolute paths): collision loadable={len(off_col)} '
          f'visual loadable={len(off_vis)}')
    print(f'ON  (moveit_mesh_uris, file:// URIs): collision loadable={len(on_col)} '
          f'visual loadable={len(on_vis)}')
    if meshes:
        ok0, reason0 = retriever_loadable(meshes[0])
        print(f'  sample OFF: {meshes[0]}\n    -> loadable={ok0} ({reason0})')
        fm0 = _all_meshes(fixed)[0]
        okf, reasonf = retriever_loadable(fm0)
        print(f'  sample ON : {fm0}\n    -> loadable={okf}')

    ok = (len(off_col) == 0 and len(off_vis) == 0
          and len(on_col) + len(on_vis) == len(meshes) and len(meshes) > 0)

    real = _real_retriever()
    if real is None:
        print('real resource_retriever: NOT importable here (model-only proof; '
              'run from a ROS-sourced shell to cross-check)')
    else:
        off_fail = sum(1 for f in meshes if not real(f)[0])
        on_ok = sum(1 for f in _all_meshes(fixed) if real(f)[0])
        print(f'real resource_retriever cross-check: OFF {off_fail}/{len(meshes)} FAIL, '
              f'ON {on_ok}/{len(meshes)} LOAD')
        if meshes:
            print(f'  real OFF sample: {real(meshes[0])}')
        ok = ok and off_fail == len(meshes) and on_ok == len(meshes)

    print('RESULT:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
