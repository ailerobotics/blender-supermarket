"""Offline commissioning-config preflight (fm460). Catches the config mistakes that each cost a
GPU commission boot: (1) a camera parent that won't resolve in the commissioning scene (no robot
prims), (2) a station support polygon/markers that project outside their camera view, (3) a station
rig that silently differs from the last known-good commissioning config.

Usage: python3 check_commissioning_config.py --config <commissioning camera_config.json>
         --known-good <last-good camera_config.json> --stations backstock_00:source,rollwagen:cart,G3_back_s2:shelf
Exit 0 iff all checks pass. Prints COMMISSION_CFG_PREFLIGHT PASS/FAIL."""
import argparse, json, math, re, sys
import numpy as np

ROBOT_PARENT = re.compile(r'(robot_base|base_link|ur_|_arm_|shoulder|forearm|wrist_[0-9])', re.I)

def _mat(m): return np.array(m, float)

def check(cfg, good, station_to_cam):
    problems, notes = [], []
    cams = cfg.get('cameras', {})
    good_parents = {c.get('parent') for c in good.get('cameras', {}).values()}
    # (1) camera parents resolve in the commissioning scene (no robot prims)
    for cid, c in cams.items():
        p = c.get('parent')
        if ROBOT_PARENT.search(str(p)) and p not in good_parents:
            problems.append(f'camera {cid!r} parent {p!r} looks like a robot prim absent from the '
                            f'commissioning scene (not in known-good parents {sorted(good_parents)})')
        elif p not in good_parents:
            problems.append(f'camera {cid!r} parent {p!r} not in the known-good commissioning parents '
                            f'{sorted(good_parents)} -- will it resolve in the commissioning scene?')
    # (2) each station's support polygon + witness project inside its camera view -- via the REAL
    #     commissioning_geometry (exact match to the GPU commissioning check; support+witness, +0.04
    #     margin, support yaw). support_uv/witness_uv are invariant to world_from_parent, so identity is
    #     sufficient for the in-frame test.
    import numpy as _np
    from twin.restock_camera.sensing import commissioning_geometry as _cg
    for station, camid in station_to_cam.items():
        if camid not in cfg.get('cameras', {}) or camid not in cfg.get('supports', {}):
            problems.append(f'station {station!r}: missing camera/support {camid!r}'); continue
        try:
            _cg(cfg, camid, world_from_parent=_np.eye(4), world_from_base=_np.eye(4))
        except Exception as e:
            problems.append(f'station {station!r} ({camid!r}): {type(e).__name__}: {e}')
    # (3) diff each station rig against known-good
    sups = cfg.get("supports", {})
    gcams, gsups = good.get('cameras', {}), good.get('supports', {})
    for station, camid in station_to_cam.items():
        if camid in gcams:
            a, b = cams.get(camid, {}), gcams[camid]
            if json.dumps(a.get('parent_from_optical'), sort_keys=True) != json.dumps(b.get('parent_from_optical'), sort_keys=True):
                pa = _mat(a['parent_from_optical'])[:3,3]; pb = _mat(b['parent_from_optical'])[:3,3]
                notes.append(f'DIFF station {station!r} camera moved {tuple(round(x,2) for x in pb)} -> {tuple(round(x,2) for x in pa)}')
            if json.dumps(a.get('intrinsics'), sort_keys=True) != json.dumps(b.get('intrinsics'), sort_keys=True):
                notes.append(f'DIFF station {station!r} camera INTRINSICS changed')
        if camid in gsups and json.dumps(sups.get(camid,{}).get('nominal_center')) != json.dumps(gsups[camid].get('nominal_center')):
            notes.append(f'DIFF station {station!r} support center {gsups[camid].get("nominal_center")} -> {sups.get(camid,{}).get("nominal_center")}')
    extra = set(cams) - set(gcams); missing = set(gcams) - set(cams)
    if extra: notes.append(f'cameras ADDED vs known-good: {sorted(extra)}')
    if missing: notes.append(f'cameras REMOVED vs known-good: {sorted(missing)}')
    return problems, notes

def v_ok(vv, H): return 0 <= vv < H

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True); ap.add_argument('--known-good', required=True)
    ap.add_argument('--stations', required=True, help='station:camera,comma-separated')
    a = ap.parse_args()
    s2c = dict(p.split(':') for p in a.stations.split(','))
    cfg = json.load(open(a.config)); good = json.load(open(a.known_good))
    problems, notes = check(cfg, good, s2c)
    for n in notes: print('NOTE', n, flush=True)
    for p in problems: print('FAIL', p, flush=True)
    ok = not problems
    print('COMMISSION_CFG_PREFLIGHT ' + ('PASS' if ok else 'FAIL'), flush=True)
    return 0 if ok else 1

if __name__ == '__main__':
    raise SystemExit(main())
