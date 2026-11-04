"""Run the store-vision realogram->planoreset pipeline, READ-ONLY (venv subprocess).

Invoked by :func:`smart_verify_storevision.cross_check_realogram` with the
store-vision ``.venv`` python (which has pydantic + store-vision installed), because
the realogram model pulls pydantic that the plain CPU test python lacks. Reads a JSON
request on stdin, builds a store-vision ``BayRealogram`` (observed) and a core
``Shelf`` (planogram target) from the smart-verify shelf observation, runs
``realogram.diff.diff_bay`` -> ``planoreset.verify.verify_from_realogram`` ->
``planoreset.verify.deviation_events``, and prints the result JSON on stdout.

It NEVER writes to or mutates store-vision; it only imports and calls its code. Any
failure is reported as ``{"available": false, "reason": ...}`` so the caller degrades.

Request shape (stdin JSON)::

    {"bay_id": str,
     "observed_slots":  [{"id","deck_index","sku","facings","store_region","level_z"}],
     "planogram_slots": [{"id","expected_sku","facings","store_region","level_z"}],
     "min_overlap": float?}
"""
import json
import sys


def _main():
    try:
        req = json.load(sys.stdin)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({'available': False, 'reason': f'bad request: {exc}'}))
        return 0
    try:
        import storevision  # noqa: F401
    except Exception:  # noqa: BLE001 - try the explicit src path once
        src = req.get('storevision_src')
        if src:
            sys.path.insert(0, src)
    try:
        from storevision.core.config import Shelf, ShelfSlot
        from storevision.realogram.diff import diff_bay, PlanogramDiff
        from storevision.realogram.model import BayRealogram, RealogramSlot
        from storevision.planoreset.verify import verify_from_realogram, deviation_events
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({'available': False,
                          'reason': f'store-vision import failed: {type(exc).__name__}: {exc}'}))
        return 0

    try:
        bay_id = req['bay_id']
        min_overlap = float(req.get('min_overlap', 0.25))

        def _region(r):
            return tuple(r) if isinstance(r, (list, tuple)) and len(r) == 4 else None

        plano_slots = [
            ShelfSlot(id=s['id'], expected_sku=s.get('expected_sku'),
                      facings=int(s.get('facings', 1)),
                      store_region=_region(s.get('store_region')),
                      attrs={'level_z': s.get('level_z')} if s.get('level_z') is not None else {})
            for s in req.get('planogram_slots', [])
        ]
        shelf = Shelf(id=bay_id, store_region=_region(req.get('store_region')),
                      level_z=req.get('level_z'), slots=plano_slots)
        obs_slots = [
            RealogramSlot(id=s['id'], deck_index=int(s.get('deck_index', 0)),
                          sku=s.get('sku'), facings=int(s.get('facings', 1)),
                          store_region=_region(s.get('store_region')),
                          level_z=s.get('level_z'),
                          confidence=float(s.get('confidence', 0.0)))
            for s in req.get('observed_slots', [])
        ]
        bay = BayRealogram(bay_id=bay_id, store_region=_region(req.get('store_region')),
                           slots=obs_slots)

        bay_diff = diff_bay(bay, shelf, min_overlap=min_overlap)
        diff = PlanogramDiff(bays=[bay_diff])
        verifications = verify_from_realogram(diff)
        out = {
            'available': True,
            'source': getattr(storevision, '__file__', 'storevision'),
            'compliant': bool(diff.is_compliant()),
            'diff': diff.to_dict(),
            'verifications': [v.to_dict() for v in verifications],
            'deviation_events': [],
        }
        try:
            out['deviation_events'] = [e.to_dict()
                                       for e in deviation_events(verifications, now=0.0)]
        except Exception as exc:  # noqa: BLE001
            out['deviation_events_note'] = f'{type(exc).__name__}: {exc}'
        print(json.dumps(out))
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({'available': False,
                          'reason': f'realogram diff failed: {type(exc).__name__}: {exc}'}))
    return 0


if __name__ == '__main__':
    raise SystemExit(_main())
