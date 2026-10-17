"""Vendored fleet human-safety decision core (ported, not re-implemented).

``messages.py`` and ``safety.py`` are copied **byte-for-byte** from the fleet
package ``ur10-setup/cams/human_safety/human_safety`` (the detector + safety layer
deployed on the real rig and exercised in the earlier Isaac port
``data/human-detect-isaac-sim-v1``). ``geometry.py`` vendors only the base-frame
distance helpers that ``safety.py`` imports. No radius, score threshold, dwell or
time budget is altered here -- ``tests/test_vendor_parity.py`` asserts the two
copied files still match the fleet source when that checkout is present, so a
future drift is caught rather than silently forked.

The sim-side bridge (:mod:`twin.humans.detection`) feeds this core either the real
detector's output on the mission camera topics or ground-truth-projected
detections (explicit option); the zone/hysteresis/dwell judgement is the fleet's,
unchanged.
"""
from .messages import (CameraHealth, Detection2D, Detection3D, DistanceStatus,
                       Method, SafetyState, Track, Zone)
from .safety import SafetyEvaluator, SafetyParams

__all__ = ['CameraHealth', 'Detection2D', 'Detection3D', 'DistanceStatus', 'Method',
           'SafetyState', 'Track', 'Zone', 'SafetyEvaluator', 'SafetyParams']

# Path to the fleet source the vendored files were copied from (for the parity
# test; absent on machines without the ur10-setup checkout, which skips the test).
FLEET_SOURCE = ('/home/manishsaini/firstmate/projects/ur10-setup/'
                'cams/human_safety/human_safety')
