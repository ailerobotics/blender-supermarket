# SPDX-License-Identifier: Apache-2.0
"""Base-frame distance helpers needed by the vendored safety machine.

These two functions are vendored VERBATIM from the fleet human-safety package
(``ur10-setup/cams/human_safety/human_safety/geometry.py``) so that
``perception/safety.py`` -- copied byte-for-byte from the same source -- imports
``distance_to_base`` unchanged. The rest of the fleet ``geometry.py`` (depth lift,
projection) is NOT vendored here: the simulation detection bridge
(:mod:`twin.humans.detection`) either consumes the real detector's already-lifted
output or projects known ground-truth poses, so only the base-frame distance math
is on the decision path that sizes the safety zone.
"""
from __future__ import annotations

import numpy as np


def distance_to_base(point, base_xy=None) -> float:
    """Horizontal distance from a base-frame point to the base origin.

    Horizontal (xy) distance is used because a person's height above the floor
    is irrelevant to whether they are inside the robot's footprint.
    """
    p = np.asarray(point, dtype=float)
    bx, by = (0.0, 0.0) if base_xy is None else base_xy
    return float(np.hypot(p[0] - bx, p[1] - by))


def distance_to_envelope(point, reach_radius, base_xy=None) -> float:
    """Distance from a base-frame point to the reach-cylinder surface (>= 0)."""
    return max(0.0, distance_to_base(point, base_xy) - float(reach_radius))
