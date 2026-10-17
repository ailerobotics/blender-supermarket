"""The ONE navigation seam: obstruction footprints -> dynobs ``Detection`` objects.

The brief is explicit that the obstruction passability layer must feed navigation
ONLY through the moving-obstacle costmap layer the ``sim-restock-dynobs-v1`` lane
owns. That layer's contract is ``twin.navigation.dynamic.Detection(position, radius,
source)`` -> ``DynamicObstacleLayer`` -> ``SensorNavigation.step(detections=...)``,
which UNIONS predicted occupancy into the sensed observation (add-only; never clears
a raw return). This module produces those ``Detection`` objects from confirmed
store-frame obstruction footprints and NOTHING ELSE -- no parallel layer, no gate, no
costmap of its own. Pure geometry; store-vision is never imported here.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

# Same-repo, pure-python import of the dynobs contract (the layer the sibling lane
# owns). Add the scripts root so ``twin.navigation.dynamic`` resolves whether this
# runs from the repo, a worktree, or the store-vision test venv.
_SCRIPTS_ROOT = str(Path(__file__).resolve().parents[2])
if _SCRIPTS_ROOT not in sys.path:
    sys.path.insert(0, _SCRIPTS_ROOT)

from twin.navigation.dynamic import Detection  # noqa: E402


def footprint_to_detection(footprint, *, source: str, pad_m: float = 0.0) -> Detection:
    """One store-frame box ``(x, y, w, h)`` metres -> a dynobs ``Detection``.

    ``position`` is the footprint centre; ``radius`` is its own plan-view extent
    (half-diagonal) plus ``pad_m`` -- the robot clearance is NOT added here, the
    nav gate adds it separately (dynobs contract). ``source`` tags the origin for
    the evidence log; it never relaxes a check.
    """
    x, y, w, h = (float(v) for v in footprint)
    if not all(math.isfinite(v) for v in (x, y, w, h)):
        raise ValueError("footprint must be finite")
    cx, cy = x + w / 2.0, y + h / 2.0
    radius = 0.5 * math.hypot(w, h) + float(pad_m)
    return Detection(position=(cx, cy), radius=radius, source=source)


def obstructions_to_detections(obstructions, *, source: str, pad_m: float = 0.0):
    """Confirmed obstructions -> dynobs ``Detection`` objects (deterministic order).

    ``obstructions`` are store-vision ``obstruction.passability.Obstruction`` objects
    (duck-typed: ``.center``, ``.radius_m``, optional ``.obstruction_id``). The
    returned list is what the caller passes verbatim to
    ``SensorNavigation.step(detections=...)``.
    """
    out = []
    for ob in obstructions:
        center = getattr(ob, "center", None)
        radius = getattr(ob, "radius_m", None)
        if center is not None and radius is not None:
            cx, cy = float(center[0]), float(center[1])
            r = float(radius) + float(pad_m)
            if not (math.isfinite(cx) and math.isfinite(cy) and math.isfinite(r)):
                continue
            out.append(Detection(position=(cx, cy), radius=r, source=source))
        else:
            out.append(footprint_to_detection(ob.footprint, source=source, pad_m=pad_m))
    return out
