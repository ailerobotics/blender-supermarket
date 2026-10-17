"""Pure axiswise slot-fit predicate shared by the place-phase gate and the preflight.

The place-phase controller refuses an undersized or occluded target (see
``controller.py`` ``_advance`` 'unreachable or undersized target'). A handling unit
fits a measured slot free space when, on EVERY axis, the free dimension is finite
and at least the unit dimension plus a fixed placement margin.

This is intentionally orientation-fixed: products keep their planogram orientation,
so ``free_dimensions`` and ``dimensions`` are compared axis-for-axis in the same
frame, never by a sorted footprint. Both sequences are the arm-base frame (x, y, z).
Keeping this one predicate shared means a ``fit_preflight`` 'ok' is exactly what the
place gate will accept and a preflight 'won't-fit' is exactly what it would reject.
"""
import math

# Per-axis clearance the place-phase gate requires beyond the unit dimension.
PLACEMENT_MARGIN_M = 0.02


def axis_fits(free_dimensions, dimensions, *, margin=PLACEMENT_MARGIN_M):
    """True iff every axis of ``free_dimensions`` is finite and >= dimension + margin.

    Mirrors the place-phase controller's size check exactly. Returns False (never
    raises) on a malformed length, a non-finite free dimension, or any axis that
    does not clear its unit dimension plus the margin.
    """
    if len(free_dimensions) != 3 or len(dimensions) != 3:
        return False
    return not any(not math.isfinite(a) or a < b + margin
                   for a, b in zip(free_dimensions, dimensions))
