"""Single source of truth for the collect PLACE support (the Rollwagen DECK) top-face height.

sim-place-support-z-consistency-v1.

The cold ground-truth collect set-down used to aim the carton at the warm-shortcut
SHELF slot -- ``support_z_world`` 0.9325 world, the Rollwagen UPPER shelf (measured
top face ~0.964 world) -- while the intended support is the DECK
(``/Rollwagen/chassis/deck``, measured top face ~0.2565 world, ~0.68 m lower). So the
place target, the planned set-down and therefore the contact-triggered weld-release
threshold all sat ~0.68 m ABOVE the real deck: the release fired in mid-air and the
carton fell off the cart (coldwin44/45 ``verify_place``: 'carton not resting on deck',
final carton z ~-0.22). Moving the place-point xy changed nothing (<2 cm) because xy
was never the cause -- the HEIGHT was.

Measured world heights (coldwin45 ``mission.usda`` /Rollwagen, cart at the station;
``ur_base_link`` world z 0.4999):

    deck prim origin / centre    0.2315   (/Rollwagen/chassis translate z + deck centre)
    deck TOP face                0.2565   (deck cube half-thickness 0.025 above the centre)
    shelf centre                 0.9515   (shelf translate z 0.72 above the chassis)
    shelf TOP face               0.9640   (shelf cube half-thickness 0.0125)
    legacy constant 0.9325       0.9325   (scenario cart_support_z_world / GT place_slot / place_select)

The "0.23 vs 0.52 vs 0.9325" numbers reconciled: 0.23 is the deck prim ORIGIN (the top
face is 0.2565); 0.9325 is the legacy shelf constant; and 0.52 was the carton-centre z
in ``ur_base_link`` where the carton was released -- i.e. 0.9325 - 0.4999 + half-height,
the legacy-shelf set-down, ~0.68 m above the deck (not a deck contact at all). In
coldwin39-42 the carton bottom at that height pressed the Rollwagen UPPER SHELF (top
0.964 world), which is the surface that produced the set-down tilt step -- never the
0.2565 deck.

This module holds the pure, CPU-checkable pieces of the fix: resolving the support
top-face z (the max world z of the support prim's AABB), substituting it into the GT
place slot so the place/descent/release/verify all share ONE height, and the
four-height consistency gate ``config_preflight`` runs so a stale shelf constant can
never feed the collect descent again. The live USD read (``BBoxCache``) lives in
``isaac_factory.CollectTargetBridge._support_top_z_world``; everything here is pure.
"""
from __future__ import annotations

import math


def support_top_z_world(aabb_min_z, aabb_max_z):
    """The support prim's TOP-face world z: the greater of its world-aligned AABB z bounds.

    The deck is a thin slab, so the top face is the AABB max-z. Pure/CPU -- the caller
    (``CollectTargetBridge``) computes the world AABB from the live stage and passes its
    z bounds. Raises on a non-finite input (fail closed; never an invented height)."""
    lo, hi = float(aabb_min_z), float(aabb_max_z)
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError('support_top_z_world needs finite AABB z bounds')
    return max(lo, hi)


def resolve_support_prim(stock_supports, selector=None):
    """sim-cart-support-select-v1: resolve WHICH Rollwagen support the collect carton is placed on,
    from the cart asset's ``stock_supports`` list (``twin.mobile_assets.author_cart`` returns
    ``[/Rollwagen/chassis/deck, /Rollwagen/chassis/shelf]``) and a RUNTIME ``selector``.

    ``selector`` (``cart_placement.support_prim``) is one of:
      * ``None``  -> ``stock_supports[0]`` (the DECK) -- the historical hardcode, so an absent key is
        byte-identical with the legacy ``RESTOCK_PLACE_ON_DECK`` path.
      * an ``int`` -> ``stock_supports[index]`` (0 = deck, 1 = shelf).
      * a ``str`` -> a full prim path present in ``stock_supports``, OR a bare prim name
        (``'deck'``/``'shelf'``) matched against each entry's basename.

    Fail closed: an out-of-range index, an unknown path/name or an empty ``stock_supports`` raises
    ``ValueError`` so a mis-set selector never silently falls back to the wrong surface. Pure/CPU."""
    supports = list(stock_supports or [])
    if not supports:
        raise ValueError('cart asset exposes no stock_supports; cannot resolve a place support prim')
    if selector is None:
        return supports[0]
    if isinstance(selector, bool):            # guard: bool is an int subclass, never a valid index
        raise ValueError(f'cart_placement.support_prim must be an index or prim path, not {selector!r}')
    if isinstance(selector, int):
        if not -len(supports) <= selector < len(supports):
            raise ValueError(
                f'cart_placement.support_prim index {selector} out of range for '
                f'{len(supports)} stock_supports ({supports})')
        return supports[selector]
    sel = str(selector).strip()
    if sel in supports:
        return sel
    by_name = [p for p in supports if p.rsplit('/', 1)[-1] == sel]
    if len(by_name) == 1:
        return by_name[0]
    raise ValueError(
        f'cart_placement.support_prim={selector!r} matches no stock_support; '
        f'known supports (path or basename): {supports}')


def resting_carton_center_z(support_top_z, carton_height):
    """The carton CENTRE z when it rests flat on a support whose top face is at
    ``support_top_z``: ``support_top_z + carton_height / 2``. Pure/CPU."""
    if not (math.isfinite(float(support_top_z)) and math.isfinite(float(carton_height))):
        raise ValueError('resting_carton_center_z needs finite inputs')
    return float(support_top_z) + float(carton_height) / 2.0


def override_slot_support_z(slot, support_z_world):
    """A COPY of a ground-truth place slot with its ``support_z_world`` replaced by the
    resolved support top-face z. Used to retarget the cold GT-collect place from the
    legacy shelf constant onto the live DECK top -- the single source of truth -- while
    keeping the slot's ``load_center_body_xy`` and ``yaw_cart_rel``. Pure/CPU."""
    if not math.isfinite(float(support_z_world)):
        raise ValueError('override_slot_support_z needs a finite support_z_world')
    out = dict(slot or {})
    out['support_z_world'] = float(support_z_world)
    return out


def height_consistency_problems(resolved_top_z, heights, *, tol_m=0.010):
    """Problems ([] == consistent) for ``config_preflight``: every named height in
    ``heights`` ({label: value|None}) must agree with the resolved support prim top
    ``resolved_top_z`` within ``tol_m`` (10 mm by default). A ``None`` value is skipped
    (not configured). A disagreement names the stale value, the resolved deck top and
    the gap, so a legacy shelf constant (0.9325) left in any of the scenario / runtime /
    GT heights is refused before a GPU boot rather than feeding a mid-air release."""
    top = float(resolved_top_z)
    if not math.isfinite(top):
        raise ValueError('height_consistency_problems needs a finite resolved_top_z')
    problems = []
    for label, value in heights.items():
        if value is None:
            continue
        v = float(value)
        if not math.isfinite(v) or abs(v - top) > float(tol_m):
            problems.append(
                f"{label}={value} disagrees with the resolved support prim top "
                f"{top:.4f} m (gap {abs(v - top) * 1000:.0f} mm > {tol_m * 1000:.0f} mm tol). "
                f"Set this height to the resolved place-support prim top (the selected Rollwagen "
                f"support, deck or shelf) so no stale constant (e.g. the legacy 0.9325) can feed the descent.")
    return problems
