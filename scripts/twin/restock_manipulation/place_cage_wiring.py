"""Boot-layer wiring for the cart cage-clearance gates (sim-place-descent-reach-wire-v1).

PR 366 added :func:`place_clearance.preplace_cage_clearance_provider` and PR 376 added
:func:`place_clearance.place_descent_cage_clearance_provider`, but NEITHER was ever constructed or
injected anywhere: there are ZERO call sites of either provider in the tree, so both controller hooks
(``TransferController._preplace_cage_clearance`` and ``._place_descent_cage_gate``) stayed ``None`` and
the in-op cage gates never ran. With every PR 376 switch ON, coldwin90 therefore repeated run 89/90
exactly -- ``PLACE_SEAT_GATE_ARM`` / ``PLACE_OBSTRUCTED +87 mm`` / the descent never started -- because
the carry-connected standoff selector never got the cage criterion and the dispatch never refused the
jamming descent by name (it fell through to the blind ``PATH_TOLERANCE`` wrist-2 jam).

This module is the glue :func:`ros_host.main` calls right after it builds the controller: it reads two
opt-in HOST blocks and, when present, builds the matching provider and installs it on the controller,
logging one ``PREPLACE_CAGE_GATE armed`` / ``PLACE_DESCENT_CAGE_GATE armed`` line. Absent blocks => the
hooks stay ``None`` => byte-identical (the whole feature is default-off). Injecting the provider is
EXACTLY the extension point both controller comments name ("the boot/integration layer supplies this
provider with the live coupled cart pose + the host XRDF").

Pure (no ROS). The gate's sweep (``place_clearance.preplace_cage_arm_clearance``) needs only the URDF FK
+ the XRDF sphere model + numpy; the cart collider AABBs come from the authored USD (usd-core) or, for a
host interpreter without pxr, a pre-baked ``collider_boxes`` override carried in the block. CPU-tested by
``scripts/twin/tests/test_place_cage_wiring.py``.

HOST block schema (both keys share it)::

    "place_descent_cage_gate": {
        "cart_pose": [x, y, z, yaw],          # REQUIRED: cart-body -> ur_base_link (or a 4x4 matrix)
        "margin": 0.03,                       # optional; default the provider's module default
        "xrdf": "/abs/path/mobile.xrdf",      # optional; default = the URDF path with a .xrdf suffix
        "collider_boxes": {                   # optional; pre-baked cart-frame AABBs for a no-pxr host
            "post_2": [[lox, loy, loz], [hix, hiy, hiz]], ...
        },
        "cal_offset_m": 0.0                   # optional SEAL-anchored calibration knob
    }

``"preplace_cage_gate"`` takes the same shape (its margin default is the penetration-only 0 mm, the PR
366 design -- a positive margin would block the standoff's inherent cage-width tightness; override via
``margin`` only with evidence). A bare ``true`` is rejected: a ``cart_pose`` is always required.
"""
from __future__ import annotations

from pathlib import Path

#: HOST config keys. Present + truthy => the gate is wired; absent => the hook stays None (byte-identical).
PREPLACE_CAGE_GATE_KEY = 'preplace_cage_gate'
PLACE_DESCENT_CAGE_GATE_KEY = 'place_descent_cage_gate'


def _coerce_block(value, key):
    """A gate HOST value -> a config dict (enabled) or ``None`` (inert). A dict stays itself; any falsey
    value (absent / ``False`` / ``None`` / ``{}``) is inert. A bare ``True`` is rejected -- a ``cart_pose``
    is always required, so an all-defaults enable cannot work."""
    if value is None or value is False:
        return None
    if value is True:
        raise ValueError('%s: enable it with a block carrying at least {"cart_pose": [x, y, z, yaw]}, '
                         'not a bare true (the cart-body -> ur_base_link pose is required)' % key)
    if isinstance(value, dict):
        return dict(value) if value else None
    raise ValueError('%s: expected a config block (dict) or false, got %r' % (key, type(value).__name__))


def _resolve_xrdf(urdf_path, block_xrdf, default_xrdf):
    """The XRDF path: an explicit block value wins, else the caller default, else the URDF path with a
    ``.xrdf`` suffix (the host ships ``mobile.urdf`` beside ``mobile.xrdf``)."""
    if block_xrdf:
        return str(block_xrdf)
    if default_xrdf:
        return str(default_xrdf)
    return str(Path(str(urdf_path)).with_suffix('.xrdf'))


def _resolve_bodies(block):
    """The cart-frame collider AABBs ``{name: (lo, hi)}``. A pre-baked ``collider_boxes`` override (offline-
    computed with usd-core, for a host interpreter without pxr) is used verbatim; otherwise the authored
    cart is read from USD (needs usd-core). Returning ``None`` lets the provider read the USD itself."""
    cb = block.get('collider_boxes')
    if cb is None:
        return None
    return {str(name): (tuple(float(c) for c in lohi[0]), tuple(float(c) for c in lohi[1]))
            for name, lohi in cb.items()}


def _require_cart_pose(block, key):
    if 'cart_pose' not in block:
        raise ValueError('%s: a cart_pose (cart-body -> ur_base_link, (x, y, z, yaw) or a 4x4 matrix) is '
                         'required to wire the cage-clearance gate' % key)
    return block['cart_pose']


def build_preplace_cage_gate(block, urdf_path, *, default_xrdf=None):
    """Build the PR 366 carried pre-place cage-clearance gate ``callable(waypoints)->reject|None`` from a
    resolved HOST block. Raises on a missing ``cart_pose``."""
    from . import place_clearance as pc
    cart_pose = _require_cart_pose(block, PREPLACE_CAGE_GATE_KEY)
    xrdf = _resolve_xrdf(urdf_path, block.get('xrdf'), default_xrdf)
    margin = float(block.get('margin', pc.DEFAULT_PREPLACE_CAGE_MARGIN_M))
    gate = pc.preplace_cage_clearance_provider(
        str(urdf_path), xrdf, cart_pose, bodies=_resolve_bodies(block), margin=margin,
        include=block.get('include'), cal_offset_m=float(block.get('cal_offset_m', 0.0)))
    return gate, {'xrdf': xrdf, 'margin': margin, 'cart_pose': cart_pose}


def build_place_descent_cage_gate(block, urdf_path, *, default_xrdf=None):
    """Build the PR 376 collect place-DESCENT cage-clearance gate ``callable(waypoints)->reject|None`` from a
    resolved HOST block, plus the raw min-clearance closure ``callable(waypoints)->metres`` the
    pad-down-branch selector ranks by (sim-place-pad-down-arrival-v1). Raises on a missing ``cart_pose``."""
    from . import place_clearance as pc
    cart_pose = _require_cart_pose(block, PLACE_DESCENT_CAGE_GATE_KEY)
    xrdf = _resolve_xrdf(urdf_path, block.get('xrdf'), default_xrdf)
    margin = float(block.get('margin', pc.DEFAULT_PLACE_DESCENT_CAGE_MARGIN_M))
    bodies = _resolve_bodies(block)
    include = block.get('include')
    cal = float(block.get('cal_offset_m', 0.0))
    gate = pc.place_descent_cage_clearance_provider(
        str(urdf_path), xrdf, cart_pose, bodies=bodies, margin=margin, include=include, cal_offset_m=cal)
    clearance = pc.place_descent_cage_min_clearance_provider(
        str(urdf_path), xrdf, cart_pose, bodies=bodies, include=include, cal_offset_m=cal)
    return gate, clearance, {'xrdf': xrdf, 'margin': margin, 'cart_pose': cart_pose}


def wire_place_cage_gates(controller, config, urdf_path, *, default_xrdf=None, log=print):
    """Install the cart cage-clearance providers on ``controller`` from the HOST ``config`` blocks
    (sim-place-descent-reach-wire-v1). Opt-in + byte-identical off: a gate is wired only when its HOST
    block is present and carries a ``cart_pose``; absent => the controller hook stays ``None``.

    Sets ``controller._preplace_cage_clearance`` (PR 366) and/or ``controller._place_descent_cage_gate``
    (PR 376) -- the exact hooks the controller dispatch + the carry-connected standoff selector guard on --
    and logs one ``armed`` line per gate so a console grep proves the previously-inert provider is now live.
    Returns the list of armed gate keys. Raises on a malformed/incomplete block (loud, never silent): the
    whole point of this lane is that a configured gate can never again be left uncalled."""
    armed = []

    pre_block = _coerce_block(config.get(PREPLACE_CAGE_GATE_KEY), PREPLACE_CAGE_GATE_KEY)
    if pre_block is not None:
        gate, meta = build_preplace_cage_gate(pre_block, urdf_path, default_xrdf=default_xrdf)
        controller._preplace_cage_clearance = gate
        armed.append(PREPLACE_CAGE_GATE_KEY)
        log('PREPLACE_CAGE_GATE armed margin=%.3f m xrdf=%s cart_pose=%s -- the carried pre-place '
            'shoulder_pan sweep is now refused BY NAME before motion if it would drive an arm link into a '
            'Rollwagen cage body (PR 366 provider, previously never constructed)'
            % (meta['margin'], meta['xrdf'], list(meta['cart_pose'])
               if not isinstance(meta['cart_pose'], (int, float)) else meta['cart_pose']), flush=True)

    desc_block = _coerce_block(config.get(PLACE_DESCENT_CAGE_GATE_KEY), PLACE_DESCENT_CAGE_GATE_KEY)
    if desc_block is not None:
        gate, clearance, meta = build_place_descent_cage_gate(desc_block, urdf_path, default_xrdf=default_xrdf)
        controller._place_descent_cage_gate = gate
        # sim-place-pad-down-arrival-v1: the raw min-clearance closure the pad-down-branch selector ranks by
        # (byte-identical when the three-stage set-down is off -- it is only read by the reorientation leg).
        controller._place_descent_cage_min_clearance = clearance
        armed.append(PLACE_DESCENT_CAGE_GATE_KEY)
        log('PLACE_DESCENT_CAGE_GATE armed margin=%.3f m xrdf=%s cart_pose=%s -- the carry-connected '
            'standoff selector now discards a cage-raking descent branch AND the place-descent dispatch '
            'refuses an intersecting set-down BY NAME before motion (PR 376 provider, previously never '
            'constructed -- coldwin90 repeated run 89/90 exactly without it)'
            % (meta['margin'], meta['xrdf'], list(meta['cart_pose'])
               if not isinstance(meta['cart_pose'], (int, float)) else meta['cart_pose']), flush=True)

    return armed
