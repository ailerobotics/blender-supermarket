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
        "cal_offset_m": 0.0,                  # optional SEAL-anchored calibration knob
        "include_support": true               # optional (sim-pad-down-descent-jam-physics-v1): also sweep
                                              #   the SUPPORT board (deck/shelf) so the pad-down gripper-
                                              #   housing-vs-shelf strike is refused BY NAME (penetration-
                                              #   only support margin). Default false => cage-only (byte-
                                              #   identical with PR 376). Only valid for the descent gate.
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
    # sim-pad-down-descent-jam-physics-v1: opt-in SHELF awareness. ``include_support`` true => the descent
    # arm-sphere sweep ALSO covers the SUPPORT board (deck/shelf), so the pad-down gripper-housing-vs-shelf
    # strike (coldwin98-100, the jam the cage-only sweep was blind to) is refused BY NAME at the penetration-
    # only support margin. Default false => cage-only => byte-identical with the PR 376 gate.
    include_support = bool(block.get('include_support', False))
    gate = pc.place_descent_cage_clearance_provider(
        str(urdf_path), xrdf, cart_pose, bodies=bodies, margin=margin, include=include, cal_offset_m=cal,
        include_support=include_support)
    clearance = pc.place_descent_cage_min_clearance_provider(
        str(urdf_path), xrdf, cart_pose, bodies=bodies, include=include, cal_offset_m=cal,
        include_support=include_support)
    return gate, clearance, {'xrdf': xrdf, 'margin': margin, 'cart_pose': cart_pose,
                             'include_support': include_support}


def wire_housing_seat(controller, *, log=print):
    """sim-place-housing-compatible-seat-v1: prove the housing-compatible pad-down seat transform is
    LIVE at boot, and log a named ``PLACE_HOUSING_SEAT armed`` line the run worker greps before trusting
    the fragment. The transform itself rides the resolved ``carton_level`` the controller already holds
    (``place_housing_seat`` + ``housing_seat_lip_overhang_m`` + ``housing_seat_z_lift_m``) and is applied
    by ``controller._housing_seat_tcp`` at BOTH the pad-down branch selector and the place descent; this
    function only asserts it is armed and logs it (so the boot path, not just a unit test, exercises the
    switch). Returns ``True`` when armed, ``False`` otherwise (absent => byte-identical, no log).

    Raises ``ValueError`` (loud, never silent) when ``place_housing_seat`` is on but the three-stage
    pad-down arrival it transforms is NOT -- exactly the ``LevelConfig.validate`` invariant, re-checked at
    the boot seam so a mis-set fragment can never leave the seat transform half-wired."""
    cl = getattr(controller, '_carton_level', None)
    if cl is None or not getattr(cl, 'place_housing_seat', False):
        return False
    if not getattr(cl, 'place_pad_down_arrival', False):
        raise ValueError('place_housing_seat is on but place_pad_down_arrival is off: the housing-seat '
                         'transform has no pad-down seat to correct (set carton_level.place_pad_down_arrival '
                         'true, or place_housing_seat false)')
    overhang = float(getattr(cl, 'housing_seat_lip_overhang_m', 0.0) or 0.0)
    lift = float(getattr(cl, 'housing_seat_z_lift_m', 0.0) or 0.0)
    log('PLACE_HOUSING_SEAT armed lip_overhang=%.4f m (mouth-out toward the +x shelf lip) z_lift=%.4f m -- '
        'the pad-down branch selector AND the place descent now target a side-grip-housing-compatible seat '
        '(sim-place-housing-compatible-seat-v1): z_lift raises the carton onto the pinned COMPOSED shelf '
        'flat top (0.9639) instead of the author_cart BOX-MODEL top (0.9325) so the pins and the descent '
        'agree, and lip_overhang hangs the gripper housing outboard of the shelf board. CPU proof + the '
        'housing-clearance-vs-CoM tradeoff: scripts/check_housing_seat_descent.py' % (overhang, lift),
        flush=True)
    return True


def build_push_in_geometry(block, urdf_path, *, default_xrdf=None):
    """Build the :class:`push_in.PushInGeometry` (the live cart geometry the controller's push-in dispatch
    reads) from a resolved ``place_descent_cage_gate`` HOST block -- the push-in needs the SAME cart pose +
    shelf collider as the descent gate (sim-place-post-release-push-in-v1). Raises on a missing ``cart_pose``.

    The two closures:
      * ``shelf_clear_fn(configs) -> min housing-vs-shelf clearance (m)`` -- the PR 383 shelf-inclusive XRDF
        arm-sphere sweep restricted to the shelf board (the stroke-cap criterion), and
      * ``carton_com_cart_x(config) -> carton CoM cart-x (m)`` -- the baked pad+carton FK in the cart frame,
        so the controller turns ``push_in_target_x_m`` into a stroke from the live seat.
    Needs the URDF FK + the XRDF sphere model + the cart collider AABBs (a pre-baked ``collider_boxes``
    override, or usd-core)."""
    import numpy as np
    from . import place_clearance as pc
    from . import push_in as pin
    from .cart_world_obstacle import cart_pose_matrix
    from .cad_tool import ARM_JOINTS
    from twin.urdf_probe import parse_urdf
    cart_pose = _require_cart_pose(block, PLACE_DESCENT_CAGE_GATE_KEY)
    xrdf = _resolve_xrdf(urdf_path, block.get('xrdf'), default_xrdf)
    bodies = _resolve_bodies(block)
    if bodies is None:
        bodies = pc.rollwagen_collider_boxes()        # author_cart box model (needs usd-core)
    T_base_cart = cart_pose_matrix(cart_pose)
    T_cart_base = np.linalg.inv(np.asarray(T_base_cart, float))
    robot = parse_urdf(str(urdf_path))
    geom = pc.GripperCartonGeometry.shipped()
    push_dir = pin.push_direction_base(T_base_cart)

    def shelf_clear_fn(configs):
        if not configs:
            return float('inf')
        per = pc.place_descent_cage_clearance(list(configs), str(urdf_path), str(xrdf), bodies,
                                              T_base_cart, include=['shelf'])
        return float(per.get('shelf', float('inf')))

    def carton_com_cart_x(config):
        cfg = {j: float(config[i]) for i, j in enumerate(ARM_JOINTS)}
        C = pc._carton_in_base(robot, cfg, geom)
        return float((T_cart_base @ np.asarray(C, float))[0, 3])

    support = None
    if 'shelf' in bodies:
        shelf_lo, shelf_hi = bodies['shelf']
        shelf_lo_xy = (float(shelf_lo[0]), float(shelf_lo[1]))
        shelf_hi_xy = (float(shelf_hi[0]), float(shelf_hi[1]))
        half = np.asarray(pc.CARTON_DIMS_M, float) / 2.0
        full = float(pc.CARTON_DIMS_M[0] * pc.CARTON_DIMS_M[1])

        def carton_support_fraction(config):
            cfg = {j: float(config[i]) for i, j in enumerate(ARM_JOINTS)}
            Cc = T_cart_base @ np.asarray(pc._carton_in_base(robot, cfg, geom), float)
            corners = [tuple((Cc @ np.array([sx * half[0], sy * half[1], -half[2], 1.0]))[:2])
                       for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
            return pin.clip_area([(float(x), float(y)) for x, y in corners],
                                 shelf_lo_xy, shelf_hi_xy) / full
        support = carton_support_fraction

    pg = pin.PushInGeometry(push_dir_base=push_dir, shelf_clear_fn=shelf_clear_fn,
                            carton_com_cart_x=carton_com_cart_x, carton_support_fraction=support,
                            meta={'xrdf': xrdf, 'cart_pose': cart_pose})
    return pg, {'xrdf': xrdf, 'cart_pose': cart_pose}


def wire_push_in(controller, config, urdf_path, *, default_xrdf=None, log=print):
    """sim-place-post-release-push-in-v1: install the live cart geometry the controller's POST-RELEASE
    push-in dispatch reads, and log a named ``PLACE_PUSH_IN armed`` line the run worker greps before
    trusting the fragment. Opt-in + byte-identical off: a controller whose ``carton_level.place_push_in`` is
    off leaves the hook ``None`` (no push-in phase runs). The push-in's knobs (``push_in_target_x_m``,
    ``push_in_shelf_clear_min_m``, ``push_in_speed``, ...) ride the resolved ``carton_level`` the controller
    already holds; this function builds + installs the cart GEOMETRY (``controller._push_in_geometry``) from
    the ``place_descent_cage_gate`` HOST block and arms the phase.

    Returns ``True`` when armed, ``False`` otherwise. Raises ``ValueError`` (loud, never silent) when
    ``place_push_in`` is on but its ``place_housing_seat`` prerequisite is off (the ``LevelConfig.validate``
    invariant, re-checked at the boot seam like ``wire_housing_seat``), or when ``place_push_in`` is on but
    the ``place_descent_cage_gate`` block that carries the cart pose is absent (the push-in cannot find the
    shelf board without it) -- so a half-wired fragment can never leave the push-in phase armed but blind."""
    cl = getattr(controller, '_carton_level', None)
    if cl is None or not getattr(cl, 'place_push_in', False):
        return False
    if not getattr(cl, 'place_housing_seat', False):
        raise ValueError('place_push_in is on but place_housing_seat is off: the push-in nudges the carton '
                         'off the housing-compatible lip seat PR 385 leaves, which only exists when '
                         'place_housing_seat is on (set carton_level.place_housing_seat true, or '
                         'place_push_in false)')
    desc_block = _coerce_block(config.get(PLACE_DESCENT_CAGE_GATE_KEY), PLACE_DESCENT_CAGE_GATE_KEY)
    if desc_block is None:
        raise ValueError('place_push_in is on but the place_descent_cage_gate HOST block (cart_pose + shelf '
                         'collider) is absent: the push-in reads the SAME cart geometry to cap the stroke by '
                         'the housing-vs-shelf clearance -- wire place_descent_cage_gate (with '
                         'include_support true), or set place_push_in false')
    pg, meta = build_push_in_geometry(desc_block, urdf_path, default_xrdf=default_xrdf)
    controller._push_in_geometry = pg
    log('PLACE_PUSH_IN armed target_x=%.4f m shelf_clear_min=%.1f mm speed=%.3f xrdf=%s cart_pose=%s -- after '
        'the PR 382 seat gate releases the carton at the PR 385 lip seat and the gripper backs clear, the arm '
        're-contacts the free carton and pushes it inboard (cart -x) to a more centred rest, the stroke CAPPED '
        'so the side-grip housing keeps the clearance to the shelf board (sim-place-post-release-push-in-v1: a '
        'bounded nudge -- a CENTRED rest needs a top-down grip, the out-of-scope follow-up)'
        % (float(getattr(cl, 'push_in_target_x_m', 0.0)),
           float(getattr(cl, 'push_in_shelf_clear_min_m', 0.010)) * 1000.0,
           float(getattr(cl, 'push_in_speed', 0.04)), meta['xrdf'],
           list(meta['cart_pose']) if not isinstance(meta['cart_pose'], (int, float)) else meta['cart_pose']),
        flush=True)
    return True


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
        log('PLACE_DESCENT_CAGE_GATE armed margin=%.3f m include_support=%s xrdf=%s cart_pose=%s -- the '
            'carry-connected standoff selector now discards a cage-raking descent branch AND the place-descent '
            'dispatch refuses an intersecting set-down BY NAME before motion (PR 376 provider, previously '
            'never constructed -- coldwin90 repeated run 89/90 exactly without it)%s'
            % (meta['margin'], meta.get('include_support', False), meta['xrdf'], list(meta['cart_pose'])
               if not isinstance(meta['cart_pose'], (int, float)) else meta['cart_pose'],
               ' + the SUPPORT board (deck/shelf) so the pad-down gripper-housing-vs-shelf jam '
               '(sim-pad-down-descent-jam-physics-v1, coldwin98-100) is now refused by name at the '
               'penetration-only support margin' if meta.get('include_support') else ''), flush=True)

    return armed
