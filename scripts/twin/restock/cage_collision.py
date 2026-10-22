"""Opt-in, SIM-ONLY carton-vs-cage collision filter (labelled demo workaround; default OFF => byte-identical).

Why (sim-video-preflight-review-s1 L1/L2): the collect place threads the carton's 0.6 m side between the
rollwagen's front corner posts with 7.5 mm per side (post inner faces at |y| = 0.3075 in the cart body
frame; the side-wall bars' inner faces at |y| = 0.314 leave 14 mm), and the transfer lift out of the same
upper-shelf slot is a joint-space swing (``carry_lift_shape``) -- while the carton is deliberately NOT a
cuMotion collision object, so the planner cannot keep it clear of the bars. Neither motion has ever run live.

``config['cage_collision'] = 'filtered'`` (the RUNTIME config: the Isaac process composes the stage) authors
a ``UsdPhysics.FilteredPairsAPI`` on the handling unit (``/RestockScenario/HandlingUnit``) against every
rollwagen CAGE collider -- the four corner posts and the side/back walls' horizontal + vertical bars -- at
scene composition, before physics starts. The carton then passes through the cage bars IN THE SIMULATION
ONLY. Kept colliding: the deck and the upper shelf (the carton rests on them), the drawbar, the casters,
the arm and the gripper. Pair filtering removes contacts only: lidar/camera rays still see the cage, and
cuMotion's ``world_obstacles`` cage (the arm's planning world) is untouched.

Labelled, never silent: :func:`stamp_summary` records ``cage_collision_mode='filtered'`` + a ``workarounds``
entry + a note in mission.json (the demo cut shows ``*_mode`` / ``workaround*`` keys as run labels), and
:func:`caption` appends the label to the burned-in walkthrough caption. An unknown value raises at boot,
so a typo cannot silently run the unfiltered cage.
"""
from __future__ import annotations

KEY = 'cage_collision'
FILTERED = 'filtered'
MODES = (FILTERED,)
_OFF = ('', '0', 'off', 'none', 'false')

# Rollwagen chassis child colliders that form the CAGE (twin.mobile_assets.author_cart): post_{0..3},
# side_{0,1}_{horizontal,vertical}_k, back_{horizontal,vertical}_k. Everything else under the chassis --
# deck, shelf (the carton's supports), drawbar -- keeps colliding with the carton.
CAGE_PREFIXES = ('post_', 'side_', 'back_')
SUPPORTS = ('deck', 'shelf')

CAPTION = 'CAGE COLLISION FILTERED (sim workaround): the carton passes through the rollwagen cage bars'
NOTE = (' CAGE_COLLISION=filtered: carton-vs-rollwagen-cage contacts are filtered in the simulation '
        '(UsdPhysics FilteredPairsAPI on the handling unit vs the cage posts/bars; deck + upper shelf still '
        'collide) so the collect place and the transfer lift cannot strike the cage (labelled workaround, '
        'not a clearance proof).')


def mode(config):
    """The active cage-collision mode, or None (absent/empty/'0'/'off'/false => off). An unknown value
    RAISES so a typo fails loudly at boot instead of silently running the unfiltered cage."""
    raw = config.get(KEY) if isinstance(config, dict) else None
    if raw is None or raw is False:
        return None
    value = str(raw).strip().lower()
    if value in _OFF:
        return None
    if value not in MODES:
        raise ValueError(f'{KEY}={raw!r}: unknown cage collision mode (known: {", ".join(MODES)})')
    return value


def cage_collider_paths(stage, cart_body):
    """The rollwagen cage collider prim paths under ``cart_body`` (authored order). Refuses (ValueError) a
    missing cart body, a cage with no colliders, or a selection that would include a carton support."""
    from pxr import UsdPhysics
    body = stage.GetPrimAtPath(cart_body)
    if not body or not body.IsValid():
        raise ValueError(f'rollwagen chassis body absent: {cart_body}')
    paths = [str(child.GetPath()) for child in body.GetChildren()
             if child.GetName().startswith(CAGE_PREFIXES) and child.HasAPI(UsdPhysics.CollisionAPI)]
    if not paths:
        raise ValueError(f'no rollwagen cage colliders under {cart_body}; refusing an empty filter')
    if any(path.rsplit('/', 1)[-1] in SUPPORTS for path in paths):
        raise ValueError('cage filter would include a carton support (deck/shelf)')
    return paths


def author_filter(stage, unit_path, cart_body):
    """Author the carton-vs-cage FilteredPairsAPI on ``unit_path`` (the handling unit's rigid body) and
    return the evidence dict (also written to the unit's customData). Pure USD; runs before physics."""
    from pxr import UsdPhysics
    unit = stage.GetPrimAtPath(unit_path)
    if not unit or not unit.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError(f'handling unit is not a rigid body: {unit_path}')
    cage = cage_collider_paths(stage, cart_body)
    rel = UsdPhysics.FilteredPairsAPI.Apply(unit).CreateFilteredPairsRel()
    for path in cage:
        rel.AddTarget(path)
    kept = [str(child.GetPath()) for child in stage.GetPrimAtPath(cart_body).GetChildren()
            if child.HasAPI(UsdPhysics.CollisionAPI) and str(child.GetPath()) not in cage]
    unit.SetCustomDataByKey('restock:cage_collision', FILTERED)
    return dict(mode=FILTERED, unit=str(unit.GetPath()), filtered=cage, kept=kept,
                scope='simulation contacts only (rays, cuMotion world untouched)')


def mission_labels(config):
    """mission.json fields recording the workaround; {} when off (byte-identical summary)."""
    return {'cage_collision_mode': FILTERED} if mode(config) == FILTERED else {}


def stamp_summary(summary, config):
    """Record the active mode in a mission.json summary dict (in place; returned). Off => untouched. Never
    raises: a bad value is recorded (the boot already refused it), not hidden."""
    try:
        labels = mission_labels(config)
    except ValueError as exc:
        summary['cage_collision_mode'] = 'invalid'
        summary['cage_collision_error'] = str(exc)
        return summary
    if not labels:
        return summary
    summary.update(labels)
    workarounds = summary.setdefault('workarounds', [])
    label = f'{KEY}={FILTERED}'
    if label not in workarounds:
        workarounds.append(label)
    if NOTE not in (summary.get('note') or ''):
        summary['note'] = (summary.get('note') or '') + NOTE
    summary['walkthrough_caption'] = caption(summary.get('walkthrough_caption'), config)
    return summary


def caption(base=None, config=None):
    """The walkthrough caption with the cage-filter label appended. Off/invalid => ``base`` unchanged."""
    try:
        active = mode(config)
    except ValueError:
        return base
    if active is None or (base and CAPTION in base):
        return base
    return f'{base} | {CAPTION}' if base else CAPTION
