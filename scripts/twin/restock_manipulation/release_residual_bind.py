"""sim-place-release-residual-bind-v1: NAME the carton's full post-detach PHYSICS state and UNDO every
sim-side coupling the release can reach, so the next run NAMES whatever still binds the released carton.

GROUND TRUTH (the fm release-jump bench, data/twin-robot-in-store-boot-v1/cw106-residual-bind-bench.md):
  With ONLY the SIM_ATTACH weld + PR 389's robust detach, the carton is FULLY FREED -- joints targeting the
  carton 1 -> 0 one tick after ``_release_fallback()``, kinematicEnabled False throughout, the carton
  free-falls (vz -> -2.78 m/s), lands and reports a real ``CONTACT_PAIR HandlingUnit<->Backroom_floor``,
  and does NOT track the lift (follow_ratio ~0). So the weld is NOT the cw106 residual bind, and the PR 384
  contact logger DOES report a free carton's support contact. The cw106 bind is a MISSION-ONLY coupling the
  bench lacks: the host stack + the cuMotion ``object_attachment`` node (mission log ``object_attachment:
  Object detached successfully``). Prime suspect: object_attachment couples the carton to the end-effector
  for collision-aware planning (a USD reparent / a kinematic pose-follow) and its detach does not drop that
  sim-side coupling. (An earlier CPU verdict blamed a surviving FixedJoint PhysX constraint; the bench
  corrected it -- the joint IS dropped. The carton prim PATH stayed ``/RestockScenario/HandlingUnit`` through
  cw106's robust detach, so a path-renaming reparent at release is ruled out; kinematic pose-follow / a
  per-tick pose write by the still-attached node remain.)

WHAT THIS MODULE IS (the pure, CPU-testable core -- mirrors ``release_detach`` / ``place_contact_logger``):
  It owns the ORDER of the post-detach carton-state QUERY, the by-name diagnostic, the safe CLEARS and the
  re-query, so the whole sequence is provable with a recording/fake physics adapter and NO Isaac/pxr/GPU. The
  ADAPTER (in production the ``NativePanelContact``) owns the actual ``omni.physx`` / ``pxr`` reads/mutations
  behind named methods, so the boot path is CPU-wiring-testable and the GPU bench validates the SAME query it
  already samples (``#joints-targeting-carton`` + ``kinematicEnabled`` are the bench's own columns).

THE QUERY (the carton's full sim state -- the bench's RESIDUAL_BIND fields, so the clear is bench-validatable):
  * ``joints_referencing_carton(body)`` -- every joint PRIM whose body0/body1 still targets the carton.
  * ``carton_kinematic(body)``          -- ``physics:kinematicEnabled`` (a kinematic carton is pose-driven,
                                           not free -- the object_attachment pose-follow signature).
  * ``carton_parent(body)``             -- the carton prim's USD PARENT path; a parent that is NOT the
                                           scenario root NAMES a reparent coupling (OBSERVED, not forcibly
                                           undone -- a runtime reparent is risky and cw106's path was stable).
  * ``carton_rigid_enabled(body)``      -- ``physics:rigidBodyEnabled``; a disabled body is frozen by USD,
                                           not owned by physics.

THE CLEAR (named, ordered, each logged; only the SAFELY undoable sim-side couplings):
  1. drop every surviving joint prim found by the query (disable + remove), then
  2. clear a stuck kinematic flag (-> a free dynamic body), then
  3. re-enable a disabled rigid body (-> physics owns the pose again), then
  4. ``physx_resync()`` -- reconcile the structural change into the running PhysX scene, then
  5. RE-QUERY the state and log ``RELEASE_RESIDUAL_BIND_STATE`` (before/after) + the residual.

Default OFF => nothing is queried, cleared or logged => byte-identical. The REAL freedom guarantee is the
companion POSITIVE free check (:mod:`release_free` ``free_proof``): a carton still bound by ANY mechanism
(including a third-party node's per-tick pose write this module cannot stop) reports neither a carton-vs-
support contact NOR a settle drop, so the clearance lift is refused by name (``RELEASE_NOT_FREE
kind=no-support-contact``) -- the carton is never carried off. This module's job is to UNDO what it can and
to NAME the residual for the next run; the free check is the safety net.
"""
from __future__ import annotations

from typing import Optional, Sequence

TOKEN = 'RELEASE_RESIDUAL_BIND'
STATE_TOKEN = 'RELEASE_RESIDUAL_BIND_STATE'

# Config/env resolution, mirroring ``release_detach.robust_detach_enabled``.
RESIDUAL_BIND_ENV = 'RESTOCK_RELEASE_RESIDUAL_BIND'
_ON = ('1', 'true', 'on', 'yes')
_OFF = ('0', 'false', 'off', 'no')

# The USD prim-path prefix a FREE carton lives under (the scenario root). A carton parented anywhere else is
# a reparent coupling (observed + named).
SCENARIO_ROOT = '/RestockScenario'


def residual_bind_clear_enabled(config=None, environ=None) -> bool:
    """The effective residual-bind-clear setting (pure/CPU-testable). DEFAULT OFF => byte-identical.

    Resolution order (first that is set wins): the runtime-config key ``release_residual_bind`` (a bool, an
    on/off string, or a dict ``{"enabled": bool}``), else the env ``RESTOCK_RELEASE_RESIDUAL_BIND``, else
    DEFAULT FALSE. A malformed value raises by name. RUNTIME key (read on the Isaac side), sibling of
    ``release_detach``."""
    if config is not None and 'release_residual_bind' in (config or {}):
        val = (config or {})['release_residual_bind']
        if isinstance(val, dict):
            val = val.get('enabled', False)
        if isinstance(val, bool):
            return val
        s = str(val).strip().lower()
        if s in _OFF:
            return False
        if s in _ON:
            return True
        raise ValueError('release_residual_bind must be a bool, a dict {"enabled": bool}, or one of '
                         'on/off/true/false/1/0/yes/no, got %r' % (val,))
    import os
    env = os.environ if environ is None else environ
    raw = str(env.get(RESIDUAL_BIND_ENV, '')).strip().lower()
    if raw in _ON:
        return True
    return False


def is_reparented(parent: Optional[str], *, scenario_root: str = SCENARIO_ROOT) -> bool:
    """True when the carton's USD parent is NOT the scenario root (a reparent coupling). None parent (no
    read) => False (cannot claim a reparent). PURE."""
    if parent is None:
        return False
    p = str(parent).rstrip('/')
    return p != str(scenario_root).rstrip('/')


def query_carton_state(adapter, body_path) -> dict:
    """Read the carton's full sim state through the adapter (best-effort; each field falls back to None/()).
    PURE dispatch. Returns ``{joints, kinematic, parent, rigid_enabled, reparented}`` -- the RESIDUAL_BIND
    fields the bench samples."""
    joints = tuple(adapter.joints_referencing_carton(body_path) or ())
    kinematic = adapter.carton_kinematic(body_path)
    parent = adapter.carton_parent(body_path) if hasattr(adapter, 'carton_parent') else None
    rigid = adapter.carton_rigid_enabled(body_path) if hasattr(adapter, 'carton_rigid_enabled') else None
    return {'joints': joints, 'kinematic': kinematic, 'parent': parent, 'rigid_enabled': rigid,
            'reparented': is_reparented(parent)}


def format_state_line(tag: str, state: dict) -> str:
    """One ``RELEASE_RESIDUAL_BIND_STATE`` line (before/after), mirroring the bench's RESIDUAL_BIND columns."""
    return ('%s %s joints=%s kinematic=%s rigid_enabled=%s parent=%s reparented=%s'
            % (STATE_TOKEN, tag, list(state.get('joints', ())), state.get('kinematic'),
               state.get('rigid_enabled'), state.get('parent'), state.get('reparented')))


def format_line(kind: str, *, cleared: int, path: Optional[str] = None, extra: str = '') -> str:
    """One ``RELEASE_RESIDUAL_BIND`` per-bind line. ``kind`` is joint/kinematic/rigid_body/reparent/none."""
    bits = ['%s kind=%s' % (TOKEN, kind)]
    if path is not None:
        bits.append('path=%s' % path)
    bits.append('cleared=%d' % int(cleared))
    if extra:
        bits.append(extra)
    return ' '.join(bits)


def clear_residual_binds(adapter, *, body_path, joint_path, log=print):
    """Query the carton's full post-detach state, log it by name, CLEAR every safely-undoable sim-side
    coupling, reconcile PhysX, re-query and log the residual. Returns a record dict (for the report + the
    bench validation). PURE dispatch -- the ``adapter`` owns every Isaac/pxr read/mutation, so a fake adapter
    proves the sequence with no GPU.

    ``adapter`` methods: ``joints_referencing_carton(body)->seq[str]``, ``carton_kinematic(body)->bool|None``,
    ``carton_parent(body)->str|None`` (optional), ``carton_rigid_enabled(body)->bool|None`` (optional),
    ``drop_joint(path)``, ``clear_carton_kinematic(body)``, ``set_carton_rigid_enabled(body, enabled)``
    (optional), ``physx_resync()``. ``body_path`` is the welded carton (None => no-op record); ``joint_path``
    the KNOWN sim-attach-fallback joint path, re-asserted (idempotent) even when the USD query finds no prim.

    The carton's PARENT and rigid-enabled flag are OBSERVED (and a reparent is NAMED), but a reparent is NOT
    forcibly undone (a runtime USD reparent is risky and cw106's path was stable). A disabled rigid body IS
    re-enabled (safe). The honest guarantee that a still-bound carton is never carried off is the companion
    positive free check."""
    if body_path is None:
        log(format_line('none', cleared=0, extra='(no carton was welded; nothing to unbind)'))
        return {'joints_before': (), 'joint_paths_dropped': (), 'kinematic_before': None,
                'kinematic_cleared': 0, 'rigid_reenabled': 0, 'reparented': False,
                'tool_collider_cleared': 0, 'gripper_residual': (),
                'joints_after': (), 'kinematic_after': None, 'residual': (), 'resynced': False}

    before = query_carton_state(adapter, body_path)
    log(format_state_line('before', before))

    dropped = []
    for jp in before['joints']:
        adapter.drop_joint(jp)
        dropped.append(jp)
        log(format_line('joint', cleared=1, path=jp, extra='(surviving joint prim targeting the carton)'))
    # Re-assert the drop on the KNOWN fallback joint path even when the USD query found no prim (idempotent):
    # the ordered detach already RemovePrim'd it, so the USD traversal cannot see a surviving PhysX
    # constraint; dropping the known path again + the resync below is a cheap, safe belt-and-suspenders.
    if joint_path is not None and joint_path not in before['joints']:
        adapter.drop_joint(joint_path)
        dropped.append(joint_path)
        log(format_line('joint', cleared=1, path=joint_path,
                        extra='(known sim-attach-fallback joint re-asserted, idempotent)'))

    kin_cleared = 0
    if before['kinematic']:
        adapter.clear_carton_kinematic(body_path)
        kin_cleared = 1
        log(format_line('kinematic', cleared=1,
                        extra='(carton was kinematic -> cleared to a free dynamic body; the '
                              'object_attachment pose-follow signature)'))

    rigid_reenabled = 0
    if before['rigid_enabled'] is False and hasattr(adapter, 'set_carton_rigid_enabled'):
        adapter.set_carton_rigid_enabled(body_path, True)
        rigid_reenabled = 1
        log(format_line('rigid_body', cleared=1,
                        extra='(rigid body was disabled -> re-enabled so physics owns the carton pose)'))

    # Gripper-collider sweep (the fm pad-friction hypothesis): the mission releases with the pad coincident
    # with the carton face (SEAL_DIAG min_gap 0.0), and the grasp line disabled only "2 collider(s) (carton +
    # tool-side)" -- so a gripper collider left ENABLED can overlap the re-enabled carton and drag it up by
    # penetration-contact friction, INVISIBLE to the carton-vs-cart contact logger. Name every gripper
    # collider's enabled state at release; DISABLE any tool-side one still enabled (restored at the next
    # grasp by the deferral), and NAME an enabled non-tool-side gripper collider as a residual the positive
    # free check must refuse. Optional (adapter may lack the method) => byte-identical where absent.
    tool_collider_cleared = 0
    gripper_residual = []
    if hasattr(adapter, 'gripper_collider_states'):
        for path, enabled, tool_side in (adapter.gripper_collider_states() or []):
            log('%s kind=gripper_collider path=%s enabled=%s tool_side=%s' % (TOKEN, path, enabled, tool_side))
            if enabled and tool_side and hasattr(adapter, 'disable_gripper_collider'):
                adapter.disable_gripper_collider(path)
                tool_collider_cleared += 1
                log(format_line('tool_collider', cleared=1, path=path,
                                extra='(tool collider still ENABLED at release -> deferred so it cannot drag '
                                      'the freed carton by contact friction)'))
            elif enabled and not tool_side:
                gripper_residual.append(path)
                log(format_line('gripper_collider', cleared=0, path=path,
                                extra='(ENABLED non-tool-side gripper collider that can overlap the carton -- '
                                      'NOT auto-deferred; the positive free check refuses the lift)'))

    if before['reparented']:
        # Named, NOT forcibly undone (runtime reparent risk; cw106 path was stable). The positive free check
        # refuses the lift if this coupling keeps the carton bound.
        log(format_line('reparent', cleared=0, path=before['parent'],
                        extra='(carton parented OFF the scenario root -- a reparent coupling; OBSERVED only, '
                              'the positive free check is the guarantee)'))

    resynced = False
    if dropped or kin_cleared or rigid_reenabled or tool_collider_cleared:
        adapter.physx_resync()
        resynced = True

    after = query_carton_state(adapter, body_path)
    log(format_state_line('after', after))
    residual = list(after['joints']) + (['kinematic'] if after['kinematic'] else []) \
        + (['rigid_disabled'] if after['rigid_enabled'] is False else []) \
        + (['reparent:%s' % after['parent']] if after['reparented'] else []) \
        + ['gripper_collider:%s' % p for p in gripper_residual]

    if (not before['joints'] and not before['kinematic'] and before['rigid_enabled'] is not False
            and not before['reparented'] and not tool_collider_cleared and not gripper_residual):
        log(format_line('none', cleared=0,
                        extra='(no residual joint/kinematic/rigid/reparent/tool-collider bind at the sim '
                              'layer; if the carton still follows, the coupling is a host/object_attachment '
                              'per-tick pose-write this module cannot stop -- the positive free check refuses '
                              'the lift)'))
    if residual:
        log('%s_RESIDUAL still_bound=%s -- a sim-side coupling survived the clear; the positive free check '
            'must refuse the clearance lift (fail-safe by name)' % (TOKEN, ','.join(str(r) for r in residual)))

    return {'joints_before': before['joints'], 'joint_paths_dropped': tuple(dropped),
            'kinematic_before': before['kinematic'], 'kinematic_cleared': kin_cleared,
            'rigid_reenabled': rigid_reenabled, 'reparented': before['reparented'],
            'tool_collider_cleared': tool_collider_cleared, 'gripper_residual': tuple(gripper_residual),
            'joints_after': after['joints'], 'kinematic_after': after['kinematic'],
            'residual': tuple(residual), 'resynced': resynced}


def joints_targeting_body(stage, body_path, *, is_joint=None, body_targets=None) -> Sequence[str]:
    """Pure USD-traversal helper: every joint prim on ``stage`` whose body0/body1 rel targets ``body_path``.
    Shared by the production adapter and the GPU bench so both enumerate the carton's joints IDENTICALLY
    (the bench's ``#joints-targeting-carton`` column).

    ``is_joint(prim)->bool`` and ``body_targets(prim)->iterable[str]`` default to the real ``UsdPhysics.Joint``
    calls but are INJECTABLE so the enumeration is testable with no pxr. Returns the matching prim paths."""
    if is_joint is None or body_targets is None:
        from pxr import UsdPhysics  # pxr only; usd-core has it

        def is_joint(prim):
            return bool(prim.HasAPI(UsdPhysics.Joint)) or bool(UsdPhysics.Joint(prim))

        def body_targets(prim):
            j = UsdPhysics.Joint(prim)
            out = []
            for rel in (j.GetBody0Rel(), j.GetBody1Rel()):
                if rel:
                    out.extend(str(t) for t in rel.GetTargets())
            return out

    want = str(body_path)
    found = []
    for prim in stage.Traverse():
        try:
            if not is_joint(prim):
                continue
            if any(str(t) == want for t in (body_targets(prim) or [])):
                found.append(str(prim.GetPath()))
        except Exception:
            continue
    return found
