"""CPU tests for sim-place-release-residual-bind-v1 (no Isaac/pxr/GPU).

coldwin106 ran with PR 389's robust detach ON (the console shows the weld joint disabled + RemovePrim + the
freed-carton velocity zeroed + cuMotion ``Object detached successfully`` + ``SEAL_DIAG status=Open gripped_n=0``)
yet the carton rose a near-pure vertical +19 mm WITH the gripper (x/y stable to 0.1 mm) and the PR 384 contact
logger saw ZERO carton-vs-support pairs -- the exact-follow + no-support-contact signature of a carton STILL
rigidly constrained to the gripper (the sim-attach-fallback FixedJoint's PhysX constraint surviving the USD
disable+RemovePrim, which does not re-sync into the tensor-backed PhysX scene).

Two coordinated opt-in fixes, proven here on CPU:
  1. PHYSICS-layer residual-bind query+clear (release_residual_bind + cad_tool): after the ordered detach,
     query the carton's surviving joints + kinematic flag, drop/clear each, reconcile PhysX, re-query and log
     RELEASE_RESIDUAL_BIND by name. Proven via the pure ordered core + a recording physics adapter.
  2. POSITIVE free check (release_free.free_proof + controller): the clearance lift is gated on a proof a
     bound carton cannot satisfy -- a carton-vs-support contact OR a measured settle drop -- else
     RELEASE_NOT_FREE kind=no-support-contact WITHOUT the destructive lift.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts python3 -m pytest \
        scripts/twin/tests/test_release_residual_bind_v1.py
"""
import sys
from types import SimpleNamespace as NS

import pytest

from twin.restock_manipulation import release_residual_bind as rrb
from twin.restock_manipulation import carton_level as cl
from twin.restock_manipulation.release_free import ReleaseFreeGate


# ============================================================ 1. residual_bind_clear_enabled (config/env)

def test_residual_bind_default_off():
    assert rrb.residual_bind_clear_enabled({}) is False
    assert rrb.residual_bind_clear_enabled(None) is False


def test_residual_bind_config_env_and_dict():
    assert rrb.residual_bind_clear_enabled({'release_residual_bind': True}) is True
    assert rrb.residual_bind_clear_enabled({'release_residual_bind': 'on'}) is True
    assert rrb.residual_bind_clear_enabled({'release_residual_bind': 'off'}) is False
    assert rrb.residual_bind_clear_enabled({'release_residual_bind': {'enabled': True}}) is True
    assert rrb.residual_bind_clear_enabled({}, environ={'RESTOCK_RELEASE_RESIDUAL_BIND': '1'}) is True
    # explicit config wins over env
    assert rrb.residual_bind_clear_enabled({'release_residual_bind': False},
                                           environ={'RESTOCK_RELEASE_RESIDUAL_BIND': '1'}) is False


def test_residual_bind_rejects_malformed():
    with pytest.raises(ValueError):
        rrb.residual_bind_clear_enabled({'release_residual_bind': 'maybe'})


# ================================================================== 2. joints_targeting_body (pure query)

def test_joints_targeting_body_finds_only_the_carton_joints():
    carton = '/RestockScenario/HandlingUnit'
    # fake prims: (path, is_joint, body targets)
    prims = [
        NS(path='/g/sim_attach_fallback', joint=True, bodies=['/robot/body', carton]),
        NS(path='/g/other_joint', joint=True, bodies=['/robot/body', '/World/wall']),
        NS(path='/RestockScenario/HandlingUnit', joint=False, bodies=[]),
        NS(path='/somewhere/second_weld', joint=True, bodies=[carton]),
    ]
    stage = NS(Traverse=lambda: [NS(GetPath=lambda p=p: p.path, _p=p) for p in prims])
    is_joint = lambda prim: prim._p.joint
    body_targets = lambda prim: prim._p.bodies
    found = rrb.joints_targeting_body(stage, carton, is_joint=is_joint, body_targets=body_targets)
    assert set(found) == {'/g/sim_attach_fallback', '/somewhere/second_weld'}


# ========================================================= 3. clear_residual_binds (ordered, recording)

class _FakeResidualAdapter:
    """Fake physics layer: records every query/clear so the ordered residual-bind sequence is provable with
    no Isaac. ``joints`` is what the query reports (first call), ``joints_after`` what a RE-query reports."""
    def __init__(self, *, joints=(), joints_after=(), kinematic=None, kinematic_after=None,
                 parent='/RestockScenario', rigid_enabled=True):
        self._joints = list(joints)
        self._joints_after = list(joints_after)
        self._kin = kinematic
        self._kin_after = kinematic_after
        self._parent = parent
        self._rigid = rigid_enabled
        self._queried = 0
        self.calls = []

    def joints_referencing_carton(self, body):
        self._queried += 1
        self.calls.append(('query_joints', body))
        return list(self._joints if self._queried == 1 else self._joints_after)

    def carton_kinematic(self, body):
        self.calls.append(('query_kinematic', body))
        # first read reports the initial flag; after a clear, report the cleared value
        drops = [c for c in self.calls if c[0] == 'clear_kinematic']
        return self._kin_after if drops else self._kin

    def carton_parent(self, body):
        self.calls.append(('query_parent', body))
        return self._parent

    def carton_rigid_enabled(self, body):
        self.calls.append(('query_rigid', body))
        return True if [c for c in self.calls if c[0] == 'set_rigid'] else self._rigid

    def drop_joint(self, jp):
        self.calls.append(('drop_joint', jp))

    def clear_carton_kinematic(self, body):
        self.calls.append(('clear_kinematic', body))

    def set_carton_rigid_enabled(self, body, enabled):
        self.calls.append(('set_rigid', body, enabled))

    def physx_resync(self):
        self.calls.append(('physx_resync', None))


def test_clear_residual_binds_drops_surviving_joint_then_reasserts_known_then_resyncs():
    a = _FakeResidualAdapter(joints=['/somewhere/second_weld'], joints_after=[], kinematic=False)
    rec = rrb.clear_residual_binds(a, body_path='/c', joint_path='/g/sim_attach_fallback', log=lambda *_: None)
    names = [c[0] for c in a.calls]
    # query -> drop the surviving joint -> re-assert the known fallback joint -> resync -> re-query
    assert names.index('query_joints') < names.index('drop_joint')
    assert ('drop_joint', '/somewhere/second_weld') in a.calls
    assert ('drop_joint', '/g/sim_attach_fallback') in a.calls   # known path re-asserted (surviving PhysX constraint)
    assert names.index('physx_resync') > names.index('drop_joint')
    assert rec['joint_paths_dropped'] == ('/somewhere/second_weld', '/g/sim_attach_fallback')
    assert rec['residual'] == ()                                  # the re-query shows clean
    assert rec['resynced'] is True


def test_clear_residual_binds_reasserts_known_joint_when_usd_query_empty():
    # the detach already RemovePrim'd the fallback joint, so the USD query finds NOTHING -- but the PhysX
    # constraint may survive, so the known path is dropped + resynced regardless.
    a = _FakeResidualAdapter(joints=[], joints_after=[], kinematic=False)
    rrb.clear_residual_binds(a, body_path='/c', joint_path='/g/sim_attach_fallback', log=lambda *_: None)
    assert ('drop_joint', '/g/sim_attach_fallback') in a.calls
    assert ('physx_resync', None) in a.calls


def test_clear_residual_binds_clears_a_stuck_kinematic_flag():
    a = _FakeResidualAdapter(joints=[], joints_after=[], kinematic=True, kinematic_after=False)
    rec = rrb.clear_residual_binds(a, body_path='/c', joint_path=None, log=lambda *_: None)
    assert ('clear_kinematic', '/c') in a.calls
    assert rec['kinematic_before'] is True and rec['kinematic_cleared'] == 1
    assert rec['kinematic_after'] is False and rec['residual'] == ()


def test_clear_residual_binds_no_body_is_noop():
    a = _FakeResidualAdapter()
    rec = rrb.clear_residual_binds(a, body_path=None, joint_path='/j', log=lambda *_: None)
    assert a.calls == []                       # nothing queried/cleared
    assert rec['joint_paths_dropped'] == () and rec['kinematic_cleared'] == 0


def test_clear_residual_binds_reports_residual_when_clear_does_not_reconcile():
    # a joint STILL references the carton after the clear (the physx resync did not drop it) -> residual named
    logs = []
    a = _FakeResidualAdapter(joints=['/g/sim_attach_fallback'], joints_after=['/g/sim_attach_fallback'],
                             kinematic=False)
    rec = rrb.clear_residual_binds(a, body_path='/c', joint_path='/g/sim_attach_fallback', log=logs.append)
    assert rec['residual'] == ('/g/sim_attach_fallback',)
    assert any('RESIDUAL' in m and 'still_bound' in m for m in logs)


def test_clear_residual_binds_logs_by_name():
    logs = []
    a = _FakeResidualAdapter(joints=['/somewhere/second_weld'], joints_after=[], kinematic=True,
                             kinematic_after=False)
    rrb.clear_residual_binds(a, body_path='/c', joint_path='/g/sim_attach_fallback', log=logs.append)
    assert any('RELEASE_RESIDUAL_BIND kind=joint' in m and 'cleared=1' in m for m in logs)
    assert any('RELEASE_RESIDUAL_BIND kind=kinematic' in m and 'cleared=1' in m for m in logs)
    # the before/after STATE line mirrors the bench's RESIDUAL_BIND columns
    assert any('RELEASE_RESIDUAL_BIND_STATE before' in m and 'joints=' in m and 'kinematic=' in m
               for m in logs)


def test_clear_residual_binds_reenables_a_disabled_rigid_body():
    a = _FakeResidualAdapter(joints=[], joints_after=[], kinematic=False, rigid_enabled=False)
    rec = rrb.clear_residual_binds(a, body_path='/c', joint_path=None, log=lambda *_: None)
    assert ('set_rigid', '/c', True) in a.calls
    assert rec['rigid_reenabled'] == 1 and rec['residual'] == ()   # re-enabled -> no residual


def test_clear_residual_binds_names_a_reparent_without_forcing_it():
    # a carton parented OFF the scenario root (the object_attachment reparent suspect) is NAMED, not forcibly
    # undone -- and it shows as a residual so the positive free check refuses the lift.
    logs = []
    a = _FakeResidualAdapter(joints=[], joints_after=[], kinematic=False,
                             parent='/World/MobileRobot/gripper')
    rec = rrb.clear_residual_binds(a, body_path='/c', joint_path=None, log=logs.append)
    assert rec['reparented'] is True
    assert any('kind=reparent' in m for m in logs)
    assert any(str(r).startswith('reparent:') for r in rec['residual'])
    # no forced reparent call was made (no such adapter method is invoked)
    assert not any(c[0] == 'reparent' for c in a.calls)


def test_is_reparented_pure():
    assert rrb.is_reparented('/World/gripper') is True
    assert rrb.is_reparented('/RestockScenario') is False
    assert rrb.is_reparented('/RestockScenario/') is False
    assert rrb.is_reparented(None) is False


class _GripperAdapter(_FakeResidualAdapter):
    """Fake adapter that also reports gripper collider states (the pad-friction sweep)."""
    def __init__(self, *, gripper_colliders, **kw):
        super().__init__(**kw)
        self._gc = list(gripper_colliders)   # list of (path, enabled, tool_side)

    def gripper_collider_states(self):
        self.calls.append(('gripper_states', None))
        return list(self._gc)

    def disable_gripper_collider(self, path):
        self.calls.append(('disable_gripper', path))


def test_gripper_sweep_disables_enabled_tool_collider_and_names_non_tool():
    # a tool-side collider left ENABLED at release (the pad-friction drag suspect) is deferred; an enabled
    # NON-tool-side gripper collider is NAMED as a residual (not auto-deferred -- the positive check refuses).
    logs = []
    a = _GripperAdapter(joints=[], joints_after=[], kinematic=False,
                        gripper_colliders=[('/RestockPanelGripper/suction_cups', True, True),
                                           ('/RestockPanelGripper/surface_plate_1', False, True),
                                           ('/RestockPanelGripper/mystery_panel', True, False)])
    rec = rrb.clear_residual_binds(a, body_path='/c', joint_path=None, log=logs.append)
    assert ('disable_gripper', '/RestockPanelGripper/suction_cups') in a.calls   # enabled tool-side -> deferred
    assert not any(c == ('disable_gripper', '/RestockPanelGripper/surface_plate_1') for c in a.calls)  # already off
    assert rec['tool_collider_cleared'] == 1
    assert rec['gripper_residual'] == ('/RestockPanelGripper/mystery_panel',)   # non-tool-side enabled -> residual
    assert any('gripper_collider:/RestockPanelGripper/mystery_panel' in str(r) for r in rec['residual'])
    assert any('kind=tool_collider' in m and 'cleared=1' in m for m in logs)
    assert any('kind=gripper_collider' in m and 'enabled=True' in m for m in logs)


def test_gripper_sweep_absent_adapter_is_noop():
    # the base fake adapter has no gripper_collider_states -> the sweep is skipped (byte-identical)
    a = _FakeResidualAdapter(joints=[], joints_after=[], kinematic=False)
    rec = rrb.clear_residual_binds(a, body_path='/c', joint_path=None, log=lambda *_: None)
    assert rec['tool_collider_cleared'] == 0 and rec['gripper_residual'] == ()


# ========================================================= 4. ReleaseFreeGate.free_proof (positive check)

def _run_free(gate, seq):
    """Drive free_proof over a sequence of (carton_z, support_contact) and return the final decision/action."""
    dec = None
    for z, sc in seq:
        dec = gate.free_proof(carton_z=z, bottom_gap=(z - 0.09) - 0.448, support_contact=sc)
        if dec.action in ('lift', 'refuse'):
            return dec
    return dec


def test_free_proof_accepts_on_a_support_contact():
    g = ReleaseFreeGate(settle_ticks=3, positive_free=True, settle_window_ticks=60)
    # carton resting (z stable) with a reported carton-vs-support contact -> lift
    dec = _run_free(g, [(0.538, True)] * 6)
    assert dec.action == 'lift' and 'support contact' in dec.reason


def test_free_proof_accepts_on_a_measured_settle_drop():
    g = ReleaseFreeGate(settle_ticks=3, positive_free=True, settle_drop_min_m=0.003, settle_window_ticks=60)
    # release height 0.540; the freed carton drops >=3 mm onto the support then rests -> lift (no contact flag)
    seq = [(0.540, False), (0.538, False), (0.536, False), (0.5355, False), (0.5355, False),
           (0.5355, False), (0.5355, False)]
    dec = _run_free(g, seq)
    assert dec.action == 'lift' and 'settle drop' in dec.reason


def test_free_proof_refuses_a_bound_carton_exact_follow_no_contact():
    # the coldwin106 signature: carton held RIGID at seat (z EXACTLY constant, no contact) -> never provably
    # free -> refuse kind=no-support-contact WITHOUT lifting.
    g = ReleaseFreeGate(settle_ticks=5, positive_free=True, settle_window_ticks=12)
    dec = _run_free(g, [(0.5306, False)] * 20)
    assert dec.action == 'refuse' and 'kind=no-support-contact' in dec.reason


def test_free_proof_bound_carton_stable_and_resting_still_refuses():
    # a bound carton also reads |bottom_gap| small (resting) -- the legacy settle() passed on exactly this;
    # free_proof still refuses because there is no contact AND no drop.
    g = ReleaseFreeGate(settle_ticks=5, positive_free=True, settle_window_ticks=10, rest_gap_m=0.03)
    # z == support_top + half (bottom_gap ~0) held constant, no contact
    dec = _run_free(g, [(0.448 + 0.09, False)] * 15)
    assert dec.action == 'refuse' and 'no-support-contact' in dec.reason


def test_free_proof_off_leaves_legacy_settle_untouched():
    # positive_free default False -> settle() is the hold (byte-identical); a stable+resting bound carton
    # would PASS the legacy gate (that is the bug the positive check closes).
    g = ReleaseFreeGate(settle_ticks=3)   # positive_free False
    assert g.positive_free is False
    d = None
    for _ in range(6):
        d = g.settle(carton_z=0.5306, bottom_gap=-0.0002)
        if d.action == 'lift':
            break
    assert d.action == 'lift'             # the legacy false-pass -- proving free_proof is the needed change


def test_gate_rejects_bad_settle_drop_min():
    with pytest.raises(ValueError):
        ReleaseFreeGate(positive_free=True, settle_drop_min_m=0.0)
    with pytest.raises(ValueError):
        ReleaseFreeGate(positive_free=True, settle_drop_min_m=-0.001)


# ====================================================== 5. controller _release_free_step with positive_free

CARTON_H = 0.18


def _free_controller(gate):
    from twin.restock_manipulation.controller import TransferController
    c = NS()
    c._release_free = gate
    c._rf_stage = 'settle'
    c._released_detached = True
    c.profile = NS(dimensions=(0.45, 0.30, CARTON_H))
    c.target_surface = (1.11, 0.08, 0.448)
    c._transitions = []
    c._transition = lambda ph: (c._transitions.append(ph), setattr(c, 'phase', ph))[1]
    c._result = lambda controls=(): ('RESULT', tuple(controls))
    c._fail = lambda r: ('FAIL', r)
    c.evidence = []
    c._begun = []
    c.motion = NS(begin_joint_goal=lambda *a, **k: c._begun.append((a, k)), poll=lambda: 'running')
    c._retreat_lifted = False
    c._collect_retreat_liftoff_goal = lambda measured: (0, 0, 0, 0, 0, 0)
    c._step = TransferController._release_free_step.__get__(c, type(c))
    return c


def _sensors(carton_z, support_contact=False, joints=(0,) * 6):
    return NS(placed_pose=(1.11, 0.08, carton_z, 0, 0, 0, 1), joint_positions=joints,
              carton_support_contact=support_contact)


def test_controller_positive_refuses_bound_carton_without_lifting():
    g = ReleaseFreeGate(settle_ticks=5, positive_free=True, settle_window_ticks=8, coupled_abort_m=0.02)
    c = _free_controller(g)
    res = None
    for _ in range(12):
        res = c._step(2.0, _sensors(0.5306, support_contact=False))   # bound: exact z, no contact
        if res[0] == 'FAIL':
            break
    assert res[0] == 'FAIL' and 'RELEASE_NOT_FREE' in res[1] and 'no-support-contact' in res[1]
    assert c._begun == []                 # never commanded the clearance lift
    assert c._rf_stage == 'settle'        # never advanced to the lift stage


def test_controller_positive_lifts_on_support_contact():
    g = ReleaseFreeGate(settle_ticks=3, positive_free=True, settle_window_ticks=60, coupled_abort_m=0.02)
    c = _free_controller(g)
    for _ in range(6):
        c._step(2.0, _sensors(0.538, support_contact=True))
        if c._rf_stage == 'lift':
            break
    assert c._rf_stage == 'lift'
    assert c._begun                       # the clearance lift was commanded (carton proved free)


# ====================================================================== 6. config wiring (carton_level etc.)

def test_carton_level_parses_positive_free_keys():
    lc = cl.resolve_level_config({'release_free_check': True, 'release_free_positive': True,
                                  'release_free_settle_drop_min_m': 0.004})
    assert lc.release_free_positive is True and lc.release_free_settle_drop_min_m == 0.004
    lc0 = cl.resolve_level_config({'release_free_check': True})
    assert lc0.release_free_positive is False and lc0.release_free_settle_drop_min_m == 0.003
    with pytest.raises(ValueError):
        cl.resolve_level_config({'release_free_check': True, 'release_free_settle_drop_min_m': -1.0})


def test_manip_config_carries_release_residual_bind_clear():
    from twin.restock_manipulation import isaac_factory as IF
    assert 'release_residual_bind_clear' in IF.ManipConfig.__dataclass_fields__
    assert IF.ManipConfig(profiles={}, stations={}, shelfbot_root='', walls={}, voxel_dims=(1, 1, 1),
                          voxel_size=0.01, grasp_mode='side').release_residual_bind_clear is False


def test_config_preflight_registers_release_residual_bind_runtime():
    from twin.restock_manipulation import config_preflight as CP
    assert CP.KEY_CONSUMER.get('release_residual_bind') == CP.RUNTIME


def test_sensors_packet_carries_support_contact_default_false():
    from twin.restock_manipulation.types import Sensors
    assert 'carton_support_contact' in Sensors.__dataclass_fields__
    assert Sensors.__dataclass_fields__['carton_support_contact'].default is False


# =================================================== 7. cad_tool real adapter method bodies (fake pxr)

def _install_fake_pxr_for_query(monkeypatch, *, joints, carton, kinematic):
    """Install a minimal fake pxr so joints_referencing_carton / carton_kinematic run with no usd-core.
    ``joints`` maps joint_path -> list of body targets; ``carton`` is the rigid carton path."""
    import types as _t

    class _Rel:
        def __init__(self, targets): self._t = list(targets)
        def __bool__(self): return True
        def GetTargets(self): return list(self._t)

    class _Joint:
        def __init__(self, prim):
            self.prim = prim
            self._b = joints.get(prim.path, [])
        def __bool__(self): return self.prim.path in joints
        def GetBody0Rel(self): return _Rel(self._b[:1])
        def GetBody1Rel(self): return _Rel(self._b[1:])

    class _Attr:
        def __init__(self, v): self._v = v
        def Get(self): return self._v
        def HasAuthoredValue(self): return self._v is not None

    class _RigidBodyAPI:
        def __init__(self, prim): self.prim = prim
        def GetKinematicEnabledAttr(self): return _Attr(kinematic)

    JointAPI = object()
    RigidBodyAPIcls = _RigidBodyAPI

    UsdPhysics = _t.SimpleNamespace(Joint=_Joint, RigidBodyAPI=RigidBodyAPIcls, JointAPIObj=JointAPI)
    monkeypatch.setitem(sys.modules, 'pxr', _t.SimpleNamespace(UsdPhysics=UsdPhysics))
    monkeypatch.setitem(sys.modules, 'pxr.UsdPhysics', UsdPhysics)
    return UsdPhysics


class _QPrim:
    def __init__(self, path, is_joint=False, rigid=False):
        self.path = path; self._joint = is_joint; self._rigid = rigid
    def GetPath(self): return self.path
    def IsValid(self): return True
    def HasAPI(self, api): return self._rigid if 'RigidBody' in str(api) else self._joint


def test_real_joints_referencing_carton_enumerates(monkeypatch):
    carton = '/RestockScenario/HandlingUnit'
    joints = {'/g/sim_attach_fallback': ['/robot/body', carton], '/g/other': ['/robot/body', '/wall']}
    up = _install_fake_pxr_for_query(monkeypatch, joints=joints, carton=carton, kinematic=False)

    def _hasapi(self, api):
        return self.path in joints if api is up.Joint else False
    prims = [_QPrim('/g/sim_attach_fallback', is_joint=True), _QPrim('/g/other', is_joint=True),
             _QPrim(carton)]
    # make Joint(prim) truthy by membership; HasAPI(Joint) returns membership
    for p in prims:
        p.HasAPI = (lambda api, pp=p: pp.path in joints)
    stage = NS(Traverse=lambda: prims)
    from twin.restock_manipulation import cad_tool as C
    g = C.NativePanelContact.__new__(C.NativePanelContact)
    g.stage = stage
    found = g.joints_referencing_carton(carton)
    assert '/g/sim_attach_fallback' in found and '/g/other' not in found


def test_real_carton_kinematic_reads_flag(monkeypatch):
    carton = '/RestockScenario/HandlingUnit'
    up = _install_fake_pxr_for_query(monkeypatch, joints={}, carton=carton, kinematic=True)
    prim = _QPrim(carton, rigid=True)
    prim.HasAPI = lambda api: 'RigidBody' in str(api) or api is up.RigidBodyAPI
    stage = NS(GetPrimAtPath=lambda p: prim if p == carton else None)
    from twin.restock_manipulation import cad_tool as C
    g = C.NativePanelContact.__new__(C.NativePanelContact)
    g.stage = stage
    assert g.carton_kinematic(carton) is True


# ============================================= 8. cad_tool _release_fallback runs the residual-bind clear

class _FakeStage:
    def __init__(self): self.removed = []
    def RemovePrim(self, p): self.removed.append(p)
    def GetPrimAtPath(self, p): return None


def test_release_fallback_runs_residual_bind_clear_when_on():
    from twin.restock_manipulation import cad_tool as C
    g = C.NativePanelContact.__new__(C.NativePanelContact)
    g.path = '/World/robot/gripper'
    g.stage = _FakeStage()
    g._sim_attach_fallback = True
    g._fallback_joint = g.path + '/sim_attach_fallback'
    g._fallback_body = '/RestockScenario/HandlingUnit'
    g._defer_tool_collision = True
    g._release_detach_robust = False
    g._release_residual_bind = True
    g._tool_collision_deferred = False
    g._set_carton_collision = lambda *a, **k: None
    cleared = {}
    # stub the residual-bind adapter methods so the clear runs with no pxr
    g.joints_referencing_carton = lambda body: cleared.setdefault('queried', []).append(body) or []
    g.carton_kinematic = lambda body: False
    g.drop_joint = lambda jp: cleared.setdefault('dropped', []).append(jp)
    g.physx_resync = lambda: cleared.__setitem__('resynced', True)
    jp = g._fallback_joint
    g._release_fallback()
    assert cleared.get('queried')                        # the residual query ran
    assert jp in cleared.get('dropped', [])              # the known joint was re-asserted
    assert cleared.get('resynced') is True


def test_release_fallback_skips_residual_bind_clear_when_off():
    from twin.restock_manipulation import cad_tool as C
    g = C.NativePanelContact.__new__(C.NativePanelContact)
    g.path = '/World/robot/gripper'
    g.stage = _FakeStage()
    g._sim_attach_fallback = True
    g._fallback_joint = g.path + '/sim_attach_fallback'
    g._fallback_body = '/RestockScenario/HandlingUnit'
    g._defer_tool_collision = True
    g._release_detach_robust = False
    g._release_residual_bind = False                     # OFF
    g._tool_collision_deferred = False
    g._set_carton_collision = lambda *a, **k: None
    ran = {'q': False}
    g.joints_referencing_carton = lambda body: ran.__setitem__('q', True) or []
    g._release_fallback()
    assert ran['q'] is False                             # the residual clear never ran (byte-identical)


# ====================================================== 9. place_contact_logger support-contact query

def test_support_contact_seen_latches_carton_vs_support_only():
    from twin.restock_manipulation.place_contact_logger import PlaceContactLogger
    lg = PlaceContactLogger(log=lambda *_: None)
    assert lg.support_contact_seen() is False
    # a carton-vs-POST contact is NOT support
    lg.on_contact('/RestockScenario/HandlingUnit', '/Rollwagen/chassis/post_2', -0.001, kind='carton')
    assert lg.support_contact_seen() is False
    # a carton-vs-DECK contact IS support
    lg.on_contact('/RestockScenario/HandlingUnit', '/Rollwagen/chassis/deck', -0.001, kind='carton')
    assert lg.support_contact_seen() is True
    lg.clear_support_contact()
    assert lg.support_contact_seen() is False


def test_support_contact_ignores_arm_gripper_contacts():
    from twin.restock_manipulation.place_contact_logger import PlaceContactLogger
    lg = PlaceContactLogger(log=lambda *_: None)
    # a GRIPPER-vs-deck contact is not the carton resting
    lg.on_contact('/MobileRobot/.../suction_cups', '/Rollwagen/chassis/deck', -0.001, kind='gripper')
    assert lg.support_contact_seen() is False


# ================================= 10. panel-collider de-instance fix (the confirmed root cause)

class _InstPrim:
    def __init__(self, name, instanceable=False, children=None):
        self._name = name
        self._inst = instanceable
        self._children = children or []
    def GetName(self): return self._name
    def IsInstanceable(self): return self._inst
    def SetInstanceable(self, v): self._inst = bool(v)
    def GetChildren(self): return list(self._children)


class _InstStage:
    def __init__(self, prims): self._prims = prims
    def Traverse(self):
        # flat traversal that (like USD) does NOT descend into instanceable prims' children
        out = []
        def walk(p):
            out.append(p)
            if not p.IsInstanceable():
                for c in p.GetChildren():
                    walk(c)
        for p in self._prims:
            walk(p)
        return out


def test_deinstance_panel_colliders_covers_surface_plates_and_cube(monkeypatch):
    # stub Usd.PrimRange so the pure helper runs with no usd-core
    import types as _t
    monkeypatch.setitem(sys.modules, 'pxr', _t.SimpleNamespace(
        Usd=_t.SimpleNamespace(PrimRange=lambda prim: [prim] + list(prim.GetChildren()))))
    from twin.restock_manipulation import cad_tool as C
    plates = [_InstPrim('surface_plate_%d' % i, instanceable=True) for i in range(1, 7)]
    cube = _InstPrim('cube_without_surface_plates', instanceable=True)
    other = _InstPrim('base_link', instanceable=False)
    stage = _InstStage(plates + [cube, other])
    done = C.deinstance_panel_colliders(stage)
    # every surface_plate + the cube is de-instanced; suction_cups is NOT in the panel set (handled at bind)
    assert set(done) == {'surface_plate_%d' % i for i in range(1, 7)} | {'cube_without_surface_plates'}
    assert all(not p.IsInstanceable() for p in plates) and not cube.IsInstanceable()
    assert other.IsInstanceable() is False and 'base_link' not in done
    assert 'suction_cups' not in C.PANEL_COLLIDERS and set(C.PANEL_COLLIDERS) < set(C.TOOL_SIDE_COLLIDERS)


def test_panel_colliders_set_excludes_suction_cups():
    from twin.restock_manipulation import cad_tool as C
    assert 'suction_cups' not in C.PANEL_COLLIDERS
    assert set(C.PANEL_COLLIDERS) == {'cube_without_surface_plates'} | {'surface_plate_%d' % i for i in range(1, 7)}


def test_author_native_gripper_accepts_deinstance_flag():
    import inspect
    from twin.restock_manipulation import cad_tool as C
    sig = inspect.signature(C.author_native_gripper)
    assert 'deinstance_tool_colliders' in sig.parameters
    assert sig.parameters['deinstance_tool_colliders'].default is False   # default off => byte-identical


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
