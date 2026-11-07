"""CPU tests for sim-place-release-carton-stays-v1.

coldwin105 -- the first collect set-down that reached the shelf seat -- acknowledged the release end to
end yet the carton stayed physically coupled to the gripper: through the 60 mm clearance lift the SEAL_DIAG
pad->carton gap held 0.0 mm and pad-vs-face 0.0 deg while the measured carton CENTRE rose 0.1494 m with the
arm, and the PR 384 contact logger saw ZERO carton-vs-cart contacts. The re-enable of the carton collider
DID take (the seal raycast went from min_gap_mm=None while welded to 0.48 mm the tick after release), so the
carton is NOT missing a collider -- it is still rigidly welded. Named mechanism: ``stage.RemovePrim`` removes
the FixedJoint PRIM (so host read() -> held=False) but does NOT drop the PhysX constraint at runtime.

Two opt-in, coordinated fixes, proven here on CPU (no Isaac/pxr/GPU):
  1. Robust ORDERED detach (release_detach + cad_tool): disable the weld joint (clear body0/body1 +
     jointEnabled=False) BEFORE RemovePrim, then make the freed carton a dynamic body with zeroed residual
     velocity. Order proven via the pure planner + a recording physics-layer adapter.
  2. Host coupled-rise verification (release_free + controller): the clearance lift is polled PER TICK and
     fails by name the instant the carton rises past the abort threshold -- naming a still-coupled carton
     early and non-destructively, before the lateral retreat, not after the 0.1494 m swing.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts python3 -m pytest \
        scripts/twin/tests/test_release_detach_carton_stays_v1.py
"""
import sys
from types import SimpleNamespace as NS

import pytest

from twin.restock_manipulation import release_detach as rd
from twin.restock_manipulation import carton_level as cl
from twin.restock_manipulation.release_free import ReleaseFreeGate


# =========================================================== 1. ordered_release_ops (the release ordering)

def test_robust_off_order_is_byte_identical_legacy():
    assert rd.ordered_release_ops(robust=False, defer_tool=True) == (
        rd.ENABLE_CARTON_COLLISION, rd.DEFER_TOOL_COLLISION, rd.REMOVE_WELD_JOINT)
    assert rd.ordered_release_ops(robust=False, defer_tool=False) == (
        rd.ENABLE_CARTON_COLLISION, rd.RESTORE_TOOL_COLLISION, rd.REMOVE_WELD_JOINT)
    assert rd.ordered_release_ops(robust=False, defer_tool=True, release_kinematic=True) == (
        rd.ENABLE_CARTON_COLLISION, rd.DEFER_TOOL_COLLISION, rd.REMOVE_WELD_JOINT, rd.SET_CARTON_KINEMATIC)


def test_robust_on_disables_joint_before_removal_then_frees_the_carton():
    ops = rd.ordered_release_ops(robust=True, defer_tool=True)
    # the carton collider is re-enabled FIRST (so it can rest); the weld joint is DISABLED before it is
    # REMOVED; the freed carton is made dynamic + velocity-zeroed AFTER the removal.
    assert ops == (rd.ENABLE_CARTON_COLLISION, rd.DISABLE_WELD_JOINT, rd.DEFER_TOOL_COLLISION,
                   rd.REMOVE_WELD_JOINT, rd.CLEAR_CARTON_KINEMATIC, rd.ZERO_CARTON_VELOCITY)
    assert ops.index(rd.DISABLE_WELD_JOINT) < ops.index(rd.REMOVE_WELD_JOINT)
    assert ops.index(rd.ENABLE_CARTON_COLLISION) < ops.index(rd.DISABLE_WELD_JOINT)
    assert ops.index(rd.REMOVE_WELD_JOINT) < ops.index(rd.ZERO_CARTON_VELOCITY)


def test_robust_never_sets_the_kinematic_flag():
    # robust means a genuinely FREE DYNAMIC body -- the proof-run kinematic shortcut is never selected.
    ops = rd.ordered_release_ops(robust=True, defer_tool=True, release_kinematic=True)
    assert rd.SET_CARTON_KINEMATIC not in ops
    assert rd.CLEAR_CARTON_KINEMATIC in ops


def test_robust_on_defers_tool_side_so_restore_cannot_overlap_the_freed_carton():
    # zero-overlap restoration of the tool colliders: with the deferral on, the tool-side colliders are
    # NOT restored at release (only on the next grasp close, gripper then clear of the carton in free space).
    ops = rd.ordered_release_ops(robust=True, defer_tool=True)
    assert rd.DEFER_TOOL_COLLISION in ops and rd.RESTORE_TOOL_COLLISION not in ops


# =========================================================== 2. robust_detach_enabled (config resolution)

def test_robust_detach_default_off():
    assert rd.robust_detach_enabled({}) is False
    assert rd.robust_detach_enabled(None) is False


def test_robust_detach_config_and_env_and_dict():
    assert rd.robust_detach_enabled({'release_detach': True}) is True
    assert rd.robust_detach_enabled({'release_detach': 'on'}) is True
    assert rd.robust_detach_enabled({'release_detach': 'off'}) is False
    assert rd.robust_detach_enabled({'release_detach': {'robust': True}}) is True
    assert rd.robust_detach_enabled({}, environ={'RESTOCK_RELEASE_DETACH_ROBUST': '1'}) is True
    # an explicit config value wins over the env
    assert rd.robust_detach_enabled({'release_detach': False},
                                    environ={'RESTOCK_RELEASE_DETACH_ROBUST': '1'}) is False


def test_robust_detach_rejects_malformed():
    with pytest.raises(ValueError):
        rd.robust_detach_enabled({'release_detach': 'maybe'})


# ================================================= 3. apply_release_ops reaches the physics layer in order

class _RecordingAdapter:
    """A fake physics layer: every per-op method records its (op, arg) so the ordered release can be
    proven without Isaac/pxr (the brief's fake-stage-for-the-ordering test)."""
    def __init__(self):
        self.calls = []

    def enable_carton_collision(self, body):
        self.calls.append(('enable_carton_collision', body))

    def disable_weld_joint(self, joint):
        self.calls.append(('disable_weld_joint', joint))

    def defer_tool_collision(self, body):
        self.calls.append(('defer_tool_collision', body))

    def restore_tool_collision(self, body):
        self.calls.append(('restore_tool_collision', body))

    def remove_weld_joint(self, joint):
        self.calls.append(('remove_weld_joint', joint))

    def set_carton_kinematic(self, body):
        self.calls.append(('set_carton_kinematic', body))

    def clear_carton_kinematic(self, body):
        self.calls.append(('clear_carton_kinematic', body))

    def zero_carton_velocity(self, body):
        self.calls.append(('zero_carton_velocity', body))


def test_apply_release_ops_robust_reaches_the_layer_in_order():
    a = _RecordingAdapter()
    ops = rd.ordered_release_ops(robust=True, defer_tool=True)
    rd.apply_release_ops(ops, a, body_path='/c', joint_path='/g/sim_attach_fallback')
    names = [c[0] for c in a.calls]
    assert names == ['enable_carton_collision', 'disable_weld_joint', 'defer_tool_collision',
                     'remove_weld_joint', 'clear_carton_kinematic', 'zero_carton_velocity']
    # the joint-disable and removal carry the joint path; the carton ops carry the body path
    assert ('disable_weld_joint', '/g/sim_attach_fallback') in a.calls
    assert ('zero_carton_velocity', '/c') in a.calls


def test_apply_release_ops_legacy_defer_reaches_the_layer_in_order():
    a = _RecordingAdapter()
    ops = rd.ordered_release_ops(robust=False, defer_tool=True)
    rd.apply_release_ops(ops, a, body_path='/c', joint_path='/j')
    assert [c[0] for c in a.calls] == ['enable_carton_collision', 'defer_tool_collision', 'remove_weld_joint']


def test_apply_release_ops_skips_carton_ops_when_no_body_but_removes_joint():
    a = _RecordingAdapter()
    ops = rd.ordered_release_ops(robust=True, defer_tool=True)
    rd.apply_release_ops(ops, a, body_path=None, joint_path='/j')
    names = [c[0] for c in a.calls]
    # no body => only the joint ops run (disable + remove); every carton-release op is skipped
    assert names == ['disable_weld_joint', 'remove_weld_joint']


# ========================================= 4. cad_tool._release_fallback dispatches the ops to the stage

class _FakeStage:
    def __init__(self):
        self.removed = []

    def RemovePrim(self, p):
        self.removed.append(p)


def _stub_contact(*, robust, defer):
    """A NativePanelContact (via __new__, no isaacsim) carrying only the attrs the release path touches.
    The pxr-backed adapter methods are stubbed to RECORD, so the dispatch order is provable with no pxr."""
    from twin.restock_manipulation import cad_tool as C
    g = C.NativePanelContact.__new__(C.NativePanelContact)
    g.path = '/World/robot/gripper'
    g.stage = _FakeStage()
    g._sim_attach_fallback = True
    g._fallback_joint = g.path + '/sim_attach_fallback'
    g._fallback_body = '/RestockScenario/HandlingUnit'
    g._defer_tool_collision = defer
    g._release_detach_robust = robust
    g._tool_collision_deferred = False
    g.ops = []
    g._set_carton_collision = lambda body, enabled, groups='both': g.ops.append(('enable', enabled, groups))
    # stub the pxr-backed robust adapter methods to record (no pxr in the test env)
    g.disable_weld_joint = lambda jp: g.ops.append(('disable_weld_joint', jp))
    g.clear_carton_kinematic = lambda bp: g.ops.append(('clear_carton_kinematic', bp))
    g.zero_carton_velocity = lambda bp: g.ops.append(('zero_carton_velocity', bp))
    return g


def test_release_fallback_robust_disables_joint_before_removal_then_frees_carton():
    g = _stub_contact(robust=True, defer=True)
    jp = g._fallback_joint                                # captured before _release_fallback clears it
    g._release_fallback()
    names = [o[0] for o in g.ops]
    assert names == ['enable', 'disable_weld_joint', 'clear_carton_kinematic', 'zero_carton_velocity']
    # the carton collider is re-enabled (carton-only, tool-side deferred)
    assert g.ops[0] == ('enable', True, 'carton')
    # the weld joint is disabled with the joint path and then the prim is removed
    assert ('disable_weld_joint', jp) in g.ops
    assert g.stage.removed == [jp]
    assert g._tool_collision_deferred is True            # tool-side held OFF (zero-overlap restore)
    assert g._fallback_joint is None and g._fallback_body is None


def test_release_fallback_robust_off_is_byte_identical_legacy():
    g = _stub_contact(robust=False, defer=True)
    jp = g._fallback_joint
    g._release_fallback()
    # legacy: enable carton-only, defer tool-side, remove -- NO disable/clear/zero
    assert [o[0] for o in g.ops] == ['enable']
    assert g.ops == [('enable', True, 'carton')]
    assert g.stage.removed == [jp]
    assert g._tool_collision_deferred is True


# ============================= 5. cad_tool real pxr adapter method bodies (fake pxr -> real mutations)

def _install_fake_pxr(monkeypatch):
    """Install a minimal fake ``pxr`` so the real adapter method bodies run (and assert their mutations)
    without usd-core. Returns a registry the test inspects."""
    import types as _t
    reg = {'joints': {}, 'bodies': {}}

    class _Attr:
        def __init__(self): self.value = None
        def Set(self, v): self.value = v

    class _Rel:
        def __init__(self): self.targets = ['/preexisting']
        def SetTargets(self, t): self.targets = list(t)

    class _Joint:
        def __init__(self, prim):
            self.prim = prim
            e = reg['joints'].setdefault(prim.path, {'b0': _Rel(), 'b1': _Rel(), 'enabled': _Attr()})
            self._e = e
        def GetBody0Rel(self): return self._e['b0']
        def GetBody1Rel(self): return self._e['b1']
        def CreateJointEnabledAttr(self): return self._e['enabled']

    class _RigidBodyAPI:
        def __init__(self, prim):
            self.prim = prim
            self._b = reg['bodies'].setdefault(prim.path, {'kinematic': _Attr(), 'vel': _Attr(), 'ang': _Attr()})
        def CreateKinematicEnabledAttr(self): return self._b['kinematic']
        def CreateVelocityAttr(self): return self._b['vel']
        def CreateAngularVelocityAttr(self): return self._b['ang']

    UsdPhysics = _t.SimpleNamespace(Joint=_Joint, RigidBodyAPI=_RigidBodyAPI)

    class _Vec3f:
        def __init__(self, x, y, z): self.t = (x, y, z)
        def __eq__(self, o): return isinstance(o, _Vec3f) and self.t == o.t

    Gf = _t.SimpleNamespace(Vec3f=_Vec3f)
    monkeypatch.setitem(sys.modules, 'pxr', _t.SimpleNamespace(UsdPhysics=UsdPhysics, Gf=Gf))
    monkeypatch.setitem(sys.modules, 'pxr.UsdPhysics', UsdPhysics)
    monkeypatch.setitem(sys.modules, 'pxr.Gf', Gf)
    return reg, _Vec3f


class _FakePrim:
    def __init__(self, path, rigid=True): self.path = path; self._rigid = rigid
    def IsValid(self): return True
    def HasAPI(self, api): return self._rigid


class _FakePrimStage:
    def __init__(self, prims): self._prims = prims; self.removed = []
    def GetPrimAtPath(self, p): return self._prims.get(p)
    def RemovePrim(self, p): self.removed.append(p)


def test_real_disable_weld_joint_clears_bodies_and_disables(monkeypatch):
    reg, _ = _install_fake_pxr(monkeypatch)
    from twin.restock_manipulation import cad_tool as C
    g = C.NativePanelContact.__new__(C.NativePanelContact)
    jp = '/g/sim_attach_fallback'
    g.stage = _FakePrimStage({jp: _FakePrim(jp)})
    g.disable_weld_joint(jp)
    assert reg['joints'][jp]['b0'].targets == []          # body0 cleared
    assert reg['joints'][jp]['b1'].targets == []          # body1 cleared
    assert reg['joints'][jp]['enabled'].value is False    # jointEnabled=False -> constraint dropped


def test_real_free_dynamic_clears_kinematic_and_zeroes_velocity(monkeypatch):
    reg, Vec3f = _install_fake_pxr(monkeypatch)
    from twin.restock_manipulation import cad_tool as C
    g = C.NativePanelContact.__new__(C.NativePanelContact)
    bp = '/RestockScenario/HandlingUnit'
    g.stage = _FakePrimStage({bp: _FakePrim(bp, rigid=True)})
    g.clear_carton_kinematic(bp)
    g.zero_carton_velocity(bp)
    assert reg['bodies'][bp]['kinematic'].value is False  # freed carton is DYNAMIC
    assert reg['bodies'][bp]['vel'].value == Vec3f(0.0, 0.0, 0.0)
    assert reg['bodies'][bp]['ang'].value == Vec3f(0.0, 0.0, 0.0)


# ================================== 6. host coupled-rise verification: ReleaseFreeGate.rising_with_gripper

def test_rising_with_gripper_off_is_inert():
    g = ReleaseFreeGate()                       # coupled_abort_m None => off
    g.lift_baseline = 0.5
    assert g.rising_with_gripper(carton_z=99.0).action == 'running'


def test_rising_with_gripper_refuses_past_the_threshold():
    g = ReleaseFreeGate(coupled_abort_m=0.015)
    g.lift_baseline = 0.5
    assert g.rising_with_gripper(carton_z=0.510).action == 'running'     # 10 mm < 15 mm abort
    d = g.rising_with_gripper(carton_z=0.520)                            # 20 mm >= 15 mm abort
    assert d.action == 'refuse' and 'still' in d.reason and 'coupled' in d.reason


def test_rising_with_gripper_no_baseline_is_running():
    g = ReleaseFreeGate(coupled_abort_m=0.015)   # lift_baseline None until settle latches it
    assert g.rising_with_gripper(carton_z=1.0).action == 'running'


def test_gate_rejects_bad_coupled_abort():
    with pytest.raises(ValueError):
        ReleaseFreeGate(coupled_abort_m=-0.01)
    with pytest.raises(ValueError):
        ReleaseFreeGate(coupled_abort_m=0.0)


# ============================================ 7. controller lift stage aborts early on a coupled rise

CARTON_H = 0.18


def _free_controller(stage='settle', gate=None, poll='running', liftoff=(0, 0, 0, 0, 0, 0)):
    from twin.restock_manipulation.controller import TransferController
    c = NS()
    c._release_free = gate
    c._rf_stage = stage
    c._released_detached = True
    c.profile = NS(dimensions=(0.45, 0.30, CARTON_H))
    c.target_surface = (1.11, 0.08, 0.448)
    c._transitions = []
    c._transition = lambda ph: (c._transitions.append(ph), setattr(c, 'phase', ph))[1]
    c._result = lambda controls=(): ('RESULT', tuple(controls))
    c._fail = lambda r: ('FAIL', r)
    c.evidence = []
    c._begun = []
    c.motion = NS(begin_joint_goal=lambda *a, **k: c._begun.append((a, k)), poll=lambda: poll)
    c._retreat_lifted = False
    c._collect_retreat_liftoff_goal = lambda measured: liftoff
    c._step = TransferController._release_free_step.__get__(c, type(c))
    return c


def _sensors(carton_z, joints=(0,) * 6):
    return NS(placed_pose=(1.11, 0.08, carton_z, 0, 0, 0, 1), joint_positions=joints)


def test_lift_aborts_early_on_a_coupled_rise_before_the_motion_completes():
    gate = ReleaseFreeGate(settle_ticks=1, lift_dz_m=0.06, follow_frac=0.5, coupled_abort_m=0.02)
    c = _free_controller(stage='settle', gate=gate)
    # settle (carton resting at a stable z) -> lift
    for _ in range(3):
        c._step(2.0, _sensors(0.538))
        if c._rf_stage == 'lift':
            break
    assert c._rf_stage == 'lift'
    # the lift is still RUNNING (poll='running'), but the carton has risen 25 mm >= 20 mm abort -> FAIL now,
    # without waiting for the full (coldwin105 149 mm) swing.
    res = c._step(2.1, _sensors(0.538 + 0.025))
    assert res[0] == 'FAIL' and 'RELEASE_NOT_FREE' in res[1] and 'clearance lift' in res[1]


def test_lift_without_abort_key_is_byte_identical_running():
    gate = ReleaseFreeGate(settle_ticks=1, lift_dz_m=0.06, follow_frac=0.5)   # no abort key
    c = _free_controller(stage='settle', gate=gate)
    for _ in range(3):
        c._step(2.0, _sensors(0.538))
        if c._rf_stage == 'lift':
            break
    assert c._rf_stage == 'lift'
    # a small rise during the lift does NOT abort (the lift-completion follow() owns the check as before)
    res = c._step(2.1, _sensors(0.538 + 0.025))
    assert res[0] == 'RESULT'        # still running, byte-identical with the pre-fix behaviour


# ========================================================= 8. config wiring (carton_level + ManipConfig)

def test_carton_level_parses_and_validates_coupled_abort():
    lc = cl.resolve_level_config({'release_free_check': True, 'release_free_coupled_abort_m': 0.02})
    assert lc.release_free_coupled_abort_m == 0.02
    # absent => None => off => byte-identical
    lc0 = cl.resolve_level_config({'release_free_check': True})
    assert lc0.release_free_coupled_abort_m is None
    with pytest.raises(ValueError):
        cl.resolve_level_config({'release_free_check': True, 'release_free_coupled_abort_m': -1.0})


def test_manip_config_carries_release_detach_robust():
    from twin.restock_manipulation import isaac_factory as IF
    assert 'release_detach_robust' in IF.ManipConfig.__dataclass_fields__
    # default off, and the resolver drives it from the runtime config
    assert IF.ManipConfig(profiles={}, stations={}, shelfbot_root='', walls={}, voxel_dims=(1, 1, 1),
                          voxel_size=0.01, grasp_mode='side').release_detach_robust is False


def test_config_preflight_registers_release_detach_runtime():
    from twin.restock_manipulation import config_preflight as CP
    assert CP.KEY_CONSUMER.get('release_detach') == CP.RUNTIME


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
