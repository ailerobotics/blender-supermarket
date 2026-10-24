"""CPU-only proof of the collect release-detach handshake robustness (sim-robust-release-detach-ack-v1).

Root cause (coldlive vs coldhonest, 2026-10-23): the collect RELEASE detach is a two-step host
handshake -- a move_group ``ApplyPlanningScene`` REMOVE, then the authoritative object_attachment
``AttachObject`` detach. move_group (launched ``respawn=True``) hit its intermittent SIGABRT double-free
AFTER the grasp attach and respawned BEFORE the detach; the respawned move_group's fresh planning scene
never held ``restock_carried_unit``, so the REMOVE returned ``success=False`` and the single-shot
handshake aborted the whole collect (``gate=scene_apply_unsuccessful``). coldhonest never crashed, so its
REMOVE succeeded first try. An unsuccessful REMOVE means there is nothing to remove (the scene is already
post-detach), and the object_attachment node survives the crash in a separate container, so the detach
must PROCEED to that authoritative action -- which is what decides the real ack.

These tests pin the pure decision helpers (``detach_scene_gate``/``attach_acked``) and re-enact the
``RosPayloadSync.__call__`` scene->attach->ack sequence with them, and pin the mission-visible controller
contract (release acks -> retreat; a genuinely unacknowledged detach fails by name). No ROS graph / GPU.
"""
import pytest

from types import SimpleNamespace

from scripts.twin.restock_manipulation.ros_payload import (
    ATTACH_ACTION_SUCCEEDED, RosPayloadSync, attach_acked, detach_scene_gate, reattach_reject_gate,
    resolve_detach_scene_tolerant, scene_bootstrap_ops)

DETACH, ATTACH = False, True


# --- pure scene gate -------------------------------------------------------------------------------

def test_detach_scene_gate_successful_apply_proceeds_with_no_note():
    # A successful ApplyPlanningScene always proceeds, attach or detach, with no diagnostic note
    # (the normal coldhonest path) -- byte-identical to the pre-fix flow.
    assert detach_scene_gate(DETACH, True, tolerant=True) == (True, None)
    assert detach_scene_gate(ATTACH, True, tolerant=True) == (True, None)
    assert detach_scene_gate(DETACH, True, tolerant=False) == (True, None)


def test_detach_scene_gate_tolerates_unsuccessful_remove_on_detach():
    # coldlive: move_group respawned, the detach REMOVE returns success=False -> PROCEED to the
    # authoritative AttachObject detach, flagged by the named tolerated-gate note.
    proceed, note = detach_scene_gate(DETACH, False, tolerant=True)
    assert proceed is True
    assert note == 'scene_apply_unsuccessful_detach_tolerated'


def test_detach_scene_gate_never_tolerates_unsuccessful_add_on_attach():
    # An ATTACH must NOT proceed on an unsuccessful scene apply: the object has to be ADDED to the
    # planning scene before the lift is planned. Aborts at the scene apply with the hard gate.
    assert detach_scene_gate(ATTACH, False, tolerant=True) == (False, 'scene_apply_unsuccessful')


def test_detach_scene_gate_tolerance_off_restores_fail_fast():
    # RESTOCK_DETACH_SCENE_TOLERANT=0 (reproduce a pre-fix run): a failed detach REMOVE aborts as before.
    assert detach_scene_gate(DETACH, False, tolerant=False) == (False, 'scene_apply_unsuccessful')


def test_attach_acked_only_on_succeeded_status():
    assert attach_acked(ATTACH_ACTION_SUCCEEDED) is True
    assert attach_acked(4) is True
    for bad in (None, 0, 1, 2, 3, 5, 6):
        assert attach_acked(bad) is False


def test_resolve_detach_scene_tolerant_env_semantics():
    assert resolve_detach_scene_tolerant(None) is True       # unset => default-on
    assert resolve_detach_scene_tolerant('0') is False       # explicit off => reproduce pre-fix
    assert resolve_detach_scene_tolerant('1') is True
    assert resolve_detach_scene_tolerant('true') is True     # any non-'0' => on


# --- the handshake sequence (faithful re-enactment of RosPayloadSync.__call__'s scene->attach tail) ---

def _handshake(held, scene_ok, attach_status, *, tolerant):
    """Exactly the sequence RosPayloadSync.__call__ runs after the precondition/service-readiness gates:
    apply the scene diff, decide via detach_scene_gate whether to proceed, send the authoritative
    AttachObject action only when proceeding, then ack via attach_acked. Returns (ack, gate) where gate
    names the stop point (matching the PAYLOAD_SYNC_DIAG gate labels) or 'ok'."""
    proceed, note = detach_scene_gate(held, scene_ok, tolerant)
    if not proceed:
        return False, note                      # aborts at the scene apply; AttachObject never sent
    if not attach_acked(attach_status):
        return False, 'attach_result_status=%s' % attach_status
    return True, note or 'ok'


def test_detach_acknowledged_under_both_configurations():
    # coldhonest config (arm feed off, no crash): the scene REMOVE succeeds first try -> ack.
    assert _handshake(DETACH, True, ATTACH_ACTION_SUCCEEDED, tolerant=True) == (True, 'ok')
    # coldlive config (live_pose + arm masked-depth feed -> move_group crash+respawn): the scene REMOVE
    # reports unsuccessful, but the authoritative object_attachment detach SUCCEEDS -> still acknowledged.
    assert _handshake(DETACH, False, ATTACH_ACTION_SUCCEEDED, tolerant=True) == (
        True, 'scene_apply_unsuccessful_detach_tolerated')


def test_genuinely_missing_detach_still_fails_by_name():
    # Tolerating the scene REMOVE never hides a REAL detach failure: if the authoritative AttachObject
    # detach does not succeed, the handshake is NOT acknowledged and names the gate.
    ack, gate = _handshake(DETACH, False, 2, tolerant=True)   # status 2 == ABORTED, not SUCCEEDED
    assert ack is False
    assert gate == 'attach_result_status=2'


def test_detach_fail_fast_when_tolerance_disabled():
    # With tolerance off, the coldlive scene-REMOVE failure aborts at the scene apply (pre-fix behaviour):
    # the authoritative detach is never even attempted.
    assert _handshake(DETACH, False, ATTACH_ACTION_SUCCEEDED, tolerant=False) == (
        False, 'scene_apply_unsuccessful')


# --- sim-robust-movegroup-reattach-after-respawn-v1: post-respawn reconcile re-attach --------------

def test_reattach_reject_gate_names_the_surviving_attach_cause():
    # The legacy first-start gate is UNCHANGED (byte-identical) when not reconciling.
    assert reattach_reject_gate(False, None) == 'attach_goal_rejected'
    # On a reconcile re-attach where the detach-first CLEARED the surviving attach yet the re-ADD was still
    # refused, the gate names that it was an unexpected post-clear rejection (not the duplicate cause).
    assert reattach_reject_gate(True, True) == 'attach_goal_rejected_after_reconcile_detach'
    # When the detach-first was NOT acked (could not clear object_attachment's surviving attach), surface the
    # KNOWN cause by name instead of the bare gate (object_attachment logs it only in its own container).
    gate = reattach_reject_gate(True, False)
    assert 'object_attachment_already_holds_restock_carried_unit' in gate


def test_scene_bootstrap_ops_first_start_equals_post_respawn():
    # The port re-establishes the SAME scene at first start and at the post-respawn resync: given the same
    # recorded state, scene_bootstrap_ops returns the same ordered plan -- which is what makes them equal.
    prof = SimpleNamespace(sku='carton-A', dimensions=(0.3, 0.3, 0.3))
    world = [('cart_cage', [((0.6, 0.6, 0.6), (0.0, -1.27, 0.42), (0, 0, 0))])]
    first_start = scene_bootstrap_ops(True, world, prof)
    post_respawn = scene_bootstrap_ops(True, world, prof)
    assert first_start == post_respawn
    assert first_start == [('base_obstacle',), ('world_obstacle', 'cart_cage'), ('attach', 'carton-A')]
    # No obstacles + nothing held (respawn2 shape with no carton) -> empty, byte-identical no-op.
    assert scene_bootstrap_ops(False, [], None) == []
    # respawn2's actual shape: no obstacles, a carton held -> only the attach, nothing else to re-publish.
    assert scene_bootstrap_ops(False, [], prof) == [('attach', 'carton-A')]


# --- _reconcile_detach: clears a SURVIVING object_attachment attach before the re-ADD ---------------

class _Fut:
    def __init__(self, val):
        self._val = val

    def result(self):
        return self._val


class _Handle:
    def __init__(self, accepted, status):
        self.accepted, self._status = accepted, status

    def get_result_async(self):
        return _Fut(SimpleNamespace(status=self._status))


class _FakeAttachObject:
    class Goal:
        def __init__(self):
            self.attach_object = None


class _AttachClient:
    """Models the object_attachment AttachObject server for a DETACH: acked SUCCEEDED iff it holds the
    carton, otherwise the goal is rejected (nothing to detach)."""
    def __init__(self, holds):
        self.holds, self.sent = holds, []

    def send_goal_async(self, goal):
        self.sent.append(goal.attach_object)
        if goal.attach_object is False and self.holds:
            self.holds = False
            return _Fut(_Handle(True, ATTACH_ACTION_SUCCEEDED))
        return _Fut(_Handle(False, None))      # nothing to detach -> rejected (tolerated upstream)


def _reconcile_self(client, wait=None):
    return SimpleNamespace(attach=client, _wait=wait or (lambda f: f.result()))


def test_reconcile_detach_clears_a_surviving_attach():
    client = _AttachClient(holds=True)
    ok = RosPayloadSync._reconcile_detach(_reconcile_self(client), _FakeAttachObject)
    assert ok is True                          # acked -> the surviving attach was cleared
    assert client.sent == [False]              # a DETACH goal (attach_object=False), not an ADD
    assert client.holds is False


def test_reconcile_detach_tolerates_nothing_attached():
    # If object_attachment DID restart (nothing attached), the detach is rejected -> tolerated (False), and
    # the caller proceeds to the ADD regardless (which then succeeds on the clean server).
    client = _AttachClient(holds=False)
    assert RosPayloadSync._reconcile_detach(_reconcile_self(client), _FakeAttachObject) is False


def test_reconcile_detach_tolerates_a_timeout():
    def _boom(_f):
        raise TimeoutError('ack deadline')
    client = _AttachClient(holds=True)
    assert RosPayloadSync._reconcile_detach(_reconcile_self(client, wait=_boom), _FakeAttachObject) is False


# --- resync_world_after_respawn: re-publish the static scene the respawned move_group lost ----------

def test_resync_world_after_respawn_republishes_recorded_obstacles_in_order():
    calls = []
    fake = SimpleNamespace(
        _published_base_obstacle=True,
        _published_world_obstacles=[('cart_cage', [('box',)]), ('overhead', [('box',)])],
        add_base_obstacle=lambda: calls.append('base'),
        add_world_obstacle=lambda i, b: calls.append(('world', i)),
    )
    ops = RosPayloadSync.resync_world_after_respawn(fake)
    assert calls == ['base', ('world', 'cart_cage'), ('world', 'overhead')]
    assert ops == [('base_obstacle',), ('world_obstacle', 'cart_cage'), ('world_obstacle', 'overhead')]


def test_resync_world_after_respawn_noop_when_nothing_published():
    # respawn2 shape (base_collision_object False, world_obstacles []): no calls, empty plan -> byte-identical.
    calls = []
    fake = SimpleNamespace(
        _published_base_obstacle=False, _published_world_obstacles=[],
        add_base_obstacle=lambda: calls.append('base'),
        add_world_obstacle=lambda i, b: calls.append(('world', i)),
    )
    assert RosPayloadSync.resync_world_after_respawn(fake) == []
    assert calls == []


def test_resync_world_after_respawn_tolerates_a_publish_raise():
    def _boom():
        raise RuntimeError('apply_planning_scene not ready yet')
    fake = SimpleNamespace(
        _published_base_obstacle=True, _published_world_obstacles=[],
        add_base_obstacle=_boom, add_world_obstacle=lambda i, b: None,
    )
    # A raise is swallowed (logged) so the carton re-attach still proceeds; the partial plan is returned.
    assert RosPayloadSync.resync_world_after_respawn(fake) == []
