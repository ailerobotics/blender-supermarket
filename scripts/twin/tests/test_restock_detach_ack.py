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

from scripts.twin.restock_manipulation.ros_payload import (
    ATTACH_ACTION_SUCCEEDED, attach_acked, detach_scene_gate, resolve_detach_scene_tolerant)

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
