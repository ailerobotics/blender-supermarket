"""fm380 stage-scoped sensing: pure teardown-id selection + isaac_port begin_action gating.

CPU-only. No Isaac: the GPU destroy (native_cameras.destroy_camera_render_product) and the actual VRAM
return are measured by the release-bench `--destroy-sensing` run, not here. These tests pin the pure
selector and the byte-identical-when-off gating of the collect->couple teardown call."""
import types

from twin.restock.isaac_port import IsaacPort, SENSING_ACTIONS
from twin.restock.sensing_sources import _stage_scoped_teardown_camera_ids


def _design(supports):
    return {'supports': {n: {'camera_id': cid} for n, cid in supports.items()}}


# ---- pure selector: which station cameras are torn down at collect end ----

def test_selector_tears_down_source_and_shelf_not_cart():
    d = _design({'source': 'source', 'cart': 'cart', 'shelf': 'shelf'})
    assert _stage_scoped_teardown_camera_ids(d) == ('shelf', 'source')


def test_selector_keeps_cart_when_source_shares_cart_camera():
    # a camera serving BOTH source and cart must NOT be torn down (cart kept for couple/nav)
    d = _design({'source': 'shared', 'cart': 'shared', 'shelf': 'shelf'})
    assert _stage_scoped_teardown_camera_ids(d) == ('shelf',)


def test_selector_empty_without_source_or_shelf():
    assert _stage_scoped_teardown_camera_ids(_design({'cart': 'cart'})) == ()
    assert _stage_scoped_teardown_camera_ids({'supports': {}}) == ()


def test_selector_source_only_when_shelf_dropped():
    # with RESTOCK_DROP_UNUSED_SENSING the shelf support may still be in the design; selector returns
    # both ids, and the teardown handle simply skips a camera absent from the live rig.
    d = _design({'source': 'source', 'cart': 'cart'})
    assert _stage_scoped_teardown_camera_ids(d) == ('source',)


# ---- begin_action gating (byte-identical when the handle is None) ----

def _port_stub(teardown):
    s = types.SimpleNamespace(_stage_scoped_teardown=teardown, _stage_scoped_done=False,
                              current_action=None, on_action=None)
    # begin_action (and the grasp path) delegate to the real IsaacPort._fire_stage_scoped_teardown;
    # bind it to the stub so the shared guarded/fail-safe fire logic is exercised exactly as in the port.
    s._fire_stage_scoped_teardown = types.MethodType(IsaacPort._fire_stage_scoped_teardown, s)
    return s


def test_fire_helper_once_byte_identical_and_idempotent():
    # byte-identical when off (no handle): fires nothing, never flags done
    s = _port_stub(None)
    s._fire_stage_scoped_teardown()
    assert s._stage_scoped_done is False
    # with a handle: fires exactly once, then idempotent
    calls = []
    s = _port_stub(lambda: calls.append('torn'))
    s._fire_stage_scoped_teardown()
    s._fire_stage_scoped_teardown()
    assert calls == ['torn'] and s._stage_scoped_done is True


def test_grasp_confirmed_triggers_teardown_before_lift():
    # fm381: the EARLIEST trigger -- the moment _carton_welded is True (grasp confirmed, before lift),
    # the step path calls _fire_stage_scoped_teardown(); it must free the source cam ONCE, right then,
    # BEFORE the collect->couple boundary. Simulate the step-path guard directly.
    calls = []
    s = _port_stub(lambda: calls.append('torn'))
    s._carton_welded = False
    # pre-grasp ticks: not welded -> no teardown
    for _ in range(3):
        if s._carton_welded:
            s._fire_stage_scoped_teardown()
    assert calls == [] and s._stage_scoped_done is False
    # grasp confirmed -> weld latches -> teardown fires ONCE on the first welded tick
    s._carton_welded = True
    for _ in range(5):           # stays welded through lift/carry; must not re-fire
        if s._carton_welded:
            s._fire_stage_scoped_teardown()
    assert calls == ['torn'] and s._stage_scoped_done is True
    # a later begin_action('couple') fallback must NOT fire again (already done)
    IsaacPort.begin_action(s, 'couple', {})
    assert calls == ['torn']


def test_begin_action_byte_identical_when_handle_none():
    calls = []
    s = _port_stub(None)
    # transition straight into couple (a non-sensing action) -- with no handle, nothing is torn down
    IsaacPort.begin_action(s, 'couple', {})
    assert s.current_action == 'couple'
    assert s._stage_scoped_done is False
    assert calls == []


def test_begin_action_tears_down_once_on_first_non_sensing_action():
    calls = []
    s = _port_stub(lambda: calls.append('torn'))
    IsaacPort.begin_action(s, 'collect', {})      # sensing action -> no teardown
    assert calls == [] and s._stage_scoped_done is False
    IsaacPort.begin_action(s, 'couple', {})        # collect->couple boundary -> teardown ONCE
    assert calls == ['torn'] and s._stage_scoped_done is True
    IsaacPort.begin_action(s, 'outbound', {})      # already done -> not again
    IsaacPort.begin_action(s, 'park', {})
    assert calls == ['torn']


def test_begin_action_never_tears_down_on_sensing_actions():
    calls = []
    s = _port_stub(lambda: calls.append('torn'))
    for a in sorted(SENSING_ACTIONS):
        IsaacPort.begin_action(s, a, {})
    assert calls == [] and s._stage_scoped_done is False


def test_begin_action_teardown_failure_is_fail_safe():
    def boom():
        raise RuntimeError('destroy failed')
    s = _port_stub(boom)
    # a teardown that raises must NOT propagate out of begin_action (mission continues)
    IsaacPort.begin_action(s, 'couple', {})
    assert s.current_action == 'couple' and s._stage_scoped_done is True
