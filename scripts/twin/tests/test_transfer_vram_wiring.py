"""Boot-path wiring test for the transfer VRAM pre-shed + spike guard (sim-transfer-vram-preshed-v1).

The pure policy is tested in ``test_transfer_vram_preshed.py``; THIS file pins the glue so a configured
guard can never be left uninstalled (the fleet-required wiring test):

  * ``wire_transfer_vram_guards`` installs the resolved pre-shed + spike guard on the port from the HOST/
    RUNTIME keys, and builds the runtime actuators from the live sensing seams;
  * absent keys => the port attributes stay None => byte-identical;
  * a malformed block raises (fails the boot);
  * ``isaac_bundle.create`` CALLS the wiring AFTER it builds the port (the merged code never does);
  * ``isaac_port.begin_action`` fires the pre-shed on the transfer entry, and the real
    ``IsaacPort._fire_transfer_preshed`` runs the sequence once and propagates the named insufficient-free
    refusal.

No Isaac / ROS / GPU: isaac_port imports on CPU; the port method is exercised on a lightweight stand-in.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts python3 -m pytest \
        scripts/twin/tests/test_transfer_vram_wiring.py
"""
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from twin.restock import isaac_bundle
from twin.restock import isaac_port
from twin.restock import transfer_preshed as tp
from twin.restock import transfer_vram_wiring as wiring


def _port():
    return types.SimpleNamespace(masked_depth=None)


# --------------------------------------------------------------------------- byte-identical off
def test_absent_keys_leave_all_attrs_none():
    p = _port()
    armed = wiring.wire_transfer_vram_guards(p, {}, sensing_sources={}, log=lambda *a: None)
    assert armed == []
    assert p._transfer_preshed is None and p._spike_guard is None


def test_false_keys_are_inert():
    p = _port()
    armed = wiring.wire_transfer_vram_guards(
        p, {'transfer_preshed': False, 'transfer_spike_guard': False},
        sensing_sources={}, log=lambda *a: None)
    assert armed == []
    assert p._transfer_preshed is None and p._spike_guard is None


# --------------------------------------------------------------------------- malformed blocks raise
def test_malformed_preshed_block_raises():
    with pytest.raises(ValueError):
        wiring.wire_transfer_vram_guards(_port(), {'transfer_preshed': {'bad': 1}},
                                         sensing_sources={}, log=lambda *a: None)


def test_malformed_spike_guard_block_raises():
    with pytest.raises(ValueError):
        wiring.wire_transfer_vram_guards(_port(), {'transfer_spike_guard': {'floor': 1}},
                                         sensing_sources={}, log=lambda *a: None)


# --------------------------------------------------------------------------- install + actuators built from seams
def test_wiring_installs_both_guards_and_builds_actuators_from_seams():
    closed = {'mast': False, 'teardown': 0}

    class _Mast:
        masked_depth = True
        def close(self):
            closed['mast'] = True

    def _teardown():
        closed['teardown'] += 1
        return 'freed'

    mast = _Mast()
    p = _port()
    p.masked_depth = mast
    sources = {'mast_depth': mast, 'stage_scoped_teardown': _teardown}
    armed = wiring.wire_transfer_vram_guards(
        p, {'transfer_preshed': True, 'transfer_spike_guard': True},
        sensing_sources=sources, log=lambda *a: None)
    assert set(armed) == {'transfer_preshed', 'transfer_spike_guard'}
    assert p._transfer_preshed.active and p._spike_guard.active
    acts = p._transfer_preshed_actuators
    assert set(acts) == {'mast_feed', 'collect_capture', 'decode_economy'}
    # firing the mast actuator closes the bridge AND drops the port publish hook
    detail = acts['mast_feed']()
    assert closed['mast'] and p.masked_depth is None and 'mast feed dropped' in detail
    # the decode-side sheds route through the one teardown
    assert 'stage_scoped_teardown' in acts['collect_capture']()
    assert closed['teardown'] == 1


def test_wiring_skips_absent_seams():
    p = _port()
    wiring.wire_transfer_vram_guards(p, {'transfer_preshed': True}, sensing_sources={},
                                     log=lambda *a: None)
    assert p._transfer_preshed_actuators == {}            # nothing to fire -> pre-shed will skip all + gate


# --------------------------------------------------------------------------- isaac_bundle CALLS the wiring
def test_isaac_bundle_calls_the_wiring_after_building_the_port():
    src = Path(isaac_bundle.__file__).read_text()
    assert 'from . import transfer_vram_wiring' in src
    assert 'wire_transfer_vram_guards(port, config' in src
    # the wiring runs after the port exists
    assert src.index('port = IsaacPort(') < src.index('wire_transfer_vram_guards(port')


# --------------------------------------------------------------------------- isaac_port fires it on transfer entry
def test_begin_action_fires_the_preshed_on_transfer_entry():
    src = Path(isaac_port.__file__).read_text()
    assert "if action=='transfer':" in src
    assert 'self._fire_transfer_preshed()' in src


def test_port_method_runs_the_sequence_once_and_propagates_the_named_refusal():
    """The real IsaacPort._fire_transfer_preshed, bound to a stand-in: it runs the pre-shed once and the
    insufficient-free refusal propagates (fails the run by name), while an installed-but-fine run no-ops
    on the second entry."""
    fire = isaac_port.IsaacPort._fire_transfer_preshed

    # (a) insufficient free -> raises by name
    cfg = tp.from_config({'transfer_preshed': {'enabled': True, 'min_free_mib': 1536}})
    frees = iter([117.0, 300.0])
    stand = types.SimpleNamespace(
        _transfer_preshed=cfg, _transfer_preshed_done=False,
        _transfer_preshed_actuators={'mast_feed': lambda: 'dropped'},
        _transfer_preshed_sampler=lambda: (0.0, next(frees)))
    with pytest.raises(tp.TransferPreshedInsufficientFree):
        fire(stand)
    assert stand._transfer_preshed_done is True

    # (b) a fine run returns, and a second entry is a no-op (idempotent)
    calls = {'n': 0}
    frees2 = iter([117.0, 2000.0])
    stand2 = types.SimpleNamespace(
        _transfer_preshed=tp.from_config({'transfer_preshed': {'enabled': True, 'min_free_mib': 1536}}),
        _transfer_preshed_done=False,
        _transfer_preshed_actuators={'mast_feed': lambda: calls.__setitem__('n', calls['n'] + 1) or 'x'},
        _transfer_preshed_sampler=lambda: (0.0, next(frees2)))
    fire(stand2)
    assert calls['n'] == 1
    fire(stand2)                                          # already done -> no second firing
    assert calls['n'] == 1

    # (c) no pre-shed installed (default) -> no-op
    fire(types.SimpleNamespace(_transfer_preshed=None))
