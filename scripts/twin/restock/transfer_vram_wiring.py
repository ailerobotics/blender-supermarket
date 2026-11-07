"""Boot-layer wiring for the transfer-entry VRAM pre-shed + spike guard (sim-transfer-vram-preshed-v1).

The pure policy lives in :mod:`transfer_preshed` and :mod:`spike_guard`; this module is the GLUE that
the mission boot (:mod:`isaac_bundle`) calls to build the runtime actuators from the live port/sensing
seams and install them on the :class:`isaac_port.IsaacPort`, so the port can fire the pre-shed on transfer
entry (``begin_action('transfer')``) and gate a new render through the spike guard. Absent config keys =>
nothing is installed => the port attributes stay None => byte-identical (asserted by the wiring test).

Runtime actuator seams available in-port today:
  * ``mast_feed``       -> close the opt-in mast masked-depth bridge (camera_1) and drop the port hook.
  * ``collect_capture`` / ``decode_economy`` -> the fm380 stage-scoped teardown, which DESTROYS the
    station decode/source/cart render products this transfer no longer reads (~0.95 GB, fm380-measured).
A shed with NO runtime seam this run (``collect_esdf_bounds``; or ``decode_economy`` when no teardown
handle exists) is SKIPPED by :func:`transfer_preshed.run_preshed` and logged -- those are boot-time levers
(set ``masked_depth.economy`` / the ESDF bounds at authoring). The >= 1.5 GB post-shed gate then refuses
the transfer render BY NAME if the available seams did not free enough, instead of crashing the allocator.
"""
from __future__ import annotations

from . import spike_guard as _spike_guard
from . import transfer_preshed as _transfer_preshed


def build_preshed_actuators(port, sensing_sources):
    """Map the resolved pre-shed shed names to zero-arg actuator callables from the live seams.

    Only the names with a real runtime seam are included; the rest are left out so the pre-shed skips
    (and logs) them. ``sensing_sources`` is the bundle's ``assembly.sensing_sources`` dict (or None)."""
    sources = sensing_sources if isinstance(sensing_sources, dict) else {}
    actuators = {}

    mast = sources.get('mast_depth')
    if mast is not None:
        def _drop_mast(_port=port, _mast=mast):
            closed = False
            if hasattr(_mast, 'close'):
                try:
                    _mast.close()
                    closed = True
                except Exception as exc:  # noqa: BLE001
                    return 'mast close FAILED %r' % (exc,)
            if getattr(_port, 'masked_depth', None) is _mast:
                _port.masked_depth = None    # stop publishing the mast feed on the sensing cadence
            return 'mast feed dropped (closed=%s)' % (closed,)
        actuators['mast_feed'] = _drop_mast

    teardown = sources.get('stage_scoped_teardown')
    if teardown is not None:
        # The same teardown frees the collect-stage decode render products; wire it to whichever of the
        # two decode-side sheds the config orders (fire-once is idempotent inside the port).
        def _teardown_decode(_t=teardown):
            return 'stage_scoped_teardown applied=%r' % (_t(),)
        actuators['collect_capture'] = _teardown_decode
        actuators['decode_economy'] = _teardown_decode

    return actuators


def wire_transfer_vram_guards(port, config, *, sensing_sources=None, sampler=None, log=print):
    """Install the transfer pre-shed + spike guard on ``port`` from ``config`` (DEFAULT OFF).

    Sets ``port._transfer_preshed`` (resolved + enabled, else None), ``port._transfer_preshed_actuators``,
    ``port._transfer_preshed_sampler``, ``port._transfer_preshed_done`` and ``port._spike_guard``. Returns
    the list of armed guard names (``[]`` when both keys are absent => byte-identical). Raises (fails the
    boot) on a malformed block, via the modules' fail-closed config resolution."""
    armed = []

    preshed_cfg = _transfer_preshed.from_config(config)
    if preshed_cfg.active:
        actuators = build_preshed_actuators(port, sensing_sources)
        port._transfer_preshed = preshed_cfg
        port._transfer_preshed_actuators = actuators
        port._transfer_preshed_sampler = sampler
        port._transfer_preshed_done = False
        log(preshed_cfg.describe())
        armed.append('transfer_preshed')
    else:
        port._transfer_preshed = None

    guard_cfg = _spike_guard.from_config(config)
    if guard_cfg.active:
        port._spike_guard = guard_cfg
        log(guard_cfg.describe())
        armed.append('transfer_spike_guard')
    else:
        port._spike_guard = None

    return armed
