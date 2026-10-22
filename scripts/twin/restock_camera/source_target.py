"""Fixed-source collect-target provider (camera lane).

The collect stage senses the pick carton with the FIXED ``source`` station camera against
its commissioned ``backstock_00`` empty-support baseline — the calibrated, pose-stable,
commissionable view verify already uses — not the arm eye-in-hand camera (which is demoted
to masked depth + fine alignment). This module wraps the existing fixed-``source``
``IsaacCartonSensors.capture(now, 'backstock_00', ...)`` and, when the handling unit is
sensed, returns its pose in the ``source`` station/world (``map``) frame with the baseline
provenance and acquisition stamp. The arm then receives a target REPROJECTED from this
result by integration (``isaac_factory``) — that reprojection and the masked-depth box are
not this module's job.

Contract: ``data/mobile-restock-v2/report.md`` section "Collect sensing contract".
No Isaac/ROS imports: the adapter is duck-typed (``capture``, ``evidence``, ``camera_id``).
"""
import math
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class SourceTarget:
    """A pick carton sensed by the fixed source camera, in the station (map) frame."""
    sku: str
    pose_station: tuple           # (x,y,z, qx,qy,qz,qw) in the source station/world (map) frame
    stamp: float                  # acquisition simulation_time of the measuring frame (or the fresh
                                  # motion-completion tick when held under render suspension)
    provenance: dict              # baseline linkage; the consumer asserts sensor provenance
    location: str = 'backstock_00'
    station_frame: str = 'map'
    occluded: bool = False        # True: held registration republished while the render is suspended
    valid_until_steady: float | None = None   # held acquisition-to-use deadline (sim clock) under
                                  # suspension; the consumer puts it on the host Sensors so the
                                  # contact/grasp suction guard (require_fresh) can fire. None when
                                  # not suspended (the rendered packet carries its own deadline).
    clean: bool = False           # fm449: True iff this is a CLEAN, committed source capture -- a fresh
                                  # synchronized rendered frame (NOT a suspended republish) whose source
                                  # support registration is NOT occluded/held, so the decoded label stands
                                  # on a freshly registered slot. The collect-source evidence gate
                                  # (build_collect_source_evidence) keys the observe->reach commit on this.


def _source_evidence(adapter, location):
    for item in adapter.evidence:
        if item['location'] == location:
            return item
    raise ValueError(f'no support evidence for source station {location!r} after capture')


def build_source_target_provider(adapter, motion, contact, *, source_location='backstock_00',
                                 station_frame='map'):
    """Return ``source_target_provider(now) -> SourceTarget | None`` over a fixed-source adapter.

    ``adapter`` is the commissioned fixed-``source`` ``IsaacCartonSensors`` (its ``source``
    context carries the ``backstock_00`` baseline); ``motion``/``contact`` are the integration
    ports, closed over so the collect stage calls the provider with ``now`` alone. The provider
    never fabricates: it returns ``None`` when no handling unit is sensed this tick, and the
    carton identity/pose come only from the fixed camera's decode + depth against the baseline.
    Freshness: ``capture`` enforces the live 0.5 s acquisition-to-use bound; the returned
    ``stamp`` lets the consumer re-validate freshness at the moment of use (base parked).

    Render-suspension (opt-in sensing gate): while integration suspends the decode render for a
    controller execution phase, ``capture`` serves the held registration whose rendered frame is
    FROZEN -- its per-observation ``simulation_time`` is the stale last-clean stamp, so the
    consumer's 0.5 s freshness check (isaac_factory) would fail every tick and starve ``/sensors``.
    When ``adapter.render_suspended`` is set, this returns the held pick with a FRESH
    motion-completion ``stamp`` (``now``) and ``occluded=True`` -- exactly mirroring the station
    held-republish -- so the collect path keeps publishing through the reach. The hold is bounded by
    the same occlusion window inside ``capture`` (it raises past it, which the consumer catches as a
    None that trips the stall), and the decode-grade stamp returns on resume. Not suspended (default/
    gate off): the measuring-frame stamp, byte-identical to before.
    """
    def source_target_provider(now):
        packet = adapter.capture(now, source_location, source_location, motion, contact)
        picks = [o for o in packet.observations if o.location == source_location]
        if not picks:
            return None                      # handling unit not sensed; absence is never a target
        if len(picks) != 1:
            raise ValueError(f'source station {source_location!r} shows more than one handling unit')
        obs = picks[0]
        if obs.source != 'sensor':
            raise ValueError('collect target must be sensor-sourced, never a fixture')
        evidence = _source_evidence(adapter, source_location)
        # Under render suspension the frozen frame's observation stamp is stale by design; the held
        # republish is fresh at the tick clock (its motion-completion fields are live), so stamp it
        # with `now` so the consumer's live-freshness gate passes while the occlusion window (inside
        # capture) bounds the hold. Byte-identical when not suspended.
        suspended = bool(getattr(adapter, 'render_suspended', False))
        stamp = float(now) if suspended else float(obs.simulation_time)
        # Under suspension carry the held packet's FRESH acquisition-to-use deadline (sim clock,
        # bounded by the occlusion window) so the host contact/grasp suction guard -- which withholds
        # the gripper until a fresh valid_until_steady is present (run-137/140) -- can fire. The
        # held republish set it Isaac-side; without threading it here the collect packet drops it and
        # suction never enables under the gate. None when not suspended (byte-identical: the rendered
        # collect packet carries no deadline, exactly as before).
        valid_until_steady = packet.valid_until_steady if suspended else None
        provenance = dict(camera_id=adapter.camera_id, source=obs.source,
                          calibration_id=evidence['baseline_calibration_id'],
                          baseline_session_id=evidence['baseline_session_id'],
                          baseline_stamp=evidence['baseline_stamp'],
                          frame_id=evidence['current_frame_id'], epoch=evidence['epoch'],
                          registration_uncertainty_m=evidence['registration_uncertainty_m'],
                          qr_rgb_sha256=evidence['qr_rgb_sha256'],
                          registration_occluded=bool(evidence.get('registration_occluded')))
        # Localisation diagnostic (opt-in, RESTOCK_SOURCE_YAW_DIAG=1; silent + byte-identical when
        # unset). The carton yaw that reaches the grasp is `yaw(pose_station)` -- and `pose_station`
        # is expressed in the adapter's LIVE `base_from_camera` frame (read off ur_base_link each
        # capture, native_cameras.CameraRig.base_from_camera), i.e. the carton yaw RELATIVE TO THE
        # ARM BASE at the capture tick. For an axis-aligned carton docked at base world-yaw b, a
        # CORRECT observation reads `-b` here (carton_world 0 - base b). Logging it next to the base
        # yaw (which integration records in nav-trace) localises a residual grasp-yaw error: if this
        # equals `-base_yaw` the camera observation is correct and the bias is downstream (grasp-pose
        # construction / execution base frame); if it stays ~0 while the base is at `b!=0`, the
        # observation is in a stale base frame (held across base motion). See run-182 diagnosis.
        if os.environ.get('RESTOCK_SOURCE_YAW_DIAG', '0') != '0':
            obs_yaw = math.degrees(2.0 * math.atan2(obs.pose[5], obs.pose[6]))
            print('SOURCE_YAW_DIAG loc=%s sku=%s obs_yaw_base_deg=%+.3f occluded=%s stamp=%.3f '
                  'frame_id=%s' % (source_location, obs.sku, obs_yaw, suspended, stamp,
                                   evidence['current_frame_id']), flush=True)
        # fm449: a CLEAN, committed source capture this tick = a fresh synchronized rendered frame (NOT a
        # suspended republish) whose source support registration is NOT occluded/held. Keyed on the SAME
        # occluded signal the camera lane already computes (`registration_occluded` = source support held
        # OR its label occluded; isaac_sensors), so a held/seeded prior never reads as clean. The collect
        # observe->reach gate (build_collect_source_evidence) consumes this.
        clean = (not suspended) and not provenance['registration_occluded']
        return SourceTarget(sku=obs.sku, pose_station=tuple(obs.pose),
                            stamp=stamp, provenance=provenance,
                            location=source_location, station_frame=station_frame,
                            occluded=suspended, valid_until_steady=valid_until_steady, clean=clean)
    return source_target_provider


def build_collect_source_evidence(source_target_provider):
    """fm449: wrap a ``source_target_provider`` into the controller's collect-source evidence gate.

    Returns ``clean(now) -> bool``: True only when the fixed source camera delivered -- this tick, at
    park -- a CLEAN synchronized frame WITH a decoded label for the calibrated source slot, i.e. a
    non-None :class:`SourceTarget` whose ``clean`` flag is set (not a suspended republish, not a
    held/occluded/seeded registration). A swallowed source-capture failure (decode / freshness /
    occlusion-window raise) is a plain False -- no clean evidence this tick -- never an exception, so the
    controller's bounded wait governs. Pass the result to
    :meth:`TransferController.set_collect_source_evidence`; the controller holds in observe until it
    returns True, then permits the reach (or fails fast on its bounded-wait timeout)."""
    def clean(now):
        try:
            target = source_target_provider(now)
        except Exception:
            return False
        return bool(target is not None and target.clean)
    return clean
