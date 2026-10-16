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
        provenance = dict(camera_id=adapter.camera_id, source=obs.source,
                          calibration_id=evidence['baseline_calibration_id'],
                          baseline_session_id=evidence['baseline_session_id'],
                          baseline_stamp=evidence['baseline_stamp'],
                          frame_id=evidence['current_frame_id'], epoch=evidence['epoch'],
                          registration_uncertainty_m=evidence['registration_uncertainty_m'],
                          qr_rgb_sha256=evidence['qr_rgb_sha256'],
                          registration_occluded=bool(evidence.get('registration_occluded')))
        return SourceTarget(sku=obs.sku, pose_station=tuple(obs.pose),
                            stamp=stamp, provenance=provenance,
                            location=source_location, station_frame=station_frame,
                            occluded=suspended)
    return source_target_provider
