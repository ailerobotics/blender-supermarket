"""Reference binding: restock_mapper_evidence/EvidenceSnapshot -> consumer envelope.

The native ``EvidenceProducerNode`` (native/restock_mapper_evidence) publishes a
typed ``restock_mapper_evidence/EvidenceSnapshot`` that atomically binds the
copied ``nvblox_msgs/DistanceMapSlice`` to the full per-sensor incorporation
ledger. Navigation's landed consumer (``twin.navigation.temporal.TemporalMap``,
origin/main PR #23) validates a pure-CPU ``EvidenceEnvelope`` and does NOT import
the generated message on its lane. This module is the small, explicit decode the
integration worker runs at launch to turn the on-the-wire native message into
that consumer envelope, then feed ``TemporalMap.accept_envelope``.

It is intentionally duck-typed on the message: it accepts either the generated
``restock_mapper_evidence.msg.EvidenceSnapshot`` (at runtime) or any object with
the same field layout (for CPU contract tests), so no generated-message import is
forced onto the navigation lane. It starts nothing and owns no ROS state.

Key invariants reproduced here (see temporal.accept_envelope):
  * generation id == producer lifetime nonce from the EmptyGeneration handshake;
  * the embedded slice is bound to the NEWEST acquisition in the ledger;
  * builtin_interfaces/Time and *_steady_ns are converted to float seconds
    explicitly (the consumer's clocks are float seconds);
  * every ledger acquisition keeps its own original acquisition time.
"""
from twin.navigation.temporal import (
    Acquisition,
    DistanceSlice,
    EvidenceEnvelope,
    MapIdentity,
)


def _time_to_seconds(stamp):
    """builtin_interfaces/Time -> float seconds."""
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _identity(ros_identity):
    return MapIdentity(
        ros_identity.session,
        ros_identity.scene,
        ros_identity.cameras,
        ros_identity.localization,
    )


def acquisition_from_message(ros_acq):
    """restock_mapper_evidence/Acquisition -> consumer Acquisition."""
    return Acquisition(
        sensor=ros_acq.sensor,
        identity=_identity(ros_acq.identity),
        sequence=int(ros_acq.sequence),
        stamp=_time_to_seconds(ros_acq.acquisition_stamp),
        steady_stamp=float(ros_acq.acquisition_steady_ns) * 1e-9,
        optical_frame=ros_acq.optical_frame,
        steady_clock_domain=ros_acq.steady_clock_domain,
    )


def envelope_from_message(msg):
    """restock_mapper_evidence/EvidenceSnapshot -> consumer EvidenceEnvelope.

    Raises ValueError (via the consumer dataclasses) on a malformed native
    message, so a garbled producer output fails closed rather than certifying.
    """
    identity = _identity(msg.identity)
    ledger = [acquisition_from_message(a) for a in msg.incorporated]
    if not ledger:
        raise ValueError('native evidence snapshot carries no incorporation ledger')
    # The producer froze the slice with the whole ledger; the slice is bound to
    # the newest incorporated acquisition (consumer rejects any older binding).
    newest = max(ledger, key=lambda a: a.stamp)
    decoded_slice = DistanceSlice.from_message(
        msg.slice, generation=msg.producer_nonce, incorporated=newest)
    return EvidenceEnvelope(
        slice=decoded_slice,
        producer_nonce=msg.producer_nonce,
        certificate_token=msg.certificate_token,
        identity=identity,
        snapshot_index=int(msg.snapshot_index),
        esdf_revision=int(msg.esdf_revision),
        incorporated=tuple(ledger),
        completed_steady=float(msg.completed_steady_ns) * 1e-9,
        steady_clock_domain=msg.steady_clock_domain,
        admission_state=int(msg.admission_state),
        empty_state=int(msg.empty_state),
        full_height_observation_verified=bool(msg.full_height_observation_verified),
    )
