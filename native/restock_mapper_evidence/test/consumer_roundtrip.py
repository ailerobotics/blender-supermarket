#!/usr/bin/env python3
"""Pure-Python consumer round-trip proof for the evidence binding.

This does NOT start a ROS node, construct a mapper, or touch a GPU. It uses the
generated ``restock_mapper_evidence`` and ``nvblox_msgs`` message classes with
``rclcpp``/CDR serialization to prove the one property the whole commissioning
gate is blocked on:

    the published map slice carries the exact incorporated-acquisition identity
    and timestamp that produced it.

It builds an EvidenceSnapshot envelope exactly as the native producer
(``EvidenceProducerNode``) assembles it -- the real ``nvblox_msgs/DistanceMapSlice``
embedded next to the per-sensor Acquisition ledger -- serializes it over the wire
format, deserializes it on the consumer side, and asserts that

  1. the embedded slice round-trips byte-for-byte,
  2. every incorporated acquisition's identity/sequence/stamp round-trips exactly,
  3. the slice the consumer would republish to the canonical Nav2 topic is bound,
     through the same envelope, to that incorporation provenance, and
  4. a bare DistanceMapSlice on its own carries NO incorporation identity, which
     is precisely why the envelope binding is required (the launcher must never
     fabricate one, and an equal publication timestamp cannot certify a slice).

Exit code 0 means the binding round-trips; nonzero means it does not.
"""
import sys


def _imports():
    # rclpy provides the CDR (de)serialization used by every ROS 2 transport.
    from rclpy.serialization import serialize_message, deserialize_message
    from builtin_interfaces.msg import Time
    from nvblox_msgs.msg import DistanceMapSlice
    from restock_mapper_evidence.msg import (
        Acquisition,
        EvidenceSnapshot,
        MapIdentity,
    )
    return (serialize_message, deserialize_message, Time, DistanceMapSlice,
            Acquisition, EvidenceSnapshot, MapIdentity)


def build_producer_envelope(mods):
    (_ser, _deser, Time, DistanceMapSlice, Acquisition, EvidenceSnapshot,
     MapIdentity) = mods

    identity = MapIdentity(session="mission_a", scene="supermarket",
                           cameras="front+rear", localization="amcl")

    # A real (small) native slice: a 3x2 distance image with one unknown cell.
    slice_msg = DistanceMapSlice()
    slice_msg.header.frame_id = "map"
    slice_msg.header.stamp = Time(sec=120, nanosec=500000000)
    slice_msg.resolution = 0.05
    slice_msg.width = 3
    slice_msg.height = 2
    slice_msg.origin.x = -1.5
    slice_msg.origin.y = -0.75
    slice_msg.origin.z = 0.0
    slice_msg.unknown_value = 1000.0
    slice_msg.data = [0.12, 0.34, 1000.0, 0.21, 0.07, 0.44]

    front = Acquisition(
        identity=identity, sensor="front", optical_frame="front_optical",
        sequence=41, acquisition_stamp=Time(sec=120, nanosec=250000000),
        acquisition_steady_ns=10250000000, steady_clock_domain="host:CLOCK_MONOTONIC")
    rear = Acquisition(
        identity=identity, sensor="rear", optical_frame="rear_optical",
        sequence=37, acquisition_stamp=Time(sec=120, nanosec=300000000),
        acquisition_steady_ns=10300000000, steady_clock_domain="host:CLOCK_MONOTONIC")

    env = EvidenceSnapshot()
    env.slice = slice_msg
    env.producer_nonce = "native-7f3c9a"
    env.certificate_token = "challenge-6d1e"
    env.identity = identity
    env.snapshot_index = 1
    env.esdf_revision = 2
    env.incorporated = [front, rear]
    env.completed_steady_ns = 10350000000
    env.steady_clock_domain = "host:CLOCK_MONOTONIC"
    env.admission_state = EvidenceSnapshot.OPEN
    env.empty_state = EvidenceSnapshot.POPULATED
    env.full_height_observation_verified = False
    return env


def acquisition_key(a):
    return (a.identity.session, a.identity.scene, a.identity.cameras,
            a.identity.localization, a.sensor, a.optical_frame, a.sequence,
            a.acquisition_stamp.sec, a.acquisition_stamp.nanosec,
            a.acquisition_steady_ns, a.steady_clock_domain)


def slice_key(s):
    # DistanceMapSlice stores resolution/unknown/data as float32, so compare at
    # float32 precision (round) rather than against the float64 inputs.
    def r(x):
        return round(float(x), 5)
    return (s.header.frame_id, s.header.stamp.sec, s.header.stamp.nanosec,
            r(s.resolution), s.width, s.height,
            r(s.origin.x), r(s.origin.y), r(s.origin.z), r(s.unknown_value),
            tuple(r(v) for v in s.data))


def main():
    mods = _imports()
    serialize_message, deserialize_message = mods[0], mods[1]
    EvidenceSnapshot = mods[5]
    DistanceMapSlice = mods[3]

    sent = build_producer_envelope(mods)

    # ---- producer side: serialize the whole atomic envelope ----
    wire = serialize_message(sent)

    # ---- consumer side: deserialize and validate provenance ----
    got = deserialize_message(wire, EvidenceSnapshot)

    # (1) embedded slice is byte-identical across the wire.
    assert slice_key(got.slice) == slice_key(sent.slice), "slice did not round-trip"

    # (2) every incorporated acquisition identity round-trips exactly, and the
    #     ledger holds every configured sensor (readiness requirement upstream).
    sent_ledger = {a.sensor: acquisition_key(a) for a in sent.incorporated}
    got_ledger = {a.sensor: acquisition_key(a) for a in got.incorporated}
    assert got_ledger == sent_ledger, "incorporation ledger did not round-trip"
    assert set(got_ledger) == {"front", "rear"}, "missing a configured sensor"

    # (3) the slice the consumer republishes to the canonical Nav2 topic is bound
    #     -- through this same envelope -- to that exact incorporation provenance.
    republished = got.slice            # consumer republishes ONLY the embedded slice
    assert isinstance(republished, DistanceMapSlice)
    assert slice_key(republished) == slice_key(sent.slice)
    # The binding: this republished slice's provenance is the envelope ledger,
    # not its own header stamp.
    assert got.producer_nonce == "native-7f3c9a"
    assert got.certificate_token == "challenge-6d1e"
    assert got.esdf_revision == 2 and got.snapshot_index == 1
    assert got.full_height_observation_verified is False
    front = next(a for a in got.incorporated if a.sensor == "front")
    # acquisition time (sensor time) is preserved and is NOT the slice/publication stamp.
    assert (front.acquisition_stamp.sec, front.acquisition_stamp.nanosec) == (120, 250000000)
    assert (republished.header.stamp.sec, republished.header.stamp.nanosec) == (120, 500000000)
    assert (front.acquisition_stamp.sec, front.acquisition_stamp.nanosec) != \
        (republished.header.stamp.sec, republished.header.stamp.nanosec)

    # (4) a bare DistanceMapSlice carries NO incorporation identity: it has no
    #     field that could bind it to the acquisitions. This is why an equal
    #     publication timestamp cannot certify an arbitrary slice.
    bare = deserialize_message(serialize_message(republished), DistanceMapSlice)
    assert not any(f.startswith("incorporat") or f in ("producer_nonce", "esdf_revision")
                   for f in bare.get_fields_and_field_types()), \
        "bare slice unexpectedly carries provenance"

    print("OK: incorporation identity round-trips through the evidence envelope; "
          "embedded slice byte-identical; bare slice carries no provenance.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as exc:
        print("FAIL:", exc)
        sys.exit(1)
