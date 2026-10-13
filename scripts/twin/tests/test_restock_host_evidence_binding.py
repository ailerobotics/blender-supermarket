"""Verify the native producer's EvidenceSnapshot binding against the LANDED
navigation consumer (twin.navigation.temporal.TemporalMap, origin/main #23).

No ROS node, mapper or GPU here: this exercises the documented host decode
(twin.restock_host.evidence_bridge) that turns the on-the-wire native
restock_mapper_evidence/EvidenceSnapshot into the consumer envelope, and proves
TemporalMap.accept_envelope accepts a well-formed producer output and rejects
malformed ones. The native producer source (native/restock_mapper_evidence) is
compiled/linked separately in the confined plugin container; see its report.

The message objects are faithful structural stand-ins for the generated
message's field layout (sec/nanosec stamps, *_steady_ns, nested identity/slice).
The in-container round-trip (native/.../test/consumer_roundtrip.py) proves the
real generated message serializes/deserializes with the same layout.
"""
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.navigation.temporal import (
    ADMISSION_CLOSED, ADMISSION_OPEN, EMPTY_POPULATED, EMPTY_VERIFIED,
    EvidenceEnvelope, MapIdentity, TemporalMap,
)
from twin.restock_host.evidence_bridge import envelope_from_message

DOMAIN = 'host:CLOCK_MONOTONIC'
NONCE = 'native-7f3c9a'
TOKEN = 'challenge-6d1e'


def _time(sec, nanosec=0):
    return SimpleNamespace(sec=sec, nanosec=nanosec)


def _identity():
    return SimpleNamespace(session='mission_a', scene='supermarket',
                           cameras='front+rear', localization='amcl')


def _acq(sensor, frame, sequence, sec, nanosec, steady_ns):
    return SimpleNamespace(
        identity=_identity(), sensor=sensor, optical_frame=frame, sequence=sequence,
        acquisition_stamp=_time(sec, nanosec), acquisition_steady_ns=steady_ns,
        steady_clock_domain=DOMAIN)


def _slice():
    return SimpleNamespace(
        header=SimpleNamespace(frame_id='map', stamp=_time(120, 500000000)),
        resolution=0.05, width=3, height=2,
        origin=SimpleNamespace(x=-1.5, y=-0.75, z=0.0),
        unknown_value=1000.0, data=[0.12, 0.34, 1000.0, 0.21, 0.07, 0.44])


def _message(**overrides):
    msg = SimpleNamespace(
        slice=_slice(), producer_nonce=NONCE, certificate_token=TOKEN,
        identity=_identity(), snapshot_index=1, esdf_revision=2,
        incorporated=[
            _acq('front', 'front_optical', 41, 120, 250000000, 10250000000),
            _acq('rear', 'rear_optical', 37, 120, 300000000, 10300000000),
        ],
        completed_steady_ns=10350000000, steady_clock_domain=DOMAIN,
        admission_state=ADMISSION_OPEN, empty_state=EMPTY_POPULATED,
        full_height_observation_verified=False)
    for key, value in overrides.items():
        setattr(msg, key, value)
    return msg


class EvidenceBindingTests(unittest.TestCase):
    def setUp(self):
        self.steady = [10.4]
        self.identity = MapIdentity('mission_a', 'supermarket', 'front+rear', 'amcl')
        # generation id == producer lifetime nonce from the EmptyGeneration handshake.
        self.gate = TemporalMap(
            self.identity, commissioned=100., required_sensors=['front', 'rear'],
            generation=NONCE, steady_clock=lambda: self.steady[0],
            steady_commissioned=0., steady_domain=DOMAIN)

    def _observe(self, envelope):
        # The consumer must have registered each acquisition (RegisterAcquisition
        # round-trips the same provenance) before the slice binding is accepted.
        for ack in envelope.incorporated:
            self.gate.observe(ack)

    def test_bridge_binds_slice_to_newest_and_converts_clocks(self):
        env = envelope_from_message(_message())
        # ns -> float seconds, preserving each original acquisition time.
        front = next(a for a in env.incorporated if a.sensor == 'front')
        rear = next(a for a in env.incorporated if a.sensor == 'rear')
        self.assertAlmostEqual(front.stamp, 120.25)
        self.assertAlmostEqual(rear.stamp, 120.30)
        self.assertAlmostEqual(front.steady_stamp, 10.25)
        self.assertAlmostEqual(env.completed_steady, 10.35)
        # slice bound to the NEWEST acquisition, generation == producer nonce.
        self.assertEqual(env.slice.incorporated, rear)
        self.assertEqual(env.slice.generation, NONCE)
        # sensor acquisition time is NOT the slice publication stamp.
        self.assertNotAlmostEqual(env.slice.incorporated.stamp, 120.5)

    def test_landed_consumer_accepts_wellformed_envelope(self):
        env = envelope_from_message(_message())
        self._observe(env)
        self.gate.accept_envelope(env)
        self.assertIs(self.gate.slice, env.slice)
        self.assertEqual(self.gate.producer_nonce, NONCE)
        self.assertEqual(self.gate.certificate_token, TOKEN)
        self.assertEqual(self.gate.snapshot_index, 1)
        self.assertEqual(self.gate.sensor_bindings,
                         {'front': 'front_optical', 'rear': 'rear_optical'})
        # provenance only: the full-height observed-column gate stays closed.
        self.assertFalse(self.gate.full_height_certified())

    def test_rejects_closed_admission(self):
        env = envelope_from_message(_message(admission_state=ADMISSION_CLOSED))
        self._observe(env)
        with self.assertRaises(ValueError):
            self.gate.accept_envelope(env)

    def test_rejects_non_populated_empty_state(self):
        env = envelope_from_message(_message(empty_state=EMPTY_VERIFIED))
        self._observe(env)
        with self.assertRaises(ValueError):
            self.gate.accept_envelope(env)

    def test_rejects_wrong_steady_domain(self):
        msg = _message()
        for ack in msg.incorporated:
            ack.steady_clock_domain = 'other:CLOCK_MONOTONIC'
        msg.steady_clock_domain = 'other:CLOCK_MONOTONIC'
        env = envelope_from_message(msg)
        # history was registered in the agreed domain, so observe the agreed ones.
        self._observe(envelope_from_message(_message()))
        with self.assertRaises(ValueError):
            self.gate.accept_envelope(env)

    def test_rejects_acquisition_absent_from_history(self):
        env = envelope_from_message(_message())
        # never observed -> not in registered acquisition history.
        with self.assertRaises(ValueError):
            self.gate.accept_envelope(env)

    def test_rejects_non_increasing_snapshot_index(self):
        env = envelope_from_message(_message())
        self._observe(env)
        self.gate.accept_envelope(env)
        again = envelope_from_message(_message(snapshot_index=1, esdf_revision=2))
        with self.assertRaises(ValueError):
            self.gate.accept_envelope(again)

    def test_rejects_duplicate_sensor_in_ledger(self):
        msg = _message(incorporated=[
            _acq('front', 'front_optical', 41, 120, 250000000, 10250000000),
            _acq('front', 'front_optical', 42, 120, 300000000, 10300000000),
        ])
        env = envelope_from_message(msg)
        self.gate.observe(env.incorporated[0])
        self.gate.observe(replace(env.incorporated[0], sequence=42,
                                   stamp=120.30, steady_stamp=10.30))
        with self.assertRaises(ValueError):
            self.gate.accept_envelope(env)

    def test_rejects_slice_bound_to_non_newest(self):
        env = envelope_from_message(_message())
        self._observe(env)
        older = min(env.incorporated, key=lambda a: a.stamp)
        bad = replace(env, slice=replace(env.slice, incorporated=older))
        with self.assertRaises(ValueError):
            self.gate.accept_envelope(bad)

    def test_rejects_empty_slice_frame_id(self):
        # Regression for the measured defect: the ESDF slice converter leaves the
        # slice header default, so the stock producer published an empty frame_id
        # and the landed consumer rejected the snapshot. The producer must set the
        # slice frame to the mapper global frame.
        bad = _slice()
        bad.header = SimpleNamespace(frame_id='', stamp=_time(120, 300000000))
        with self.assertRaisesRegex(ValueError, 'invalid native distance slice'):
            envelope_from_message(_message(slice=bad))

    def test_accepts_global_frame_slice_frame_id(self):
        # The fix: slice.header.frame_id == mapper global frame -> the envelope
        # decodes and the landed consumer accepts it, binding the slice frame.
        good = _slice()
        good.header = SimpleNamespace(frame_id='map', stamp=_time(120, 300000000))
        env = envelope_from_message(_message(slice=good))
        self.assertEqual(env.slice.frame, 'map')
        self._observe(env)
        self.gate.accept_envelope(env)
        self.assertIs(self.gate.slice, env.slice)


if __name__ == '__main__':
    unittest.main()
