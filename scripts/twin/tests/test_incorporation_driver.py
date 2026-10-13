"""CPU tests for the sensor+handshake sequencer and its consumer float-identity.

Pure logic only: no ROS, no GPU, no mapper. These assert the ordering and clock
invariants the driver enforces locally BEFORE a request reaches the native
EvidenceProducerNode, and that the consumer-side Acquisition the driver observes is
bit-identical (in float seconds) to the one the mapper echoes back in the snapshot
ledger -- the property accept_envelope's `ack in history` check depends on. They are
NOT a substitute for the guarded native run (run_incorporation_identity_gpu.py).
"""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.restock_host.incorporation_driver import (
    IncorporationSequencer, MapIdentity, SensorBinding, HandshakeError,
    STEADY_CLOCK_DOMAIN, FRESHNESS_NS)
from twin.restock_host import evidence_bridge

S = 1_000_000_000  # one second in ns
IDENT = MapIdentity('mission_a', 'supermarket', 'front', 'amcl')


def seq(generation='mission_incorp_a', challenge='challenge-1', sensors=None):
    sensors = sensors or [SensorBinding('front', 'front_optical')]
    return IncorporationSequencer(generation=generation, challenge=challenge,
                                  identity=IDENT, sensors=sensors)


# ---- minimal fakes mirroring the generated restock_mapper_evidence layout ----
class FTime:
    __slots__ = ('sec', 'nanosec')

    def __init__(self, sec=0, nanosec=0):
        self.sec, self.nanosec = sec, nanosec


class FIdent:
    def __init__(self, d):
        self.session, self.scene = d['session'], d['scene']
        self.cameras, self.localization = d['cameras'], d['localization']


class FAcq:
    """Same field names acquisition_from_message reads."""
    def __init__(self, ad):
        self.identity = FIdent(ad['identity'])
        self.sensor = ad['sensor']
        self.optical_frame = ad['optical_frame']
        self.sequence = ad['sequence']
        self.acquisition_stamp = FTime(ad['acquisition_stamp_ns'] // S, ad['acquisition_stamp_ns'] % S)
        self.acquisition_steady_ns = ad['acquisition_steady_ns']
        self.steady_clock_domain = ad['steady_clock_domain']


def echo_through_mapper(ad):
    """Model the native fill_ledger: a.stamp = Time(sec,nanosec).nanoseconds(),
    then to_ros_time re-splits it. Returns a fresh FAcq, as the snapshot would."""
    stamp_ns = ad['acquisition_stamp_ns'] // S * S + ad['acquisition_stamp_ns'] % S
    echoed = dict(ad, acquisition_stamp_ns=stamp_ns)
    return FAcq(echoed)


class SequencerHappyPath(unittest.TestCase):
    def test_full_sequence(self):
        s = seq()
        er = s.empty_request()
        self.assertEqual(er['challenge'], 'challenge-1')
        self.assertEqual(er['expected_identity']['session'], 'mission_a')
        s.certified(producer_nonce='mission_incorp_a', certificate_token='challenge-1')
        orq = s.open_request(cutoff_stamp_ns=1 * S, cutoff_steady_ns=10 * S)
        self.assertEqual(orq['producer_nonce'], 'mission_incorp_a')
        self.assertEqual(orq['certificate_token'], 'challenge-1')
        self.assertEqual(orq['steady_clock_domain'], STEADY_CLOCK_DOMAIN)
        self.assertEqual(orq['required_sensors'], [{'sensor': 'front', 'optical_frame': 'front_optical'}])
        rr = s.register_request(sensor='front', stamp_ns=1 * S + 90_000_000,
                                steady_ns=10 * S + 90_000_000,
                                now_ns=1 * S + 100_000_000, steady_now_ns=10 * S + 100_000_000)
        self.assertEqual(rr['acquisition']['sequence'], 1)
        self.assertEqual(rr['acquisition']['sensor'], 'front')
        self.assertEqual(rr['acquisition']['steady_clock_domain'], STEADY_CLOCK_DOMAIN)

    def test_sequence_increments_per_sensor(self):
        s = seq()
        s.certified(producer_nonce='mission_incorp_a', certificate_token='challenge-1')
        s.open_request(cutoff_stamp_ns=0, cutoff_steady_ns=S)
        seqs = []
        for i in range(1, 4):
            now = i * 100_000_000
            rr = s.register_request(sensor='front', stamp_ns=now - 1_000_000,
                                    steady_ns=S + now - 1_000_000,
                                    now_ns=now, steady_now_ns=S + now)
            seqs.append(rr['acquisition']['sequence'])
        self.assertEqual(seqs, [1, 2, 3])


class SequencerRejections(unittest.TestCase):
    def test_open_before_certify(self):
        with self.assertRaises(HandshakeError):
            seq().open_request(cutoff_stamp_ns=0, cutoff_steady_ns=S)

    def test_register_before_open(self):
        s = seq()
        s.certified(producer_nonce='mission_incorp_a', certificate_token='challenge-1')
        with self.assertRaises(HandshakeError):
            s.register_request(sensor='front', stamp_ns=S, steady_ns=S,
                               now_ns=S, steady_now_ns=S)

    def test_certificate_echo_mismatch(self):
        s = seq()
        with self.assertRaises(HandshakeError):
            s.certified(producer_nonce='mission_incorp_a', certificate_token='wrong')
        s2 = seq()
        with self.assertRaises(HandshakeError):
            s2.certified(producer_nonce='other', certificate_token='challenge-1')

    def _open(self):
        s = seq()
        s.certified(producer_nonce='mission_incorp_a', certificate_token='challenge-1')
        s.open_request(cutoff_stamp_ns=1 * S, cutoff_steady_ns=10 * S)
        return s

    def test_stamp_not_after_cutoff(self):
        s = self._open()
        with self.assertRaises(HandshakeError):
            s.register_request(sensor='front', stamp_ns=1 * S, steady_ns=10 * S + 1,
                               now_ns=1 * S + 10_000_000, steady_now_ns=10 * S + 10_000_000)

    def test_steady_not_after_cutoff(self):
        s = self._open()
        with self.assertRaises(HandshakeError):
            s.register_request(sensor='front', stamp_ns=1 * S + 1, steady_ns=10 * S,
                               now_ns=1 * S + 10_000_000, steady_now_ns=10 * S + 10_000_000)

    def test_stale_beyond_freshness(self):
        s = self._open()
        stale = FRESHNESS_NS + 10_000_000
        with self.assertRaises(HandshakeError):
            s.register_request(sensor='front', stamp_ns=1 * S + 1,
                               steady_ns=10 * S + 1,
                               now_ns=1 * S + 1 + stale, steady_now_ns=10 * S + 1 + stale)

    def test_out_of_order_stamp(self):
        s = self._open()
        s.register_request(sensor='front', stamp_ns=1 * S + 90_000_000, steady_ns=10 * S + 90_000_000,
                           now_ns=1 * S + 100_000_000, steady_now_ns=10 * S + 100_000_000)
        with self.assertRaises(HandshakeError):
            s.register_request(sensor='front', stamp_ns=1 * S + 90_000_000, steady_ns=10 * S + 190_000_000,
                               now_ns=1 * S + 200_000_000, steady_now_ns=10 * S + 200_000_000)

    def test_service_clock_regression(self):
        s = self._open()
        s.register_request(sensor='front', stamp_ns=1 * S + 190_000_000, steady_ns=10 * S + 190_000_000,
                           now_ns=1 * S + 200_000_000, steady_now_ns=10 * S + 200_000_000)
        with self.assertRaises(HandshakeError):
            s.register_request(sensor='front', stamp_ns=1 * S + 290_000_000, steady_ns=10 * S + 290_000_000,
                               now_ns=1 * S + 100_000_000, steady_now_ns=10 * S + 300_000_000)

    def test_unconfigured_sensor(self):
        s = self._open()
        with self.assertRaises(HandshakeError):
            s.register_request(sensor='rear', stamp_ns=1 * S + 1, steady_ns=10 * S + 1,
                               now_ns=1 * S + 10_000_000, steady_now_ns=10 * S + 10_000_000)

    def test_bad_sensor_config(self):
        with self.assertRaises(HandshakeError):
            seq(sensors=[SensorBinding('a', 'f'), SensorBinding('b', 'f')])  # dup frame
        with self.assertRaises(HandshakeError):
            seq(sensors=[SensorBinding('a', 'f1'), SensorBinding('a', 'f2')])  # dup sensor
        with self.assertRaises(HandshakeError):
            seq(sensors=[SensorBinding(f's{i}', f'f{i}') for i in range(5)])  # >4


class ConsumerFloatIdentity(unittest.TestCase):
    """The driver observes a consumer Acquisition built from its OWN request msg;
    it must equal the one decoded from the mapper's echoed ledger entry."""

    def test_observed_matches_echoed(self):
        s = seq(sensors=[SensorBinding('front', 'front_optical')])
        s.certified(producer_nonce='mission_incorp_a', certificate_token='challenge-1')
        s.open_request(cutoff_stamp_ns=1 * S, cutoff_steady_ns=10 * S)
        rr = s.register_request(sensor='front', stamp_ns=1 * S + 123_456_789,
                                steady_ns=10 * S + 123_456_789,
                                now_ns=1 * S + 150_000_000, steady_now_ns=10 * S + 150_000_000)
        ad = rr['acquisition']
        observed = evidence_bridge.acquisition_from_message(FAcq(ad))
        echoed = evidence_bridge.acquisition_from_message(echo_through_mapper(ad))
        self.assertEqual(observed, echoed)
        self.assertEqual(observed.sensor, 'front')
        self.assertEqual(observed.optical_frame, 'front_optical')
        self.assertEqual(observed.steady_clock_domain, STEADY_CLOCK_DOMAIN)


if __name__ == '__main__':
    unittest.main()
