"""Pure-logic sensor+handshake sequencer for the restock_mapper_evidence producer.

No ROS, no GPU, no clock of its own: the caller supplies a sim clock (ROS message
stamps, nanoseconds) and a steady CLOCK_MONOTONIC clock (nanoseconds), so this
module is deterministic under test. It produces the exact field payloads for the
three-step native handshake the ``EvidenceProducerNode`` serves

    EmptyGeneration  -> certifies the freshly launched map is empty, returns the
                        producer_nonce (== the launch generation) and binds the
                        certificate_token (== our challenge);
    OpenAdmission    -> fixes the per-sensor bindings and the acquisition cutoff
                        (acquisitions must be strictly newer than it in BOTH
                        clocks);
    RegisterAcquisition -> declares each incoming sensor frame BEFORE it is
                        integrated, keyed by (optical_frame, sim-stamp).

and it mirrors, locally and before any request leaves the host, the same ordering
and clock invariants the native pure policy enforces
(native/restock_mapper_evidence/state.hpp): identity and sensor/frame bindings are
fixed for the generation, admission opens exactly once, every acquisition is newer
than the cutoff and within the 0.3 s freshness window in both clocks, and per
sensor the sequence/stamp/steady strictly increase. A local rejection here means
the request would have been refused natively; failing before the wire keeps a
misordered driver from ever touching the certified map.

The ROS runtime (sensor_handshake.py) turns these payload dicts into generated
messages and, crucially, derives the consumer-side ``temporal.Acquisition`` from
the SAME generated message it sends, so the float-seconds the TemporalMap observes
are bit-identical to the ones the mapper will echo in the EvidenceSnapshot ledger.
"""
from dataclasses import dataclass

# Mirror of restock_mapper_evidence/state.hpp constants.
FRESHNESS_NS = 300_000_000        # 0.3 s, both clocks
EPOCH_NS = 600_000_000_000        # 600 s admission epoch

STEADY_CLOCK_DOMAIN = 'host_monotonic_shared'  # == navigation mapper_launch plan


class HandshakeError(ValueError):
    """Raised when a step would violate the native admission/clock policy."""


@dataclass(frozen=True)
class MapIdentity:
    session: str
    scene: str
    cameras: str
    localization: str

    def as_fields(self):
        return {'session': self.session, 'scene': self.scene,
                'cameras': self.cameras, 'localization': self.localization}


@dataclass(frozen=True)
class SensorBinding:
    sensor: str
    optical_frame: str


class IncorporationSequencer:
    """One generation's handshake, from empty certification to each acquisition.

    ``generation`` is the producer_nonce the mapper was launched with; it is the
    value EmptyGeneration echoes back. ``challenge`` becomes the certificate_token
    once the native side certifies the map empty.
    """

    CLOSED, CERTIFIED, OPEN = 'closed', 'certified', 'open'

    def __init__(self, *, generation, challenge, identity, sensors,
                 steady_domain=STEADY_CLOCK_DOMAIN):
        if not isinstance(generation, str) or not generation:
            raise HandshakeError('generation (producer_nonce) must be a non-empty id')
        if not isinstance(challenge, str) or not challenge:
            raise HandshakeError('challenge must be a non-empty token')
        if not isinstance(identity, MapIdentity):
            raise HandshakeError('identity must be a MapIdentity')
        bindings = {}
        frames = set()
        for binding in sensors:
            if not isinstance(binding, SensorBinding) or not binding.sensor or not binding.optical_frame:
                raise HandshakeError('each sensor needs a non-empty sensor and optical_frame')
            if binding.sensor in bindings:
                raise HandshakeError(f'duplicate sensor {binding.sensor!r}')
            if binding.optical_frame in frames:
                raise HandshakeError(f'ambiguous optical frame {binding.optical_frame!r}')
            bindings[binding.sensor] = binding.optical_frame
            frames.add(binding.optical_frame)
        if not 1 <= len(bindings) <= 4:
            raise HandshakeError('1-4 sensors required')
        if not steady_domain:
            raise HandshakeError('steady_clock_domain required')
        self.generation = generation
        self.challenge = challenge
        self.identity = identity
        self.sensors = bindings
        self.steady_domain = steady_domain
        self.phase = self.CLOSED
        self._cutoff_stamp = None
        self._cutoff_steady = None
        self._last_now = None
        self._last_steady = None
        self._seq = {s: 0 for s in bindings}
        self._last_reg = {}  # sensor -> (sequence, stamp, steady)

    # ---- step 1: certify the empty generation -------------------------------
    def empty_request(self):
        if self.phase != self.CLOSED:
            raise HandshakeError('empty generation already requested')
        return {'challenge': self.challenge,
                'expected_identity': self.identity.as_fields()}

    def certified(self, *, producer_nonce, certificate_token):
        """Record the native EmptyGeneration response, enforcing the echo."""
        if self.phase != self.CLOSED:
            raise HandshakeError('not awaiting empty-generation response')
        if producer_nonce != self.generation:
            raise HandshakeError('producer_nonce does not match launched generation')
        if certificate_token != self.challenge:
            raise HandshakeError('certificate_token must echo our challenge')
        self.phase = self.CERTIFIED

    # ---- step 2: open the admission window ----------------------------------
    def open_request(self, *, cutoff_stamp_ns, cutoff_steady_ns):
        if self.phase != self.CERTIFIED:
            raise HandshakeError('must certify an empty generation before opening admission')
        if not isinstance(cutoff_stamp_ns, int) or not isinstance(cutoff_steady_ns, int):
            raise HandshakeError('cutoff clocks must be integer nanoseconds')
        if cutoff_stamp_ns < 0 or cutoff_steady_ns <= 0:
            raise HandshakeError('invalid acquisition cutoff clocks')
        self._cutoff_stamp = cutoff_stamp_ns
        self._cutoff_steady = cutoff_steady_ns
        self._last_now = cutoff_stamp_ns
        self._last_steady = cutoff_steady_ns
        self.phase = self.OPEN
        return {
            'producer_nonce': self.generation,
            'certificate_token': self.challenge,
            'identity': self.identity.as_fields(),
            'required_sensors': [{'sensor': s, 'optical_frame': f}
                                 for s, f in sorted(self.sensors.items())],
            'acquisition_cutoff_stamp_ns': cutoff_stamp_ns,
            'acquisition_cutoff_steady_ns': cutoff_steady_ns,
            'steady_clock_domain': self.steady_domain,
        }

    # ---- step 3: register each acquisition before it is integrated ----------
    def register_request(self, *, sensor, stamp_ns, steady_ns, now_ns, steady_now_ns):
        if self.phase != self.OPEN:
            raise HandshakeError('admission is not open')
        for v in (stamp_ns, steady_ns, now_ns, steady_now_ns):
            if not isinstance(v, int):
                raise HandshakeError('clocks must be integer nanoseconds')
        if sensor not in self.sensors:
            raise HandshakeError(f'unconfigured sensor {sensor!r}')
        frame = self.sensors[sensor]
        # Monotonic service clocks (mirrors State::clocks).
        if now_ns < self._last_now or steady_now_ns < self._last_steady:
            raise HandshakeError('service clock regression')
        if now_ns - self._cutoff_stamp > EPOCH_NS or steady_now_ns - self._cutoff_steady > EPOCH_NS:
            raise HandshakeError('admission epoch expired')
        # Acquisition must sit strictly after the cutoff, no later than now, and
        # be fresh in BOTH clocks (mirrors State::register_acquisition).
        if not (self._cutoff_stamp < stamp_ns <= now_ns):
            raise HandshakeError('sim stamp must be after the cutoff and no later than now')
        if not (self._cutoff_steady < steady_ns <= steady_now_ns):
            raise HandshakeError('steady stamp must be after the cutoff and no later than now')
        if now_ns - stamp_ns > FRESHNESS_NS or steady_now_ns - steady_ns > FRESHNESS_NS:
            raise HandshakeError('acquisition is stale beyond the 0.3 s freshness window')
        last = self._last_reg.get(sensor)
        seq = self._seq[sensor] + 1
        if last is not None and (seq <= last[0] or stamp_ns <= last[1] or steady_ns <= last[2]):
            raise HandshakeError('duplicate or out-of-order acquisition for sensor')
        self._seq[sensor] = seq
        self._last_reg[sensor] = (seq, stamp_ns, steady_ns)
        self._last_now = now_ns
        self._last_steady = steady_now_ns
        return {
            'producer_nonce': self.generation,
            'certificate_token': self.challenge,
            'acquisition': {
                'identity': self.identity.as_fields(),
                'sensor': sensor,
                'optical_frame': frame,
                'sequence': seq,
                'acquisition_stamp_ns': stamp_ns,
                'acquisition_steady_ns': steady_ns,
                'steady_clock_domain': self.steady_domain,
            },
        }

    @property
    def cutoff_stamp_ns(self):
        return self._cutoff_stamp

    @property
    def cutoff_steady_ns(self):
        return self._cutoff_steady
