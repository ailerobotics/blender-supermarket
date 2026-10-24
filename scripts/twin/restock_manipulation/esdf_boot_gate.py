"""Opt-in boot-time nvblox ESDF incorporation gate for the arm host (sim-robust-esdf-boot-gate-v1).

A never-acknowledged nvblox ESDF incorporation used to be discovered only ~40 s into a staged run,
as an opaque mid-stage controller invariant trip (``CONTROLLER_INVARIANT_TRIP ... collision_dt=40.0``,
with the mirrored station ``collision_time`` stuck at the ``-1`` not-ready sentinel from the first
tick -- meshuris1). The real robot would behave the same way if nvblox never incorporated at boot.

This pure state machine names that failure BEFORE the first arm motion streams, on the SIM clock.
From the first controller step of an operation it gives a bounded grace for the first VALID
incorporation acknowledgement -- the arm host sees that as ``collision_time`` leaving the ``-1``
not-ready sentinel for a real integration stamp (the parked station publishes a finite ``>=0`` stamp
only once the ESDF has incorporated fresh and strictly after the parked epoch, so "ever left ``-1``"
is a faithful "first valid incorporation ack"; freshness AFTER that is the ``collision_dt``
invariant's job). Until that ack arrives the gate HOLDs the operation (the controller streams no
motion); if the grace elapses with the stamp still stuck it returns a named fault. Once a valid ack
has been seen it disarms permanently and the controller's existing ``collision_dt`` freshness
invariant is the sole owner (its band untouched).

Disabled (``grace_s <= 0``) it always returns :data:`PROCEED` => byte-identical. Pure: no ROS /
Isaac / clock -- the caller supplies ``now`` (sim seconds) and the observed ``collision_time``.
"""
import math

# The not-ready sentinel the parked station publishes for collision_time until a real NVIDIA ESDF
# integration stamp (> parked_since) is available -- see restock.station.snapshot and
# restock_host.incorporation_ack (both emit collision_time=-1 while unready).
NOT_READY = -1.0


class _Signal:
    """Identity sentinel for the gate decision, distinguishable from a fault reason str."""
    __slots__ = ('_name',)

    def __init__(self, name):
        self._name = name

    def __repr__(self):
        return '<EsdfBootGate.%s>' % self._name


#: Disabled, or a valid ack has arrived -> the controller takes its normal path.
PROCEED = _Signal('PROCEED')
#: Enabled, no valid ack yet, still within grace -> hold the op (stream no motion).
HOLD = _Signal('HOLD')


class EsdfBootGate:
    """Bounded sim-clock boot grace for the first valid nvblox ESDF incorporation ack.

    :param grace_s: sim-seconds of boot grace; ``<= 0`` disables the gate (byte-identical).
    """
    def __init__(self, grace_s):
        if not (isinstance(grace_s, (int, float)) and math.isfinite(grace_s)):
            raise ValueError('esdf_boot_ack_grace_s must be a finite number of sim seconds')
        self.grace_s = float(grace_s)
        self.enabled = self.grace_s > 0.0
        self._op_start = None   # sim time of the first observed step this operation
        self._satisfied = False  # a valid incorporation ack has been seen (disarmed for this op)

    def reset(self):
        """Re-arm for a fresh operation (called from TransferController.start)."""
        self._op_start = None
        self._satisfied = False

    @staticmethod
    def _ack_valid(collision_time):
        # A valid incorporation ack = collision_time left the -1 not-ready sentinel for a real
        # integration stamp (>= 0). The station only publishes such a stamp once the ESDF has
        # incorporated fresh and strictly after the parked epoch, so this is enough to mean "the
        # ESDF has incorporated at least once"; per-step freshness afterwards is the collision_dt
        # invariant's job, not this gate's.
        return isinstance(collision_time, (int, float)) and math.isfinite(collision_time) \
            and collision_time >= 0.0

    def observe(self, now, collision_time):
        """Decide the gate action for this step.

        Returns :data:`PROCEED` (normal processing), :data:`HOLD` (running, no motion), or a
        non-empty fault reason ``str`` (fail fast by name).
        """
        if not self.enabled or self._satisfied:
            return PROCEED
        if not (isinstance(now, (int, float)) and math.isfinite(now)):
            # A bad/regressing clock is the controller's own invariant to reject, not this gate's.
            return PROCEED
        if self._op_start is None:
            self._op_start = now
        if self._ack_valid(collision_time):
            self._satisfied = True
            return PROCEED
        elapsed = now - self._op_start
        if elapsed > self.grace_s:
            return ('nvblox ESDF incorporation never acknowledged within %.1fs sim boot grace '
                    '(collision_time stuck at %.3f -- the not-ready sentinel; no valid incorporation '
                    'ack reached the arm host): likely nvblox not incorporating (masked-depth feed or '
                    'mapper) or a stale/uncommissioned station baseline -- failing fast before the '
                    'first arm motion instead of the mid-stage collision_dt invariant'
                    % (elapsed, collision_time if isinstance(collision_time, (int, float)) else float('nan')))
        return HOLD
