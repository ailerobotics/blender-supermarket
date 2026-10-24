"""Opt-in boot-time nvblox ESDF incorporation gate for the arm host (sim-robust-esdf-boot-gate-v1,
facts/proof corrected in -v2).

At boot the mirrored station ``collision_time`` sits at the ``-1`` not-ready sentinel until the first
nvblox ESDF integration lands, and the first controller step of an operation that reads ``-1`` trips
the opaque ``collision_dt`` invariant (``CONTROLLER_INVARIANT_TRIP ... collision_dt=40.0`` = ``now -
(-1)``) and e-stops. Two distinct cases produce that read, and the gate serves BOTH:

* a **one-tick boot race** -- the incorporation lands one controller step AFTER the op's first
  observe step (measured with the stale ``cc-bench1`` baseline: ``meshuris1`` died at the first
  observe step ``now=39`` before it could land; ``esdfgate1``, same baseline + gate on, held exactly
  ONE tick at ``now=39`` then ran to completion at ``sim 174`` with ZERO ``collision_dt`` trips --
  only possible with continuous REAL incorporation). The gate's bounded HOLD absorbs that race, so
  a run that would otherwise e-stop one step too early instead proceeds the moment the stamp lands.
* a **genuine never-incorporation** -- nvblox never integrates (no masked-depth feed, a dead mapper,
  the ``RESTOCK_ESDF_FORCE_NEVER_ACK`` proof switch). The same ``-1`` read then never clears; the
  gate names it by fault once the sim-clock grace elapses, BEFORE any arm motion streams, instead of
  the opaque mid-stage ``collision_dt`` trip. (Note: the stale baseline is NOT such a reproduction --
  it incorporates one tick late; the fail-fast path needs a feed/mapper actually disabled.)

This pure state machine: from the first controller step of an operation it gives a bounded grace for
the first VALID incorporation ack -- ``collision_time`` leaving the ``-1`` not-ready sentinel for a
real integration stamp. The arm-host chain only ever produces a finite ``>= 0`` from a genuine nvblox
integration (``incorporation_ack.acknowledge`` -> station snapshot ``ready`` -> ``StationClient`` ->
feedback mirror -> ``hold_last_good_collision`` -> ``sync_packet_clock``; every hop passes ``-1``
through untouched and the only ``>= 0`` source is a real integration stamp strictly after the parked
epoch -- v2 traced this on CPU and ruled out any default/wall/fabricated ``>= 0``). So "left ``-1``"
is a faithful "first valid incorporation ack", and ``>= 0`` is exactly the handoff point to the
``collision_dt`` freshness invariant (which owns staleness thereafter, its band untouched). Until the
ack arrives the gate HOLDs (the controller streams no motion); if the grace elapses still stuck it
returns a named fault; once a valid ack is seen it disarms permanently (a later transient ``-1`` never
re-arms it -- that is the invariant's job).

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
        # One-shot: set True on the SINGLE step that flips _satisfied False->True (the collision_time
        # sentinel -> first valid stamp transition). The controller consumes it to log that handoff
        # once per op (the proceed decision was otherwise invisible -- only HOLD/TRIP were logged).
        self._just_satisfied = False

    def reset(self):
        """Re-arm for a fresh operation (called from TransferController.start)."""
        self._op_start = None
        self._satisfied = False
        self._just_satisfied = False

    def consume_just_satisfied(self):
        """Pop the one-shot sentinel->valid transition (True at most once per op, on the step the first
        valid incorporation ack arrived). Lets the controller log that handoff exactly once, cheaply."""
        j = self._just_satisfied
        self._just_satisfied = False
        return j

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
            self._just_satisfied = True   # one-shot: the sentinel -> first valid stamp transition
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
