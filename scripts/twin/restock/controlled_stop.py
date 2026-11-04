"""sim-host-fault-controlled-stop-v1 -- a controlled stop on a host fault while a carton is welded.

Pure decision logic, no Isaac / ROS / USD imports, so it is CPU-testable with a fake host and is
shared verbatim by:

  * the mission runtime (``restock.runtime.RuntimeAdapter``) -- which, on a manipulation failure while
    the gripper still holds a WELDED carton, enters the controlled stop and records the FIRST trip;
  * the base + weld owners (``mobile_runtime`` / ``restock.isaac_port``) -- which keep the base
    parking brake engaged and hold the arm at its joint targets (drive still energised), never going
    slack while a welded body interpenetrates a surface;
  * the behaviour-tree retry decorator (``restock.bt.leaves.StageRetry``) -- which refuses a collect
    re-begin when this stop is active or the chassis attitude is outside a sane band, and names the
    first trip (``CARTON_TILT 2.06``) as the terminal mission reason instead of the downstream
    ``arm command heartbeat lost``.

Why this module exists (coldwin33): a carried-carton CARTON_TILT trip of 2.06 deg (0.06 over the
captain-fixed 2.00 deg bound) fired the host emergency stop at the place set-down. The arm was welded
to the carton and fully extended over the cart deck with the carton interpenetrating the deck; the
subsequent collect RE-BEGIN drove the base, whose ``command()`` released every brake, and PhysX
resolved the penetration by flinging the base and the cart (roll 138 deg, pitch 47 deg, z 2.11 m).
Collect then re-began twice on the flipped robot and the run ended with the generic
``arm command heartbeat lost`` -- a marginal, recoverable trip turned into a wrecked scene and a
misleading terminal reason. The rule this module encodes: NEVER cut the drive / release the base
brake while a welded body interpenetrates; release the weld only once the carton is supported or
clear; and preserve the first trip as the terminal reason.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


# Downstream symptoms a later (re-begin) failure reports that MUST NOT overwrite the first real trip.
# Mirrors ``native_transport._DOWNSTREAM_ARM_SYMPTOMS`` (duplicated to keep this module import-light,
# with no Isaac/ROS dependency); a drift between the two is harmless (this only picks which reason to
# surface) and is guarded by ``test_controlled_stop``.
DOWNSTREAM_SYMPTOMS = ('arm command heartbeat lost', 'StationAckStreamStalled',
                       'arm command feedback', 'writer_eof', 'MASKED_DEPTH_SERVE_EXIT',
                       'controller_not_ready')

# Substrings that mark a reason as a REAL first trip worth preserving over a later downstream symptom.
TRIP_MARKERS = ('carton_tilt', 'carton off level', 'host emergency stop', 'host fault')

# Sane base-attitude band for a collect re-begin, as absolute tolerances around the robot's NOMINAL
# (level-chassis) pose. The nominal pitch is NOT zero on this rig: the caster-lift static tilt reads
# ~-1.2 deg pitch at z ~= 0.24 m (BASE_ATT, fm437), so the band is anchored on the measured nominal
# pose (``AttitudeBand.from_nominal``), not on 0. It must comfortably pass that level pose yet reject
# the coldwin33 fling (roll 138 deg, pitch 47 deg, z +1.87 m above nominal). Generous, because the
# only thing on the wrong side of it is a wrecked chassis.
DEFAULT_ROLL_TOL_DEG = 20.0
DEFAULT_PITCH_TOL_DEG = 20.0
DEFAULT_Z_TOL_M = 0.15


@dataclass(frozen=True)
class AttitudeBand:
    """A sane chassis-attitude band = the nominal (level) pose +- per-axis tolerances."""
    nominal_roll_deg: float
    nominal_pitch_deg: float
    nominal_z_m: float
    roll_tol_deg: float = DEFAULT_ROLL_TOL_DEG
    pitch_tol_deg: float = DEFAULT_PITCH_TOL_DEG
    z_tol_m: float = DEFAULT_Z_TOL_M

    @classmethod
    def from_nominal(cls, roll_deg, pitch_deg, z_m, *, roll_tol_deg=DEFAULT_ROLL_TOL_DEG,
                     pitch_tol_deg=DEFAULT_PITCH_TOL_DEG, z_tol_m=DEFAULT_Z_TOL_M):
        return cls(float(roll_deg), float(pitch_deg), float(z_m),
                   float(roll_tol_deg), float(pitch_tol_deg), float(z_tol_m))

    def describe(self):
        return (f'roll {self.nominal_roll_deg:+.2f}+-{self.roll_tol_deg:.1f} '
                f'pitch {self.nominal_pitch_deg:+.2f}+-{self.pitch_tol_deg:.1f} deg, '
                f'z {self.nominal_z_m:.3f}+-{self.z_tol_m:.3f} m')


def attitude_ok(roll_deg, pitch_deg, z_m, band):
    """True iff the measured chassis attitude is within ``band``. A non-finite reading is NOT ok
    (fail safe: a bad measurement must refuse a re-begin, never silently pass)."""
    vals = (roll_deg, pitch_deg, z_m)
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
               for v in vals):
        return False
    return (abs(roll_deg - band.nominal_roll_deg) <= band.roll_tol_deg
            and abs(pitch_deg - band.nominal_pitch_deg) <= band.pitch_tol_deg
            and abs(z_m - band.nominal_z_m) <= band.z_tol_m)


def attitude_detail(roll_deg, pitch_deg, z_m, band):
    return f'roll={roll_deg:+.2f} pitch={pitch_deg:+.2f} deg z={z_m:.3f} m vs band {band.describe()}'


def is_trip_reason(reason):
    """A reason that names a REAL first trip (a carried-carton / host fault), not a downstream symptom."""
    text = (reason or '').lower()
    return any(marker in text for marker in TRIP_MARKERS)


def is_downstream_symptom(reason):
    text = reason or ''
    return any(symptom in text for symptom in DOWNSTREAM_SYMPTOMS)


def prefer_first_trip(first, latest):
    """Return the terminal reason that names the FIRST trip over a later downstream symptom.

    Keep ``first`` only when it names a real trip AND ``latest`` is empty / a downstream symptom / not
    itself a trip; otherwise keep ``latest`` (the normal last-reason behaviour, so every unrelated
    failure chain is unchanged)."""
    if not first:
        return latest
    if first == latest:
        return first
    if is_trip_reason(first) and (not latest or is_downstream_symptom(latest)
                                  or not is_trip_reason(latest)):
        return first
    return latest


@dataclass(frozen=True)
class WeldedStopDecision:
    """The controlled-stop action set on a host fault while a carton is welded."""
    hold_arm: bool            # hold the current joint targets with the drive still energised
    keep_base_brake: bool     # keep the base parking brake engaged (never release while welded)
    allow_base_drive: bool    # may the base drive? NEVER during a welded controlled stop
    release_weld: bool        # release the weld ONLY once the carton is supported or clear of the deck
    retreat: bool             # lift / back off along the approach, only if a clear retreat exists
    reason: str


def decide_welded_stop(*, welded, interpenetrating, supported, retreat_clear, trip_reason=''):
    """Decide the controlled stop. While a carton is WELDED:

      * always hold the arm energised and keep the base brake engaged;
      * NEVER allow a base drive (this is a STOP) -- and the base brake stays no matter what while a
        welded body interpenetrates (that release is exactly the coldwin33 fling);
      * release the weld ONLY once the carton is supported on a surface or clear of the deck;
      * lift / back off along the approach only if the carton still interpenetrates AND a clear
        retreat is known (otherwise hold in place -- the safe default).

    Not welded => no controlled stop (the normal failure path, every field inert)."""
    if not welded:
        return WeldedStopDecision(hold_arm=False, keep_base_brake=False, allow_base_drive=True,
                                  release_weld=False, retreat=False, reason=trip_reason or '')
    supported_or_clear = bool(supported) or not bool(interpenetrating)
    return WeldedStopDecision(
        hold_arm=True,
        keep_base_brake=True,
        allow_base_drive=False,
        release_weld=supported_or_clear,
        retreat=bool(retreat_clear) and bool(interpenetrating) and not bool(supported),
        reason=trip_reason or '')
