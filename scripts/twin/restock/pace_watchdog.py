"""Pace-aware liveness predicate shared by the mission's producer watchdogs.

Under render-paced simulation the WALL clock runs far ahead of the sim/stage clock
(measured sim-to-wall ratio 0.17-0.31x in the twin-robot-in-store runs), so a watchdog
timed on WALL alone false-trips a healthy-but-slow producer, while one timed on the SIM
clock alone can mis-read a sim-advancing-but-producer-render-lagged startup as a death
(run-341/346/348/350/354/356: the station-ack stamp sat frozen for ~5 STAGE seconds at
warm start while the still-alive host had not yet received the relayed /clock).

This predicate states the audit's liveness rule once (sim-robust-audit-v1, mode
``wall-clock-watchdog-vs-render-pacing``): fire ONLY when the producer is truly dead --
its signal frozen AND, depending on the combinator, sim time advanced past a budget
and/or enough WALL time elapsed that an alive producer would certainly have recovered.
Pure arithmetic: no clock, no Isaac, no I/O, so every watchdog site is CPU-testable by
replaying a run's (sim_elapsed, wall_elapsed) signature.
"""


def is_dead(*, sim_elapsed, wall_elapsed, sim_budget, wall_backstop, mode):
    """Return True when the monitored producer should be declared dead.

    ``sim_elapsed``/``wall_elapsed`` are seconds the producer's signal has been frozen
    (no fresh stamp / no base progress) on the sim and wall clocks respectively; either
    may be ``None`` when that clock is not wired. ``sim_budget``/``wall_backstop`` are
    the bounds (``None`` disables that leg).

    mode='either' -- dead if EITHER the sim budget OR the wall backstop is exceeded. Use
      when the sim clock advancing past the budget is ITSELF proof of death (the producer
      stopped while the sim kept stepping); the wall backstop then only guards the
      frozen-sim / wholly-hung-process case the sim leg cannot see.
    mode='both' -- dead ONLY when BOTH are exceeded. Use when a sim-advancing-but-
      producer-render-lagged transient must not trip: require real wall time to pass so
      an alive-but-lagged producer has a chance to recover before we call it dead.
    """
    sim_hit = (sim_budget is not None and sim_elapsed is not None
               and sim_elapsed > sim_budget)
    wall_hit = (wall_backstop is not None and wall_elapsed is not None
                and wall_elapsed > wall_backstop)
    if mode == 'both':
        return sim_hit and wall_hit
    if mode == 'either':
        return sim_hit or wall_hit
    raise ValueError("mode must be 'either' or 'both'")
