"""CPU pacing harness for the manipulation-host sim-clock timers (sim-robust-host-timers-audit-fix-v1).

The honest live_pose cold cycle enables a second full-resolution render (the masked_depth arm feed) that
slows the simulator's pacing to a measured 16-30x wall-to-sim during manipulation. Every host timer that
this lane converted -- the total-move PLANNING budget (RosMotionPort.planning_wall_exceeded), the
first-frame STARTUP grace, and the payload/attach ACK wait -- reduces to the SAME shape: a SIM-primary
budget with a derived-generous frozen-sim WALL backstop, evaluated by the shared pace_watchdog predicate
(mode='either'). A pure-wall timer at the historical bound false-trips a healthy-but-render-slow producer
at 30x (coldlive sample-4); a sim-primary one tolerates it while still catching a genuinely hung producer
(sim advancing, producer never finishing) and a frozen sim (the WALL backstop).

This module drives those SHIPPED predicates (``frozen_sim_wall_backstop`` from the host adapter and
``pace_watchdog.is_dead``) against a fake wall/sim clock pair at configurable pacing, so the whole family
is provable on CPU in milliseconds -- no ROS, no Isaac, no GPU. It is pure: no real clock, no I/O.

sim-robust-station-heartbeat-simclock-v1 extends it to the PARKED-STATION heartbeat guard: the same
SIM-primary + frozen-sim-wall-backstop shape lives in ``restock_host.incorporation_ack.IncorporationAckState.
snapshot`` (the host-stack parked-heartbeat liveness that e-stopped coldliverecovery at ``wall gap 30.75 s``
while the sim gap was healthy). ``simulate_station_heartbeat`` drives that SHIPPED state object against the
same fake clock pair, so the station timer is proven on CPU alongside the manipulation-host timers.

sim-robust-station-pause-on-transfer-v1 adds the FAR-STATION scenario (``simulate_far_station_paused``): the
parked-heartbeat guard's SIM-primary budget alone STILL trips a station left active through a long far op
(coldwin2: the collect station un-heartbeat-ed through the whole couple->outbound->park->transfer span hit a
30.05 s SIM gap, 0.05 s over the 30 s budget). The fix PAUSES that station before towing away; a paused
station must accrue NO gap and publish NO stop for ANY op length at ANY pacing. This harness drives the same
SHIPPED ``IncorporationAckState`` with an explicit ``pause`` command, proving it on CPU.
"""

from twin.restock_manipulation.ros_adapter import frozen_sim_wall_backstop
from twin.restock import pace_watchdog


class FakeClockPair:
    """A wall clock and a render-paced sim clock. ``pacing`` is wall-seconds per sim-second (1.0 == real
    rig; 16-30 == the live_pose arm-feed economy). ``frozen`` holds the sim clock still (a dead/stalled
    simulator) while wall keeps advancing. Pure arithmetic."""

    def __init__(self, pacing=1.0, frozen=False):
        if pacing < 1.0:
            raise ValueError('pacing is wall-per-sim seconds, >= 1.0')
        self.pacing = float(pacing)
        self.frozen = bool(frozen)
        self.wall = 0.0
        self.sim = 0.0

    def advance(self, dwall):
        """Advance the wall clock by ``dwall`` seconds; the sim clock advances dwall/pacing unless frozen."""
        if dwall < 0:
            raise ValueError('time only moves forward')
        self.wall += dwall
        if not self.frozen:
            self.sim += dwall / self.pacing


def simulate_sim_primary_budget(*, pacing, sim_budget, wall_floor, producer_sim_duration,
                                frozen=False, dwall=0.1, max_wall=100000.0, pacing_factor=50.0):
    """Drive ONE sim-primary budget (planning / startup / ack) to its outcome under ``pacing``.

    The monitored producer (a plan, the first frame, an attach ack) completes after
    ``producer_sim_duration`` SIM seconds; ``None`` means it never completes (a genuine hang/death). The
    budget is the one the host ships: a SIM leg at ``sim_budget`` and a frozen-sim WALL backstop derived by
    ``frozen_sim_wall_backstop(sim_budget, wall_floor, pacing_factor)``. Steps the fake clock by ``dwall``
    wall-seconds and evaluates the shipped ``pace_watchdog.is_dead(mode='either')`` each tick.

    Returns a dict: outcome in {'completed','tripped_sim','tripped_wall'}, and the wall/sim readings at the
    event. 'completed' means the producer finished before any leg fired (no false trip). 'tripped_sim' /
    'tripped_wall' name which leg caught a hang. Raises if neither completes nor trips within ``max_wall``
    (that would itself be a bug -- an unbounded wait)."""
    clock = FakeClockPair(pacing=pacing, frozen=frozen)
    wall_backstop = frozen_sim_wall_backstop(sim_budget, wall_floor, pacing_factor)
    while clock.wall <= max_wall:
        # Producer completion is checked BEFORE the watchdog, exactly as the host consumes a delivered
        # result before the bridge-watchdog cap (a delivered success never counts as exhausted).
        if producer_sim_duration is not None and clock.sim >= producer_sim_duration:
            return {'outcome': 'completed', 'wall': clock.wall, 'sim': clock.sim,
                    'wall_backstop': wall_backstop}
        sim_elapsed = None if frozen and clock.sim == 0.0 else clock.sim
        if pace_watchdog.is_dead(sim_elapsed=sim_elapsed, wall_elapsed=clock.wall,
                                 sim_budget=sim_budget, wall_backstop=wall_backstop, mode='either'):
            leg = 'tripped_sim' if (sim_elapsed is not None and sim_elapsed > sim_budget) else 'tripped_wall'
            return {'outcome': leg, 'wall': clock.wall, 'sim': clock.sim, 'wall_backstop': wall_backstop}
        clock.advance(dwall)
    raise AssertionError('neither completed nor tripped within max_wall -- unbounded wait is a bug')


def simulate_station_heartbeat(*, pacing, sim_gap_budget, wall_floor, no_heartbeat_sim,
                               frozen=False, dwall=0.05, max_wall=200000.0):
    """Drive the SHIPPED parked-station heartbeat guard to its outcome under ``pacing``.

    Exercises the REAL ``restock_host.incorporation_ack.IncorporationAckState.snapshot`` (SIM-primary gap +
    derived frozen-sim WALL backstop, via ``pace_watchdog.is_dead(mode='either')``) against the fake clock
    pair. A parked station is prepared + stamped once, then the mission runs ONE manipulation op lasting
    ``no_heartbeat_sim`` SIM seconds during which it sends NO station heartbeat (the live_pose transfer's
    multi-capture gap). Under ``pacing`` wall-per-sim that gap stretches to ``no_heartbeat_sim x pacing``
    WALL seconds -- the stretch that false-tripped the OLD flat wall guard. ``frozen`` holds the sim clock
    still for the whole op (a dead simulator) so only the WALL backstop can catch it.

    The guard's budgets are set to (``sim_gap_budget``, ``wall_floor``) directly on the shipped state object
    and its ``wall_backstop_s`` is re-derived with the shipped ``frozen_sim_wall_backstop`` -- the same
    derivation ``__init__`` runs, driven here without env mutation so the harness stays pure.

    Returns a dict: outcome in {'survived','tripped_sim','tripped_wall'} with the wall/sim readings and the
    guard's reason. 'survived' == the guard did NOT fault across the whole healthy op (no false stop).
    Raises if nothing happens within ``max_wall`` (an unbounded wait would itself be a bug)."""
    from twin.restock_host.incorporation_ack import IncorporationAckState
    clock = FakeClockPair(pacing=pacing, frozen=False)
    st = IncorporationAckState(wall_clock=lambda: clock.wall)
    st.sim_gap_s = float(sim_gap_budget)
    st.wall_gap_s = float(wall_floor)
    st.wall_backstop_s = frozen_sim_wall_backstop(sim_gap_budget, wall_floor)
    # Prepare the parked epoch at sim 0 (sets last_heartbeat = 0 sim, wall_heartbeat = 0 wall). The gap
    # guard runs on an active epoch whether or not an ESDF ack has landed, so no acknowledge() is needed.
    st.accept(dict(action='prepare', epoch=1, sequence=0, stamp=0.0, parked=True,
                   camera_frame='restock_camera_transfer'), 0.0)
    clock.frozen = bool(frozen)                       # the op begins; a frozen sim stalls /clock here
    op_start_sim = clock.sim
    while clock.wall <= max_wall:
        snap = st.snapshot(clock.sim, endpoints_ready=True, process_alive=True)
        if snap['state'] == 'failed':
            leg = 'tripped_sim' if 'sim gap' in snap['reason'] else 'tripped_wall'
            return {'outcome': leg, 'wall': clock.wall, 'sim': clock.sim, 'reason': snap['reason'],
                    'wall_backstop': st.wall_backstop_s}
        # A healthy op completes after no_heartbeat_sim SIM seconds (a frozen sim never gets here and must
        # be caught by the WALL backstop instead).
        if not frozen and (clock.sim - op_start_sim) >= no_heartbeat_sim:
            return {'outcome': 'survived', 'wall': clock.wall, 'sim': clock.sim, 'reason': '',
                    'wall_backstop': st.wall_backstop_s}
        clock.advance(dwall)
    raise AssertionError('station guard neither survived nor tripped within max_wall -- unbounded wait')


def simulate_far_station_paused(*, pacing, sim_gap_budget, wall_floor, far_op_sim, pause=True,
                                dwall=0.5, max_wall=2000000.0):
    """Drive the SHIPPED parked-station heartbeat guard with the station PAUSED for a long FAR op.

    Models sim-robust-station-pause-on-transfer-v1: the mission completes the collect op, PAUSES the parked
    collect/backstock station (one ``pause`` command, accepted while still parked), then tows AWAY and runs a
    long FAR op (couple -> outbound -> park -> transfer) lasting ``far_op_sim`` SIM seconds during which the
    station gets NO heartbeat. A PAUSED station must accrue no gap and publish no stop no matter how long the
    far op runs -- even stretched to ``far_op_sim x pacing`` WALL seconds. Set ``pause=False`` to model the
    pre-fix behaviour (station left ACTIVE through the far op), which must still trip on the SIM leg (the
    coldwin2 30.05 s gap) -- the contrast that proves the pause is what prevents the false stop.

    Drives the REAL ``restock_host.incorporation_ack.IncorporationAckState`` (budgets set directly on the
    shipped object; ``wall_backstop_s`` re-derived with the shipped ``frozen_sim_wall_backstop``) against the
    fake clock pair. Returns a dict: outcome in {'survived','tripped_sim','tripped_wall'} with the final
    wall/sim readings, the guard's reason and its published state. Raises on an unbounded wait."""
    from twin.restock_host.incorporation_ack import IncorporationAckState
    clock = FakeClockPair(pacing=pacing, frozen=False)
    st = IncorporationAckState(wall_clock=lambda: clock.wall)
    st.sim_gap_s = float(sim_gap_budget)
    st.wall_gap_s = float(wall_floor)
    st.wall_backstop_s = frozen_sim_wall_backstop(sim_gap_budget, wall_floor)
    st.accept(dict(action='prepare', epoch=1, sequence=0, stamp=0.0, parked=True,
                   camera_frame='restock_camera_collect'), 0.0)
    if pause:
        # The collect op completes; the mission SUSPENDS the station before towing away (stamp fresh off the
        # stage clock, exactly as StationClient.pause does). The host state becomes 'paused'.
        st.accept(dict(action='pause', epoch=1, sequence=1, stamp=clock.sim, parked=True), clock.sim)
    op_start_sim = clock.sim
    while clock.wall <= max_wall:
        snap = st.snapshot(clock.sim, endpoints_ready=True, process_alive=True)
        if snap['state'] == 'failed':
            leg = 'tripped_sim' if 'sim gap' in snap['reason'] else 'tripped_wall'
            return {'outcome': leg, 'wall': clock.wall, 'sim': clock.sim, 'reason': snap['reason'],
                    'state': snap['state']}
        if (clock.sim - op_start_sim) >= far_op_sim:
            return {'outcome': 'survived', 'wall': clock.wall, 'sim': clock.sim, 'reason': snap['reason'],
                    'state': snap['state']}
        clock.advance(dwall)
    raise AssertionError('far-station guard neither survived nor tripped within max_wall -- unbounded wait')


def simulate_capture_silent_station_suspend(*, pacing, sim_gap_budget, wall_floor, retreat_sim,
                                            suspend=True, frozen=False, enter_phase='retreat',
                                            exit_phase='verify', dwall=0.5, max_wall=4000000.0):
    """Drive the SHIPPED parked-station heartbeat guard through a CAPTURE-SILENT retreat/stow window.

    Models sim-retreat-station-heartbeat-scope-v1 (coldwin81): after the loaded place + contact release the
    mission executes a long, SILENT empty retreat (its first trajectory segment unwinds wrist_2 ~1.33 rad at
    ~0.013 rad/s => ~98 sim-s) during which it does NOT heartbeat the still-parked collect station. With the
    60 s SIM-gap budget that false-trips a healthy run ('parked heartbeat lost: sim gap 60.02 s' -> latched
    global stop -> collect StageRetry loop). With ``suspend=True`` the mission's phase goes capture-silent
    (``enter_phase``) and the node suspends ONLY the SIM leg (the frozen-sim WALL backstop stays armed), then
    re-arms on the next capture-bearing phase (``exit_phase``). ``suspend=False`` reproduces the pre-fix fault
    (key off => the SIM gap still trips). ``frozen=True`` holds the sim still for the whole window (a genuinely
    dead simulator) so the WALL backstop must still catch it even while suspended.

    Drives the REAL ``restock_host.incorporation_ack.IncorporationAckState`` (budgets set directly on the
    shipped object; ``wall_backstop_s`` re-derived with the shipped ``frozen_sim_wall_backstop``; the suspend
    key toggled on the object, as the env would). Returns a dict: outcome in
    {'survived','tripped_sim','tripped_wall'} with the final wall/sim readings, the reason, the published
    state and whether the guard was suspended across the window. Raises on an unbounded wait."""
    from twin.restock_host.incorporation_ack import IncorporationAckState
    clock = FakeClockPair(pacing=pacing, frozen=False)
    st = IncorporationAckState(wall_clock=lambda: clock.wall)
    st.sim_gap_s = float(sim_gap_budget)
    st.wall_gap_s = float(wall_floor)
    st.wall_backstop_s = frozen_sim_wall_backstop(sim_gap_budget, wall_floor)
    st.hb_capture_silent_suspend = bool(suspend)
    st.accept(dict(action='prepare', epoch=1, sequence=0, stamp=0.0, parked=True,
                   camera_frame='restock_camera_collect'), 0.0)
    # The mission's phase enters the capture-silent retreat/stow window (STATION_HEARTBEAT_SUSPENDED edge).
    suspended_line = st.note_heartbeat_phase(enter_phase)
    clock.frozen = bool(frozen)                       # the retreat begins; a frozen sim stalls /clock here
    op_start_sim = clock.sim
    while clock.wall <= max_wall:
        snap = st.snapshot(clock.sim, endpoints_ready=True, process_alive=True)
        if snap['state'] == 'failed':
            leg = 'tripped_sim' if 'sim gap' in snap['reason'] else 'tripped_wall'
            return {'outcome': leg, 'wall': clock.wall, 'sim': clock.sim, 'reason': snap['reason'],
                    'state': snap['state'], 'suspended_entry': bool(suspended_line)}
        # The healthy retreat completes after retreat_sim SIM seconds; the mission then re-enters a
        # capture-bearing phase (verify), which re-arms the SIM gap (STATION_HEARTBEAT_REARMED edge).
        if not frozen and (clock.sim - op_start_sim) >= retreat_sim:
            rearmed_line = st.note_heartbeat_phase(exit_phase)
            snap = st.snapshot(clock.sim, endpoints_ready=True, process_alive=True)
            return {'outcome': 'tripped_sim' if snap['state'] == 'failed' and 'sim gap' in snap['reason']
                    else 'tripped_wall' if snap['state'] == 'failed' else 'survived',
                    'wall': clock.wall, 'sim': clock.sim, 'reason': snap['reason'], 'state': snap['state'],
                    'suspended_entry': bool(suspended_line), 'rearmed_exit': bool(rearmed_line)}
        clock.advance(dwall)
    raise AssertionError('capture-silent station guard neither survived nor tripped -- unbounded wait')


def simulate_loaded_collect_heartbeat_read_lag(*, pacing, sim_gap_budget, wall_floor, read_lag_sim,
                                               op_sim, dwall=0.5, max_wall=2000000.0):
    """Reproduce the intermittent COLLECT host e-stop (sim-collect-host-estop-v1) on CPU.

    The mission heart-beats the parked collect/source station every tick while it is actively picking, but
    under a loaded-host executor / render spike in a long collect the host READS each heartbeat LATE -- it
    drains a backlog oldest-first, so the heartbeat it processes this service cycle was stamped
    ``read_lag_sim`` SIM seconds earlier (the oldest, stalest backlogged command), while the mission is
    alive and still sending strictly-increasing sequences. Models exactly that: each host service cycle
    (``dwall`` wall apart) the host calls ``accept`` with a NEWER-sequence heartbeat whose stamp lags the
    host's own /clock by ``read_lag_sim``. A ``read_lag_sim`` > the ``now - stamp`` upper bound reproduced
    the trip on the pre-fix host (``accept`` rejected the valid late read -> fault -> bare global e-stop);
    the shipped pace-safe ``accept`` must instead ACCEPT it and the station must survive the whole op.

    Drives the REAL ``restock_host.incorporation_ack.IncorporationAckState`` against the fake clock pair.
    Returns a dict: outcome in {'survived','tripped_reject','tripped_gap'} with the final wall/sim readings,
    the reason, and the count of accepted heartbeats. 'survived' == every late heartbeat was accepted AND the
    liveness gap never faulted across the whole op (no false stop). Raises on an unbounded wait."""
    from twin.restock_host.incorporation_ack import IncorporationAckState
    clock = FakeClockPair(pacing=pacing, frozen=False)
    st = IncorporationAckState(wall_clock=lambda: clock.wall)
    st.sim_gap_s = float(sim_gap_budget)
    st.wall_gap_s = float(wall_floor)
    st.wall_backstop_s = frozen_sim_wall_backstop(sim_gap_budget, wall_floor)
    # The parked epoch is prepared FRESH while the base first parks (prepare keeps the strict freshness
    # window -- the nvblox-incorporation contract); only the recurring heartbeats are later read late.
    st.accept(dict(action='prepare', epoch=1, sequence=0, stamp=0.0, parked=True,
                   camera_frame='restock_camera_collect'), 0.0)
    op_start_sim = clock.sim
    seq = 0
    accepted = 0
    while clock.wall <= max_wall:
        # The mission stamped this heartbeat read_lag_sim SIM-seconds ago (newer sequence each cycle); the
        # loaded host only gets to it now.
        seq += 1
        stamp = max(0.0, clock.sim - read_lag_sim)
        try:
            action = st.accept(dict(action='heartbeat', epoch=1, sequence=seq, stamp=stamp, parked=True),
                               clock.sim)
            if action == 'heartbeat':
                accepted += 1
        except ValueError as exc:
            # Pre-fix: the valid-but-late heartbeat is rejected; the node turns this into the fault + stop.
            return {'outcome': 'tripped_reject', 'wall': clock.wall, 'sim': clock.sim,
                    'reason': str(exc), 'accepted': accepted}
        snap = st.snapshot(clock.sim, endpoints_ready=True, process_alive=True)
        if snap['state'] == 'failed':
            return {'outcome': 'tripped_gap', 'wall': clock.wall, 'sim': clock.sim,
                    'reason': snap['reason'], 'accepted': accepted}
        if (clock.sim - op_start_sim) >= op_sim:
            return {'outcome': 'survived', 'wall': clock.wall, 'sim': clock.sim,
                    'reason': snap['reason'], 'accepted': accepted}
        clock.advance(dwall)
    raise AssertionError('loaded-collect heartbeat guard neither survived nor tripped -- unbounded wait')
