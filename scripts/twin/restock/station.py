"""Parked-station mapper lifecycle contract, independent of ROS and Isaac."""
import math
import os
from pathlib import Path
import signal
import subprocess
import time


class OwnedProcess:
    """One task-created process session. Never discovers or stops live services."""
    def __init__(self, command, log, *, env):
        self.command,self.log,self.env=list(command),Path(log),dict(env)
        self.process=None
        self.stream=None

    def start(self):
        if self.process is not None:
            raise RuntimeError('owned process already started')
        self.log.parent.mkdir(parents=True,exist_ok=True)
        self.stream=self.log.open('xb')
        try:
            self.process=subprocess.Popen(self.command,stdin=subprocess.DEVNULL,
                stdout=self.stream,stderr=subprocess.STDOUT,env=self.env,start_new_session=True)
        except BaseException:
            self.stream.close()
            raise

    @property
    def alive(self):
        return self.process is not None and self.process.poll() is None

    def close(self):
        if self.alive:
            # start_new_session makes the recorded child PID its own group ID.
            os.killpg(self.process.pid,signal.SIGINT)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid,signal.SIGTERM)
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid,signal.SIGKILL)
                    self.process.wait(timeout=2)
        if self.stream is not None:
            self.stream.close()


# Loose wall-clock liveness backstop (s). Only guards against a FROZEN sim clock; the sim-clock
# heartbeat gap (0.5 s) + process_alive are the primary liveness checks. Kept well above the slowest
# render-paced mission tick so a heavy full-res sensing render cannot false-trip a station fault.
_WALL_HEARTBEAT_BACKSTOP_S = 15.0

# Bounded fault recovery: a station fault (mapper child died/segfaulted, incorporation service
# unavailable, a rejected command) latches ``self.fault`` and emergency-stops. The mission client
# re-prepares under a FRESH measured parked epoch when it sees the 'failed'/'idle' reply; that fresh
# prepare is a legitimate recovery, not a protocol violation, so accept() clears the fault and the
# supervisor restarts the mapper child (station_host.command). Bounded so a PERSISTENTLY broken mapper
# still surfaces honestly instead of looping forever. Only a 'prepare' recovers (it carries a fresh
# parked epoch); heartbeat/stop against a faulted station are still refused. See ds-015 (station faulted
# then rejected re-prepare forever) and freshness-sim-clock-not-wall.
_MAX_RECOVERIES = 3


class StationState:
    """Fresh parked epochs and real mapper acknowledgements; no receipt shortcuts."""
    def __init__(self, *, wall_clock=time.monotonic):
        self.wall_clock=wall_clock
        self.epoch=0
        self.sequence=-1
        self.parked_since=-1.
        self.last_heartbeat=-1.
        self.wall_heartbeat=wall_clock()
        self.collision_time=-1.
        self.active=False
        self.fault=None
        self.camera_frame=None
        # Bounded recovery bookkeeping (requirement: fault-recoverable station).
        self.recoveries=0
        self.recovered_from=None

    def accept(self, command, now):
        recovering=False
        if self.fault:
            # Only a fresh measured parked epoch ('prepare') recovers a faulted station; a stray
            # heartbeat/stop against the dead epoch stays refused. Bound the recoveries so a mapper
            # that keeps dying is surfaced rather than retried forever.
            if command.get('action')!='prepare':
                raise RuntimeError('station fault requires explicit recovery')
            if self.recoveries>=_MAX_RECOVERIES:
                raise RuntimeError('station fault recovery budget exhausted: '+str(self.fault))
            recovering=True
        epoch=command.get('epoch')
        stamp=command.get('stamp')
        seq=command.get('sequence')
        action=command.get('action')
        if (type(epoch) is not int or epoch<=0 or type(seq) is not int or seq<0
                or not isinstance(stamp,(int,float)) or not math.isfinite(stamp)
                or not math.isfinite(now) or not 0<=now-stamp<=.5):
            raise ValueError('invalid station epoch, sequence or timestamp')
        if epoch<self.epoch or (epoch==self.epoch and seq<=self.sequence):
            return 'ignored'
        if action=='prepare':
            if epoch<=self.epoch or command.get('parked') is not True:
                raise ValueError('new measured parked epoch required')
            frame=command.get('camera_frame')
            if not isinstance(frame,str) or not frame.startswith('restock_camera_'):
                raise ValueError('calibrated mapper camera frame required')
            # The fresh epoch validated: clear the latched fault and count the recovery. Done here,
            # after every field check, so a malformed recovery prepare leaves the fault latched.
            if recovering:
                self.recovered_from=self.fault
                self.fault=None
                self.recoveries+=1
            self.epoch,self.parked_since,self.camera_frame=epoch,stamp,frame
            self.active=True
            self.collision_time=-1.
        elif action in ('heartbeat','stop'):
            if epoch!=self.epoch or not self.active:
                raise ValueError('station command has no active matching epoch')
            if action=='heartbeat' and command.get('parked') is not True:
                raise ValueError('base/cart motion invalidates parked mapper')
            if action=='stop':
                self.active=False
                self.collision_time=-1.
        else:
            raise ValueError('unsupported station action')
        self.sequence=seq
        self.last_heartbeat=stamp
        self.wall_heartbeat=self.wall_clock()
        return action

    def acknowledge(self, stamp, epoch):
        if (not self.active or epoch!=self.epoch or not math.isfinite(stamp)
                or stamp<=self.parked_since or stamp<self.collision_time):
            raise ValueError('mapper acknowledgement does not belong to current parked epoch')
        self.collision_time=stamp

    def snapshot(self, now, *, endpoints_ready, process_alive):
        # Liveness is primarily the SIM/node-clock heartbeat gap (0.5 s; the mission heartbeats every
        # physics tick, so this trips only on a genuinely stalled mission whose sim clock still
        # advances) plus process_alive. The WALL-clock check is only a loose backstop for a frozen
        # sim clock; at the real-rig 1 s it FALSE-TRIPS under render pacing -- a single full-res
        # sensing render during the transfer warm pushes one tick past 1 s wall and faulted the
        # station mid-warm (ds-011/013). Widen the wall backstop so it no longer false-trips while
        # the sim-clock gap + process_alive remain the real guards. See freshness-sim-clock-not-wall.
        if self.active and (not process_alive or now-self.last_heartbeat>.5
                            or self.wall_clock()-self.wall_heartbeat>_WALL_HEARTBEAT_BACKSTOP_S):
            self.fault='station process or parked heartbeat lost'
        ready=(self.active and not self.fault and endpoints_ready
               and 0<=now-self.collision_time<=.5 and self.collision_time>self.parked_since)
        return dict(epoch=self.epoch,sequence=self.sequence,stamp=now,
            state='failed' if self.fault else ('ready' if ready else ('warming' if self.active else 'idle')),
            reason=self.fault or '',collision_time=self.collision_time if ready else -1.,
            parked_since=self.parked_since,camera_frame=self.camera_frame)
