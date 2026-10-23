"""Isaac native ROS bridge: no Python rclpy, cuRobo or nvblox imports.

Construct inside integration's guarded app after enabling isaacsim.ros2.bridge.
Host Jazzy runs ros_host separately. Existing clock, TF, joint trajectory and
RGB-D graphs remain integration-owned; this graph adds manipulation messages.
"""
from dataclasses import asdict
import json
import time

from .ros_adapter import require_sim_domain
from ..restock_camera.freshness import require_fresh


TOPICS = {'sensors': '/restock_sim/sensors',
          'job': '/restock_sim/manipulation/job',
          'status': '/restock_sim/manipulation/status',
          'gripper': '/restock_sim/gripper/enable',
          'stop': '/restock_sim/manipulation/stop'}


class IsaacMeasuredFeedback:
    """Planner-free sensor adapter for the real imported articulation.

    Integration sets collision_time only from mapper incorporation acknowledgement,
    and resets it to -1 after docking/frame changes. No depth-receipt shortcut.
    """
    def __init__(self, articulation):
        from .cad_tool import ARM_JOINTS
        self.robot = articulation
        self.indices = [articulation.get_dof_index(name) for name in ARM_JOINTS]
        self.collision_time = -1.
        self.source = None

    def observe_source(self, observation):
        if observation.source != 'sensor' or observation.frame != 'ur_base_link':
            raise ValueError('source must be sensor-derived in current arm frame')
        self.source = observation

    def joint_positions(self):
        import numpy as np
        joints = np.asarray(self.robot.get_joint_positions(joint_indices=self.indices),float)
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise ValueError('invalid measured arm encoders')
        return tuple(joints)

    def parked(self):
        import numpy as np
        linear = np.asarray(self.robot.get_linear_velocity(),float)
        angular = np.asarray(self.robot.get_angular_velocity(),float)
        return bool(np.isfinite(linear).all() and np.isfinite(angular).all()
                    and np.linalg.norm(linear)<.005 and np.linalg.norm(angular)<.005)


class IsaacRosGraph:
    def __init__(self, *, contact, hold, root='/RestockManipulationROS', timeout=5.,
                 wall_backstop=180., host_control=None):
        require_sim_domain()
        import omni.graph.core as og
        import omni.usd
        if omni.usd.get_context().get_stage().GetPrimAtPath(root):
            raise ValueError('refusing to overwrite an existing ROS graph')
        self.og, self.contact, self.hold = og, contact, hold
        self.root, self.timeout = root, timeout
        # Node-clock (sim-time) status bound + a generous wall backstop for a frozen sim clock.
        self.wall_backstop = wall_backstop
        self.last_receive_sim = None
        # Isaac->host PUBLISHES (job, sensors) go through the host-DDS subprocess, since the
        # OmniGraph ROS2 publisher does not reach the host UDPv4 DDS (run-59). The host->Isaac
        # SUBSCRIBERS (status, stop, gripper) stay on OmniGraph, which discovers the host writers.
        self.host_control = host_control
        self.active, self.last_status = False, ''
        self.last_sequence = -1
        self.faulted, self.job_id = False, None
        self.last_receive = time.monotonic()
        for key, topic in TOPICS.items():
            publish = key in ('sensors', 'job')
            # job/sensors are published out-of-Isaac via the control subprocess; /manipulation/status
            # is forwarded back by it too (the OmniGraph subscriber delivered it too slowly for the
            # 5 s manip watchdog). stop + gripper stay on OmniGraph subscribers.
            if host_control is not None and key in ('sensors', 'job', 'status'):
                continue
            node_type = 'ROS2Publisher' if publish else 'ROS2Subscriber'
            tick_type = 'OnImpulseEvent' if publish else 'OnPlaybackTick'
            tick_port = 'execOut' if publish else 'tick'
            graph, _, _, _ = og.Controller.edit(
                {'graph_path': root+'/'+key, 'evaluator_name': 'execution'}, {
                og.Controller.Keys.CREATE_NODES: [
                    ('tick','omni.graph.action.'+tick_type),
                    ('context','isaacsim.ros2.bridge.ROS2Context'),
                    ('wire','isaacsim.ros2.bridge.'+node_type)],
                og.Controller.Keys.SET_VALUES: [
                    ('context.inputs:useDomainIDEnvVar',False),
                    ('context.inputs:domain_id',88),
                    ('wire.inputs:topicName',topic),
                    ('wire.inputs:messagePackage','std_msgs'),
                    ('wire.inputs:messageSubfolder','msg'),
                    ('wire.inputs:messageName','Bool' if key in ('stop','gripper') else 'String')],
                og.Controller.Keys.CONNECT: [
                    ('tick.outputs:'+tick_port,'wire.inputs:execIn'),
                    ('context.outputs:context','wire.inputs:context')]})
            # MessageName materializes dynamic data attributes in the native node.
            og.Controller.evaluate_sync(graph)

    def _publish(self, key, data):
        if self.host_control is not None and key in ('sensors', 'job'):
            self.host_control.publish(TOPICS[key], data)  # host-DDS subprocess (reaches the host)
            return
        og = self.og
        path = self.root+'/'+key
        og.Controller.attribute(path+'/wire.inputs:data').set(data)
        og.Controller.attribute(path+'/tick.state:enableImpulse').set(True)
        og.Controller.evaluate_sync(path)

    def start(self, operation, job):
        if self.active or self.faulted:
            raise RuntimeError('transfer active or explicit fault recovery required')
        # sim-robust-transfer-start-estop-v1: re-derive the global-stop wire at the start of a NEW op,
        # exactly as a fresh (staged-start) graph begins -- AFTER the explicit-recovery guard above (a
        # genuine in-op fault still demands explicit recovery). See _clear_stale_global_stop.
        self._clear_stale_global_stop(operation)
        self._publish('job', json.dumps({'operation':operation,'job':job},allow_nan=False))
        self.active = True
        self.job_id = job['job_id']
        self.last_receive = time.monotonic()
        self.last_receive_sim = None   # re-seeded from the first poll of this job

    def publish_sensors(self, sensors):
        # Attach-fallback propagation probe (run-213): log the contact bool that actually reaches the
        # wire, throttled to transitions. If this prints contact=True after the weld but the controller
        # still cap-faults, the break is host-side (deserialize/freshness); if it never prints True while
        # the welded instance shows fb=True, the capture read a different NativePanelContact.
        c = bool(getattr(sensors, 'contact', False))
        if c != getattr(self, '_last_pub_contact', None):
            self._last_pub_contact = c
            print('PUBLISH_SENSORS contact=%s stamp=%s' % (c, getattr(sensors, 'simulation_time', None)),
                  flush=True)
        self._publish('sensors',json.dumps(asdict(sensors),allow_nan=False))

    def _clear_stale_global_stop(self, operation):
        """sim-robust-transfer-start-estop-v1: re-derive the latched global-stop wire at a NEW op start.

        The global ``/manipulation/stop`` wire LATCHES permanently on this OmniGraph subscriber --
        nothing ever publishes ``False`` (every emergency_stop publishes ``Bool(data=True)``), and
        check_health sets ``self.faulted`` once it reads True. A BARE stop published by a STATION-
        liveness guard (``restock_host.incorporation_ack`` / ``restock.station_host``) or the trajectory
        host DURING the between-ops nav gap (couple -> outbound -> park, when the manipulation state
        machine is NOT polling, so check_health never consumes it) therefore survives into the NEXT op's
        first poll -- even after that station RECOVERS at this op's fresh ``prepare``. The full chain then
        e-stops at transfer's MANIP_START with a bare 'host emergency stop' and no host-controller fault
        (coldlivefull: host-stack.log 'station recovered: epoch=3 from="parked heartbeat lost: wall gap
        30.64 s"'). PR 231's named triggers + _host_fault_suffix only cover the MOTION controller/port
        (HOST_FAULT_ESTOP / PORT_FAULT_ESTOP on ``/manipulation/status``), so a station/trajectory stop
        reports bare. The STAGED start (start_at=transfer) never hits this: it begins with a FRESH graph
        whose stop wire was never True.

        Mirror that fresh-graph start. Called AFTER start()'s explicit-recovery guard, so a genuine in-op
        fault (``self.faulted`` latched by a check_health raise during the PRIOR op) still demands explicit
        recovery -- this only ever sees ``faulted == False``. In coldlivefull that is exactly the case: the
        nav stages did not poll check_health, so the stale stop never latched ``faulted``; it simply sat on
        the wire until transfer's first poll. A stop latched True AT op start is therefore a stale carry-
        over (a genuine op-FATAL stop during the PRIOR op would have failed the mission before a new op's
        start()), so clear it -- write the SAME Controller.attribute check_health reads. A genuinely
        ONGOING stop is re-taken by the ROS2Subscriber on the very next tick (it overwrites outputs:data
        only when it takes a NEW message, so a cleared-but-not-republished wire stays False), so
        check_health still fires WITHIN this op -- now with a named host-stack.log trigger (the
        *_ESTOP_PUBLISH lines added in this lane) and the _host_fault_suffix hint below. A fresh graph
        (stop wire already False) => no write, no log => byte-identical (collect op start; every staged
        start). ``self.faulted`` is re-cleared defensively (a no-op given the guard above)."""
        attr = self.og.Controller.attribute(self.root + '/stop/wire.outputs:data')
        if attr.get():
            attr.set(False)
            self.faulted = False
            print('STALE_OP_START_STOP_CLEARED op=%s (global /manipulation/stop was latched True at op '
                  'start from a station-liveness/trajectory stop during the between-ops nav gap, now '
                  'recovered; cleared so the new op begins like a fresh staged start -- a genuine ongoing '
                  'stop re-arms on the next tick)' % operation, flush=True)

    def check_health(self):
        """Check global stop every physics tick without consuming job replies.

        Integration also calls this while navigating or warming the mapper,
        when the manipulation state machine does not call poll().
        """
        if self.faulted:
            self.hold()
            raise RuntimeError('explicit fault recovery required')
        og = self.og
        if og.Controller.attribute(self.root+'/stop/wire.outputs:data').get():
            self.hold()
            self.faulted = True
            self.active = False
            # sim-robust-host-timers-audit-fix-v1: the stop WIRE is a bare Bool, so the mission historically
            # recorded only 'host emergency stop' with no cause (coldlive sample-4). The host also publishes
            # the NAMED fault reason on /manipulation/status; peek the latest (idempotent read, host-relay
            # path only) and append it so the mission reports WHICH host trigger fired, not a bare stop. Best
            # effort: if the failed status has not yet been relayed when the wire reads True, fall back to the
            # bare string (host-stack.log still carries HOST_FAULT_ESTOP/PORT_FAULT_ESTOP).
            raise RuntimeError('host emergency stop' + self._host_fault_suffix())

    # sim-robust-transfer-start-estop-v1: when /manipulation/status carries no matching MOTION-controller
    # fault, the stop came from a non-controller publisher (a station-liveness guard or the trajectory
    # host). Those do NOT set /manipulation/status, so the suffix would be bare; instead point the reader
    # at the named host-stack.log trigger so no op-start stop is ever reported with zero direction.
    _NON_CONTROLLER_STOP_HINT = (' (no matching host-controller fault on /manipulation/status: a station-'
                                 'liveness/trajectory stop, or a controller fault whose status has not yet '
                                 'relayed -- see the named trigger in host-stack.log: HOST_FAULT_ESTOP / '
                                 'PORT_FAULT_ESTOP / *_ESTOP_PUBLISH)')

    def _host_fault_suffix(self):
        """' (host fault: <reason>)' from the latest failed /manipulation/status; else a hint pointing at
        the named host-stack.log trigger for a station/trajectory stop. Read-only; never raises (a stop
        must still surface even if the peek fails)."""
        try:
            host_control = getattr(self, 'host_control', None)
            if host_control is None:
                return ''
            raw = host_control.read(TOPICS['status'])
            if raw:
                result = json.loads(raw)
                reason = result.get('reason')
                # Only THIS job's failed status (a lingering failed status from a prior, recovered op must
                # not mislabel the current stop).
                if (result.get('status') == 'failed' and reason
                        and result.get('job_id') == getattr(self, 'job_id', None)):
                    return ' (host fault: %s)' % reason
        except Exception:
            return ''
        return self._NON_CONTROLLER_STOP_HINT

    def _status_fault(self, reason):
        self.hold()
        self.faulted = True
        self.active = False
        return {'status':'failed','reason':reason}

    def poll(self, simulation_time=None):
        """Call every physics step, including steps without a fresh image.

        ``simulation_time`` (the node/sim clock, seconds) node-clocks the status watchdog.
        The host goes legitimately silent during a long render-paced cuMotion multi-seed IK
        retry: many WALL seconds pass while the sim clock barely advances across each render
        tick, so a wall-time 5 s bound false-trips and kills the mission mid-grasp (run-82).
        Measuring the bound in SIM time keeps exactly the 5 s semantics on real hardware
        (sim == wall) without false-tripping under render pacing; a generous WALL backstop
        still hard-stops a genuinely dead host when the sim clock is frozen (the sim bound can
        never advance to trip then). Called with no sim clock => legacy wall behaviour.
        """
        try:
            self.check_health()
        except RuntimeError as exc:
            return {'status':'failed','reason':str(exc)}
        og=self.og
        # /manipulation/status arrives via the control subprocess (host-side latency) when a
        # host_control relay is wired; otherwise the OmniGraph subscriber.
        host_control = getattr(self, 'host_control', None)
        if host_control is not None:
            raw = host_control.read(TOPICS['status'])
        else:
            raw = og.Controller.attribute(self.root+'/status/wire.outputs:data').get()
        result = None
        if raw and raw != self.last_status:
            result = json.loads(raw)
            sequence = result.get('sequence')
            if (not self.active or result.get('job_id') != self.job_id or
                    type(sequence) is not int or sequence <= self.last_sequence):
                result = None
            if result is None:
                raw = ''
        if result is not None:
            if result.get('status')!='failed' and any(c.get('kind')=='gripper' for c in result.get('controls',[])):
                try:
                    deadline=result.get('valid_until_steady')
                    if deadline is None:
                        raise ValueError('missing acquisition deadline for gripper command')
                    # The gripper acquisition deadline is a SIM-time budget (sensing_sources steady_clock
                    # = world.current_time + window); check it on the SIM/node clock, not wall. Under
                    # render pacing wall races far ahead of sim, so a wall check expires instantly even
                    # though the registration is fresh in sim time (run-144: suction enabled at
                    # valid_until=20.78 sim then this check wall-expired). Mirrors the controller's
                    # sim-clocked require_fresh. Legacy wall only when no sim clock is supplied.
                    require_fresh(deadline, (lambda: simulation_time) if simulation_time is not None
                                  else time.monotonic)
                except (ValueError,TypeError) as exc:
                    self.hold()
                    self.faulted,self.active=True,False
                    return dict(status='failed',reason=str(exc))
            self.last_sequence = result['sequence']
            self.last_status, self.last_receive = raw, time.monotonic()
            if simulation_time is not None:
                self.last_receive_sim = simulation_time
            for control in result.get('controls',[]):
                if control.get('kind') == 'gripper':
                    self.contact.command(control['enabled'])
            if result.get('status') in ('succeeded','failed','rejected'):
                self.active = False
        if self.active:
            wall_now = time.monotonic()
            if simulation_time is None:
                if wall_now - self.last_receive > self.timeout:
                    return self._status_fault('host process/status watchdog')
            else:
                last_sim = getattr(self, 'last_receive_sim', None)
                if last_sim is None:
                    self.last_receive_sim = simulation_time   # seed from the first poll of the job
                elif simulation_time - last_sim > self.timeout:
                    return self._status_fault('host process/status watchdog')
                elif wall_now - self.last_receive > getattr(self, 'wall_backstop', 180.):
                    return self._status_fault('host process/status wall backstop')
        return result

    def close(self, *, close_control=True):
        """Hold the held payload sealed and fault the graph. ``close_control`` gates tearing down the
        out-of-Isaac control publisher subprocess (``host_control``).

        That subprocess is a SHARED, STATION-level resource: it both PUBLISHES job/sensors to the host
        AND forwards the host->Isaac return topics -- including ``/restock_sim/station/status``, which the
        StationClient reads for its parked-mapper ack. Terminating it on a PER-OP manipulation failure
        (``ros_manipulation.stop(close_sensing=False)``) killed the station/status forwarding mid-transfer:
        the StationClient's reader thread hit EOF, kept reading the LAST ack stamp forever, and the
        frozen-stamp watchdog raised ``StationAckStreamStalled`` -- masking the true stage reason
        (``collision planning or execution failed``; armcam/transfer2/bench1 all froze at ~sim 31 s this
        way). Like the masked-depth feed (``close_sensing``), the relay is a per-mission resource: keep it
        alive across a per-op failure so the ack stays fresh and the true reason surfaces; tear it down
        exactly once, at the genuine mission teardown (``close_control=True``, the default)."""
        self.hold()  # leave any physically held payload sealed
        self.active = False
        self.faulted = True
        if close_control and self.host_control is not None:
            try:
                self.host_control.close()  # terminate the out-of-Isaac control publisher subprocess
            except Exception:
                pass
