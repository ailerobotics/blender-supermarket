"""Gap-filling liveness publisher for /manipulation/status (sim-preplace-branch-select-bounded-v1).

coldwin58/coldwin59 died on the Isaac-side HOST_STATUS_WATCHDOG (``isaac_ros_graph.poll``, 5 sim-s
bound on a fresh ``/manipulation/status``). The host publishes that status only from its SINGLE
rclpy executor thread -- ``RosTransferBridge._emit`` (per ``/sensors`` tick, inside ``controller.step``)
and ``RosTransferBridge.heartbeat`` (inside a bounded payload wait). Any computation that runs
synchronously on that executor -- the reaction-branch standoff selection ran >5 s at the preplace
transition -- blocks ALL of them, so the status stream goes silent and the watchdog kills a live host.

The reaction-branch selection itself is now cheap and bounded (``carton_level``), which removes the
coldwin59 cause. This class is the STRUCTURAL guarantee the captain asked for on top of that fix: the
status heartbeat keeps publishing from a SEPARATE daemon thread, so no future planner that blocks the
executor can ever starve it again. It publishes a liveness fill only when the last publish is older
than ``interval`` AND the controller is in an active phase, so during normal operation it simply tops
up the gaps between the real executor-driven publishes; it never clears ``active`` (it publishes the
same liveness message ``RosTransferBridge`` already emits) and it never advances the sensor-stream
anchors (so the sensor watchdog is untouched).

Pure Python: the ROS wiring lives in ``ros_host``; this module takes plain ``publish``/``now``/
``is_active`` callables so it is unit-testable with no ROS, Isaac or GPU (the mock-planner /
coldwin59-reproduction tests drive it directly).
"""
import threading
import time


class StatusHeartbeat:
    """Keep a liveness heartbeat publishing while the executor thread is busy.

    ``publish()``    -- called from the filler thread to emit one liveness heartbeat.
    ``now()``        -- a monotonic wall clock (seconds); the filler is a WALL-time gap filler
                        (a CPU/executor block is a wall event), and a wall publish maps to a fresh
                        SIM-clocked message on the Isaac side (each arrival reseeds its sim bound).
    ``is_active()``  -- True while a job is running (controller phase not idle/failed/succeeded); the
                        filler stays silent otherwise so an idle host publishes nothing (byte-identical).
    ``interval``     -- the maximum intended gap between publishes (seconds). The filler wakes four
                        times per interval, so the worst-case gap after a real publish is about
                        ``interval * 1.25`` -- well under the 5 sim-s Isaac watchdog for the 0.5 s default.
    """

    def __init__(self, publish, now=time.monotonic, is_active=lambda: True, interval=0.5):
        if not callable(publish):
            raise ValueError('publish must be callable')
        if not (isinstance(interval, (int, float)) and interval > 0):
            raise ValueError('interval must be a finite positive number of seconds')
        self._publish = publish
        self._now = now
        self._is_active = is_active
        self._interval = float(interval)
        self._last = self._now()
        self._run = False
        self._thread = None

    def note_publish(self):
        """Record that a real (executor-driven) publish just happened, so the filler only tops up
        genuine gaps. Called from ``RosTransferBridge._emit`` / ``.heartbeat`` on the executor thread;
        a plain float store is atomic under the GIL."""
        self._last = self._now()

    def start(self):
        if self._thread is not None:
            return
        self._run = True
        self._thread = threading.Thread(target=self._loop, name='status-heartbeat', daemon=True)
        self._thread.start()

    def stop(self):
        self._run = False
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
            self._thread = None

    def _loop(self):
        wake = max(self._interval / 4.0, 0.01)
        while self._run:
            try:
                if self._is_active() and (self._now() - self._last) >= self._interval:
                    self._publish()
                    self._last = self._now()
            except Exception:  # noqa: BLE001 -- a liveness fill must never crash the host thread
                pass
            time.sleep(wake)
