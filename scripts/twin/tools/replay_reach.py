#!/usr/bin/env python3
"""Offline (CPU, no GPU) reach-replay harness for the restock capture+hold pipeline.

A camera-side change (``restock_manipulation/isaac_sensors.py``,
``restock_camera/sensed.py``, ``restock_camera/binding.py``, ``restock_camera/qr.py``)
can be proven against the exact recorded frame cadence that failed in a live mission run
in **seconds**, instead of a 12-20 minute GPU run. It drives the **production** capture
entry (``IsaacCartonSensors.capture``), the real QR replay-refusal and the real
occlusion-hold state machine, imported and run **unchanged** -- this module forks no gate
logic -- over the controller phase sequence ``observe -> approach -> contact -> lift`` on a
virtual clock, and reports per-call outcomes in the same form as the live ``COLLECT_CAPTURE``
log lines.

### What is replayed, and the honest limit

A live run persists (``restock/recording.py``) only the **walkthrough** camera's RGB with
per-frame identity (``render_id``), acquisition stamp (``simulation_time``) and pixel
``sha256`` -- see ``mission-frames/frames.json``. The **sensing** camera's depth / RGB /
registration context are ephemeral and never hit disk, so a byte-for-byte replay of
``capture()`` on the sensing pixels is not possible from on-disk artifacts.

What *is* on disk and decisive is the **frame cadence and identity schedule**: the order of
``(frame_id, stamp, sha256)`` the renderer produced, paired with the control loop's poll
cadence. That schedule is exactly what drives the timestamp/identity gates that dominate
runs 99-102's ``provider_error``s: ``no new synchronized rendered camera frame`` (the
control loop polls faster than the renderer, so ``get_current_frame`` returns an
un-advanced frame -> the monotonic gate refuses), ``QR decoder refuses replayed camera
frame/session`` (the anti-replay gate), and the occlusion-hold window refusals. The harness
replays that recorded schedule against a **committed analytic empty-support bench frame**
(labelled, not sensor evidence) through the real gates. The bench frame stands in for the
unrecorded sensing pixels; the *timing/identity* comes from the run. This reproduces the
run's failure **modes** deterministically and makes a camera-side fix provable in seconds.

Verification target: replay run-101 and run-102 tapes; show the modes they failed on, and
that PR #80's sustained occlusion hold (window configured) keeps ``/sensors`` publishing on
the same frames where strict mode (window ``None``, the observe/commissioning default)
refuses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np


# --- recorded frame tape (with input hashes) --------------------------------

@dataclass(frozen=True)
class FrameRecord:
    index: int
    frame_id: int        # render numerator (monotonic acquisition id)
    stamp: float         # simulation_time of acquisition
    rgb_sha256: str      # recorded pixel hash (input hash)
    file: str = ""


def load_frame_tape(run_dir) -> list[FrameRecord]:
    """Load a run's recorded native-frame tape from ``mission-frames/frames.json``."""
    frames_json = Path(run_dir) / "mission-frames" / "frames.json"
    data = json.loads(frames_json.read_text())
    tape = []
    for i, f in enumerate(data.get("frames", [])):
        rid = f.get("render_id", "")
        try:
            frame_id = int(json.loads(rid).get("referenceTimeNumerator"))
        except (ValueError, TypeError):
            frame_id = i
        tape.append(FrameRecord(i, frame_id, float(f["simulation_time"]),
                                f.get("sha256", ""), f.get("file", "")))
    return tape


def tape_digest(tape: list[FrameRecord]) -> str:
    """Stable hash of the ordered recorded-frame set (its identity+timing+pixel hashes)."""
    h = hashlib.sha256()
    for r in tape:
        h.update(f"{r.frame_id}:{r.stamp!r}:{r.rgb_sha256}\n".encode())
    return h.hexdigest()


# --- phase plan + replay schedule -------------------------------------------

PHASES = ("observe", "approach", "contact", "lift")
_OCCLUDED_PHASES = {"approach", "contact", "lift"}  # the reaching arm occludes the support


# The bench baseline is commissioned at this sim stamp; the replayed clock is rebased so the
# first recorded frame lands here (the live mission loads a prior-session baseline whose stamp
# precedes the run clock -- rebasing preserves every frame-to-frame freshness delta while
# fixing the absolute ordering vs the baseline lifetime gate).
COMMISSION_STAMP = 1.0


@dataclass(frozen=True)
class PhasePlan:
    """Fraction of the tape spent in each controller phase (observe -> approach ->
    contact -> lift). Observe must be clean long enough to seed a good registration; the
    reach phases are occluded (the reaching arm hides the fixed camera's support view)."""
    observe: float = 0.30
    approach: float = 0.40
    contact: float = 0.15
    lift: float = 0.15

    def phase_for(self, frac: float) -> str:
        acc = 0.0
        for name in PHASES:
            acc += getattr(self, name)
            if frac < acc:
                return name
        return "lift"


@dataclass(frozen=True)
class ReplayStep:
    step: int
    now: float
    frame: FrameRecord
    phase: str
    occluded: bool


def build_schedule(tape: list[FrameRecord], *, plan: PhasePlan | None = None,
                   repeats: int = 1, max_frames: int | None = None) -> list[ReplayStep]:
    """Walk the recorded tape one capture per frame (optionally re-presenting each frame
    ``repeats`` times to model the control loop outrunning the renderer -- the live "no new
    synchronized rendered camera frame" / replay-refusal race). ``now`` tracks each frame's
    (rebased) acquisition stamp so observe frames are fresh and in-sync; the reach phases are
    occluded. The absolute clock is rebased so the first frame lands at ``COMMISSION_STAMP``.
    """
    if not tape:
        return []
    plan = plan or PhasePlan()
    offset = COMMISSION_STAMP - tape[0].stamp
    if max_frames is not None:
        tape = tape[:max_frames]
    n = len(tape)
    steps: list[ReplayStep] = []
    for i, rec in enumerate(tape):
        now = rec.stamp + offset
        rec = FrameRecord(rec.index, rec.frame_id, now, rec.rgb_sha256, rec.file)
        phase = plan.phase_for(i / n)
        occluded = phase in _OCCLUDED_PHASES
        for _ in range(max(1, repeats)):
            steps.append(ReplayStep(len(steps), now, rec, phase, occluded))
    return steps


# --- replay camera: presents a recorded frame's identity on the bench pixels -

class ReplayCamera:
    """A production-shaped camera that re-presents the bench RGB-D under a recorded
    frame's identity (``rendering_frame``/``rendering_time``), optionally occluding the
    support footprint depth (the reaching arm) to drive the hold path."""

    def __init__(self, base_frame: dict, intrinsics, occluded_depth):
        self._base = base_frame
        self._k = intrinsics
        self._clean_depth = np.asarray(base_frame["distance_to_image_plane"], float)
        self._occluded_depth = occluded_depth
        self._frame = base_frame

    def present(self, record: FrameRecord, *, occluded: bool):
        depth = self._occluded_depth if occluded else self._clean_depth
        self._frame = dict(self._base, distance_to_image_plane=depth,
                           rendering_frame=int(record.frame_id),
                           rendering_time=float(record.stamp))

    def get_current_frame(self, **_kw):
        return self._frame

    def get_intrinsics_matrix(self):
        k = self._k
        return np.array([[k.fx, 0, k.cx], [0, k.fy, k.cy], [0, 0, 1.0]])

    def add_distance_to_image_plane_to_frame(self):
        pass


# --- replay report -----------------------------------------------------------

@dataclass
class ReplayReport:
    tape_digest: str = ""
    frames: int = 0
    steps: int = 0
    ok: int = 0
    outcomes: list = field(default_factory=list)       # per-step dict rows
    errors: dict = field(default_factory=dict)          # message -> count
    log_lines: list = field(default_factory=list)       # COLLECT_CAPTURE-form lines
    occluded_published: int = 0                         # held (flagged-occluded) successes
    rendered: int = 0                                   # frames actually presented (render cost)

    def summary(self) -> str:
        parts = [f"frames={self.frames} steps={self.steps} ok={self.ok} "
                 f"held={self.occluded_published}"]
        for msg, n in sorted(self.errors.items(), key=lambda kv: -kv[1]):
            parts.append(f"  provider_error x{n}: {msg}")
        return "\n".join(parts)


def replay(adapter, camera: ReplayCamera, motion, contact, schedule, *,
           source: str, destination: str, gate: bool = False) -> ReplayReport:
    """Drive the production ``adapter.capture`` over the replay schedule.

    Emits one ``COLLECT_CAPTURE``-form line per step (identical in form to the live log),
    tallies outcomes, and flags held (occluded) successes -- the PR #80 sustain behaviour.

    ``gate`` exercises the opt-in SENSING GATE: on an occluded (execution-phase) step the harness
    signals ``adapter.set_render_suspended(True)`` and does NOT present a new frame, so the adapter
    republishes the last validated registration (flagged occluded) WITHOUT rendering -- the wall
    lever. Observe (clean) steps render as before. ``rep.rendered`` counts actual renders; with the
    gate it drops to the observe prefix while ``/sensors`` keeps publishing through the reach.
    """
    rep = ReplayReport(frames=len({s.frame.frame_id for s in schedule}), steps=len(schedule))
    calls = 0
    for step in schedule:
        calls += 1
        suspended = gate and step.occluded
        if gate:
            adapter.set_render_suspended(suspended)
        if not suspended:
            camera.present(step.frame, occluded=step.occluded)   # a fresh render
            rep.rendered += 1
        # collision map keeps pace with the packet clock (we are not exercising that gate)
        motion.collision_time = step.now
        try:
            packet = adapter.capture(step.now, source, destination, motion, contact)
        except Exception as exc:                       # the production gate refused this frame
            msg = f"{type(exc).__name__}:{exc}"
            rep.errors[str(exc)] = rep.errors.get(str(exc), 0) + 1
            rep.log_lines.append(
                f"COLLECT_CAPTURE calls={calls} phase={step.phase} provider_error={msg}")
            rep.outcomes.append(dict(step=step.step, phase=step.phase,
                                     frame_id=step.frame.frame_id, now=round(step.now, 3),
                                     outcome="error", detail=str(exc)))
            continue
        if packet is None:
            rep.log_lines.append(
                f"COLLECT_CAPTURE calls={calls} phase={step.phase} target=None")
            rep.outcomes.append(dict(step=step.step, phase=step.phase,
                                     frame_id=step.frame.frame_id, now=round(step.now, 3),
                                     outcome="none"))
            continue
        rep.ok += 1
        occ = bool(adapter.evidence and adapter.evidence[0].get("registration_occluded"))
        if occ:
            rep.occluded_published += 1
        rep.log_lines.append(
            f"COLLECT_CAPTURE calls={calls} phase={step.phase} sku=ok "
            f"held={occ} now={step.now:.3f} stamp={step.frame.stamp:.3f}")
        rep.outcomes.append(dict(step=step.step, phase=step.phase,
                                 frame_id=step.frame.frame_id, now=round(step.now, 3),
                                 outcome="ok", held=occ))
    return rep


def replay_source_target(adapter, camera: ReplayCamera, motion, contact, schedule, *,
                         location: str, gate: bool = False, max_age: float = 0.5):
    """Replay the COLLECT ``source_target_provider`` over the schedule (observe -> occluded reach).

    Mirrors the integration consumer (``isaac_factory``): a target is *fresh* iff
    ``-0.02 <= now - target.stamp <= max_age``. With the gate, the occluded reach steps suspend the
    render and the provider republishes the held pick with a FRESH stamp, so the collect path keeps
    delivering a fresh target through the reach instead of the run-131 starvation (a frozen frame's
    stale observation stamp failing this gate every tick). Returns fresh / held / rendered tallies.
    """
    from scripts.twin.restock_camera.source_target import build_source_target_provider
    provider = build_source_target_provider(adapter, motion, contact, source_location=location)
    rep = SimpleNamespace(steps=len(schedule), fresh=0, stale=0, held=0, rendered=0,
                          none=0, errors={})
    for step in schedule:
        suspended = gate and step.occluded
        if gate:
            adapter.set_render_suspended(suspended)
        if not suspended:
            camera.present(step.frame, occluded=step.occluded)
            rep.rendered += 1
        motion.collision_time = step.now
        try:
            target = provider(step.now)
        except Exception as exc:
            rep.errors[str(exc)] = rep.errors.get(str(exc), 0) + 1
            continue
        if target is None:
            rep.none += 1
            continue
        if -0.02 <= step.now - target.stamp <= max_age:
            rep.fresh += 1
            rep.held += bool(target.occluded)
        else:
            rep.stale += 1                              # the consumer would reject -> /sensors starves
    return rep


# --- analytic source bench (production-API construction; NOT sensor evidence) -

# A minimal pinhole pose bridge for the bench: this is projection MATH only (the same
# backproject/plane-fit the real bridge does for a flat support), never capture/hold logic.
_BENCH_POSE_BRIDGE = '''\
import numpy as np
class CameraIntrinsics:
    def __init__(self, *k): self.k = k
def CameraIntrinsics_factory(*k): return CameraIntrinsics(*k)
def backproject_mask(mask, depth, k):
    fx, fy, cx, cy = k.k if hasattr(k, "k") else k
    v, u = np.nonzero(mask)
    z = depth[v, u]
    return np.column_stack(((u - cx) * z / fx, (v - cy) * z / fy, z))
class _Pose:
    def __init__(self, t, n): self.translation, self.normal = t, n
def estimate_pose(points, min_points):
    return _Pose(points.mean(0), np.array([0., 0., 1.]))
'''


def _bench_pose_bridge_root() -> Path:
    root = Path(tempfile.mkdtemp(prefix="reach-bench-bridge-"))
    (root / "labeling").mkdir()
    (root / "labeling" / "pose_bridge.py").write_text(_BENCH_POSE_BRIDGE)
    return root


def _quad(center, size):
    return np.asarray(center) + np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]) \
        * [*np.asarray(size) / 2, 0]


def build_source_bench(*, occlusion_window_s=20.0, location="backstock_00", execution_gate=False,
                       with_pick=False):
    """Build a REAL commissioned source ``IsaacCartonSensors`` over an analytic empty
    support, via production commissioning APIs. Returns (adapter, camera, motion, contact).

    The RGB-D here is a committed analytic bench frame, explicitly not sensor/physics
    evidence -- the same honesty caveat the repo's CPU fixtures carry. It stands in for the
    unrecorded sensing pixels so the recorded frame *cadence/identity* can drive the real
    gates. The pose bridge is the bench projection bridge above.
    """
    import cv2
    from scripts.twin.restock_camera.geometry import Intrinsics, project, transform_points
    from scripts.twin.restock_camera import commissioning as C
    from scripts.twin.restock_camera.sensed import polygon_mask
    from scripts.twin.restock_camera.binding import RenderedSupportContext, SupportSlot
    from scripts.twin.restock_manipulation import isaac_sensors

    scale = 2
    k = Intrinsics(1280 * scale, 960 * scale, 640. * scale, 640. * scale,
                   640. * scale - .5, 480. * scale - .5)
    camera = np.diag([1., -1., -1., 1.])
    camera[2, 3] = 1.
    depth = np.full((k.height, k.width), .8)

    def pix(points):
        uv, valid = project(transform_points(np.linalg.inv(camera), points), k)
        assert valid.all(), "bench fixture outside view"
        return uv

    support = pix(_quad([0, 0, .2], [.70, .50]))
    witness = pix(_quad([-.48, 0, .2], [.08, .25]))
    a = pix(_quad([-.43, -.17, .2], [.06, .06])[[3, 2, 1, 0]])
    b = pix(_quad([-.43, .17, .2], [.06, .06])[[3, 2, 1, 0]])
    gray = np.full(depth.shape, 255, np.uint8)
    for code, dest in [("support:a", a), ("support:b", b)]:
        qr = cv2.copyMakeBorder(cv2.QRCodeEncoder_create().encode(code), 4, 4, 4, 4,
                                cv2.BORDER_CONSTANT, value=255)
        qr = cv2.resize(qr, (300, 300), interpolation=cv2.INTER_NEAREST)
        _, corners, _ = cv2.QRCodeDetector().detectAndDecode(qr)
        hom = cv2.getPerspectiveTransform(corners[0].astype(np.float32), dest.astype(np.float32))
        gray = np.minimum(gray, cv2.warpPerspective(qr, hom, (k.width, k.height),
                                                    flags=cv2.INTER_NEAREST, borderValue=255))
    frame = dict(rgb=np.repeat(gray[:, :, None], 3, axis=2),
                 distance_to_image_plane=depth, rendering_frame=1, rendering_time=1.)

    def _context():
        slot = SupportSlot(location, "deck", "support:a", .06, "bolted1",
                           np.ones((k.height, k.width), bool),
                           auxiliary_id="support:b", auxiliary_offset_xy=(0, .34))
        return RenderedSupportContext(
            camera_id="source", intrinsics=k, slots=[slot],
            measure_context=lambda fr, now: (camera, now, "park1", True),
            clearance_for_slot=lambda loc, view, fr, now: None,
            depth_error_m=.0002, camera_revision="lens1")

    dataset = Path(tempfile.mkdtemp(prefix="reach-bench-baseline-"))
    C.commission_and_persist(_context(), frame, 1., {location: (support, witness)},
                             dataset_dir=dataset)
    boot = _context()
    C.apply_baselines(boot, C.load_baselines(dataset), required_locations=[location])
    boot.occlusion_window_s = occlusion_window_s

    # Optional pick carton on the commissioned support, so the COLLECT source_target_provider path
    # (restock_camera/source_target.py) can be replayed: a raised box with a decodable top-face SKU
    # QR, authored into a SEPARATE live frame (commission stays on the empty support above). Its top
    # sits one carton-height above the support plane, so the depth patch + height gate pass and the
    # production decode reads the SKU as inventory (not a support marker).
    profiles = {}
    live_frame, live_depth = frame, depth
    if with_pick:
        from scripts.twin.restock_manipulation.types import ProductProfile
        sku, dims = "bench_sku", (.24, .18, .16)
        top_z = .2 + dims[2]                              # carton top in the base/map frame
        top_quad = pix(_quad([0, 0, top_z], [dims[0], dims[1]]))   # full top footprint (depth patch)
        # Warp the (square) QR onto a centred SQUARE region of the top face -- a non-square dest
        # would stretch the modules and make it undecodable. Match the support-marker corner order
        # ([3,2,1,0]) so it is not mirrored.
        qr_side = min(dims[0], dims[1]) - .02
        dest = pix(_quad([0, 0, top_z], [qr_side, qr_side])[[3, 2, 1, 0]])
        live_depth = depth.copy()
        live_depth[polygon_mask(top_quad, k)] = 1. - top_z   # camera-frame depth of the raised top
        pick_gray = gray.copy()
        qr = cv2.copyMakeBorder(cv2.QRCodeEncoder_create().encode(sku), 4, 4, 4, 4,
                                cv2.BORDER_CONSTANT, value=255)
        qr = cv2.resize(qr, (400, 400), interpolation=cv2.INTER_NEAREST)
        _, corners, _ = cv2.QRCodeDetector().detectAndDecode(qr)
        hom = cv2.getPerspectiveTransform(corners[0].astype(np.float32), dest.astype(np.float32))
        pick_gray = np.minimum(pick_gray, cv2.warpPerspective(qr, hom, (k.width, k.height),
                                                              flags=cv2.INTER_NEAREST, borderValue=255))
        live_frame = dict(rgb=np.repeat(pick_gray[:, :, None], 3, axis=2),
                          distance_to_image_plane=live_depth, rendering_frame=1, rendering_time=1.)
        profiles = {sku: ProductProfile(sku, dims, (0., 0., 0., 1.))}

    occluded_depth = live_depth.copy()
    # The reaching arm hides the fixed camera's independent witness patch (markers still
    # register): the fresh full-footprint view refuses, so the capture must hold or go strict.
    occluded_depth[polygon_mask(witness, k)] = np.nan

    replay_camera = ReplayCamera(live_frame, k, occluded_depth)
    adapter = isaac_sensors.IsaacCartonSensors(
        replay_camera, pose_bridge_root=str(_bench_pose_bridge_root()), profiles=profiles,
        context_provider=boot, camera_id="source", steady_clock=lambda: 0.,
        execution_gate=execution_gate)
    motion = SimpleNamespace(parked=lambda: True, collision_time=1.,
                             joint_positions=lambda: (0.,) * 6, observe_source=lambda o: None)
    contact = SimpleNamespace(read=lambda: False)
    return adapter, replay_camera, motion, contact, location


# --- CLI ---------------------------------------------------------------------

def run_tape(run_dir, *, occlusion_window_s=20.0, repeats=1, max_frames=None,
             plan: PhasePlan | None = None) -> ReplayReport:
    tape = load_frame_tape(run_dir)
    adapter, camera, motion, contact, location = build_source_bench(
        occlusion_window_s=occlusion_window_s)
    schedule = build_schedule(tape, plan=plan, repeats=repeats, max_frames=max_frames)
    rep = replay(adapter, camera, motion, contact, schedule,
                 source=location, destination=location)
    rep.tape_digest = tape_digest(tape)
    rep.frames = len(tape)
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", help="a mission run dir with mission-frames/frames.json")
    ap.add_argument("--window", type=float, default=20.0,
                    help="occlusion hold window seconds (None-like: 0 disables -> strict)")
    ap.add_argument("--strict", action="store_true",
                    help="observe/commissioning strict mode (occlusion_window_s=None)")
    ap.add_argument("--repeats", type=int, default=1,
                    help="re-present each recorded frame N times (models control>render rate)")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="replay only the first N recorded frames (full tape by default)")
    ap.add_argument("--verbose", action="store_true", help="print every per-call log line")
    args = ap.parse_args(argv)

    window = None if args.strict else args.window
    rep = run_tape(args.run_dir, occlusion_window_s=window, repeats=args.repeats,
                   max_frames=args.max_frames)
    print(f"tape {Path(args.run_dir).name}  digest={rep.tape_digest[:16]}  "
          f"window={'strict(None)' if window is None else window}")
    if args.verbose:
        for line in rep.log_lines:
            print(line)
    print(rep.summary())
    return 0


if __name__ == "__main__":
    import sys
    # Running as a script puts scripts/twin/tools on sys.path, not the repo root the
    # production ``scripts.twin...`` imports need; add the repo root (tools->twin->scripts->repo).
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    raise SystemExit(main())
