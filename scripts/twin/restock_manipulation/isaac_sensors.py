"""Rendered RGB-D inventory for a small, physically labelled carton slice.

Uses actual camera pixels plus the existing masked depth/PCA implementation.
QR text is product identity; no USD object names, semantic IDs, poses or colors
enter perception. A registered carton has a centred top label reading its SKU.
"""
from dataclasses import dataclass, replace
import os
SENSOR_WALL_BACKSTOP_S = max(5.0, min(600.0, float(os.environ.get('RESTOCK_SENSOR_WALL_BACKSTOP_S', 60.0))))  # fm362 sim-only env widen
import math
import time

import numpy as np

from .perception import DepthPerception, load_pose_bridge
from .types import Sensors
from ..restock_camera.geometry import rigid
from ..restock_camera.freshness import frame_deadline, require_fresh
from ..restock_camera.sensed import FrameEvidence, SupportView, masked_cloud, contains, depth_digest
from ..restock_camera.qr import FixedTileQR, FrameQR, rgb_digest


#: Controller phases whose decode-camera render is unused: observe has validated the registration
#: and cuMotion is executing the reach, so the arm occludes the fixed camera (the OBSERVE-THEN-
#: EXECUTE lock). The opt-in sensing gate suspends the render for these and republishes the held
#: registration. 'couple' is the cart-coupling action's execution phase.
#: The WHOLE cart holding chain (preplace/place/retreat) is suspended too (firstmate 286, observe-
#: then-execute choreography extension): once lift has the carton, the arm reaching to the cart
#: occludes the fixed source camera just as badly as the pick reach, so a preplace re-validation that
#: RESUMED the render stalled /sensors to the wall backstop (run-227). The cart placement target is
#: DERIVED from the cart prim (not camera-sensed), so the held re-validation fabricates nothing;
#: the fixed camera resumes only after retreat clears the view (next non-suspended phase: verify).
#: The staged cart-swing via points (firstmate 292: tuck/pan on the way out, retreat_tuck on the way
#: back) are part of the held carry chain -- the arm swings the carton and occludes the fixed camera --
#: so they suspend the render exactly like lift/preplace/retreat.
_SENSING_SUSPEND_PHASES = frozenset({'approach', 'contact', 'grasp', 'lift', 'preplace', 'place',
                                     'retreat', 'release', 'stow', 'couple',
                                     'tuck', 'pan', 'retreat_tuck'})


def sensing_render_required(phase):
    """Whether the decode camera must render for this controller phase.

    False only for the execution phases in ``_SENSING_SUSPEND_PHASES`` (observe already validated the
    registration and the rendered frame is unused while the arm occludes the fixed camera). True for
    observe and the verify re-validation, and for any other/unknown phase -- fail safe:
    never suspend the render unless the phase is explicitly a validated-reach execution phase. The
    integration runtime maps the controller phase through this each tick to drive
    ``set_render_suspended(not sensing_render_required(phase))``."""
    return str(phase) not in _SENSING_SUSPEND_PHASES


def _reject_unsynced(gate, **values):
    """Raise the standing ``no new synchronized rendered camera frame`` capture refusal.

    ``capture`` refuses a frame at THREE distinct gates that historically shared this one opaque
    message, so a cold-collect render-sync failure could not be pinned to a gate (nor its magnitude
    read) from the run log -- which is why the flaky COLD collect ``no new synchronized`` needed a
    live dig (``gate='future'`` = render clock leads ``now`` / non-finite; ``gate='nonmonotonic'`` =
    the render product's frame_id/stamp did not advance and it is not an exact reuse; ``gate='stale'``
    = ``now-stamp`` exceeded the acquisition-to-use budget, i.e. the source render did not deliver a
    fresh frame within the window -- the stride/async-RTX-latency case). ``RESTOCK_RENDER_SYNC_DIAG=1``
    prints a ``RENDER_SYNC_DIAG`` line naming the gate + the discriminating values; default off =>
    silent. This runs ONLY on the refusal path (a successful capture never calls it), and the raised
    message is UNCHANGED, so a default run -- and every existing matcher/tolerant retry -- is
    byte-identical."""
    if os.environ.get('RESTOCK_RENDER_SYNC_DIAG', '0') != '0':
        detail = ' '.join(f'{k}={v!r}' for k, v in values.items())
        print(f'RENDER_SYNC_DIAG reject gate={gate} {detail}', flush=True)
    raise ValueError('no new synchronized rendered camera frame')


@dataclass(frozen=True)
class SlotView:
    location: str
    roi: tuple[int, int, int, int]
    support_z: float      # previous empty-slot depth calibration in arm frame
    free_dimensions: tuple[float, float, float]
    yaw_rad: float = 0.0  # calibrated slot orientation in current arm frame
    label_mask: object = None
    support_view: SupportView | None = None

    @classmethod
    def from_support(cls, location, view, label_mask):
        mask = np.asarray(label_mask)
        k = view.geometry.intrinsics
        if mask.dtype != bool or mask.shape != (k.height,k.width) or not mask.any():
            raise ValueError('separate label search mask required')
        support = view.mask(k)
        v,u = np.where(support)
        roi = (int(u.min()),int(v.min()),int(u.max()+1),int(v.max()+1))
        return cls(location,roi,view.support_z,
                   (*view.geometry.required_xy,0. if view.clearance is None else view.clearance.height_m),view.yaw_rad,
                   mask.copy(),view)


@dataclass(frozen=True)
class CameraContext:
    base_from_camera: object
    transform_stamp: float
    epoch: str
    slots: tuple
    parked: bool
    session_id: str = 'default'
    steady_domain: str = ''
    acquired_steady: float | None = None
    qr_observation: FrameQR | None = None
    occlusion_window_s: float | None = None   # None: label in a calibrated slot must decode every frame


class IsaacCartonSensors:
    def __init__(self, camera, *, pose_bridge_root, profiles, slots=(), base_from_camera=None,
                 context_provider=None, camera_id='camera', steady_clock=time.monotonic,
                 wall_backstop_s=SENSOR_WALL_BACKSTOP_S, wall_clock=time.monotonic, execution_gate=False):
        # Injectable monotonic clock so acquisition-to-use freshness (and tests of it) do not
        # depend on real decode wall time; defaults to the process clock for native capture. For
        # the render-paced mission, build_sensing supplies the SIM/node clock here so a slow
        # full-resolution render is not misread as stale; wall_backstop_s hard-stops a hung render
        # whose sim clock is frozen (wall_clock is injectable for tests of the backstop).
        self.steady_clock = steady_clock
        self.wall_backstop_s = wall_backstop_s
        self.wall_clock = wall_clock
        self.camera, self.profiles = camera, profiles
        self.slots = tuple(slots)
        self.transform = None if base_from_camera is None else rigid(base_from_camera)
        self.context_provider, self.camera_id = context_provider, camera_id
        self.depth = DepthPerception(load_pose_bridge(pose_bridge_root))
        self.decoder = None  # created only when the context has no shared QR result
        self.last_frame = None
        self.last_stamp = None
        self.last_session = None
        self.seen_sessions = set()
        # Last SUCCESSFULLY captured frame, kept only to tolerate a reduced sensing render
        # cadence (IsaacPort.sensing_stride): when integration renders the sensing camera every
        # Nth physics tick, get_current_frame() returns the SAME frame on the intervening ticks.
        # That is not a replay — identical frame id, stamp AND pixels — so the decoded support
        # context + QR are reused (the costly full-tile decode is not repeated) while every
        # time-sensitive gate is re-run below against the live clock/clearance. Set only after a
        # full success so a failed frame is never reusable.
        self._reuse_key = None
        self._reuse_context = None
        self._reuse_qr = None
        # Opt-in integration-controlled SENSING GATE (default off): during a controller EXECUTION
        # phase (approach/contact/lift/place/retreat/release/stow/couple) AFTER observe validated the
        # registration, the arm occludes the fixed camera and the rendered frame is unused, yet the
        # whole-store RTX render runs every sensing tick -- the dominant wall cost (run-126). When the
        # gate is on and integration signals suspension (set_render_suspended(True)), capture skips
        # get_current_frame entirely and REPUBLISHES the last validated registration flagged occluded
        # (PR 80 hold), with FRESH motion-completion fields (parked/contact/collision/joints) so
        # /sensors stays alive for motion polling. Rendering resumes (set_render_suspended(False)) on
        # the controller re-validation signal (preplace after lift, verify) and for every observe.
        # Bounded by the same occlusion window as the hold: suspended past it refuses, so a genuinely
        # stalled execution still trips a fault. Default off => every gate and the exact-frame reuse
        # key below are byte-identical to before.
        self.execution_gate = bool(execution_gate)
        self._render_suspended = False
        # Tracked desired state of the decode camera's render product (None = never driven yet). The
        # gate drives it to match (enabled when NOT suspended), idempotently, so the FIRST
        # non-suspended tick at mission start guarantees the product is ON regardless of the native
        # camera's default or a disable persisted from a prior run -- not an edge-only toggle that
        # no-ops when was==suspended at init (run-146..148: product left off -> no first frame).
        self._products_enabled = None
        self._held_source = None   # last validated registration republished while suspended
        # Per-location sim time of the last frame a calibrated slot was cleanly read (support not
        # held AND no undecoded label in it). While the reaching arm occludes a slot's carton label
        # during approach/contact/lift, an undecoded label is tolerated (held, flagged occluded)
        # for up to context.occlusion_window_s seconds since that last clean read; past the window
        # it refuses. Seeded by the clean observe before the reach.
        self._label_clean = {}
        self.sensed_target_bboxes = ()
        self.evidence = ()
        # Integration initializes actual cameras; every frame uses measured TF.
        camera.add_distance_to_image_plane_to_frame()

    @property
    def render_suspended(self):
        """True iff the opt-in gate is on AND integration has suspended the render this tick, i.e.
        ``capture`` is serving the held republish (frozen frame, no new decode). A consumer that
        reads the per-observation acquisition stamp (e.g. the collect ``source_target_provider``)
        checks this to switch to the fresh motion-completion stamp while held. False (default/gate
        off) => the normal rendered path, byte-identical to before."""
        return bool(self.execution_gate and self._render_suspended)

    def set_render_suspended(self, suspended):
        """Integration-controlled sensing gate (effective only when ``execution_gate`` is on).

        Integration toggles this each tick from the controller phase: True while the arm EXECUTES a
        validated reach (the rendered frame is unused and the full-res render is the wall cost),
        False for observe / the preplace+verify re-validations (a fresh render is required).

        This DRIVES the decode camera's render product(s) to match the gate state via the camera's
        optional ``set_render_products_enabled`` hook: DISABLED while suspended (so a
        ``world.step(render=True)`` during suspension skips the expensive whole-store decode render
        and pays only for the cheap non-decode products -- arm depth / masked_depth / chase recorder,
        separate cameras the host nvblox ESDF needs; run-139 died when integration skipped ALL
        rendering), ENABLED otherwise. It is IDEMPOTENT and tracked (``_products_enabled``), not
        edge-only: the FIRST non-suspended call at mission start enables the product even though it is
        not a transition, so a native default-off (or a disable persisted from a prior run) can no
        longer leave the decode camera dark at startup (run-146..148: no first frame). On a genuine
        resume (was suspended, now not) it also drops the exact-frame reuse snapshot, so the first
        post-resume capture requires a NEW render (a stale decode-product frame is refused by the
        monotonic-frame gate, not reused). When the gate is off this just records the flag and never
        touches the product (byte-identical). See ``sensing_render_required``."""
        suspended = bool(suspended)
        was = self._render_suspended
        self._render_suspended = suspended
        if not self.execution_gate:
            return
        toggle = getattr(self.camera, 'set_render_products_enabled', None)
        desired_enabled = not suspended
        if self._products_enabled is not desired_enabled:   # None (startup) always drives it
            if toggle is not None:
                toggle(desired_enabled)
            self._products_enabled = desired_enabled
        if was and not suspended:
            # Genuine resume: a stale decode-product frame must not be reused as a fresh capture.
            self._reuse_key = self._reuse_context = self._reuse_qr = None

    def _held_capture(self, now, source, destination, motion, contact):
        """Republish the last validated registration (PR 80 occlusion hold) WITHOUT rendering.

        The integration runtime has suspended the decode-camera render for a controller execution
        phase, so there is no new frame to fetch/decode. Serve the last validated registration flagged
        occluded, with FRESH motion-completion fields so /sensors keeps feeding the controller's
        parked/contact/joint/collision polling. Every non-visibility bound still holds: a prior clean
        registration must exist, the configured occlusion window bounds how long the render may be
        suspended (past it this refuses, so a stalled execution trips a fault, never an endless held
        publish), and the base must still be parked (a moving base refuses -- the held pose is not
        re-measured). The published packet's simulation_time is the fresh tick time; its sensed
        registration/target are held, exactly as a PR 80 occluded frame."""
        held = self._held_source
        if held is None:
            raise ValueError('no validated registration to hold while the sensing render is suspended')
        window = held['window']
        if window is None or not np.isfinite(window) or window < 0:
            raise ValueError('sensing gate requires a configured occlusion window to hold')
        if not np.isfinite(now) or now < held['clean_now']:
            raise ValueError('held sensing publish needs a non-regressing clock')
        if now - held['clean_now'] > window:
            raise ValueError('sensing render suspended beyond the occlusion window')
        if not (motion.parked() and held['parked']):
            raise ValueError('held sensing publish requires a parked base')
        collision_time = motion.collision_time
        joints = tuple(motion.joint_positions())
        contact_state = bool(contact.read())
        if (not np.isfinite(collision_time) or not 0 <= now - collision_time <= .5
                or len(joints) != 6 or not np.isfinite(joints).all()):
            raise ValueError('stale or missing motion-completion feedback while sensing is held')
        # Fresh steady-clock deadline = the window remaining since the last clean registration, so the
        # controller's require_fresh(valid_until_steady) passes while the sim-time window still bounds
        # the hold. feedback/collision_time are fresh (motion-completion); the sensed target/obs held.
        remaining = window - (now - held['clean_now'])
        valid_until = self.steady_clock() + remaining
        self.sensed_target_bboxes = ()                      # no fresh detection is claimed while held
        self.evidence = tuple(dict(item, registration_occluded=True, current_stamp=now)
                              for item in held['evidence'])
        return Sensors(now, held['observations'], held['target'], bool(motion.parked()),
                       contact_state, now, collision_time, joints,
                       held['scanned'], held['scan_time'], valid_until)

    def capture(self, now, source, destination, motion, contact):
        if self.execution_gate and self._render_suspended:
            return self._held_capture(now, source, destination, motion, contact)
        import cv2
        frame = self.camera.get_current_frame(clone=True)
        frame_id, stamp = frame['rendering_frame'], float(frame['rendering_time'])
        self.sensed_target_bboxes = ()
        self.evidence = ()
        # Tolerate the camera render leading the dt-accumulated sim clock by up to ~1 physics
        # step (matches frame_deadline's _RENDER_LEAD in the context freshness gate); genuine
        # future/stale frames are still rejected. Without this a sub-step-leading frame is
        # dropped here before the context (which tolerates it) ever sees it.
        # From-the-future / non-finite frames are always rejected here. The STALENESS bound is
        # deferred until the context tells us whether this is HELD output under approach occlusion
        # (then the bound is the hold window, not 0.5 s — a deliberately held frame is stale by
        # design); see the budget block after the context is resolved.
        if not np.isfinite([now,stamp]).all() or now-stamp < -0.02:
            _reject_unsynced('future', now=now, stamp=stamp, delta=now-stamp, frame_id=frame_id)
        # Anchor the acquisition-to-use deadline on the steady clock BEFORE processing; the budget
        # (0.5 s live, or the hold window for held output) is applied once known, below.
        started = self.steady_clock()
        wall_started = self.wall_clock()
        rgb = np.asarray(frame['rgb'])[:, :, :3]
        depth = np.asarray(frame['distance_to_image_plane'], float)
        if rgb.shape[:2] != depth.shape or depth.ndim != 2:
            raise ValueError('unregistered camera RGB/depth')
        matrix = self.camera.get_intrinsics_matrix()
        intr = (matrix[0,0], matrix[1,1], matrix[0,2], matrix[1,2])
        # Exact-frame key for stride reuse: identical frame id, stamp AND pixels (rgb+depth) to the
        # last captured frame. Hashing the actual pixels (never array/frame identity) also detects a
        # tampered repeat — a changed buffer under an unchanged frame id fails the key and falls to
        # the normal path, where the decoder's monotonic-frame gate refuses it.
        key = (int(frame_id), stamp, rgb_digest(rgb), depth_digest(depth))
        reuse = self._reuse_key is not None and self._reuse_key == key
        if reuse:
            # Stride-paced re-presentation of the exact current frame. Reuse the decoded support
            # context + QR rather than calling the context provider / full-tile decoder again (that
            # would refuse the repeat as a replayed frame), but re-validate the frame against the
            # live clock below: the frame-age window was checked above, and the per-slot support /
            # baseline-lifetime / freshness validation, the live collision-map clearance gate, the
            # QR pixel re-hash and the acquisition-to-use deadline all re-run. A frame that has aged
            # out of the 0.5 s window, whose baseline expired, or that is no longer parked refuses.
            context = self._reuse_context
            self.transform = rigid(context.base_from_camera)
            self.slots = tuple(context.slots)
        elif self.context_provider is not None:
            context = self.context_provider(frame,now)
            self.transform = rigid(context.base_from_camera)
            self.slots = tuple(context.slots)
        elif self.slots and all(s.support_view is not None for s in self.slots):
            e = self.slots[0].support_view.registration.evidence
            context = CameraContext(self.transform,e.transform_stamp,e.epoch,self.slots,motion.parked(),
                                    e.session_id,e.steady_domain,e.acquired_steady)
        else:
            raise ValueError('fresh registered support views and camera context required')
        if not reuse and (self.last_session == context.session_id
                          and (frame_id <= self.last_frame or stamp <= self.last_stamp)
                          or self.last_session != context.session_id and context.session_id in self.seen_sessions):
            _reject_unsynced('nonmonotonic', now=now, stamp=stamp, frame_id=frame_id,
                             last_frame=self.last_frame, last_stamp=self.last_stamp,
                             session=context.session_id, last_session=self.last_session)
        self.last_frame,self.last_stamp,self.last_session = frame_id,stamp,context.session_id
        self.seen_sessions.add(context.session_id)
        if not self.slots or len({s.location for s in self.slots}) != len(self.slots):
            raise ValueError('camera needs unique calibrated slots')
        # HELD output (approach occlusion): any slot whose registration the context provider held
        # from the last clean frame. Its per-frame acquisition staleness is bounded by the hold
        # window, not the 0.5 s live budget -- a deliberately held, flagged-occluded frame is stale
        # by design, and the window (vs the last clean registration) is the real bound, enforced in
        # RegisteredSupport.hold. Clean/observe output stays on the strict 0.5 s budget. This is what
        # lets /sensors keep publishing through the multi-second reach instead of the frame-age and
        # monotonic/reuse gates stopping it once no fresh frame registers.
        held = any(getattr(s.support_view, 'occluded', False) for s in self.slots)
        window = getattr(context, 'occlusion_window_s', None)
        budget = float(window) if (held and window and np.isfinite(window)) else .5
        if now-stamp > budget:
            _reject_unsynced('stale', now=now, stamp=stamp, delta=now-stamp, budget=budget,
                             held=held, frame_id=frame_id)
        deadline = frame_deadline(frame, now, started, budget_s=budget, use_acquired=False)
        camera_evidence = None
        for slot in self.slots:
            if slot.support_view is None:
                raise ValueError('fresh registered support view required')
            k = slot.support_view.geometry.intrinsics
            if (depth.shape != (k.height,k.width) or not np.allclose(intr,[k.fx,k.fy,k.cx,k.cy],atol=1e-9,rtol=0)):
                raise ValueError('camera intrinsics changed after support calibration')
            evidence = FrameEvidence(self.camera_id,int(frame_id),stamp,context.transform_stamp,
                context.epoch,slot.support_view.registration.evidence.camera_revision,
                context.session_id,context.steady_domain,context.acquired_steady)
            if camera_evidence is not None and evidence != camera_evidence:
                raise ValueError('slot camera-frame evidence disagrees')
            camera_evidence = evidence
            slot.support_view.validate(depth,k,self.transform,evidence,now,
                                       bool(context.parked and motion.parked()),max_age=budget)
            if (not np.isfinite(motion.collision_time) or not 0 <= now-motion.collision_time <= .5
                    or (slot.support_view.clearance is not None
                        and motion.collision_time < slot.support_view.clearance.incorporated_time)):
                raise ValueError('collision map has not incorporated current clearance evidence')
            if (slot.support_z != slot.support_view.support_z
                    or slot.free_dimensions != (*slot.support_view.geometry.required_xy,
                        0. if slot.support_view.clearance is None else slot.support_view.clearance.height_m)
                    or slot.yaw_rad != slot.support_view.yaw_rad):
                raise ValueError('slot geometry differs from sensed support evidence')
            if (np.asarray(slot.label_mask).dtype != bool or np.shape(slot.label_mask) != depth.shape):
                raise ValueError('separate registered label mask required')
        marker_ids = {marker for s in self.slots for marker in s.support_view.registration.marker_ids}
        # fm466: a support carries TWO printed registration markers (support:<station>:0/1), but the
        # baseline only registers the markers that cleanly decoded AT COMMISSION TIME -- so a support
        # whose auxiliary marker was marginal when commissioned (e.g. at a dimmer commissioning
        # intensity_scale) stores only the primary in marker_ids. Once the mission lights brighter the
        # auxiliary decodes robustly and, under the all-support label search mask, lands in a calibrated
        # slot: it is neither in marker_ids nor a SKU profile, so it tripped 'unknown or undecoded label
        # in calibrated slot' (run-343). A decoded code in the reserved `support:` namespace is ALWAYS a
        # registration marker, never inventory, so skip it regardless of which subset the baseline
        # registered -- support identity never becomes a SKU (the invariant line 384 already asserts).
        qr = context.qr_observation
        if qr is None and reuse:
            qr = self._reuse_qr  # directly-decoded frame (no shared context) re-presented under stride
        if qr is None:
            if self.decoder is None:
                self.decoder = FixedTileQR(self.camera_id)
            qr = self.decoder.decode(rgb,camera_evidence,now=now,parked=bool(context.parked and motion.parked()))
        else:
            qr.validate(rgb,camera_evidence,now=now,parked=bool(context.parked and motion.parked()),max_age=budget)
        observations, rejected = [], []
        occluded_labels = set()        # calibrated slots whose label is occluded (held) this frame
        window = context.occlusion_window_s
        mask_shape = depth.shape
        candidates = [(d.text,np.asarray(d.corners)) for d in qr.detections]
        # fm467: the tiled decoder (FixedTileQR, 640 px tiles / 480 stride) can see one physical QR in
        # TWO tiles -- cleanly decoded in the tile that contains it whole, and as an UNREADABLE partial
        # quad at the adjacent tile seam. Under the all-support label mask that unreadable partial lands
        # in a calibrated slot and, having no SKU profile and no prior clean read, tripped 'unknown or
        # undecoded label' (run-344: the bottom-right support marker's seam partial). An unreadable quad
        # that spatially overlaps a cleanly DECODED quad is that same QR seen twice, never a distinct
        # label, so drop it. A GENUINE occlusion (gripper over the carton -> no decode for it) produces an
        # unreadable quad that overlaps NO decode, so it is kept and still handled by the occlusion window.
        def _decoded_box(poly):
            poly = np.asarray(poly, float)
            return (poly[:,0].min(), poly[:,1].min(), poly[:,0].max(), poly[:,1].max())
        _decoded_boxes = [_decoded_box(d.corners) for d in qr.detections]
        def _overlaps_decoded(poly):
            x0, y0, x1, y1 = _decoded_box(poly)
            for bx0, by0, bx1, by1 in _decoded_boxes:
                if x0 <= bx1 and bx0 <= x1 and y0 <= by1 and by0 <= y1:   # bbox intersection (same QR)
                    return True
            return False
        candidates.extend(('',np.asarray(p)) for p in qr.unreadable_corners
                          if not _overlaps_decoded(p))
        for index, (code, polygon) in enumerate(candidates):
            if code in marker_ids or (isinstance(code, str) and code.startswith('support:')):
                continue  # support identity never becomes inventory/SKU
            centre = np.mean(polygon, axis=0)
            ui,vi = np.rint(centre).astype(int)
            matches = [s for s in self.slots if 0 <= ui < depth.shape[1]
                       and 0 <= vi < depth.shape[0] and s.label_mask[vi,ui]]
            if not matches:
                continue
            if code not in self.profiles:
                # fm469: if the slot's carton SKU already decoded into an OBSERVATION this frame (the
                # candidate loop processes decoded detections before unreadable corners), then an
                # unreadable quad landing in that same slot is NOT the carton -- the carton decoded -- but
                # a spurious non-QR quad the detector mis-quads on a textured surface (run-346: the robot
                # gripper's fine grid at the top of the view + small stray quads, both caught by the
                # all-support label mask). The carton is already observed, so ignore it. ONLY when the
                # carton is NOT observed (genuine gripper occlusion of the label) does an unreadable quad
                # in the slot matter -> the occlusion window / strict-refuse path below. This is robust to
                # a spurious quad of ANY size or location (size/overlap filters are not), while preserving
                # the occlusion hold.
                if any(any(o.location == s.location for o in observations) for s in matches):
                    continue
                # The reaching arm/gripper occludes the carton label (on the carton top) during the
                # grasp, so it decodes as undecoded/unknown. While within the occlusion window since
                # the slot was last cleanly read, hold it (flag occluded) instead of refusing, so
                # /sensors keeps publishing and the controller can poll its locked grasp to
                # completion. A slot never cleanly read, or lost longer than the window, still refuses.
                tolerant = (window is not None and any(
                    s.location in self._label_clean and now-self._label_clean[s.location] <= window
                    for s in matches))
                if tolerant:
                    occluded_labels.update(s.location for s in matches)
                    continue
                # Decisive decode diagnostic (fires ONLY on this rare refusal, so always-on is cheap):
                # name WHY the label in a calibrated slot did not decode, so the root (too few px/module
                # i.e. framing/distance vs enough px but low contrast/blur vs a spurious non-QR quad vs
                # the markers decoded but the carton did not) is readable from the run log without a
                # saved sensing frame. `quad_px` is the failing polygon's pixel bbox; `code` is '' when it
                # came from an UNREADABLE quad (qr.unreadable_corners) vs a decoded-but-unknown text.
                try:
                    _poly = np.asarray(polygon, float)
                    _w = float(_poly[:, 0].max() - _poly[:, 0].min())
                    _h = float(_poly[:, 1].max() - _poly[:, 1].min())
                    _decoded = [getattr(d, 'text', '') for d in qr.detections]
                    _ever = {s.location: (s.location in self._label_clean) for s in matches}
                    print(
                        'DECODE_FAIL cam=%s slots=%s code=%r quad_px=%.0fx%.0f '
                        'decoded=%s n_unreadable=%d ever_clean=%s window=%s rgb=%sx%s' % (
                            self.camera_id, [s.location for s in matches], code, _w, _h,
                            _decoded, len(qr.unreadable_corners), _ever, window,
                            depth.shape[1], depth.shape[0]),
                        flush=True)
                    # fm465 (captain A): persist the ACTUAL sensing-camera RGB + the failing quad crop so
                    # the decode root can be SEEN (the walkthrough RGB is not the sensing frame). Opt-in via
                    # RESTOCK_DECODE_DUMP_DIR, bounded to the first few refusals so it never fills disk.
                    _dump = os.environ.get('RESTOCK_DECODE_DUMP_DIR')
                    if _dump:
                        n = getattr(self, '_decode_dump_count', 0)
                        if n < 6:
                            self._decode_dump_count = n + 1
                            os.makedirs(_dump, exist_ok=True)
                            bgr = np.ascontiguousarray(rgb[:, :, 2::-1]) if rgb.ndim == 3 and rgb.shape[2] >= 3 else rgb
                            tag = '%s_%s_%03d' % (self.camera_id, matches[0].location, n)
                            cv2.imwrite(os.path.join(_dump, 'decodefail_%s_full.png' % tag), bgr)
                            pad = 24
                            x0 = max(0, int(_poly[:, 0].min()) - pad); x1 = min(bgr.shape[1], int(_poly[:, 0].max()) + pad)
                            y0 = max(0, int(_poly[:, 1].min()) - pad); y1 = min(bgr.shape[0], int(_poly[:, 1].max()) + pad)
                            if x1 > x0 and y1 > y0:
                                cv2.imwrite(os.path.join(_dump, 'decodefail_%s_crop.png' % tag), bgr[y0:y1, x0:x1])
                            print('DECODE_FAIL dumped %s (quad bbox %d,%d..%d,%d) to %s' % (tag, x0, y0, x1, y1, _dump),
                                  flush=True)
                except Exception:
                    pass
                raise ValueError('unknown or undecoded label in calibrated slot')
            if len(matches) != 1:
                rejected.append('label outside unique calibrated slot')
                continue
            mask = np.zeros(mask_shape, np.uint8)
            cv2.fillConvexPoly(mask, np.rint(polygon).astype(np.int32), 1)
            try:
                obs = self.depth.observe(depth_m=depth, mask=mask,
                    intrinsics=intr, base_from_camera=self.transform,
                    classification=([(code,1.)], 'OK', 1., 1.),
                    observation_id=f'camera:{frame_id}:qr:{index}', simulation_time=stamp,
                    location=matches[0].location, rgb_time=stamp, depth_time=stamp, transform_time=stamp)
                angle = self._label_yaw(polygon, depth, intr)
                obs = replace(obs, pose=(*obs.pose[:3],0.,0.,math.sin(angle/2),math.cos(angle/2)))
                if abs(obs.pose[2]-matches[0].support_z-self.profiles[code].dimensions[2]) > .01:
                    raise ValueError('observed carton height differs from registered handling unit')
                # Verify the complete physical suction patch in depth, beyond the
                # small QR marker. A label floating over a gap cannot certify a seal.
                self._check_top_patch(obs, self.profiles[code], depth, intr)
                self._check_slot_footprint(obs,self.profiles[code],matches[0])
                observations.append(obs)
            except ValueError as exc:
                rejected.append(str(exc))
        targets, scanned = {}, []
        for slot in self.slots:
            mask = slot.support_view.mask(slot.support_view.geometry.intrinsics)
            cloud = masked_cloud(depth,mask,slot.support_view.geometry.intrinsics,self.transform)
            target = self.depth.target(depth_m=depth, mask=mask, intrinsics=intr,
                base_from_camera=self.transform, support_z=slot.support_z,
                free_dimensions=slot.free_dimensions, observation_id=f'camera:{frame_id}:slot:{slot.location}',
                simulation_time=stamp, location=slot.location)
            # The footprint centre comes from the measured quad/registration,
            # not the perspective-biased median of its image pixels.
            target = replace(target,clear=bool(slot.support_view.clearance is not None
                                               and np.all(np.abs(cloud[:,2]-slot.support_z) < .004)
                                               and not any(o.location == slot.location for o in observations)),
                pose=(*slot.support_view.center_base,0.,0.,math.sin(slot.yaw_rad/2),math.cos(slot.yaw_rad/2)))
            targets[slot.location] = target
            # Only a fully observed empty plane proves vacancy. A decoded carton
            # proves presence, not visibility of every hidden product behind it.
            if target.clear:
                scanned.append(slot.location)
            elif (slot.location == destination and not rejected and
                  sum(o.location == destination for o in observations) == 1
                  and self._known_foreground(slot, observations, depth, intr)):
                # Supported slice is a single dedicated carton slot, never a bin
                # of loose/stacked products. Target verification checks carton pose.
                scanned.append(slot.location)
        candidates = [o for o in observations if o.location == source]
        if len(candidates) == 1:
            motion.observe_source(candidates[0])
        # Seed/refresh the per-slot clean-read time for the label occlusion window: a slot is clean
        # this frame when its support is not held AND no label in it was occluded (held).
        for slot in self.slots:
            if not slot.support_view.occluded and slot.location not in occluded_labels:
                self._label_clean[slot.location] = now
        # The acquisition must still be fresh now that processing is done; refuse before publishing
        # any evidence/bbox (they stay empty on expiry) so a stale frame never reaches the mission.
        require_fresh(deadline, self.steady_clock, wall_started=wall_started,
                      wall_backstop_s=self.wall_backstop_s, wall_clock=self.wall_clock)
        self.sensed_target_bboxes = tuple(self._object_bbox(o,self.profiles[o.sku],intr,depth.shape)
                                          for o in observations)
        self.evidence = tuple(dict(location=s.location,
            baseline_calibration_id=s.support_view.geometry.calibration_id,
            baseline_stamp=s.support_view.geometry.baseline_stamp,
            baseline_session_id=s.support_view.geometry.baseline_session_id,
            baseline_acquired_steady=s.support_view.geometry.baseline_acquired_steady,
            current_frame_id=frame_id,current_stamp=stamp,epoch=context.epoch,
            registration_uncertainty_m=s.support_view.uncertainty_m,
            clearance_observation_id=None if s.support_view.clearance is None else s.support_view.clearance.observation_id,
            qr_processing_seconds=qr.processing_seconds,qr_rgb_sha256=qr.rgb_sha256,
            registration_occluded=bool(getattr(s.support_view,'occluded',False) or s.location in occluded_labels),
            provenance='dated_rigid_geometry_with_fresh_marker_witness_and_depth') for s in self.slots)
        # Record the just-validated frame so a stride-paced re-presentation of these exact pixels
        # reuses this decode (only now that every gate has passed; a failed frame stays unreusable).
        self._reuse_key, self._reuse_context, self._reuse_qr = key, context, qr
        # Snapshot this validated registration so the opt-in sensing gate can republish it (flagged
        # occluded) while the render is suspended for a controller execution phase. Guarded by the
        # opt-in flag so a default (gate-off) run is byte-identical. clean_now anchors the hold window.
        if self.execution_gate:
            self._held_source = dict(
                observations=tuple(observations), target=targets.get(destination),
                scanned=tuple(scanned), scan_time=stamp, evidence=self.evidence,
                clean_now=now, window=context.occlusion_window_s,
                parked=bool(context.parked and motion.parked()))
        return Sensors(now, tuple(observations), targets.get(destination),
                       motion.parked(), contact.read(), now, motion.collision_time,
                       motion.joint_positions(), tuple(scanned), stamp, deadline)

    def _known_foreground(self, slot, observations, depth, intr):
        mask = slot.support_view.mask(slot.support_view.geometry.intrinsics)
        cloud = masked_cloud(depth,mask,slot.support_view.geometry.intrinsics,self.transform)
        foreground = cloud[np.abs(cloud[:,2]-slot.support_z) >= .004]
        explained = np.zeros(len(foreground), bool)
        for obs in observations:
            if obs.location != slot.location:
                continue
            size = np.asarray(self.profiles[obs.sku].dimensions)
            centre = np.asarray(obs.pose[:3]) - [0,0,size[2]/2]
            angle = 2*math.atan2(obs.pose[5],obs.pose[6])
            c,s = math.cos(angle),math.sin(angle)
            local = (foreground-centre) @ np.array([[c,-s,0],[s,c,0],[0,0,1]])
            explained |= np.all(np.abs(local) <= size/2+.01, axis=1)
        return bool(explained.all())

    def _check_slot_footprint(self, observation, profile, slot):
        x,y,z = observation.pose[:3]
        yaw = 2*math.atan2(observation.pose[5],observation.pose[6])
        c,s = math.cos(yaw),math.sin(yaw)
        points = np.array([[x+c*dx-s*dy,y+s*dx+c*dy,z]
            for dx,dy in [(-profile.dimensions[0]/2,-profile.dimensions[1]/2),
                          (profile.dimensions[0]/2,-profile.dimensions[1]/2),
                          (profile.dimensions[0]/2,profile.dimensions[1]/2),
                          (-profile.dimensions[0]/2,profile.dimensions[1]/2)]])
        inverse = np.linalg.inv(slot.support_view.registration.base_from_support)
        local = points @ inverse[:3,:3].T+inverse[:3,3]
        if not contains(np.asarray(slot.support_view.geometry.polygon_support)[:,:2],local[:,:2],
                        slot.support_view.uncertainty_m):
            raise ValueError('sensed carton footprint outside the measured slot')

    def _object_bbox(self, observation, profile, intr, shape):
        """Project accepted sensed carton volume, never a support/label ROI.

        This is a catalog-sized box anchored to fresh QR/depth pose. Consumers
        must choose only the active target and retain mapper coverage/ack gates.
        """
        x,y,z = observation.pose[:3]
        yaw = 2*math.atan2(observation.pose[5],observation.pose[6])
        c,s = math.cos(yaw),math.sin(yaw)
        points = np.array([[x+c*dx-s*dy,y+s*dx+c*dy,z-dz]
            for dx in (-profile.dimensions[0]/2,profile.dimensions[0]/2)
            for dy in (-profile.dimensions[1]/2,profile.dimensions[1]/2)
            for dz in (0,profile.dimensions[2])])
        transform = np.linalg.inv(self.transform)
        p = points @ transform[:3,:3].T+transform[:3,3]
        if np.any(p[:,2] <= 0):
            raise ValueError('sensed object bounding box behind camera')
        uv = p[:,:2]/p[:,2,None]*intr[:2]+intr[2:]
        lo,hi = np.floor(uv.min(axis=0)).astype(int),np.ceil(uv.max(axis=0)).astype(int)+1
        if np.any(lo < 0) or np.any(hi > [shape[1],shape[0]]):
            raise ValueError('sensed object bounding box truncated')
        return dict(camera_id=self.camera_id,observation_id=observation.observation_id,
                    stamp=observation.simulation_time,location=observation.location,
                    roi=tuple(map(int,[*lo,*hi])),source='sensor_pose_registered_catalog_volume')

    def _label_yaw(self, polygon, depth, intr):
        """Fit the carton-top QR yaw from ALL FOUR decoded corners, not one top edge.

        The printed label is a square centred on the carton top (marker axes +X right,
        +Y up, +Z out of the top). Each corner is backprojected with a 3x3 depth patch
        that rejects a corner straddling a depth discontinuity (the carton rim / a gap),
        then a single in-plane rotation is fit to the four map-frame corners by orthogonal
        Procrustes -- the same rigid fit ``sensed.register_marker`` uses for supports -- and
        its yaw is the carton yaw. Using four neighbourhood-sampled corners averages
        per-corner pixel/depth quantisation and refuses a mis-sampled corner, so one noisy
        or boundary corner cannot rotate the fitted yaw by degrees the way the former two
        single-pixel top-edge corners could (run-177 collect seal-alignment hardening).
        The returned angle is the SAME marker-+X direction in the map frame as before: on a
        clean flat top the four-corner fit and the old top edge agree to < 0.05 deg.
        """
        fx,fy,cx,cy = intr
        depth = np.asarray(depth)
        corners = []
        for u,v in polygon:
            ui,vi = int(round(u)),int(round(v))
            if not (1 <= ui < depth.shape[1]-1 and 1 <= vi < depth.shape[0]-1):
                raise ValueError('QR corner outside depth image')
            patch = depth[vi-1:vi+2,ui-1:ui+2]
            if not np.isfinite(patch).all() or np.any(patch <= 0):
                raise ValueError('QR orientation lacks valid depth')
            if np.ptp(patch) > .015:
                raise ValueError('QR corner straddles a depth discontinuity')
            z = float(depth[vi,ui])
            corners.append((self.transform @ np.array([(ui-cx)*z/fx,(vi-cy)*z/fy,z,1.]))[:3])
        corners = np.asarray(corners)
        if np.linalg.norm(np.ptp(corners[:,:2],axis=0)) < .01:
            raise ValueError('QR top orientation is uncertain/tilted')
        # A flat carton top keeps the four corners within a few mm in world Z; constrain to
        # gravity exactly as the former abs(edge_z) edge-tilt guard did (dense top-patch depth
        # independently re-checks this plane in _check_top_patch).
        if np.ptp(corners[:,2]) > .006:
            raise ValueError('QR top orientation is uncertain/tilted')
        model = np.array([[-1,1],[1,1],[1,-1],[-1,-1]],float)   # TL,TR,BR,BL, marker +X right/+Y up
        xy = corners[:,:2]
        model_center,center = model.mean(0),xy.mean(0)
        u,_,vt = np.linalg.svd((model-model_center).T @ (xy-center))
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0:
            vt[-1] *= -1
            rotation = vt.T @ u.T
        scale = np.linalg.norm(xy-center,axis=1).sum()/np.linalg.norm(model-model_center,axis=1).sum()
        residual = float(np.max(np.linalg.norm((model-model_center)*scale @ rotation.T+center-xy,axis=1)))
        if not np.isfinite(residual) or residual > .01:
            raise ValueError('QR top orientation is uncertain/tilted')  # corners are not a rigid square
        return math.atan2(rotation[1,0],rotation[0,0])

    def _check_top_patch(self, observation, profile, depth, intr):
        # A dense interior grid requires depth agreement over the registered
        # carton footprint. It also rejects tilted/small/occluded surfaces.
        fx,fy,cx,cy = intr
        camera_from_base = np.linalg.inv(self.transform)
        x,y,z = observation.pose[:3]
        angle = 2*math.atan2(observation.pose[5], observation.pose[6])
        c,s = math.cos(angle),math.sin(angle)
        # QR centre is mechanically centred on the top; unknown mounting is unsupported.
        for dx in np.linspace(-profile.dimensions[0]/2+.01, profile.dimensions[0]/2-.01, 9):
            for dy in np.linspace(-profile.dimensions[1]/2+.01, profile.dimensions[1]/2-.01, 7):
                p = camera_from_base @ np.array([x+c*dx-s*dy,y+s*dx+c*dy,z,1.])
                if p[2] <= 0:
                    raise ValueError('carton top behind camera')
                u,v = int(round(fx*p[0]/p[2]+cx)), int(round(fy*p[1]/p[2]+cy))
                if not (0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]):
                    raise ValueError('carton top outside depth image')
                if not np.isfinite(depth[v,u]) or abs(depth[v,u]-p[2]) > .004:
                    raise ValueError('carton top patch occluded, tilted or smaller than catalog')


def write_carton_label(sku, output_path, *, pixels=512):
    """Generate the actual scannable top texture; caller applies it to the carton.

    The label must be centred, flat, visible and unwarped on the carton top. This
    writes a PNG only; it never injects a decoded observation or semantic label.
    """
    import cv2
    if not sku or not isinstance(sku, str) or pixels < 128:
        raise ValueError('SKU and label resolution required')
    qr = cv2.QRCodeEncoder_create().encode(sku)
    qr = cv2.copyMakeBorder(qr, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)
    qr = cv2.resize(qr, (pixels,pixels), interpolation=cv2.INTER_NEAREST)
    if not cv2.imwrite(str(output_path), qr):
        raise OSError('could not write carton label PNG')
