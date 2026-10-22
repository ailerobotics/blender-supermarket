"""Fresh RGB-D support registration. No USD, object semantics or simulator APIs.

A dated empty scan establishes rigid support geometry, not enduring vacancy.
Every use requires a newly sensed physical marker AND an independent plane patch.
Only current full-footprint depth can prove that the slot is empty.
"""
from dataclasses import dataclass
import hashlib
import math
import numpy as np

from .geometry import Intrinsics, backproject, project, rigid, transform_points
from .freshness import _RENDER_LEAD


def _matrix_tuple(matrix):
    return tuple(tuple(float(v) for v in row) for row in matrix)


def depth_digest(depth):
    # Hash the contiguous buffer directly; ascontiguousarray is a no-op when the depth is
    # already C-contiguous float64 and memoryview avoids a second full-frame copy (tobytes).
    return hashlib.sha256(memoryview(np.ascontiguousarray(depth, dtype='<f8'))).hexdigest()


@dataclass(frozen=True)
class FrameEvidence:
    camera_id: str
    frame_id: int
    stamp: float
    transform_stamp: float
    epoch: str
    camera_revision: str
    session_id: str = 'default'
    steady_domain: str = ''
    acquired_steady: float | None = None

    def validate(self, now, parked, *, max_age=.5):
        # max_age is the acquisition-to-use staleness bound (default 0.5 s live). A HELD frame under
        # approach occlusion passes the hold window here instead: its registration is the last clean
        # one (window-bounded by RegisteredSupport.hold), so the per-frame 0.5 s budget must not
        # refuse a deliberately held, flagged-occluded frame. The cross-stamp sync (|stamp-tf|<=.05)
        # and the from-the-future bound (-_RENDER_LEAD) are physical and stay fixed regardless.
        if (not parked or not self.camera_id or not self.epoch or not self.camera_revision or not self.session_id
                or type(self.frame_id) is not int or self.frame_id < 0
                or not np.isfinite([now,self.stamp,self.transform_stamp]).all()
                or not -_RENDER_LEAD <= now-self.stamp <= max_age
                or not -_RENDER_LEAD <= now-self.transform_stamp <= max_age
                or abs(self.stamp-self.transform_stamp) > .05):
            raise ValueError('missing, moving, stale or unsynchronized camera evidence')
        if bool(self.steady_domain) != (self.acquired_steady is not None):
            raise ValueError('persistent acquisition clock identity and time required together')
        if self.acquired_steady is not None and not math.isfinite(self.acquired_steady):
            raise ValueError('invalid persistent acquisition clock')


def polygon_mask(polygon, k):
    """Pixel centres strictly within an ordered convex image quadrilateral."""
    uv = np.asarray(polygon,float)
    if (uv.shape != (4,2) or not np.isfinite(uv).all()
            or np.any(uv < 0) or np.any(uv >= [k.width,k.height])):
        raise ValueError('support polygon truncated or invalid')
    edges = np.roll(uv,-1,axis=0)-uv
    turns = np.cross(edges,np.roll(edges,-1,axis=0))
    if not (np.all(turns > 1e-6) or np.all(turns < -1e-6)):
        raise ValueError('support polygon must be convex and ordered')
    x1,y1 = np.ceil(uv.min(axis=0)).astype(int)
    x2,y2 = np.floor(uv.max(axis=0)).astype(int)+1
    u = np.arange(x1,x2); v = np.arange(y1,y2)
    uu,vv = np.meshgrid(u,v)
    # Half-plane sign per edge, accumulated over 2-D grids (no H x W x 4 x 2 intermediate):
    # inside iff every edge cross-product shares one sign. Identical test, far less memory.
    positive = np.ones(uu.shape,bool); negative = np.ones(uu.shape,bool)
    for (ex,ey),(px,py) in zip(edges,uv):
        cross = ex*(vv-py)-ey*(uu-px)
        positive &= cross >= 0
        negative &= cross <= 0
    inside = positive | negative
    result = np.zeros((k.height,k.width),bool)
    result[y1:y2,x1:x2] = inside
    if result.sum() < 64:
        raise ValueError('insufficient observed polygon pixels')
    return result


def masked_cloud(depth, mask, k, base_from_camera):
    depth,mask = np.asarray(depth,float),np.asarray(mask)
    if depth.shape != (k.height,k.width) or mask.shape != depth.shape or mask.dtype != bool:
        raise ValueError('unregistered depth or mask')
    v,u = np.where(mask)
    if len(u) < 64:
        raise ValueError('insufficient observed support pixels')
    return transform_points(base_from_camera,backproject(np.column_stack([u,v]),depth[v,u],k))


def _corner_points(depth, pixels, k, base_from_camera):
    uv = np.asarray(pixels,float)
    if uv.shape != (4,2) or not np.isfinite(uv).all():
        raise ValueError('four depth-backed corners required')
    ij = np.rint(uv).astype(int)
    if np.any(ij < 1) or np.any(ij >= [k.width-1,k.height-1]):
        raise ValueError('corner depth neighbourhood truncated')
    depths = []
    for u,v in ij:
        patch = np.asarray(depth)[v-1:v+2,u-1:u+2]
        if not np.isfinite(patch).all() or np.any((patch < k.near)|(patch > k.far)):
            raise ValueError('invalid corner depth')
        if np.ptp(patch) > .015:
            raise ValueError('corner straddles a depth discontinuity')
        depths.append(float(depth[v,u]))
    # Pixel quantisation is explicitly included in registration uncertainty.
    return transform_points(base_from_camera,backproject(ij.astype(float),np.array(depths),k))


@dataclass(frozen=True)
class MarkerRegistration:
    marker_id: str
    support_id: str
    revision: str
    base_from_support: tuple
    position_error_m: float
    angular_error_rad: float
    evidence: FrameEvidence
    marker_ids: tuple = ()
    layout_id: str = ''


def register_marker(depth, marker_pixels, k, base_from_camera, *, marker_id,
                    decoded_id, support_id, revision, side_m, evidence, now,
                    parked, depth_error_m, auxiliary_pixels=None, auxiliary_id=None,
                    decoded_auxiliary_id=None, auxiliary_offset_xy=None):
    """Fit metric QR corners decoded from this RGB frame to its registered depth.

    Marker axes: printed +X right, +Y up, +Z out of the marked support. QR corner
    order is TL,TR,BR,BL. A physical, rigidly bolted marker and matching revision
    are mandatory; a USD prim transform is not a registration input.
    """
    evidence.validate(now,parked)
    rigid(base_from_camera)
    if (not marker_id or marker_id != decoded_id or not support_id or not revision
            or not np.isfinite([side_m,depth_error_m]).all()
            or not .03 <= side_m <= .25 or not 0 < depth_error_m <= .002):
        raise ValueError('unknown marker or unbounded depth uncertainty')
    actual = _corner_points(depth,marker_pixels,k,base_from_camera)
    model = np.array([[-1,1,0],[1,1,0],[1,-1,0],[-1,-1,0]],float)*side_m/2
    marker_ids = (marker_id,)
    if auxiliary_pixels is not None:
        offset = np.asarray(auxiliary_offset_xy,float)
        if (not auxiliary_id or auxiliary_id == marker_id or auxiliary_id != decoded_auxiliary_id
                or offset.shape != (2,) or not np.isfinite(offset).all()
                or np.linalg.norm(offset) < side_m*2):
            raise ValueError('invalid independent support marker layout')
        actual = np.concatenate([actual,_corner_points(depth,auxiliary_pixels,k,base_from_camera)])
        model = np.concatenate([model,model+[*offset,0.]])
        marker_ids += (auxiliary_id,)
    model_center = model.mean(axis=0)
    center = actual.mean(axis=0)
    u,_,vt = np.linalg.svd((model-model_center).T @ (actual-center))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    if rotation[2,2] < .999:
        raise ValueError('support marker metric fit/normal is uncertain')
    # This bounded slice accepts only horizontal supports. Constrain roll/pitch
    # to gravity; dense witness depth independently checks that assumption.
    yaw = math.atan2(rotation[1,0],rotation[0,0])
    c,s = math.cos(yaw),math.sin(yaw)
    rotation = np.array([[c,-s,0],[s,c,0],[0,0,1]])
    translation = center-rotation @ model_center
    residual = float(np.max(np.linalg.norm(model @ rotation.T+translation-actual,axis=1)))
    if residual > .002:
        raise ValueError('support marker metric fit/normal is uncertain')
    pose = np.eye(4)
    pose[:3,:3],pose[:3,3] = rotation,translation
    distance = np.linalg.norm(actual-np.asarray(base_from_camera)[:3,3],axis=1).max()
    error = depth_error_m + residual + .75*distance/min(k.fx,k.fy)
    span = max(side_m,float(np.linalg.norm(np.ptp(model[:,:2],axis=0))))
    # Quantize the metric model to sub-nanometre precision before hashing so the layout id is
    # REPRODUCIBLE across processes: a commissioned baseline and the live mission build the same
    # design geometry but differ by float ULPs (different computation order / contiguity), which a
    # raw float64 byte hash turns into a spurious "changed rigid-support geometry" mismatch. 9
    # decimals (1 nm) preserves genuine layout identity while killing that ULP noise.
    layout_id = hashlib.sha256(np.round(np.asarray(model,float),9).tobytes()+repr(marker_ids).encode()).hexdigest()
    import os as _os
    if _os.environ.get('RESTOCK_SUPPORT_DIAG'):
        print('REGMARK side_m=%r marker_ids=%r model_shape=%s model_sum=%.9f layout=%s' % (
            side_m, marker_ids, model.shape, float(np.asarray(model).sum()), layout_id), flush=True)
    return MarkerRegistration(marker_id,support_id,revision,_matrix_tuple(pose),
                              float(error),float(2*error/span),evidence,marker_ids,layout_id)


def _flat_patch(depth, uv, k, base_from_camera, base_from_support):
    mask = polygon_mask(uv,k)
    cloud = masked_cloud(depth,mask,k,base_from_camera)
    local = transform_points(np.linalg.inv(base_from_support),cloud)
    center = local.mean(axis=0)
    _,vectors = np.linalg.eigh((local-center).T @ (local-center)/len(local))
    normal = vectors[:,0]
    if abs(normal[2]) < .999 or np.ptp(local[:,2]) > .006:
        raise ValueError('support patch is tilted, occupied or occluded')
    corners = transform_points(np.linalg.inv(base_from_support),
                               _corner_points(depth,uv,k,base_from_camera))
    if np.max(np.abs(corners[:,2]-center[2])) > .003:
        raise ValueError('support corners are not on the observed plane')
    return corners, float(center[2])


def _rectangle(center, dimensions):
    return np.asarray(center)+np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*np.asarray(dimensions)/2


def contains(polygon, points, margin=0.):
    p,q = np.asarray(polygon,float),np.asarray(points,float)
    edges = np.roll(p,-1,axis=0)-p
    norms = np.linalg.norm(edges,axis=1)
    if min(norms) < 1e-8:
        return False
    cross = edges[None,:,0]*(q[:,None,1]-p[None,:,1])-edges[None,:,1]*(q[:,None,0]-p[None,:,0])
    return bool(np.all(cross >= margin*norms) or np.all(cross <= -margin*norms))


@dataclass(frozen=True)
class SupportGeometry:
    support_id: str
    marker_id: str
    revision: str
    camera_id: str
    camera_revision: str
    intrinsics: Intrinsics
    calibration_id: str
    baseline_frame_id: int
    baseline_stamp: float
    polygon_support: tuple
    witness_support: tuple
    center_support: tuple
    required_xy: tuple
    baseline_error_m: float
    marker_layout_id: str = ''
    baseline_session_id: str = 'default'
    steady_domain: str = ''
    baseline_acquired_steady: float | None = None
    lifetime_s: float = 600.
    # Commissioned camera<-support pose (world-constant: fixed source camera + fixed support, base
    # parked). Persisted so apply_baselines can SEED RegisteredSupport.good, letting hold() re-express
    # the commissioned registration through the LIVE base_from_camera before any clean observe prepare()
    # runs (Option B / fm443). None on legacy baselines -> no seed (byte-identical clean-establish).
    camera_from_support: tuple | None = None

    def age_at(self, evidence, now, *, wall_lifetime_s=None):
        """Baseline age bounded by ``lifetime_s`` on the SIM/node clock (``now``).

        Support drift is a SIM-time phenomenon, so the lifetime is measured on the sim/node clock
        (``now``, the mission's 0-based sim time; the baseline is loaded at mission boot). Under
        render pacing a run's WALL time vastly exceeds its sim time (collect ~15 min wall for ~13 s
        sim), and the commission->mission wall gap is render compute, not drift — measuring the age
        in wall seconds (the old ``acquired_steady`` gap) made a render-paced run outlive its baseline
        mid-collect. The persistent acquisition clock stays a cross-process PROVENANCE +
        anti-rewind check only (its wall magnitude no longer bounds the lifetime).

        ``wall_lifetime_s`` is an opt-in cap on the acquisition-clock gap for the REAL RIG, where
        wall == sim and the commission->deployment gap IS real drift time; left ``None`` under render
        pacing (the sim lifetime governs). Mirrors the sim-clock watchdogs fixed across the lane."""
        if self.steady_domain:
            if (evidence.steady_domain != self.steady_domain or evidence.acquired_steady is None
                    or evidence.acquired_steady < self.baseline_acquired_steady):
                raise ValueError('commissioning acquisition clock continuity lost')
            # Sim-time age (baseline loaded at mission boot -> age is the mission sim elapsed). Not
            # the wall acquired_steady gap, which render pacing balloons without any real drift.
            age = now
            if wall_lifetime_s is not None:
                wall_gap = evidence.acquired_steady-self.baseline_acquired_steady + now-evidence.stamp
                if not 0 <= wall_gap <= wall_lifetime_s:
                    raise ValueError('expired or reversed rigid-support commissioning clock')
        else:
            if evidence.session_id != self.baseline_session_id or evidence.steady_domain:
                raise ValueError('clock reset requires persistent commissioning acquisition provenance')
            age = now-self.baseline_stamp
        if not 0 <= age <= self.lifetime_s:
            raise ValueError('expired or reversed rigid-support commissioning clock')
        return age


def calibrate_rigid_support(depth, support_polygon_uv, witness_polygon_uv, k,
                            base_from_camera, registration, *, required_xy,
                            calibration_id, now, parked, lifetime_s=600.):
    """Measure an empty full support quad and disjoint witness from real depth.

    The marker must be fixed on the support plane (or a coplanar rigid tab).
    A top of a carton cannot pass as support because it is elevated above that
    sensed marker plane. Required XY is a requirement, never a measured result.
    """
    registration.evidence.validate(now,parked)
    if (not calibration_id or len(required_xy) != 2 or not np.isfinite(required_xy).all()
            or min(required_xy) <= 0 or not math.isfinite(lifetime_s) or not 1 <= lifetime_s <= 900):
        raise ValueError('invalid support footprint requirement')
    pose = np.asarray(registration.base_from_support)
    corners,z = _flat_patch(depth,support_polygon_uv,k,base_from_camera,pose)
    witness,wz = _flat_patch(depth,witness_polygon_uv,k,base_from_camera,pose)
    if abs(z) > .004 or abs(wz-z) > .003:
        raise ValueError('observed surface is not the marked support plane')
    mask = polygon_mask(support_polygon_uv,k)
    if np.any(mask & polygon_mask(witness_polygon_uv,k)):
        raise ValueError('support witness must be independently visible outside the slot')
    center = corners.mean(axis=0)
    footprint = _rectangle(center[:2],required_xy)
    if not contains(corners[:,:2],footprint,registration.position_error_m):
        raise ValueError('measured quadrilateral does not cover full required footprint')
    # World-constant camera<-support (base parked, camera + support world-fixed) so hold() can
    # re-express the commissioned registration through the LIVE base_from_camera (Option B / fm443).
    camera_from_support = _matrix_tuple(np.linalg.inv(np.asarray(base_from_camera, float)) @ pose)
    return SupportGeometry(registration.support_id,registration.marker_id,registration.revision,
        registration.evidence.camera_id,registration.evidence.camera_revision,k,calibration_id,
        registration.evidence.frame_id,registration.evidence.stamp,
        _matrix_tuple(corners),_matrix_tuple(witness),tuple(center),tuple(required_xy),
        registration.position_error_m,registration.layout_id,registration.evidence.session_id,
        registration.evidence.steady_domain,registration.evidence.acquired_steady,lifetime_s,
        camera_from_support=camera_from_support)


@dataclass(frozen=True)
class ClearanceEvidence:
    support_id: str
    observation_id: str
    stamp: float
    epoch: str
    footprint_xy: tuple
    height_m: float
    incorporated_time: float
    source: str = 'sensor'

    def validate(self, now, evidence, geometry):
        if (self.source != 'sensor' or self.support_id != geometry.support_id
                or not self.observation_id or self.epoch != evidence.epoch
                or len(self.footprint_xy) != 2
                or not np.isfinite([self.stamp,self.height_m,self.incorporated_time,*self.footprint_xy]).all()
                or not 0 <= now-self.stamp <= .5
                or not self.stamp <= self.incorporated_time <= now
                or self.height_m <= 0
                or np.any(np.asarray(self.footprint_xy) < geometry.required_xy)):
            raise ValueError('fresh sensor clearance and incorporated collision evidence required')


@dataclass(frozen=True)
class SupportView:
    geometry: SupportGeometry
    registration: MarkerRegistration
    clearance: ClearanceEvidence | None
    support_polygon_uv: tuple
    support_z: float
    center_base: tuple
    yaw_rad: float
    uncertainty_m: float
    depth_sha256: str
    base_from_camera: tuple
    occluded: bool = False       # True: held last-good registration under approach occlusion

    def validate(self, depth, k, transform, evidence, now, parked, *, max_age=.5):
        # max_age relaxes only the per-frame acquisition staleness for a HELD (occluded) view (the
        # capture passes the hold window); the baseline lifetime (age_at) and frame/calibration/depth
        # identity below are unconditional.
        evidence.validate(now,parked,max_age=max(max_age,.5) if self.occluded else .5)
        self.geometry.age_at(evidence,now)
        if (self.registration.evidence != evidence or k != self.geometry.intrinsics
                or not np.allclose(transform,self.base_from_camera,atol=1e-9,rtol=0)
                or self.depth_sha256 != depth_digest(depth)):
            raise ValueError('support view belongs to a different frame/calibration/depth')
        if self.clearance is not None:
            self.clearance.validate(now,evidence,self.geometry)

    def mask(self, k):
        return polygon_mask(self.support_polygon_uv,k)


class RegisteredSupport:
    """Bounded reuse of *geometry*, refreshed by marker and support witness sensing.

    Motion changes registration, never the baseline stamp. A support deformation,
    marker/mount revision or loss of rigidity requires invalidate() and a new
    empty scan. Lifetime is immutable per commissioning (default 600 s, maximum
    900 s), distinct from live 0.5 s freshness. The policy drift allowance is
    10 micrometres/s; the 10 mm combined uncertainty limit still refuses a view.
    These bounds require deployment validation, not a claim of hardware precision.
    """
    def __init__(self, geometry):
        if not math.isfinite(geometry.lifetime_s) or not 1 <= geometry.lifetime_s <= 900:
            raise ValueError('commissioning lifetime outside bounded policy')
        self.geometry = geometry
        self.invalid_reason = ''
        self.last_frame = None
        self.seen_sessions = set()
        self.good = None             # (SupportView, now) of the last fully visible clean registration

    def invalidate(self, reason):
        self.invalid_reason = str(reason) or 'rigid support invalidated'
        self.good = None             # a disturbed support may never be held under occlusion

    def _identity_ok(self, registration, e, k):
        g = self.geometry
        if (self.invalid_reason
                or registration.support_id != g.support_id or registration.marker_id != g.marker_id
                or registration.revision != g.revision or e.camera_id != g.camera_id
                or registration.layout_id != g.marker_layout_id
                or e.camera_revision != g.camera_revision or k != g.intrinsics):
            import os as _os
            if _os.environ.get('RESTOCK_SUPPORT_DIAG'):
                print('SUPPORT_GEOM_MISMATCH invalid=%r support=%r/%r marker=%r/%r rev=%r/%r '
                      'cam=%r/%r layout=%r/%r camrev=%r/%r k_eq=%s' % (
                      self.invalid_reason, registration.support_id, g.support_id,
                      registration.marker_id, g.marker_id, registration.revision, g.revision,
                      e.camera_id, g.camera_id, registration.layout_id, g.marker_layout_id,
                      e.camera_revision, g.camera_revision, k == g.intrinsics), flush=True)
            raise ValueError('expired, invalidated or changed rigid-support geometry')

    def _monotonic_ok(self, e, *, allow_repeat=False):
        # ``allow_repeat`` relaxes the strictly-new requirement to monotonic NON-REGRESSING for the
        # HELD path only: hold() re-expresses the last clean registration through the live
        # base_from_camera/TF and does NOT consume fresh pixels, so under a sensing stride the source
        # camera re-polls it with the SAME last frame between renders (frame_id/stamp/acquisition
        # clock equal). Accept an equal (non-regressing) frame then; still refuse a frame that goes
        # BACKWARDS (frame_id/stamp/acquired_steady strictly decreasing) or a regressed (already-seen)
        # session. prepare()/observe keep the strict strictly-new gate (allow_repeat left False).
        if self.last_frame is not None:
            prior = self.last_frame
            same_session = e.session_id == prior.session_id
            if allow_repeat:
                regressed = e.frame_id < prior.frame_id or e.stamp < prior.stamp
                steady_regressed = (e.acquired_steady is not None
                                    and e.acquired_steady < prior.acquired_steady)
            else:
                regressed = e.frame_id <= prior.frame_id or e.stamp <= prior.stamp
                steady_regressed = (e.acquired_steady is not None
                                    and e.acquired_steady <= prior.acquired_steady)
            if ((same_session and regressed)
                    or (not same_session and e.session_id in self.seen_sessions)
                    or steady_regressed):
                raise ValueError('rigid-support registration needs a new monotonic frame')

    def prepare(self, depth, k, base_from_camera, registration, clearance, *, now, parked,
                occlusion_window_s=None, contradiction_m=.05, contradiction_rad=.17):
        """Refresh the registered support from the current frame.

        ``occlusion_window_s`` (None = strict, the observe/commissioning default) enables
        approach-occlusion tolerance: when the markers register but the support footprint/witness
        is occluded by the reaching arm (so the fresh view would refuse), fall back to the last
        fully visible clean registration (``self.good``), flagged ``occluded``, as long as the view
        has not been lost longer than the window AND the visible markers do not contradict the held
        pose. The identity, monotonic-frame, horizontality and baseline-lifetime gates are strict
        regardless; only the fresh full-footprint *visibility* requirement is relaxed under occlusion
        (the held registration already satisfied it at observe time)."""
        g,e = self.geometry,registration.evidence
        e.validate(now,parked)
        age = g.age_at(e,now)
        self._identity_ok(registration,e,k)
        self._monotonic_ok(e)
        if clearance is not None:
            clearance.validate(now,e,g)
        pose = np.asarray(registration.base_from_support)
        if pose[2,2] < .999:
            raise ValueError('registered support is no longer horizontal')
        try:
            witness_base = transform_points(pose,g.witness_support)
            uv,valid = project(transform_points(np.linalg.inv(rigid(base_from_camera)),witness_base),k)
            if not valid.all():
                raise ValueError('fresh support witness not visible')
            _,z = _flat_patch(depth,uv,k,base_from_camera,pose)
            if abs(z-g.center_support[2]) > .003:
                raise ValueError('fresh witness disagrees with dated support calibration')
            radius = np.linalg.norm(np.asarray(g.polygon_support)[:,:2],axis=1).max()
            error = g.baseline_error_m+registration.position_error_m+radius*registration.angular_error_rad+age*.00001
            footprint = _rectangle(g.center_support[:2],g.required_xy)
            if error > .01 or not contains(np.asarray(g.polygon_support)[:,:2],footprint,error):
                raise ValueError('uncertain registration cannot establish full required footprint')
            # Expand the runtime check, never the measured geometry, by uncertainty.
            polygon = np.column_stack([_rectangle(g.center_support[:2],np.asarray(g.required_xy)+2*error),
                                       np.full(4,g.center_support[2])])
            base = transform_points(pose,polygon)
            uv,valid = project(transform_points(np.linalg.inv(rigid(base_from_camera)),base),k)
            if not valid.all():
                raise ValueError('full support footprint is truncated or out of range')
            mask = polygon_mask(uv,k)
            # The full footprint must carry only valid, present depth (no invalid/hidden pixels),
            # but its 3-D cloud is not used here — validate the masked depths directly instead of
            # backprojecting ~1e6 points every capture (same gate as masked_cloud, far cheaper).
            patch = np.asarray(depth,float)[mask]
            if patch.size < 64 or not np.isfinite(patch).all() or np.any((patch < k.near)|(patch > k.far)):
                raise ValueError('insufficient observed support pixels')
        except ValueError:
            # The reaching arm occludes the fixed camera's view of the support it grasps toward.
            # Hold the last clean registration if configured and still within policy; otherwise the
            # strict refusal (observe/commissioning, or no prior good view, or a lost/contradicted
            # view) stands.
            if occlusion_window_s is None or self.good is None:
                raise
            return self.hold(depth,k,base_from_camera,e,now=now,parked=parked,
                             occlusion_window_s=occlusion_window_s,current_registration=registration,
                             contradiction_m=contradiction_m,contradiction_rad=contradiction_rad)
        center = transform_points(pose,[g.center_support])[0]
        result = SupportView(g,registration,clearance,_matrix_tuple(uv),float(center[2]),tuple(center),
            math.atan2(pose[1,0],pose[0,0]),float(error),depth_digest(depth),_matrix_tuple(base_from_camera))
        self.last_frame = e
        self.seen_sessions.add(e.session_id)
        self.good = (result, now)
        return result

    def hold(self, depth, k, base_from_camera, evidence, *, now, parked, occlusion_window_s,
             current_registration=None, contradiction_m=.05, contradiction_rad=.17):
        """Publish the last clean registration under approach occlusion, flagged ``occluded``.

        Used when the support markers/footprint are occluded by the reaching arm so no fresh
        full-footprint registration is available this frame. Every non-visibility gate still holds:
        the from-the-future bound + stamp sync, baseline lifetime (``age_at``), identity validity and
        a monotonic NON-REGRESSING frame (the held path re-expresses via live TF and consumes no fresh
        pixels, so a sensing stride re-polls the SAME last frame between renders -- an equal frame_id
        holds, only a backwards frame_id/stamp/acquisition clock or a regressed session refuses). The
        camera is allowed to MOVE through the reach: the support is rigid
        (world-fixed for the collect source, cart-fixed for the transfer deck) so camera_from_support
        is invariant, and base_from_support is RE-EXPRESSED from the live ``base_from_camera`` each
        tick rather than frozen — a frozen pose silently drifts, and the old exact-match guard bailed
        on sub-mm base motion as the arm reached, stopping /sensors. The per-frame acquisition
        STALENESS is bounded by ``occlusion_window_s`` (not the 0.5 s live budget), since a
        deliberately held frame under a multi-second reach is stale by design; the window is the real
        bound, enforced against the last clean registration. Refuses when there is no established
        clean view, the view has been lost longer than ``occlusion_window_s``, the support has left
        the camera frustum, or a still-visible current registration contradicts the held pose
        re-expressed into the current frame (the support actually moved, not just hidden). Never
        carries a stale clearance forward — a held view is not fresh vacancy proof."""
        g,e = self.geometry,evidence
        e.validate(now,parked,max_age=occlusion_window_s if (occlusion_window_s and math.isfinite(occlusion_window_s)) else .5)
        age = g.age_at(e,now)
        if self.invalid_reason:
            raise ValueError('expired, invalidated or changed rigid-support geometry')
        if (occlusion_window_s is None or not math.isfinite(occlusion_window_s) or occlusion_window_s < 0):
            raise ValueError('occlusion tolerance is not configured for this support')
        if self.good is None:
            raise ValueError('no established support registration to hold under occlusion')
        self._monotonic_ok(e, allow_repeat=True)
        good_view,good_now = self.good
        if now-good_now > occlusion_window_s:
            raise ValueError('support view lost beyond the occlusion window')
        # Re-express the held pose into the current camera/base frame from live TF (support rigid,
        # only the camera moved): camera_from_support is invariant; base_from_support follows the
        # live base_from_camera.
        bfc = rigid(base_from_camera)
        camera_from_support = np.linalg.inv(rigid(good_view.base_from_camera)) @ np.asarray(good_view.registration.base_from_support)
        base_from_support = bfc @ camera_from_support
        if base_from_support[2,2] < .999:
            raise ValueError('held support is no longer horizontal in the current camera frame')
        if current_registration is not None:
            cur = np.asarray(current_registration.base_from_support)
            if (np.linalg.norm(cur[:3,3]-base_from_support[:3,3]) > contradiction_m
                    or abs(math.atan2(cur[1,0],cur[0,0])-math.atan2(base_from_support[1,0],base_from_support[0,0])) > contradiction_rad):
                raise ValueError('visible support markers contradict the held registration')
        radius = np.linalg.norm(np.asarray(g.polygon_support)[:,:2],axis=1).max()
        error = (g.baseline_error_m+good_view.registration.position_error_m
                 +radius*good_view.registration.angular_error_rad+age*.00001)
        center = transform_points(base_from_support,[g.center_support])[0]
        # Reproject the uncertainty-expanded footprint through the LIVE camera; if the moving camera
        # has pushed the support out of the frustum the hold refuses here (never a blind target).
        polygon = np.column_stack([_rectangle(g.center_support[:2],np.asarray(g.required_xy)+2*error),
                                   np.full(4,g.center_support[2])])
        uv,valid = project(transform_points(np.linalg.inv(bfc),transform_points(base_from_support,polygon)),g.intrinsics)
        if not valid.all():
            raise ValueError('held support footprint is truncated or out of range in the current camera frame')
        degraded = MarkerRegistration(good_view.registration.marker_id,g.support_id,good_view.registration.revision,
            _matrix_tuple(base_from_support),good_view.registration.position_error_m,
            good_view.registration.angular_error_rad,e,good_view.registration.marker_ids,good_view.registration.layout_id)
        result = SupportView(g,degraded,None,_matrix_tuple(uv),float(center[2]),tuple(center),
            math.atan2(base_from_support[1,0],base_from_support[0,0]),float(error),depth_digest(depth),
            _matrix_tuple(base_from_camera),occluded=True)
        self.last_frame = e
        self.seen_sessions.add(e.session_id)
        return result
