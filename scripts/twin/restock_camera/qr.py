"""One bounded, pixel-derived QR observation shared by support and inventory.

No marker-specific crop, expected identity, geometry lookup or resampling enters
detection. The 2560x1920 design uses exactly the measured 20-tile grid.
"""
from dataclasses import dataclass
import hashlib
import time

import numpy as np

from .sensed import FrameEvidence


def rgb_digest(rgb):
    pixels = np.asarray(rgb)
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError('QR observation requires original uint8 RGB pixels')
    # Snapshot on every call, including validation: never cache by array/frame
    # identity. Packing strided RGBA->RGB views as an ndarray avoids the slower
    # tobytes serializer; hashlib consumes the contiguous buffer directly.
    # Always copy even contiguous inputs so hashing retains snapshot semantics.
    snapshot = pixels.copy(order='C')
    return hashlib.sha256(snapshot).hexdigest()


def _tile_overlaps(origin, roi, width, height):
    """Whether the 640-px tile at ``origin`` overlaps the ROI box (x1,y1,x2,y2)."""
    x,y = origin
    x1,y1,x2,y2 = roi
    return x < x2 and min(x+640,width) > x1 and y < y2 and min(y+640,height) > y1


def tile_origins(width, height):
    if not 1 <= width <= 2560 or not 1 <= height <= 1920:
        raise ValueError('image exceeds bounded QR search dimensions')
    def starts(size):
        last = max(0,size-640)
        return sorted(set([*range(0,last+1,480),last]))
    return tuple((x,y) for y in starts(height) for x in starts(width))


@dataclass(frozen=True)
class QRDetection:
    text: str
    corners: tuple
    sightings: int = 1
    duplicate_corner_rms_pixels: float = 0.


@dataclass(frozen=True)
class FrameQR:
    evidence: FrameEvidence
    rgb_shape: tuple
    rgb_sha256: str
    detections: tuple[QRDetection, ...]
    unreadable_corners: tuple
    processing_seconds: float
    tile_count: int

    def validate(self, rgb, evidence, *, now, parked, max_age=.5):
        evidence.validate(now,parked,max_age=max_age)
        if (evidence != self.evidence or tuple(np.shape(rgb)) != self.rgb_shape
                or rgb_digest(rgb) != self.rgb_sha256
                or not np.isfinite(self.processing_seconds)
                or not 0 <= self.processing_seconds <= .5):
            raise ValueError('QR observation belongs to another frame/pixels or exceeds latency budget')


def _corners(points, x, y, width, height):
    p = np.asarray(points,float).reshape(-1,2)
    if p.shape != (4,2) or not np.isfinite(p).all():
        raise ValueError('invalid decoded QR quadrilateral')
    p = p+[x,y]
    edges = np.roll(p,-1,axis=0)-p
    cross = edges[:,0]*np.roll(edges,-1,axis=0)[:,1]-edges[:,1]*np.roll(edges,-1,axis=0)[:,0]
    if (np.any(p < [x,y]) or np.any(p >= [min(x+640,width),min(y+640,height)])
            or not (np.all(cross>0) or np.all(cross<0))):
        raise ValueError('truncated or ambiguous decoded QR quadrilateral')
    return p


def _overlap(a,b):
    low = np.maximum(a.min(0),b.min(0)); high = np.minimum(a.max(0),b.max(0))
    intersection = float(np.prod(np.maximum(0,high-low)))
    area = min(float(np.prod(np.ptp(a,axis=0))),float(np.prod(np.ptp(b,axis=0))))
    return intersection > .2*area


class FixedTileQR:
    """Single-code tile detection with spatial deduplication and replay refusal.

    Processing budget is wall time, distinct from acquisition timestamps and
    their unchanged freshness semantics. A failed attempt consumes its frame;
    callers must obtain new pixels, never retry/restamp old data to pass.
    """
    def __init__(self, camera_id):
        import cv2
        if not camera_id:
            raise ValueError('camera identity required')
        self.camera_id = camera_id
        self.detector = cv2.QRCodeDetector()
        self.last = None
        self.seen_sessions = set()

    def _scan(self, bgr, origins, width, height, start):
        import cv2
        groups,unreadable = [],[]
        for x,y in origins:
            if time.perf_counter()-start > .5:
                raise ValueError('QR detection/merge exceeds 500 ms budget')
            try:
                text,points,_ = self.detector.detectAndDecode(bgr[y:y+640,x:x+640].copy())
            except cv2.error as exc:
                raise ValueError('QR tile decoder failed; no partial observation') from exc
            if points is None:
                if text:
                    raise ValueError('decoded identity lacks pixel corners')
                continue
            p = _corners(points,x,y,width,height)
            if not text:
                unreadable.append(p)
                continue
            matches = [g for g in groups if np.linalg.norm(p.mean(0)-g['corners'].mean(0)) <= 5.]
            if matches:
                g = matches[0]
                rms = float(np.sqrt(np.mean(np.sum((p-g['corners'])**2,axis=1))))
                if len(matches)!=1 or text!=g['text'] or rms>2.:
                    raise ValueError('conflicting QR identity or duplicate pixel corners')
                g['sightings'] += 1
                g['rms'] = max(g['rms'],rms)
            else:
                if any(g['text']==text or _overlap(p,g['corners']) for g in groups):
                    raise ValueError('duplicate QR identity or overlapping conflicting labels')
                groups.append(dict(text=text,corners=p,sightings=1,rms=0.))
        return groups,unreadable

    def decode(self, rgb, evidence, *, now, parked, roi=None):
        """Decode QR tiles. ``roi``=(x1,y1,x2,y2) restricts the first pass to the certified
        tiles overlapping that region (the expected handling-unit/support area from the
        commissioned station geometry + pose); a miss (no readable AND no unreadable candidate
        in the subset) falls back to the full certified tile grid, so correctness is unchanged
        and only the common case is cheaper."""
        import cv2
        start = time.perf_counter()
        evidence.validate(now,parked)
        if evidence.camera_id != self.camera_id:
            raise ValueError('QR decoder belongs to another camera')
        if self.last is not None:
            same = self.last.session_id == evidence.session_id
            if ((same and (evidence.frame_id <= self.last.frame_id or evidence.stamp <= self.last.stamp))
                    or (not same and evidence.session_id in self.seen_sessions)):
                raise ValueError('QR decoder refuses replayed camera frame/session')
        self.last = evidence
        self.seen_sessions.add(evidence.session_id)
        digest = rgb_digest(rgb)
        height,width = rgb.shape[:2]
        origins = tile_origins(width,height)
        bgr = cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        scanned = origins
        if roi is not None:
            subset = tuple(o for o in origins if _tile_overlaps(o,roi,width,height))
            groups,unreadable = self._scan(bgr,subset,width,height,start)
            if not groups and not unreadable:        # total miss in the ROI: full-frame fallback
                groups,unreadable = self._scan(bgr,origins,width,height,start)
            else:
                scanned = subset
        else:
            groups,unreadable = self._scan(bgr,origins,width,height,start)
        # A failed tile may contain a code successfully decoded in another tile.
        # Preserve all other unreadable candidates so in-slot inventory refuses.
        unresolved = [p for p in unreadable if not any(
            np.linalg.norm(p.mean(0)-g['corners'].mean(0))<=5. and
            np.sqrt(np.mean(np.sum((p-g['corners'])**2,axis=1)))<=2. for g in groups)]
        detections = tuple(QRDetection(g['text'],tuple(map(tuple,g['corners'])),g['sightings'],g['rms'])
                           for g in groups)
        elapsed = time.perf_counter()-start
        if elapsed > .5:
            raise ValueError('QR detection/merge exceeds 500 ms budget')
        return FrameQR(evidence,tuple(rgb.shape),digest,detections,
                       tuple(tuple(map(tuple,p)) for p in unresolved),elapsed,len(scanned))
