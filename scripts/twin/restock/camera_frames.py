"""Bounded acquired-frame queues behind actual mapper acknowledgements.

No timestamp or transform is synthesized on retrieval. Integration records the
measured transform when each native camera acquisition is received while parked.
This module does not establish that the selected mapper sees the required volume.
"""
from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from twin.restock_camera.geometry import rigid


@dataclass(frozen=True)
class AcquiredView:
    frame: dict
    base_from_camera: np.ndarray
    transform_stamp: float
    epoch: int
    parked: bool


class SnapshotCamera:
    def __init__(self, camera):
        self.camera,self.selected=camera,None

    def get_current_frame(self, *, clone=False):
        if self.selected is None:
            raise ValueError('no mapper-incorporated camera acquisition selected')
        # Owned immutable arrays may be shared: downstream sensor adapters read.
        return dict(self.selected.frame)

    def __getattr__(self, name):
        return getattr(self.camera,name)


class CameraFrames:
    def __init__(self, cameras, location_cameras, *, capacity=8, max_bytes=768*1024**2):
        if type(capacity) is not int or not 2<=capacity<=30 or max_bytes<=0:
            raise ValueError('bounded camera queue required')
        if not cameras or not set(location_cameras.values())<=cameras.keys():
            raise ValueError('complete camera/location assignment required')
        self.cameras={name:SnapshotCamera(camera) for name,camera in cameras.items()}
        self.locations=dict(location_cameras)
        self.queues={name:deque(maxlen=capacity) for name in cameras}
        self.consumed={}
        self.last_seen={}
        self.epoch=None
        self.max_bytes=max_bytes

    def begin_epoch(self, epoch):
        if type(epoch) is not int or epoch<=0 or self.epoch is not None and epoch<=self.epoch:
            raise ValueError('new monotonically increasing parked epoch required')
        self.epoch=epoch
        for queue in self.queues.values():
            queue.clear()
        for camera in self.cameras.values():
            camera.selected=None
        # Keep consumed identities across epochs; old frames cannot become new.

    def append(self, name, frame, *, base_from_camera, transform_stamp, now, parked):
        if self.epoch is None or not parked:
            raise ValueError('current measured parked epoch required')
        stamp=float(frame['rendering_time'])
        identity=(frame['session_id'],frame['rendering_frame'])
        if (not frame['session_id'] or type(identity[1]) is not int
                or not all(math.isfinite(t) for t in (stamp,transform_stamp,now))
                or not 0<=now-stamp<=.5 or abs(stamp-transform_stamp)>.02):
            raise ValueError('fresh synchronized acquired frame and transform required')
        queue=self.queues[name]
        if name in self.last_seen:
            prior_id,prior_stamp=self.last_seen[name]
            if identity==prior_id:
                if stamp!=prior_stamp:
                    raise ValueError('cached acquisition was restamped')
                return False
            if identity[0]!=prior_id[0] or identity[1]<=prior_id[1] or stamp<=prior_stamp:
                raise ValueError('acquisition identity or clock moved backwards or session changed')
        if identity==self.consumed.get(name):
            return False
        saved={key:frame[key] for key in ('session_id','steady_domain','acquired_steady',
            'rendering_frame','rendering_time')}
        for key in ('rgb','distance_to_image_plane'):
            pixels=np.array(frame[key],copy=True)
            pixels.flags.writeable=False
            saved[key]=pixels
        if saved['rgb'].shape[:2]!=saved['distance_to_image_plane'].shape:
            raise ValueError('registered RGB-D shapes required')
        view=AcquiredView(saved,rigid(base_from_camera).copy(),float(transform_stamp),self.epoch,True)
        queue.append(view)
        self.last_seen[name]=(identity,stamp)
        while self._bytes()>self.max_bytes:
            populated=[q for q in self.queues.values() if q]
            oldest=min(populated,key=lambda q:q[0].frame['rendering_time'])
            oldest.popleft()
        return any(item is view for item in queue)

    def _bytes(self):
        return sum(v.frame['rgb'].nbytes+v.frame['distance_to_image_plane'].nbytes
                   for queue in self.queues.values() for v in queue)

    def select(self, source, destination, *, now, collision_time):
        if not all(math.isfinite(t) for t in (now,collision_time)) or not 0<=now-collision_time<=.5:
            return False
        names=sorted({self.locations[source],self.locations[destination]})
        candidates={name:[v for v in self.queues[name]
            if v.epoch==self.epoch and 0<=now-v.frame['rendering_time']<=.5
            and v.frame['rendering_time']<=collision_time
            and (v.frame['session_id'],v.frame['rendering_frame'])!=self.consumed.get(name)]
            for name in names}
        if not all(candidates.values()):
            return False
        # Search newest acknowledged pair; camera phases need not be identical.
        selected=None
        horizons=sorted({v.frame['rendering_time'] for values in candidates.values()
                         for v in values},reverse=True)
        for horizon in horizons:
            pair={}
            for name,values in candidates.items():
                choices=[v for v in values if 0<=horizon-v.frame['rendering_time']<=.05]
                if not choices:
                    break
                pair[name]=choices[-1]
            if len(pair)==len(names):
                selected=pair
                break
        if selected is None:
            return False
        for name,view in selected.items():
            self.cameras[name].selected=view
            self.consumed[name]=(view.frame['session_id'],view.frame['rendering_frame'])
            # Never allow an older queued frame after a newer selection.
            while self.queues[name] and self.queues[name][0].frame['rendering_time']<=view.frame['rendering_time']:
                self.queues[name].popleft()
        return True

    def context(self, name, frame, now):
        view=self.cameras[name].selected
        if (view is None or view.epoch!=self.epoch or not 0<=now-view.frame['rendering_time']<=.5
                or (frame['session_id'],frame['rendering_frame'])!=
                   (view.frame['session_id'],view.frame['rendering_frame'])):
            raise ValueError('context does not belong to selected acquired frame')
        return view.base_from_camera.copy(),view.transform_stamp,str(view.epoch),view.parked
