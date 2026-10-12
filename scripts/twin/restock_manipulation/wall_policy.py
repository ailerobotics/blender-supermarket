"""Mobile station adapter around shelfbot's actual tray-wall pose primitives.

Keeps the same carton face sealed from source to destination. World offsets are
derived from current observed poses, never the external fixed desk coordinates.
The planner targets gripper_pad_tcp, so upstream's pad->platform offset is zero.
"""
import importlib
import math
from pathlib import Path
import sys
from types import SimpleNamespace


def yaw(pose):
    x,y,z,w = pose[3:]
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


class WallCartonPolicy:
    def __init__(self, source_root, tool, *, walls, lift=.03):
        root = str(Path(source_root).resolve())
        sys.path.insert(0, root)
        old = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            self.geometry = importlib.import_module('shelfbot.motion_planning.geometry')
            self.primitives = importlib.import_module('shelfbot.motion_planning.primitives')
        finally:
            sys.dont_write_bytecode = old
            sys.path.remove(root)
        if not Path(self.geometry.__file__).resolve().is_relative_to(root):
            raise ValueError('a different shelfbot source is already loaded')
        self.tool, self.walls, self.lift = tool, dict(walls), float(lift)
        if not 0 < self.lift <= .05 or any(type(w) is not int or w not in range(4) for w in self.walls.values()):
            raise ValueError('invalid wall or bounded lift')

    def build(self, source, target, profile):
        self.tool.validate_profile(profile)
        if profile.grasp_mode != 'side':
            raise ValueError('wall policy requires side mode')
        wall = self.walls[source.location]
        if self.walls[target.location] != wall:
            raise ValueError('changing carton wall requires a separately verified regrasp')
        lateral = profile.dimensions[1 if wall % 2 == 0 else 0]
        if lateral < self.tool.panel_size[0] + .02:
            raise ValueError('selected wall is too narrow for the actual CAD panel')
        if target.free_dimensions[2] < profile.dimensions[2] + self.lift + .02:
            raise ValueError('carton plus lift and margin exceed shelf vertical gap')
        # This tests all three slot dimensions as well as the additional lift.
        if any(room < size+.02 for room,size in zip(target.free_dimensions, profile.dimensions)):
            raise ValueError('handling unit does not fit the measured slot in all three dimensions')
        variant = SimpleNamespace(tray_size=profile.dimensions,
            contact_margin=profile.tcp_surface_offset,
            approach_offset_x_m=profile.approach, grasp_quat=(.5,.5,.5,.5))
        def stations(observation, centre_z):
            placement = SimpleNamespace(yaw=yaw(observation.pose))
            ox,oy,rx,ry,q = self.geometry.wall_grip(variant, placement, wall)
            return self.primitives.approach_contact(
                observation.pose[0]+ox, observation.pose[1]+oy, centre_z,
                rx,ry,q,(0.,0.,0.))
        approach, contact = stations(source, source.pose[2]-profile.dimensions[2]/2)
        retreat, place = stations(target, target.pose[2]+profile.dimensions[2]/2)
        lift = list(contact)
        lift[2] += self.lift
        preplace = list(retreat)
        preplace[2] += self.lift
        return {name:tuple(p) for name,p in dict(approach=approach, contact=contact,
            lift=lift, preplace=preplace, place=place, retreat=retreat).items()}
