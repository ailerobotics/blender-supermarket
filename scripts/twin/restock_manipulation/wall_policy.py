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


class SimCartPlacePolicy:
    """SIM-ONLY place-pose policy (firstmate 282). Keeps the proven side-grip PICK poses (and the
    grip TCP orientation) from the real-rig WallCartonPolicy, but RE-CHAINS the cart PLACE so the
    pre/post standoff offsets TOWARD the robot along the approach axis instead of the wall policy's
    outward standoff -- which points away from the base and is cuMotion-IK-infeasible onto the cart
    (runs 218-222: side-grip place preplace/retreat ~1.3 m IK-fail, deck AND shelf). The carton is
    placed with the SAME side-grip orientation it was picked with; only the approach/retreat
    direction is mirrored to the reachable (base-ward) side. The real-rig wall policy is untouched;
    this wraps it and only rewrites preplace/place/retreat, gated on a sim config flag.

    place    = the wall place POSITION with the side-grip orientation SAGITTAL-MIRRORED (see below)
    retreat  = place offset by `profile.approach` TOWARD ur_base_link (origin) in the horizontal plane
    preplace = retreat raised by the lift margin

    Orientation mirror (firstmate msg 285 step B): the raw side-grip TCP orientation has NO KDL
    joint-space solution at the cart's -y side (CPU KDL-verified: cuMotion task-space AND KDL both
    find none) -- the wrist cannot achieve the +y-grip orientation mirrored across the base. Its
    SAGITTAL mirror (reflect about the base xz-plane, M=diag(1,-1,1): quat (x,y,z,w)->(-x,y,-z,w))
    IS reachable at -y by the UR10's symmetry (CPU KDL-verified: preplace/place/retreat all solve).
    The carton is placed in that mirrored side pose on the cart shelf.
    """
    def __init__(self, wall, lift=.03):
        self.wall, self.lift = wall, float(lift)

    @staticmethod
    def _sagittal_mirror(quat):
        x, y, z, w = quat
        return (-x, y, -z, w)

    def build(self, source, target, profile):
        poses = dict(self.wall.build(source, target, profile))   # proven pick + grip orientation
        wp = tuple(poses['place'])
        q = self._sagittal_mirror(wp[3:])                        # -y-reachable orientation
        x, y = wp[0], wp[1]
        n = math.hypot(x, y) or 1.0
        d = profile.approach
        place = (wp[0], wp[1], wp[2], *q)
        retreat = (x - x / n * d, y - y / n * d, wp[2], *q)       # toward the origin (robot)
        preplace = (retreat[0], retreat[1], retreat[2] + self.lift, *q)
        poses['place'], poses['retreat'], poses['preplace'] = place, retreat, preplace
        return poses
