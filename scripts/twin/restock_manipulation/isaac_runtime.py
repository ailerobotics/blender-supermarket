"""Assembly for integration's booted Isaac loop. No independent GPU launch."""
from .cad_tool import inspect_tool, NativePanelContact
from .controller import TransferController
from .isaac_motion import build_cad_planner, IsaacCadMotion
from .isaac_sensors import IsaacCartonSensors
from .wall_policy import WallCartonPolicy


class CadRestockRuntime:
    def __init__(self, *, stage, articulation, camera, gripper_path, urdf_path,
                 shelfbot_root, pose_bridge_root, profiles, slots, base_from_camera,
                 voxel_dims=(2.4,2.4,2.4), voxel_size=.02, walls=None):
        """Call only under integration's GPU ownership, after native gripper authoring.

        camera is an initialized Isaac Camera (registered RGB+depth), articulation
        is the real imported mobile URDF's SingleArticulation. base_from_camera is
        its calibrated mount in ur_base_link, not a product pose. All writable
        authoring belongs in integration's composed output, never source assets.
        """
        tool = inspect_tool(urdf_path)
        for profile in profiles.values():
            tool.validate_profile(profile)
        self.contact = NativePanelContact(stage, gripper_path)
        self.planner = build_cad_planner(shelfbot_root, urdf_path,
                                        voxel_dims=voxel_dims, voxel_size=voxel_size)
        self.motion = IsaacCadMotion(articulation, self.planner, tool, self.contact)
        modes = {p.grasp_mode for p in profiles.values()}
        if len(modes) != 1:
            raise ValueError('use one calibrated grasp mode per runtime')
        policy = WallCartonPolicy(shelfbot_root, tool, walls=walls or {}) if modes == {'side'} else None
        self.controller = TransferController(self.motion, profiles, phase_timeout=120.,
                                              frame=tool.base_frame, pose_policy=policy)
        self.sensors = IsaacCartonSensors(camera, pose_bridge_root=pose_bridge_root,
                                          profiles=profiles, slots=slots, base_from_camera=base_from_camera)
        self.latest = None
        self.last_sensor_stamp = -1.

    def start(self, operation, job):
        self.controller.start(operation, job)
        self.latest = None
        self.last_sensor_stamp = -1.

    def configure_station(self, *, slots, base_from_camera):
        """Invalidate old arm-frame maps and ROIs after parking at a new dock."""
        import numpy as np
        if self.controller.phase not in ('idle','succeeded','failed') or self.controller.held:
            raise RuntimeError('cannot change station during transfer or payload recovery')
        self.sensors.slots = tuple(slots)
        self.sensors.transform = np.asarray(base_from_camera,float)
        self.sensors.last_frame = None
        self.motion.collision_time = -1.
        self.motion.last_grid = None
        self.motion.source = None
        self.last_sensor_stamp = -1.

    def step(self, dt, simulation_time):
        """Call before each physics step. Refresh sensor ESDF through motion or
        NvbloxWorldBridge. Returns the shared status/controls/evidence contract.
        """
        c = self.controller
        if c.phase in ('idle', 'failed', 'succeeded'):
            return c._result()
        try:
            if not self.motion.parked():
                return c._fail('mobile base moved during arm transfer')
            if c.held and c.phase != 'release' and not self.contact.read():
                return c._fail('native CAD seal lost')
            frame = self.sensors.camera.get_current_frame()
            stamp = float(frame['rendering_time'])
            if not 0 <= simulation_time-stamp <= c.max_age:
                return c._fail('rendered camera stream stale')
            if not 0 <= simulation_time-self.motion.collision_time <= c.max_age:
                return c._fail('sensor collision map stale or absent')
            if frame['rendering_frame'] != self.sensors.last_frame:
                packet = self.sensors.capture(simulation_time, c.source, c.destination,
                                              self.motion, self.contact)
                elapsed = dt if self.last_sensor_stamp < 0 else simulation_time-self.last_sensor_stamp
                self.latest = c.step(elapsed, packet)
                self.last_sensor_stamp = simulation_time
                for command in self.latest.controls:
                    if command['kind'] == 'gripper':
                        self.contact.command(command['enabled'])
            if c.phase not in ('failed', 'succeeded'):
                self.motion.tick(dt)
            return self.latest or c._result()
        except Exception as exc:
            return c._fail(f'CAD runtime adapter: {type(exc).__name__}: {exc}')
