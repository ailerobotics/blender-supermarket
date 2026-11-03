"""vr1: labelled GROUND-TRUTH mission verify for the backroom-bay demo (default OFF).

The mission's ``verify`` stage normally commits stock only from the camera ``InventoryVerifier`` (two
fresh post-transfer scans of the destination). For the fm459/fm461 backroom-bay demo the bay's shelf
station camera is NOT commissioned (one bay marker never decodes), so that verifier can never confirm
and the mission would sit in ``verify`` until the stage timeout. This drop-in replacement (same
``begin`` / ``stop`` / ``__call__(now)`` contract the port drives) confirms the placement from the SIM
GROUND TRUTH instead: the handling unit's measured centre must rest ON the bay top (centre over the
top inside a margin, bottom on the surface, level), with the gripper released, for ``frames`` distinct
post-boundary frames. Its evidence is honestly labelled ``source='ground_truth'`` +
``verify_mode='ground_truth'`` -- never ``'sensor'`` -- so mission.json, the inventory log and the
walkthrough caption cannot present it as a camera measurement. It is enabled ONLY by the runtime
config ``verify_ground_truth: {"enabled": true}`` (see :func:`from_config`); absent => the camera
verifier is used, byte-identical. A unit clearly OFF the bay fails fast with a named reason instead of
waiting out the stage timeout.

Pure Python (no Isaac/USD imports): the Isaac bundle supplies ``read_unit_pose`` (a stage read of the
handling-unit prim) and ``read_contact``; CPU tests supply fixtures.
"""
from __future__ import annotations

import math

VERIFY_MODE = 'ground_truth'
DEFAULT_XY_MARGIN_M = 0.02     # the unit's centre must sit this far inside the bay top edges
DEFAULT_Z_TOL_M = 0.04         # |unit bottom - bay top|: resting on the surface, not held above / sunk
DEFAULT_MAX_TILT_DEG = 10.0    # a tipped unit is not 'resting on the bay'
DEFAULT_FRAMES = 2             # distinct post-boundary frames, as the camera verifier requires
DEFAULT_OFF_BAY_FRAMES = 3     # consecutive clearly-off-bay frames before failing fast


class GroundTruthVerifyError(ValueError):
    """The unit is clearly not on the bay (or released); raised so the stage fails with this reason."""


class GroundTruthInventoryVerifier:
    """Sim ground-truth stand-in for :class:`verification.InventoryVerifier` (vr1, labelled).

    ``read_unit_pose()`` -> ``(x, y, z, tilt_rad)`` world centre of the handling unit + its tilt from
    level, or None when unreadable. ``read_contact()`` -> True while the gripper still holds it."""

    def __init__(self, read_unit_pose, *, bay_center_xy, bay_size_xy, bay_top_z, unit_height,
                 read_contact=None, source='rollwagen', bay_yaw=0.0,
                 xy_margin=DEFAULT_XY_MARGIN_M, z_tol=DEFAULT_Z_TOL_M,
                 max_tilt_deg=DEFAULT_MAX_TILT_DEG, frames=DEFAULT_FRAMES,
                 off_bay_frames=DEFAULT_OFF_BAY_FRAMES):
        vals = (*bay_center_xy, *bay_size_xy, bay_top_z, unit_height, bay_yaw, xy_margin, z_tol,
                max_tilt_deg)
        if len(bay_center_xy) != 2 or len(bay_size_xy) != 2 or not all(math.isfinite(v) for v in vals):
            raise ValueError('ground-truth verify needs a finite bay centre/size/top and unit height')
        if min(bay_size_xy) <= 2 * xy_margin or unit_height <= 0 or z_tol <= 0 or max_tilt_deg <= 0:
            raise ValueError('ground-truth verify: bay smaller than its margin or non-positive tolerance')
        if int(frames) < 1 or int(off_bay_frames) < 1:
            raise ValueError('ground-truth verify frames must be >= 1')
        self.read_unit_pose, self.read_contact = read_unit_pose, read_contact
        self.source = source
        self.bay_center = (float(bay_center_xy[0]), float(bay_center_xy[1]))
        self.bay_half = (float(bay_size_xy[0]) / 2.0, float(bay_size_xy[1]) / 2.0)
        self.bay_top_z, self.bay_yaw = float(bay_top_z), float(bay_yaw)
        self.unit_half_h = float(unit_height) / 2.0
        self.xy_margin, self.z_tol = float(xy_margin), float(z_tol)
        self.max_tilt = math.radians(float(max_tilt_deg))
        self.frames, self.off_bay_frames = int(frames), int(off_bay_frames)
        self.active = False
        self.last_check = None

    def tolerances(self):
        return {'xy_margin_m': self.xy_margin, 'z_tol_m': self.z_tol,
                'max_tilt_deg': round(math.degrees(self.max_tilt), 3), 'frames': self.frames,
                'bay_center_xy': list(self.bay_center), 'bay_half_xy': list(self.bay_half),
                'bay_top_z': self.bay_top_z}

    def begin(self, job, now):
        if not math.isfinite(now):
            raise ValueError('finite verification boundary required')
        self.job, self.started = job, float(now)
        self.last_stamp = float(now)
        self.confirmations = []
        self._off = 0
        self.active = True

    def stop(self):
        self.active = False

    def check(self, pose):
        """Pure geometry: is a unit at ``pose`` = (x, y, z, tilt) resting on the bay top? Returns a dict
        with the bay-frame offsets, the bottom gap and the verdicts (``on_bay`` / ``far``)."""
        x, y, z, tilt = (float(v) for v in pose)
        c, s = math.cos(self.bay_yaw), math.sin(self.bay_yaw)
        dx0, dy0 = x - self.bay_center[0], y - self.bay_center[1]
        dx, dy = c * dx0 + s * dy0, -s * dx0 + c * dy0          # offset in the bay frame
        gap = (z - self.unit_half_h) - self.bay_top_z
        lim_x, lim_y = self.bay_half[0] - self.xy_margin, self.bay_half[1] - self.xy_margin
        on_bay = abs(dx) <= lim_x and abs(dy) <= lim_y and abs(gap) <= self.z_tol and abs(tilt) <= self.max_tilt
        # Clearly NOT on the bay: centre beyond the top's edge, or the bottom well off the surface (fell to
        # the floor / still carried high) -- 2x the accept band, like the controller's place verify.
        far = (abs(dx) > self.bay_half[0] or abs(dy) > self.bay_half[1] or abs(gap) > 2 * self.z_tol
               or abs(tilt) > 2 * self.max_tilt)
        return {'offset_bay_xy': [round(dx, 4), round(dy, 4)], 'bottom_gap_m': round(gap, 4),
                'tilt_deg': round(math.degrees(tilt), 3), 'on_bay': on_bay, 'far': far,
                'unit_center_world': [round(x, 4), round(y, 4), round(z, 4)]}

    def __call__(self, now):
        if not self.active:
            return None
        if not math.isfinite(now) or now <= self.last_stamp:
            return None                                       # one confirmation per new frame
        self.last_stamp = float(now)
        pose = self.read_unit_pose()
        if pose is None:
            self.confirmations = []
            return None
        held = bool(self.read_contact()) if self.read_contact is not None else False
        verdict = self.check(pose)
        verdict['gripper_contact'] = held
        self.last_check = verdict
        if held:
            raise GroundTruthVerifyError('ground-truth verify: handling unit still held by the gripper')
        if verdict['far']:
            self._off += 1
            self.confirmations = []
            if self._off >= self.off_bay_frames:
                raise GroundTruthVerifyError(
                    'ground-truth verify: handling unit not resting on the bay '
                    f'(offset_bay_xy={verdict["offset_bay_xy"]} bottom_gap={verdict["bottom_gap_m"]} '
                    f'tilt_deg={verdict["tilt_deg"]})')
            return None
        self._off = 0
        if not verdict['on_bay']:
            self.confirmations = []                            # settling: neither on nor clearly off
            return None
        self.confirmations.append(f'ground_truth_verify:{now:.4f}')
        if len(self.confirmations) < self.frames:
            return None
        self.active = False
        return {'observation_id': self.confirmations[-1], 'source': 'ground_truth',
                'verify_mode': VERIFY_MODE, 'simulation_time': float(now),
                'sku': self.job.sku, 'quantity': self.job.quantity, 'location': self.job.shelf_id,
                'unit_kind': 'handling_unit', 'transfer_confirmed': True,
                'source_vacancy_location': self.source,
                'source_vacancy': 'implied: the single handling unit rests on the bay (sim ground truth)',
                'verification_started': self.started,
                'confirming_observation_ids': list(self.confirmations),
                'ground_truth_check': verdict, 'tolerances': self.tolerances(),
                'note': ('vr1 labelled demo workaround: SIM GROUND-TRUTH verify (handling-unit pose vs '
                         'the authored bay top); the bay shelf camera is not commissioned. NOT a camera '
                         'measurement.')}


def config_block(config):
    """The enabled ``verify_ground_truth`` block, or None when off/absent (byte-identical default)."""
    block = config.get('verify_ground_truth') if isinstance(config, dict) else None
    if isinstance(block, bool):
        block = {'enabled': block}
    if not isinstance(block, dict) or not block.get('enabled'):
        return None
    return block


def from_config(config, scenario, read_unit_pose, *, read_contact=None, source='rollwagen'):
    """Build the verifier from the runtime config + mission scenario, or None when not enabled.

    Bay geometry comes from ``scenario['transfer_bay']`` (``center_xy``, ``dimensions_m[:2]``,
    ``top_z``) unless the block overrides it with ``bay: {center_xy, size_xy, top_z, yaw}``; the unit
    height from ``scenario['handling_unit']['dimensions_m'][2]``. Tolerances: ``xy_margin_m``,
    ``z_tol_m``, ``max_tilt_deg``, ``frames`` (defaults above)."""
    block = config_block(config)
    if block is None:
        return None
    bay = dict(block.get('bay') or {})
    authored = scenario.get('transfer_bay') or {}
    center = bay.get('center_xy') or authored.get('center_xy')
    size = bay.get('size_xy') or (authored.get('dimensions_m') or [None, None])[:2]
    top = bay.get('top_z', authored.get('top_z'))
    unit = scenario.get('handling_unit') or {}
    dims = unit.get('dimensions_m') or []
    if not center or not size or None in size or top is None or len(dims) != 3:
        raise ValueError('verify_ground_truth needs scenario.transfer_bay (or a bay block) and '
                         'scenario.handling_unit.dimensions_m')
    return GroundTruthInventoryVerifier(
        read_unit_pose, bay_center_xy=center, bay_size_xy=size, bay_top_z=top,
        unit_height=dims[2], read_contact=read_contact, source=source,
        bay_yaw=float(bay.get('yaw', 0.0)),
        xy_margin=float(block.get('xy_margin_m', DEFAULT_XY_MARGIN_M)),
        z_tol=float(block.get('z_tol_m', DEFAULT_Z_TOL_M)),
        max_tilt_deg=float(block.get('max_tilt_deg', DEFAULT_MAX_TILT_DEG)),
        frames=int(block.get('frames', DEFAULT_FRAMES)))


def mission_modes(config):
    """The labelled demo workarounds a runtime config turns on, for mission.json + stage labels.

    {} for a config with none of them (byte-identical summaries)."""
    modes = {}
    if config_block(config) is not None:
        modes['verify_mode'] = VERIFY_MODE
    gt = config.get('transfer_ground_truth') if isinstance(config, dict) else None
    if gt and (gt.get('enabled') if isinstance(gt, dict) else gt):
        modes['transfer_verify_mode'] = VERIFY_MODE
    if isinstance(config, dict) and config.get('uncouple_on_return'):
        modes['return_mode'] = 'uncouple_at_cart_parking'
    if isinstance(config, dict) and config.get('lidar_sensor_yaw_fix', True) is False:
        # Provenance, not a workaround (a geometry-wiring choice; deliberately not a *_mode key so the
        # walkthrough does not caption it): the vr1 yaw-0 fix is the DEFAULT now, so record only the
        # EXCEPTIONAL legacy wiring (lidar_sensor_yaw_fix: false) -- the rotated pre-fix offset this run's
        # nav consumed. {} (no key) means the default, correct yaw-0 wiring.
        modes['lidar_sensor_to_base'] = 'xy_z_legacy (pre-vr1 rotated wiring; lidar_sensor_yaw_fix=false)'
    # sim-video-speedup-v1: the opt-in speed profile (sim_speed_mode + its resolved levers); {} when off.
    from .sim_speed import summary_fields as _sim_speed_fields
    modes.update(_sim_speed_fields(config))
    # sim-robust-vram-headroom-v1: the opt-in VRAM headroom monitor (vram_headroom_mode + levers); {} off.
    from .vram_headroom import summary_fields as _vram_fields
    modes.update(_vram_fields(config))
    return modes


def ground_truth_place_actions(config):
    """Stages whose fm461 ground-truth place check the RuntimeAdapter accepts as stage evidence."""
    modes = mission_modes(config)
    # transfer only: collect's ground-truth place evidence is collect-gt's (RESTOCK_COLLECT_GROUND_TRUTH,
    # twin.restock_manipulation.ground_truth_collect), not duplicated here.
    return ('transfer',) if 'transfer_verify_mode' in modes else ()


def mission_ground_truth_sources(config):
    """``('ground_truth',)`` when any ground-truth workaround is on (Mission opt-in), else ()."""
    modes = mission_modes(config)
    return (('ground_truth',) if any(k in modes for k in
                                     ('verify_mode', 'transfer_verify_mode'))
            else ())


def stage_modes(config):
    """``{stage: mode}`` labels for the behaviour-tree steps (verify/transfer/collect) from the modes."""
    modes = mission_modes(config)
    return {stage: modes[key] for stage, key in (('verify', 'verify_mode'),
                                                 ('transfer', 'transfer_verify_mode')) if key in modes}
