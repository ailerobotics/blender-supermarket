"""sim-chassis-pitch-stability-v1: the chassis see-saw cause + the compliant-suspension fix.

Pins the pure analytical design (twin.traction), the opt-in MobileSpec suspension field + its byte-
identical-when-absent URDF authoring (twin.mobile_assets), the config_preflight pitch-stability guard,
and (pxr-gated) the sim-only station pitch-hold brake. The named cause: an unsprung two-drive-wheel base
with both casters recessed caster_lift_m above the drive plane free-pitches +-atan(lift/spacing) until a
caster touches down (coldwin37 end stops -4.10 / +4.89 deg ~ +-4.51 predicted), so the welded carton
pitches by the chassis rock-back after a level descent was planned -> the CARTON_TILT +2.06 trip.
"""
import math
import xml.etree.ElementTree as ET

import pytest

from twin.mobile import MobileSpec
from twin.mobile_assets import build_mobile_urdf
from twin.urdf_probe import parse_urdf, find_roots, link_world_transforms
from twin.traction import (free_pitch_travel_deg, caster_spacing_m, CasterSuspension,
                           sprung_chassis_pitch_deg, suspension_meets_pitch_bound,
                           suspension_drive_wheel_share, static_traction_budget,
                           suspension_solver_stability_ratio, suspension_solver_stable,
                           min_suspender_mass_kg, SUSPENSION_STABILITY_RATIO_MAX)
from twin.restock_manipulation.config_preflight import check_pitch_stability

WORST_CASE_MOMENT_NM = 250.0   # UR10 own mass out at reach + 0.45 kg carton (FK sweep measured 241)
RECOMMENDED = CasterSuspension(spring_n_per_m=3.0e5, preload_n=120.0,
                               damping_n_s_per_m=2000.0, travel_m=0.02)
SUSPENDER_MASS_KG = 4.0        # coldwin38 fix: a solver-stable sprung caster mass (was 0.2 kg)


# ---- (1) the unsprung see-saw geometry ----------------------------------------------------------

def test_free_pitch_travel_matches_measured_end_stops():
    d = caster_spacing_m()
    assert d == pytest.approx(0.38)
    # caster_lift_m 0.03 (the live rig) -> +-4.51 deg: the coldwin37 end stops were -4.10 / +4.89.
    assert free_pitch_travel_deg(0.03, d) == pytest.approx(4.514, abs=1e-2)
    assert -4.27 >= -free_pitch_travel_deg(0.03, d) - 0.5    # -4.10 end stop sits inside the travel
    # co-planar / sprung (lift 0) -> no free travel.
    assert free_pitch_travel_deg(0.0, d) == 0.0
    # monotone non-decreasing in the lift.
    tr = [free_pitch_travel_deg(l, d) for l in (0.0, 0.005, 0.01, 0.02, 0.03)]
    assert all(b >= a for a, b in zip(tr, tr[1:]))
    with pytest.raises(ValueError):
        free_pitch_travel_deg(-0.01, d)


# ---- (2) the compliant-suspension fix: criterion (a) pitch bound --------------------------------

def test_suspension_bounds_worstcase_pitch_under_half_degree():
    d = caster_spacing_m()
    pitch = abs(sprung_chassis_pitch_deg(WORST_CASE_MOMENT_NM, RECOMMENDED, d))
    assert pitch <= 0.5
    assert pitch == pytest.approx(0.27, abs=0.03)
    assert suspension_meets_pitch_bound(WORST_CASE_MOMENT_NM, RECOMMENDED, d, bound_deg=0.5)
    # vs the unsprung rig, which rocks the full +-4.5 deg -> >> the 2.0 carton bound.
    assert free_pitch_travel_deg(0.03, d) > 2.0


def test_sprung_pitch_is_monotone_signed_and_bilinear():
    d = caster_spacing_m()
    assert sprung_chassis_pitch_deg(0.0, RECOMMENDED, d) == 0.0
    # sign tracks the moment.
    assert sprung_chassis_pitch_deg(100.0, RECOMMENDED, d) > 0
    assert sprung_chassis_pitch_deg(-100.0, RECOMMENDED, d) < 0
    # monotone increasing in |moment|.
    vals = [abs(sprung_chassis_pitch_deg(m, RECOMMENDED, d)) for m in (0, 50, 91.2, 150, 250)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))
    # a stiffer spring gives a smaller pitch at the same moment.
    stiffer = CasterSuspension(spring_n_per_m=6.0e5, preload_n=120.0, travel_m=0.02)
    assert abs(sprung_chassis_pitch_deg(250.0, stiffer, d)) < abs(sprung_chassis_pitch_deg(250.0, RECOMMENDED, d))


def test_caster_suspension_rejects_bad_params():
    for bad in (dict(spring_n_per_m=0.0), dict(spring_n_per_m=-1.0), dict(travel_m=0.0),
                dict(preload_n=-1.0), dict(damping_n_s_per_m=-1.0)):
        with pytest.raises(ValueError):
            CasterSuspension(**{**dict(spring_n_per_m=3e5, preload_n=120.0, travel_m=0.02), **bad})


# ---- (3) criterion (b): traction share + budget dominance ---------------------------------------

def test_suspension_share_light_preload_keeps_traction():
    # light preload -> most of the weight stays on the drive wheels (near the caster_lift 0.03's 0.95).
    assert suspension_drive_wheel_share(120.0) == pytest.approx(0.902, abs=1e-3)
    # monotone DECREASING in the preload (heavier preload unloads the drive wheels).
    shares = [suspension_drive_wheel_share(p) for p in (0, 60, 120, 240, 480)]
    assert all(b <= a for a, b in zip(shares, shares[1:]))
    with pytest.raises(ValueError):
        suspension_drive_wheel_share(-1.0)


def test_suspension_budget_strictly_dominates_stuck_and_shows_lift03_margin():
    stuck = static_traction_budget(wheel_friction=1.0, caster_lift_m=0.005, drive_max_force_nm=80.0)
    lift03 = static_traction_budget(wheel_friction=1.0, caster_lift_m=0.03, drive_max_force_nm=200.0)
    susp = static_traction_budget(wheel_friction=1.0, caster_lift_m=0.0, drive_max_force_nm=200.0,
                                  suspension_preload_n=120.0)
    assert susp['margin'] > stuck['margin']
    assert susp['drive_wheel_share'] > stuck['drive_wheel_share']
    assert susp['suspension_preload_n'] == 120.0
    # the suspension sits just below the caster_lift 0.03 config (it does not fully match the airborne
    # share, but keeps the tow traction well above the stuck rig).
    assert stuck['margin'] < susp['margin'] <= lift03['margin']


def test_budget_without_preload_is_byte_identical():
    a = static_traction_budget(wheel_friction=1.0, caster_lift_m=0.03, drive_max_force_nm=200.0)
    assert a['suspension_preload_n'] is None
    assert a['drive_wheel_share'] == pytest.approx(0.95)


# ---- (3b) coldwin38: the sprung-caster DYNAMIC solver-stability ratio -------------------------------

def test_shipped_light_suspender_is_solver_unstable():
    # coldwin38 FAIL reproduction: k=3e5 on the original 0.2 kg suspender (m_sprung ~1.0) gives
    # k*dt^2/m ~= 83 -- far above the stable ceiling and the arm's GPU-stable ~40 -- so the drive
    # exploded at reset (BASE_ATT -7.4..+12.2 deg before any arm motion).
    ratio = suspension_solver_stability_ratio(3.0e5, 0.2)
    assert ratio == pytest.approx(83.3, abs=1.0)
    assert ratio > SUSPENSION_STABILITY_RATIO_MAX
    assert not suspension_solver_stable(3.0e5, 0.2)


def test_corrected_heavier_suspender_is_solver_stable():
    # The fix: keep k=3e5 (so the steady pitch bound 0.26 deg is preserved) but raise the sprung mass
    # into the solver-stable band.
    ratio = suspension_solver_stability_ratio(3.0e5, SUSPENDER_MASS_KG)
    assert ratio <= SUSPENSION_STABILITY_RATIO_MAX
    assert suspension_solver_stable(3.0e5, SUSPENDER_MASS_KG)
    # the pitch bound is untouched by the mass (stiffness unchanged).
    assert abs(sprung_chassis_pitch_deg(WORST_CASE_MOMENT_NM, RECOMMENDED, caster_spacing_m())) <= 0.5


def test_min_suspender_mass_sizes_the_fix():
    # the sizing helper returns the least suspender mass that is stable at k, and that mass IS stable.
    m = min_suspender_mass_kg(3.0e5)
    assert m == pytest.approx(1.98, abs=0.05)
    assert suspension_solver_stable(3.0e5, m + 1e-6)
    # a raised physics_dt relaxes the ratio (dt^2), so a lighter mass becomes stable at a coarser step.
    assert suspension_solver_stability_ratio(3.0e5, 1.0, dt_s=1/120.) < \
           suspension_solver_stability_ratio(3.0e5, 1.0, dt_s=1/60.)


# ---- (4) the opt-in MobileSpec field + byte-identical-when-absent asset authoring ----------------

def _source(tmp_path):
    p = tmp_path / 'arm.urdf'
    p.write_text('''<robot name="test_fixture">
      <link name="world"/><link name="pedestal_link"/><link name="pedestal_top"/>
      <link name="ur_base_link"/><link name="arm"/><link name="ur_tool0"/>
      <joint name="pedestal_joint" type="fixed"><parent link="world"/><child link="pedestal_link"/></joint>
      <joint name="pedestal_top_joint" type="fixed"><parent link="pedestal_link"/><child link="pedestal_top"/></joint>
      <joint name="ur_base_joint" type="fixed"><parent link="pedestal_top"/><child link="ur_base_link"/></joint>
      <joint name="shoulder" type="revolute"><parent link="ur_base_link"/><child link="arm"/>
      <limit lower="-3" upper="3" effort="100" velocity="1"/></joint>
      <joint name="tool0_joint" type="fixed"><parent link="arm"/><child link="ur_tool0"/>
      <origin xyz="0 0 0.3" rpy="0 0 0"/></joint>
      <ros2_control name="fixture"><joint name="shoulder"><state_interface name="position">
      <param name="initial_value">-2.5</param></state_interface></joint></ros2_control>
      </robot>''')
    return p


def _susp_spec(**kw):
    base = dict(caster_suspension=True, caster_spring_n_per_m=3e5, caster_preload_n=120.0,
                caster_damping_n_s_per_m=2000.0, caster_travel_m=0.02,
                caster_suspender_mass_kg=SUSPENDER_MASS_KG)
    base.update(kw)
    return MobileSpec(**base)


def test_spec_suspension_opt_in_and_mutually_exclusive_with_lift():
    s = _susp_spec()
    assert s.caster_suspension and s.caster_lift_m == 0.0 and s.caster_suspender_mass_kg == SUSPENDER_MASS_KG
    # a sprung caster cannot also be recessed.
    with pytest.raises(ValueError):
        _susp_spec(caster_lift_m=0.02)
    # enabling needs a positive spring + stroke.
    with pytest.raises(ValueError):
        MobileSpec(caster_suspension=True)
    # enabling needs a positive sprung mass (coldwin38: a too-light mass is refused).
    with pytest.raises(ValueError):
        MobileSpec(caster_suspension=True, caster_spring_n_per_m=3e5, caster_travel_m=0.02)
    # suspension params set while off is a config slip, refused.
    with pytest.raises(ValueError):
        MobileSpec(caster_spring_n_per_m=3e5)
    with pytest.raises(ValueError):
        MobileSpec(caster_suspender_mass_kg=4.0)
    # the default spec never carries the suspension (byte-identical legacy).
    assert MobileSpec().caster_suspension is False


def test_spec_refuses_solver_unstable_suspension():
    # coldwin38 guard: the shipped-FAIL light sprung mass (0.2 kg) at k=3e5 is refused by MobileSpec.
    with pytest.raises(ValueError, match='solver-UNSTABLE'):
        _susp_spec(caster_suspender_mass_kg=0.2)
    # a sufficiently heavy sprung mass is accepted.
    assert _susp_spec(caster_suspender_mass_kg=4.0).caster_suspender_mass_kg == 4.0


def test_suspension_off_urdf_is_byte_identical(tmp_path):
    src = _source(tmp_path)
    build_mobile_urdf(src, tmp_path / 'off.urdf')               # default spec
    links = parse_urdf(tmp_path / 'off.urdf').links
    assert 'front_caster_suspender' not in links and 'rear_caster_suspender' not in links
    root = ET.parse(tmp_path / 'off.urdf').getroot()
    assert root.find("joint[@name='front_caster_lift']") is None
    # the legacy swivel parents straight off base_link.
    assert root.find("joint[@name='front_caster_swivel']/parent").get('link') == 'base_link'


def test_suspension_on_inserts_sprung_prismatic(tmp_path):
    src = _source(tmp_path)
    spec = _susp_spec()
    out = tmp_path / 'on.urdf'
    build_mobile_urdf(src, out, spec)
    robot = parse_urdf(out)
    assert find_roots(robot) == ['base_link']
    assert len(link_world_transforms(robot, {})) == len(robot.links)   # connected tree
    assert 'front_caster_suspender' in robot.links and 'rear_caster_suspender' in robot.links
    root = ET.parse(out).getroot()
    for label in ('front', 'rear'):
        lift = root.find(f"joint[@name='{label}_caster_lift']")
        assert lift.get('type') == 'prismatic'
        assert lift.find('parent').get('link') == 'base_link'
        assert lift.find('child').get('link') == f'{label}_caster_suspender'
        assert lift.find('axis').get('xyz') == '0 0 1'
        lim = lift.find('limit')
        assert float(lim.get('lower')) == pytest.approx(-0.02) and float(lim.get('upper')) == pytest.approx(0.02)
        # the swivel now hangs off the suspender, not base_link.
        assert root.find(f"joint[@name='{label}_caster_swivel']/parent").get('link') == f'{label}_caster_suspender'
    # sprung caster wheel bottom sits at the drive-wheel plane (contact, lift forced 0).
    t = link_world_transforms(robot, {})
    drive_bottom = t['left_wheel'][2, 3] - spec.wheel_radius
    assert (t['front_caster_wheel'][2, 3] - 0.065) == pytest.approx(drive_bottom, abs=1e-6)


# ---- (5) the config_preflight pitch-stability guard ---------------------------------------------

def test_preflight_refuses_unsprung_seesaw_rig():
    problems = check_pitch_stability({'length': 1.0, 'caster_lift_m': 0.03}, {}, {})
    assert problems and 'PITCH-STABILITY' in problems[0]


def test_preflight_allows_suspension_or_station_hold():
    # compliant suspension in the spec clears it.
    assert check_pitch_stability(
        {'length': 1.0, 'caster_lift_m': 0.0, 'caster_suspension': True}, {}, {}) == []
    # the sim-only station pitch hold (either config layer) clears it.
    assert check_pitch_stability({'length': 1.0, 'caster_lift_m': 0.03}, {},
                                 {'chassis_station_pitch_hold': True}) == []
    # a near-coplanar rig (free travel under the bound-margin) is allowed without a stabiliser.
    assert check_pitch_stability({'length': 1.0, 'caster_lift_m': 0.005}, {}, {}) == []
    # no manifest -> no-op (existing invocations unchanged).
    assert check_pitch_stability({}, {}, {}) == []
