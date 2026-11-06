"""CPU tests for the startup-stall release-height guard (sim-place-verify-rest-tolerance-v1).

coldwin85 (an exact re-run of coldwin83) set the carton down, released it, finished the long retreat,
then the collect PLACE verify failed: ``carton not resting on deck (xy=1.219 bottom_gap=-0.954)`` -- the
carton had come to rest ~1 m below the shelf, on the FLOOR. coldwin83 passed the SAME verify.

Root cause (verified from each run's host-stack.log PLACE_RELEASE_TICK + nav-trace RESTOCK_TRACE_POSES):
the contact-gated place release fired on its STALL trigger at the descent START height, ~0.06 m ABOVE the
planned set-down, in BOTH runs. The render-paced descent trajectory had not produced motion yet, so the
measured carton centre z was FROZEN at the standoff (cw85 0.6251, cw83 0.6256) for the first few ticks;
the "no downward progress for stall_ticks" rule read that frozen startup as "support met" and released the
weld while the carton was still cradled ~0.06 m above the shelf (cw85 release_z 0.5628, cw83 0.5635):

    cw85  PLACE_RELEASE_TICK now=245.250 carton_z=0.6251 stall=0 action=press
          PLACE_CONTACT_RELEASE_ARM start_carton_z=0.6251 -> release at carton_z<=0.5628
          PLACE_RELEASE_TICK now=245.517 carton_z=0.6251 stall=5 action=release trigger=stall   (0.062 m HIGH)
    cw83  PLACE_RELEASE_TICK now=257.300 carton_z=0.6257 stall=0 action=press
          PLACE_CONTACT_RELEASE_ARM start_carton_z=0.6257 -> release at carton_z<=0.5635
          PLACE_RELEASE_TICK now=257.517 carton_z=0.6256 stall=5 action=release trigger=stall   (0.062 m HIGH)

With the carton released high and still cradled in the side grip, whether it then slips free onto the shelf
(cw83: dropped 0.125 m, settled at world z=1.022, stable through verify) or stays perched and is carried off
by the retreat fold (cw85: rode up to world z=1.77 as the arm folded, then fell to the floor at z=0.08) is a
coin flip -- the rest is non-deterministic. nav-trace confirms the cart and base did NOT move in either run,
so the verify frame is correct; the carton genuinely left the shelf.

The fix (opt-in ``carton_level.place_release_stall_height_margin_m``, default None => byte-identical): a
descent STALL only counts as "support met" when the carton centre z is within ``release_z + margin`` of the
planned set-down height. The startup stall ~0.06 m HIGH is suppressed, the descent keeps pressing, and the
weld releases at the TRUE set-down (the z / effort / support-contact triggers), so the carton rests on the
shelf deterministically. Never touches the 2.0 deg carton tilt bound and never makes verify pass a carton
that is not resting on the support (it makes the carton actually rest).

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts python3 -m pytest \
        scripts/twin/tests/test_place_verify_rest_tolerance_v1.py
"""
import pytest

from twin.restock_manipulation import carton_level as cl
from twin.restock_manipulation.place_release import PlaceReleaseGate

# Exact recorded numbers (host-stack.log PLACE_CONTACT_RELEASE_ARM / PLACE_RELEASE_TICK).
CW85 = dict(start_z=0.6251, release_z=0.5628)   # stall fired at 0.6251, 0.0623 m above set-down
CW83 = dict(start_z=0.6256, release_z=0.5635)   # stall fired at 0.6256, 0.0621 m above set-down
MARGIN = 0.02                                   # the ON value shipped in the example fragment


def _gate(release_z, margin=None, stall_ticks=5):
    return PlaceReleaseGate(release_z=release_z, effort_margin_nm=100.0, effort_consecutive=3,
                            stall_eps_m=0.001, stall_ticks=stall_ticks, support_tokens=('shelf', 'deck'),
                            obstruction_gap_mm=2.0, stall_height_margin_m=margin)


def _frozen_descent(gate, z, ticks):
    """Feed ``ticks`` descent observations with the carton centre FROZEN at ``z`` (the render-paced
    startup stall: trajectory not moving yet, no effort rise, no seal hit). Return the last decision."""
    d = None
    for i in range(ticks):
        d = gate.observe(now=i * 0.05, carton_z=z, commanded_down=True, arm_effort_max=124.0,
                         seal_gap_mm=None, seal_hit_prims=None)
    return d


# --------------------------------------------------------------- the bug reproduces with the guard OFF

@pytest.mark.parametrize('run', [CW85, CW83])
def test_startup_stall_releases_without_guard(run):
    """Default (margin None) => the frozen startup stall fires release:stall ~0.06 m above set-down
    (reproduces the coldwin85 / coldwin83 release-too-high that left the carton cradled)."""
    g = _gate(run['release_z'], margin=None)
    d = _frozen_descent(g, run['start_z'], 7)
    assert d.action == 'release' and d.trigger == 'stall'
    assert d.carton_z > run['release_z'] + 0.04     # released well ABOVE the planned set-down


# --------------------------------------------------------------- the guard suppresses the startup stall

@pytest.mark.parametrize('run', [CW85, CW83])
def test_startup_stall_suppressed_with_guard(run):
    """With the height guard on, a stall whose carton_z is still above release_z + margin NEVER releases
    -- the gate keeps pressing so the descent can continue to the real set-down."""
    g = _gate(run['release_z'], margin=MARGIN)
    d = _frozen_descent(g, run['start_z'], 20)      # far more than stall_ticks
    assert d.action == 'press' and d.trigger is None
    assert g._stall >= g.stall_ticks                # the stall WAS counted -- only the release is gated


@pytest.mark.parametrize('run', [CW85, CW83])
def test_release_fires_at_true_setdown_with_guard(run):
    """With the guard on, once the descent actually reaches the set-down height the z trigger fires -- the
    weld releases at the TRUE set-down, not 0.06 m high. (cw85/cw83 both set down on the shelf.)"""
    g = _gate(run['release_z'], margin=MARGIN)
    _frozen_descent(g, run['start_z'], 7)           # startup stall -- suppressed, still pressing
    # The trajectory now moves: the carton descends smoothly to the planned set-down height.
    d = None
    z = run['start_z']
    while z > run['release_z'] - 0.005:
        z -= 0.002
        d = g.observe(now=1.0 + z, carton_z=z, commanded_down=True, arm_effort_max=124.0,
                      seal_gap_mm=None, seal_hit_prims=None)
        if d.action == 'release':
            break
    assert d.action == 'release' and d.trigger == 'z'
    assert d.carton_z <= run['release_z'] + 1e-9    # released AT the set-down height, not above it


# --------------------------------------------------- a GENUINE stall near the set-down still fires (no over-suppression)

def test_real_stall_near_setdown_still_releases_with_guard():
    """The guard must not suppress a real support contact: a stall WITHIN release_z + margin (but still
    above release_z, so the z trigger has not fired) releases on stall as before."""
    rz = 0.5624
    g = _gate(rz, margin=MARGIN)
    # Descend to just inside the window (rz + 0.012 < rz + margin 0.02), then stall there.
    z = 0.60
    d = None
    for i in range(6):                              # descending -- stall stays 0
        z -= 0.005
        d = g.observe(now=i * 0.05, carton_z=max(z, rz + 0.012), commanded_down=True,
                      arm_effort_max=124.0)
    for i in range(6, 14):                          # frozen at rz+0.012 (inside the window) -> stall fires
        d = g.observe(now=i * 0.05, carton_z=rz + 0.012, commanded_down=True, arm_effort_max=124.0)
    assert d.action == 'release' and d.trigger == 'stall'
    assert rz < d.carton_z <= rz + MARGIN           # fired inside the window, above release_z


def test_guard_inert_when_release_z_unknown():
    """With no release_z reference, the height guard cannot gate -- the stall behaves as default (fires)."""
    g = _gate(release_z=None, margin=MARGIN)
    d = _frozen_descent(g, 0.60, 7)
    assert d.action == 'release' and d.trigger == 'stall'


# ----------------------------------------------------------------------- config resolve + validation + off

def test_config_default_off_byte_identical():
    c = cl.resolve_level_config({'place_release_on_contact': True})
    assert c.place_release_stall_height_margin_m is None


def test_config_resolve_margin():
    c = cl.resolve_level_config({'place_release_on_contact': True,
                                 'place_release_stall_height_margin_m': 0.02})
    assert c.place_release_stall_height_margin_m == 0.02


@pytest.mark.parametrize('bad', [0, -0.01, float('nan'), float('inf')])
def test_config_validation_rejects_bad_margin(bad):
    with pytest.raises(ValueError):
        cl.resolve_level_config({'place_release_on_contact': True,
                                 'place_release_stall_height_margin_m': bad})


def test_off_matches_legacy_gate_byte_identical():
    """A gate built with margin None is byte-identical to one built without the kwarg at all."""
    legacy = PlaceReleaseGate(release_z=0.5628, support_tokens=('shelf', 'deck'))
    guarded_off = _gate(0.5628, margin=None)
    seqs = []
    for g in (legacy, guarded_off):
        out = []
        for i in range(7):
            out.append(g.observe(now=i * 0.05, carton_z=0.6251, commanded_down=True,
                                 arm_effort_max=124.0).action)
        seqs.append(out)
    assert seqs[0] == seqs[1]
    assert seqs[0][-1] == 'release'
