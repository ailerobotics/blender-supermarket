"""sim-warm-reset-stage-clean-v1: prove the warm-session reset leaves the stage CLEAN so the
between-runs census passes and the second attempt warm re-seats instead of paying the ~215 s cold boot
(CPU; no Isaac/ROS/GPU).

warmtwo1 (PR 299 warm two-run, --warm-attempts 2, collect-only) cold-booted attempt 2 with:

    WARM_RESET decision=cold_boot cold boot: stage_clean: leaked prims: /CartParkingBrake,
    /MobileGraspBrake, /RestockPanelGripper/sim_attach_fallback; missing prims:
    /Render/OmniverseKit/HydraTextures/Replicator_01[...], .../Replicator_03[...],
    /World/restock_camera_shelf/Camera, /World/restock_camera_source/Camera

because (a) the failed collect attempt left the base/grasp brakes engaged and the attach-fallback weld
held (never released on the fail teardown), and (b) the stage-scoped / VRAM teardown destroyed the shelf
+ source station cameras (and their RTX render products), and the launcher read the census BEFORE the
reset ran -- so ``decide_reset`` cold-booted on the dirty post-run stage and ``reset_host_state`` never
ran.

These tests seed a FAKE stage with exactly that recorded dirty state and prove the fix:
  * ``reset_host_state`` releases each leaked brake/weld at its OWNER and restores each torn-down camera;
  * the census (read AFTER the reset) is then clean and ``plan_and_run_reset`` warm re-seats;
  * a stage the reset cannot clean still fires the same named ``WARM_RESET decision=cold_boot`` line;
  * the volatile ``/Render/OmniverseKit/HydraTextures/*`` render-target prims are excluded from the
    census (re-created on demand), while the stable restock-camera prims are matched + restored.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.restock import warm_reset_hooks as wrh  # noqa: E402
from twin.restock import warm_reset_provider as wrp  # noqa: E402
from twin.restock import warm_session as ws  # noqa: E402


# --------------------------------------------------------------------------- recorded warmtwo1 census

LEAKED = ('/CartParkingBrake', '/MobileGraspBrake', '/RestockPanelGripper/sim_attach_fallback')
MISSING_CAMERAS = ('/World/restock_camera_shelf/Camera', '/World/restock_camera_source/Camera')
# The Isaac RTX render-target prims the teardown destroyed (a subset of the recorded list; the full
# SDGPipeline children all hang under these two Replicator roots). All volatile -> census-excluded.
MISSING_RENDER = (
    '/Render/OmniverseKit/HydraTextures/Replicator_01',
    '/Render/OmniverseKit/HydraTextures/Replicator_01/PostRender/SDGPipeline',
    '/Render/OmniverseKit/HydraTextures/Replicator_03',
    '/Render/OmniverseKit/HydraTextures/Replicator_03/PostRender/SDGPipeline',
)
# A representative clean post-boot stage: the store + cart + the two station cameras + their render
# products + the backstock support. No brakes (they only engage mid-collect), no weld.
CLEAN_STAGE = ('/Supermarket', '/Rollwagen', '/RestockPanelGripper',
               '/RestockScenario/HandlingUnit') + MISSING_CAMERAS + MISSING_RENDER


# --------------------------------------------------------------------------- fakes

class _Prim:
    def __init__(self, path):
        self._p = path

    def GetPath(self):
        return type('P', (), {'pathString': self._p})()


class _FakeStage:
    """A prim-path set with the USD methods the census + reset touch. RemovePrim tolerates an absent
    path (returns False, like USD), so a prim pruned by reset_stage and then released by its owner does
    not raise."""
    def __init__(self, paths):
        self._paths = set(paths)
        self.defined = []

    def Traverse(self):
        return [_Prim(p) for p in sorted(self._paths)]

    def GetPrimAtPath(self, path):
        return _Prim(path) if str(path) in self._paths else None

    def RemovePrim(self, path):
        if str(path) in self._paths:
            self._paths.discard(str(path))
            return True
        return False

    def add(self, path):
        self._paths.add(str(path))


class _FakeMobile:
    """Owns the sim brakes: each release RemovePrim's its brake and clears its own engaged flag, so a
    future engage does not no-op on a stale flag (the warmtwo1 defect)."""
    def __init__(self, stage):
        self.stage = stage
        self._cart_braked = '/CartParkingBrake' in stage._paths
        self._base_full_braked = '/MobileGraspBrake' in stage._paths
        self._braked = '/MobileBrake' in stage._paths

    def release_cart_brake(self, *, force=False):
        self.stage.RemovePrim('/CartParkingBrake')
        self._cart_braked = False

    def release_base_full_brake(self):
        self.stage.RemovePrim('/MobileGraspBrake')
        self._base_full_braked = False

    def release_brake(self):
        self.stage.RemovePrim('/MobileBrake')
        self._braked = False


class _FakeContact:
    """Owns the attach-fallback weld; release RemovePrim's the FixedJoint + clears the held body."""
    def __init__(self, stage, joint='/RestockPanelGripper/sim_attach_fallback'):
        self.stage = stage
        self._fallback_joint = joint if joint in stage._paths else None
        self._fallback_body = '/RestockScenario/HandlingUnit' if self._fallback_joint else None

    def release_sim_attach(self):
        if self._fallback_joint is not None:
            self.stage.RemovePrim(self._fallback_joint)
        self._fallback_joint = self._fallback_body = None


def _sensing(stage, cameras=MISSING_CAMERAS):
    """A sensing_sources mapping whose restore_cameras re-authors the torn-down camera prims (the OWNER,
    CameraRig.recreate_destroyed, does this on the live rig)."""
    def restore_cameras():
        restored = [c for c in cameras if c not in stage._paths]
        for c in restored:
            stage.add(c)
        return restored
    return {'restore_cameras': restore_cameras}


class _FakeWorld:
    def __init__(self, stage):
        self.current_time = 3.0
        self.scene = type('S', (), {'remove_object': lambda self, n: None})()

    def reset(self):
        self.current_time = 0.0


class _FakeApp:
    def is_running(self):
        return True


class _FakeGate:
    def __init__(self):
        self.faulted = True

    def resume(self):
        self.faulted = False

    def hold(self):
        self.faulted = True


class _FakeAssembly:
    def __init__(self, stage):
        self.stage = stage
        self.world = _FakeWorld(stage)
        self.mobile = _FakeMobile(stage)
        self.contact = _FakeContact(stage)
        self.sensing_sources = _sensing(stage)


def _dirty_stage():
    """The recorded post-run dirty stage: the clean set minus the two destroyed cameras + their render
    products, plus the three leaked brake/weld prims."""
    present = set(CLEAN_STAGE) - set(MISSING_CAMERAS) - set(MISSING_RENDER) | set(LEAKED)
    return _FakeStage(present)


def _parts(gate):
    return {'port': type('P', (), {'command_gate': gate})(), 'manipulation': None, 'adapter': None}


# --------------------------------------------------------------------------- reset_host_state cleanup

def test_reset_host_state_releases_every_leaked_prim_at_its_owner():
    stage = _dirty_stage()
    assembly = _FakeAssembly(stage)
    hooks = wrh.build(app=_FakeApp(), assembly=assembly, parts=_parts(_FakeGate()))
    assert hooks.reset_host_state(attempt=2, now=2.0) is True
    for leaked in LEAKED:
        assert leaked not in stage._paths, f'{leaked} not released by its owner'
    # The owners cleared their engaged flags too (so a future brake does not leak again).
    assert assembly.mobile._cart_braked is False and assembly.mobile._base_full_braked is False
    assert assembly.contact._fallback_joint is None


def test_reset_host_state_restores_every_torn_down_camera():
    stage = _dirty_stage()
    assembly = _FakeAssembly(stage)
    wrh.build(app=_FakeApp(), assembly=assembly,
              parts=_parts(_FakeGate())).reset_host_state(attempt=2, now=2.0)
    for cam in MISSING_CAMERAS:
        assert cam in stage._paths, f'{cam} not restored'


def test_reset_host_state_release_is_best_effort_when_an_owner_is_absent():
    # No mobile / contact on the assembly: the step does not crash (the post-reset census is the
    # authority and will cold-boot on the leftover). Proven by it completing + leaving the leak.
    stage = _dirty_stage()
    assembly = type('A', (), {'stage': stage, 'world': _FakeWorld(stage),
                              'sensing_sources': _sensing(stage)})()
    hooks = wrh.build(app=_FakeApp(), assembly=assembly, parts=_parts(_FakeGate()))
    assert hooks.reset_host_state(attempt=1, now=1.0) is True
    assert '/CartParkingBrake' in stage._paths   # nothing released it -> census will catch it


# --------------------------------------------------------------------------- provider census + exclusion

def _provider(stage, assembly, *, config=None):
    config = config or {}
    hooks = wrp._resolve_hooks(config, app=_FakeApp(), assembly=assembly,
                               parts=_parts(_FakeGate()), plan=None, output='/x')
    ops = wrp._build_production_ops(app=_FakeApp(), assembly=assembly, parts=_parts(_FakeGate()),
                                    plan=None, config=config, output='/x', scenario={}, hooks=hooks)
    return wrp.build(app=_FakeApp(), assembly=assembly, parts=_parts(_FakeGate()), plan=None,
                     config=config, output='/x', scenario={}, _ops=ops)


def test_volatile_render_prims_are_excluded_from_the_census():
    # A stage differing from the baseline ONLY in the Replicator render-target subtree is NOT dirty
    # (those are re-created on demand when the camera render product re-initializes).
    base = _FakeStage(CLEAN_STAGE)
    prov = _provider(base, _FakeAssembly(base))
    # Drop every render prim: still clean (excluded on both sides).
    for p in MISSING_RENDER:
        base.RemovePrim(p)
    st = prov.stage_census()
    assert st.leaked() == () and st.missing() == ()
    # But a leaked WORLD prim is still caught (the exclusion is scoped to the RTX namespace).
    base.add('/World/ghost')
    assert '/World/ghost' in prov.stage_census().leaked()


def test_full_provider_reset_then_census_is_clean_on_the_warmtwo1_stage():
    # Baseline captured from the CLEAN stage; then make it dirty exactly like warmtwo1 and run the FULL
    # provider reset (stop_runtime -> reset_stage -> reseat -> reset_host_state -> verify). The census
    # read AFTER the reset is clean.
    stage = _FakeStage(CLEAN_STAGE)
    assembly = _FakeAssembly(stage)
    prov = _provider(stage, assembly)          # baseline snapshot = clean stage (render prims excluded)
    for cam in MISSING_CAMERAS:                 # camera teardown
        stage.RemovePrim(cam)
    for p in MISSING_RENDER:                    # render-product teardown
        stage.RemovePrim(p)
    for leaked in LEAKED:                        # brakes + weld left engaged
        stage.add(leaked)
    assert prov.stage_census().leaked() and prov.stage_census().missing()  # dirty before the reset
    assert prov(attempt=2, now=2.0) is True
    st = prov.stage_census()
    assert st.leaked() == () and st.missing() == (), (st.leaked(), st.missing())


# --------------------------------------------------- plan_and_run_reset: reset-then-census ordering

def _run(prov, *, prev=None, nxt=None, logs=None):
    steps = [ws.SequenceStep('reset_stage_and_reseed', lambda: prov(attempt=2, now=2.0)),
             ws.SequenceStep('preflight_verify', lambda: None)]
    return ws.plan_and_run_reset(
        prev or {}, nxt or {}, steps, census=prov.stage_census,
        vram_used_mib=8300, post_boot_baseline_mib=8200,
        on_log=(logs.append if logs is not None else None))


def test_warmtwo1_replay_warm_reseats_after_the_fixed_reset():
    stage = _FakeStage(CLEAN_STAGE)
    assembly = _FakeAssembly(stage)
    prov = _provider(stage, assembly)
    for cam in MISSING_CAMERAS:
        stage.RemovePrim(cam)
    for p in MISSING_RENDER:
        stage.RemovePrim(p)
    for leaked in LEAKED:
        stage.add(leaked)
    steps = [ws.SequenceStep('reset_stage_and_reseed', lambda: prov(attempt=2, now=2.0)),
             ws.SequenceStep('preflight_verify', lambda: None)]
    logs = []
    out = ws.plan_and_run_reset({}, {}, steps, census=prov.stage_census,
                                vram_used_mib=8300, post_boot_baseline_mib=8200, on_log=logs.append)
    assert out.reused is True, out.reason
    # The reset sequence actually RAN (the WARM_RESET step lines warmtwo1 never reached).
    assert any('WARM_RESET step=reset_stage_and_reseed' in m for m in logs)
    assert not any('decision=cold_boot' in m for m in logs)


def test_dirty_stage_the_reset_cannot_clean_still_fires_the_named_cold_boot_line():
    # A non-camera baseline prim torn down and NOT restorable by the reset (restore_cameras only
    # re-authors the station cameras; reset_stage prunes leaks but never restores a missing prim): the
    # reset runs but the post-reset census is still dirty, so the SAME named WARM_RESET decision=cold_boot
    # line fires and the session is NOT reused (the census is never loosened to tolerate it).
    stage = _FakeStage(CLEAN_STAGE)
    assembly = _FakeAssembly(stage)
    prov = _provider(stage, assembly)            # baseline snapshot = the full clean stage
    stage.RemovePrim('/Supermarket')             # the store root torn down; nothing re-creates it
    logs = []
    out = _run(prov, logs=logs)
    assert out.reused is False
    cold = [m for m in logs if 'WARM_RESET decision=cold_boot' in m]
    assert cold and 'stage_clean' in cold[-1] and 'missing prims: /Supermarket' in cold[-1]


def test_host_key_change_cold_boots_without_running_the_reset():
    # The non-stage gate runs FIRST: a HOST/boot-fixed key delta cold-boots without touching the stage.
    stage = _FakeStage(CLEAN_STAGE)
    assembly = _FakeAssembly(stage)
    prov = _provider(stage, assembly)
    ran = []
    steps = [ws.SequenceStep('reset_stage_and_reseed', lambda: ran.append('reset')),
             ws.SequenceStep('preflight_verify', lambda: None)]
    logs = []
    out = ws.plan_and_run_reset({'render_width': 960}, {'render_width': 640}, steps,
                                census=prov.stage_census, vram_used_mib=8300,
                                post_boot_baseline_mib=8200, on_log=logs.append)
    assert out.reused is False and ran == []     # the reset never ran (stage untouched)
    assert any('render_width' in m for m in logs)
