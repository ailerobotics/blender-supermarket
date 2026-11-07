"""CPU tests for the place-descent PhysX contact-pair logger (sim-place-contact-logger-view-safe-v1,
following sim-pad-down-descent-jam-physics-v1).

The PURE core (path classification, first/deepen log decision, line format) is tested directly. The
boot-path wiring is VIEW-SAFE: the contact-report API is AUTHORED before the tensor-backed simulation view
is created (``wire_place_contact_logger_pre_reset``, from ``physics_setup.assemble``) and only the callback
is SUBSCRIBED after reset (``wire_place_contact_logger``, from ``isaac_factory.create``). The ordering test
drives a fake world that RAISES on any prim authoring after the view exists (reproducing the coldwin101
invalidation) and proves an arm DOF read still works -- with injected fakes it needs no pxr/Isaac/GPU.

``attach`` and the end-to-end wire test exercise the real ``pxr.UsdPhysics`` import, so they skip cleanly
when usd-core (``pxr``) is absent rather than fail. No Isaac / ROS / GPU anywhere.
"""
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from twin.restock_manipulation import place_contact_logger as pcl


# --------------------------------------------------------------------------- pure classification
def test_classify_gripper_vs_shelf():
    hit = pcl.classify_contact(['/MobileRobot/suction_cups', '/Rollwagen/chassis/shelf'])
    assert hit == ('/MobileRobot/suction_cups', '/Rollwagen/chassis/shelf', 'gripper')


def test_classify_housing_cube_is_gripper_kind():
    hit = pcl.classify_contact(['/Rollwagen/chassis/shelf', '/MobileRobot/cube_without_surface_plates'])
    assert hit[2] == 'gripper' and hit[1] == '/Rollwagen/chassis/shelf'


def test_classify_non_housing_arm_link_is_arm_kind():
    hit = pcl.classify_contact(['/MobileRobot/ur_forearm_link', '/Rollwagen/chassis/post_2'])
    assert hit == ('/MobileRobot/ur_forearm_link', '/Rollwagen/chassis/post_2', 'arm')


def test_classify_carton_vs_cart_is_carton_kind():
    hit = pcl.classify_contact(['/RestockScenario/HandlingUnit', '/Rollwagen/chassis/deck'])
    assert hit == ('/RestockScenario/HandlingUnit', '/Rollwagen/chassis/deck', 'carton')


def test_classify_ignores_non_cart_contacts():
    # arm vs floor, cart vs floor, arm vs arm -> not a robot/carton-vs-cart pair
    assert pcl.classify_contact(['/MobileRobot/ur_wrist_2_link', '/Supermarket/Floor/Floor']) is None
    assert pcl.classify_contact(['/Rollwagen/chassis/deck', '/Supermarket/Floor/Floor']) is None
    assert pcl.classify_contact(['/MobileRobot/ur_forearm_link', '/MobileRobot/ur_upper_arm_link']) is None
    assert pcl.classify_contact([]) is None


# --------------------------------------------------------------------------- first/deepen + format
def test_on_contact_logs_first_then_only_on_deepen():
    lines = []
    lg = pcl.PlaceContactLogger(deepen_step_m=0.005, log=lines.append)
    a, b = '/MobileRobot/suction_cups', '/Rollwagen/chassis/shelf'
    assert lg.on_contact(a, b, -0.010, kind='gripper') is not None      # first
    assert lg.on_contact(a, b, -0.012, kind='gripper') is None          # +2 mm < 5 mm step -> quiet
    assert lg.on_contact(a, b, -0.020, kind='gripper') is not None      # +10 mm -> deepen
    assert len(lines) == 2
    assert lines[0].startswith(pcl.CONTACT_PAIR) and 'event=first' in lines[0] and 'depth=10.0mm' in lines[0]
    assert 'event=deepen' in lines[1] and 'depth=20.0mm' in lines[1]
    assert 'a=/MobileRobot/suction_cups' in lines[0] and 'b=/Rollwagen/chassis/shelf' in lines[0]


def test_format_line_penetration_is_negative_separation():
    line = pcl.format_contact_line('/MobileRobot/suction_cups', '/Rollwagen/chassis/shelf', -0.0523,
                                   phase='place', event='first', kind='gripper')
    assert 'depth=52.3mm' in line and 'sep=-0.0523m' in line and 'phase=place' in line


def test_handle_header_paths_classifies_and_logs():
    lines = []
    lg = pcl.PlaceContactLogger(log=lines.append)
    assert lg.handle_header_paths(['/MobileRobot/surface_plate_6', '/Rollwagen/chassis/shelf'], -0.04)
    assert lines and 'kind=gripper' in lines[0] and 'depth=40.0mm' in lines[0]
    # a non-cart contact logs nothing
    assert lg.handle_header_paths(['/MobileRobot/ur_forearm_link', '/Supermarket/Floor/Floor'], -0.1) is None
    assert len(lines) == 1


# --------------------------------------------------------------------------- Isaac attach (injected fakes)
class _FakePrim:
    def __init__(self, path, rigid):
        self._path = path
        self._rigid = rigid

    def GetPath(self):
        return self._path

    def HasAPI(self, _api):
        return self._rigid


class _FakeStage:
    def __init__(self, prims):
        self._prims = prims

    def Traverse(self):
        return list(self._prims)


class _FakeHeader:
    def __init__(self, c0, c1, offset, n):
        self.collider0, self.collider1, self.actor0, self.actor1 = c0, c1, c0, c1
        self.contact_data_offset, self.num_contact_data = offset, n


class _FakePoint:
    def __init__(self, sep):
        self.separation = sep


class _FakeSim:
    def __init__(self):
        self.callback = None

    def subscribe_contact_report_events(self, cb):
        self.callback = cb
        return 'subscription-token'


def test_attach_instruments_bodies_subscribes_and_logs_contact():
    pytest.importorskip('pxr')  # attach() imports pxr.UsdPhysics for the RigidBodyAPI check; skip if absent
    # rigid bodies under the robot/carton/cart prefixes get the contact-report API; a non-matching prim and
    # a non-rigid prim do not.
    prims = [
        _FakePrim('/MobileRobot/suction_cups', True),
        _FakePrim('/MobileRobot/ur_tool0', False),               # not rigid -> skipped
        _FakePrim('/RestockScenario/HandlingUnit', True),
        _FakePrim('/Rollwagen/chassis/shelf', True),
        _FakePrim('/Supermarket/Floor/Floor', True),             # not a tracked prefix -> skipped
    ]
    applied = []
    lines = []
    # int code -> path, for the synthetic header
    paths = {1: '/MobileRobot/suction_cups', 2: '/Rollwagen/chassis/shelf'}
    lg = pcl.PlaceContactLogger(log=lines.append)
    lg.set_phase('place')
    sim = _FakeSim()
    n = lg.attach(_FakeStage(prims), sim,
                  apply_contact_report=lambda prim, thr: applied.append(str(prim.GetPath())),
                  decode_path=lambda i: paths.get(i, '/x/%d' % i))
    assert n == 3                                                # 3 rigid bodies under tracked prefixes
    assert set(applied) == {'/MobileRobot/suction_cups', '/RestockScenario/HandlingUnit',
                            '/Rollwagen/chassis/shelf'}
    assert sim.callback is not None                              # the logger subscribed
    # drive the subscribed callback with a synthetic gripper-vs-shelf contact (-52.3 mm penetration)
    sim.callback([_FakeHeader(1, 2, 0, 1)], [_FakePoint(-0.0523)])
    assert lines and lines[0].startswith(pcl.CONTACT_PAIR)
    assert 'a=/MobileRobot/suction_cups' in lines[0] and 'b=/Rollwagen/chassis/shelf' in lines[0]
    assert 'kind=gripper' in lines[0] and 'depth=52.3mm' in lines[0] and 'phase=place' in lines[0]


# --------------------------------------------------------------------------- boot-layer wire + byte-identical off
def test_wire_pre_reset_absent_block_is_none_no_isaac_import():
    # no block -> pre-reset wiring returns None WITHOUT importing pxr/omni.physx (byte-identical off)
    assert pcl.wire_place_contact_logger_pre_reset(_FakeStage([]), {}) is None
    assert pcl.wire_place_contact_logger_pre_reset(_FakeStage([]),
                                                   {'place_contact_logger': False}) is None


def test_wire_post_reset_absent_logger_is_none_no_isaac_import():
    # the post-reset subscribe is None (no omni.physx import) when the assembly carries no pre-authored logger
    assert pcl.wire_place_contact_logger(types.SimpleNamespace(), {}) is None
    assert pcl.wire_place_contact_logger(
        types.SimpleNamespace(place_contact_logger=None), {'place_contact_logger': True}) is None


class _OrderingWorld:
    """Model of the boot sequence: authoring a contact-report API is only legal BEFORE the tensor-backed
    simulation view is created. ANY prim authoring after the view exists is the coldwin101 defect -- it
    invalidates the view, after which a DOF read raises. The fake raises on any such late authoring."""

    def __init__(self):
        self.view_created = False
        self.view_valid = False
        self.authored = []          # (path, view_created_at_that_time)
        self._dofs = [0.11, 0.22, 0.33]

    def apply_contact_report(self, prim, thr):
        path = str(prim.GetPath())
        self.authored.append((path, self.view_created))
        if self.view_created:
            self.view_valid = False  # re-authoring a covered prim invalidated the view (coldwin101)
            raise AssertionError('prim %s authored AFTER the simulation view was created' % path)

    @staticmethod
    def is_rigid_body(prim):
        return prim.HasAPI(None)    # _FakePrim.HasAPI ignores the arg -> returns its rigid flag; no pxr

    def reset(self):
        # world.reset() builds the tensor-backed view over the prims authored so far
        self.view_created = True
        self.view_valid = True

    def get_dof_positions(self):
        if not self.view_valid:
            raise RuntimeError('Failed to get DOF positions from backend')
        return list(self._dofs)


def test_boot_ordering_authors_before_view_then_dof_read_survives(monkeypatch):
    # The view-safe boot path: author pre-reset (physics_setup.assemble), create the view (world.reset()),
    # subscribe post-reset (isaac_factory.create). No pxr needed -- apply_contact_report + is_rigid_body are
    # injected. Proves the fix: no authoring after the view, and an arm DOF read after attach succeeds.
    world = _OrderingWorld()
    prims = [_FakePrim('/MobileRobot/suction_cups', True),
             _FakePrim('/MobileRobot/ur_tool0', False),            # not rigid -> skipped
             _FakePrim('/RestockScenario/HandlingUnit', True),
             _FakePrim('/Rollwagen/chassis/shelf', True),
             _FakePrim('/Supermarket/Floor/Floor', True)]          # not a tracked prefix -> skipped
    stage = _FakeStage(prims)
    lines = []

    # --- PRE-RESET: author the contact-report API (before the view exists) ---
    logger = pcl.wire_place_contact_logger_pre_reset(
        stage, {'place_contact_logger': True}, log=lines.append,
        apply_contact_report=world.apply_contact_report, is_rigid_body=world.is_rigid_body)
    assert isinstance(logger, pcl.PlaceContactLogger)
    assert logger._armed_bodies == 3 and logger._attach_phase == 'pre-reset'
    assert {p for p, _ in world.authored} == {'/MobileRobot/suction_cups',
                                              '/RestockScenario/HandlingUnit', '/Rollwagen/chassis/shelf'}
    # every authoring happened BEFORE the view was created
    assert all(not created_at for _, created_at in world.authored)
    # the armed line names the attach phase + body count for the run worker to verify from the boot log
    armed = [ln for ln in lines if 'PLACE_CONTACT_LOGGER armed' in ln]
    assert armed and 'phase=pre-reset' in armed[0] and 'bodies=3' in armed[0]

    # --- world.reset() creates the tensor-backed view; it is VALID because nothing re-authored a prim ---
    world.reset()
    assert world.view_valid

    # --- POST-RESET: subscribe the callback (re-authors NO prim) via an injected omni.physx ---
    fake_sim = _FakeSim()
    fake_mod = types.ModuleType('omni.physx')
    fake_mod.get_physx_simulation_interface = lambda: fake_sim
    monkeypatch.setitem(sys.modules, 'omni', types.ModuleType('omni'))
    monkeypatch.setitem(sys.modules, 'omni.physx', fake_mod)
    assembly = types.SimpleNamespace(place_contact_logger=logger)
    subscribed = pcl.wire_place_contact_logger(assembly, {'place_contact_logger': True}, log=lines.append)
    assert subscribed is logger
    assert fake_sim.callback is not None                          # subscribed through the injected interface

    # the subscribe step authored NOTHING after the view, so the view stayed valid and a DOF read works --
    # this is the exact read (getDofPositions) that failed for the whole mission in coldwin101.
    assert all(not created_at for _, created_at in world.authored)
    assert world.view_valid
    assert world.get_dof_positions() == [0.11, 0.22, 0.33]


def test_subscribe_after_view_never_reauthors_prims(monkeypatch):
    # guard the invariant directly: wire_place_contact_logger (post-reset) must only subscribe, never author.
    world = _OrderingWorld()
    stage = _FakeStage([_FakePrim('/MobileRobot/suction_cups', True)])
    logger = pcl.wire_place_contact_logger_pre_reset(
        stage, {'place_contact_logger': True},
        apply_contact_report=world.apply_contact_report, is_rigid_body=world.is_rigid_body)
    world.reset()
    before = list(world.authored)
    fake_sim = _FakeSim()
    fake_mod = types.ModuleType('omni.physx')
    fake_mod.get_physx_simulation_interface = lambda: fake_sim
    monkeypatch.setitem(sys.modules, 'omni', types.ModuleType('omni'))
    monkeypatch.setitem(sys.modules, 'omni.physx', fake_mod)
    pcl.wire_place_contact_logger(types.SimpleNamespace(place_contact_logger=logger), {})
    assert world.authored == before                              # no new authoring during subscribe
    assert world.view_valid


def test_wire_present_block_authors_and_subscribes_end_to_end(monkeypatch):
    # end-to-end on a box WITH usd-core: pre-reset author (real pxr RigidBodyAPI path, apply injected) then
    # post-reset subscribe. Skips cleanly when pxr is absent rather than failing.
    pytest.importorskip('pxr')
    applied = []
    stage = _FakeStage([_FakePrim('/Rollwagen/chassis/shelf', True),
                        _FakePrim('/MobileRobot/suction_cups', True)])
    lines = []
    logger = pcl.wire_place_contact_logger_pre_reset(
        stage, {'place_contact_logger': True}, log=lines.append,
        apply_contact_report=lambda prim, thr: applied.append(str(prim.GetPath())))
    assert isinstance(logger, pcl.PlaceContactLogger)
    assert set(applied) == {'/Rollwagen/chassis/shelf', '/MobileRobot/suction_cups'}
    assert any('PLACE_CONTACT_LOGGER armed' in ln and 'phase=pre-reset' in ln for ln in lines)

    fake_sim = _FakeSim()
    fake_mod = types.ModuleType('omni.physx')
    fake_mod.get_physx_simulation_interface = lambda: fake_sim
    monkeypatch.setitem(sys.modules, 'omni', types.ModuleType('omni'))
    monkeypatch.setitem(sys.modules, 'omni.physx', fake_mod)
    subscribed = pcl.wire_place_contact_logger(types.SimpleNamespace(place_contact_logger=logger), {})
    assert subscribed is logger and fake_sim.callback is not None


def test_isaac_factory_subscribes_and_physics_setup_authors_on_the_boot_path():
    # pin the two-phase boot wiring: physics_setup.assemble authors PRE-RESET; isaac_factory.create
    # subscribes POST-RESET. A configured logger can never be left unattached, and the authoring never runs
    # after the view exists.
    from twin.restock_manipulation import isaac_factory
    src = Path(isaac_factory.__file__).read_text()
    assert 'wire_place_contact_logger' in src and 'place_contact_logger' in src
    from twin.restock import physics_setup
    psrc = Path(physics_setup.__file__).read_text()
    assert 'wire_place_contact_logger_pre_reset' in psrc
    # the pre-reset authoring must be wired BEFORE the world.scene.add()/world.reset() that build the view
    assert psrc.index('place_contact_logger=wire_place_contact_logger_pre_reset') \
        < psrc.index('robot=world.scene.add(SingleArticulation')
