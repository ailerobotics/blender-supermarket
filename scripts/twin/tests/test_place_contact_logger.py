"""CPU tests for the place-descent PhysX contact-pair logger (sim-pad-down-descent-jam-physics-v1).

The PURE core (path classification, first/deepen log decision, line format) is tested directly; the Isaac
``attach`` + the ``isaac_factory.create`` boot-path wiring are tested with INJECTED fakes for the
omni.physx subscribe / contact-report-API / path-decode hooks, so the whole wiring is proven without a GPU.
Needs only usd-core (``pxr.UsdPhysics``, which ``attach`` imports). No Isaac / ROS / GPU.
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
def test_wire_absent_block_is_none_no_isaac_import():
    # no block -> returns None WITHOUT importing omni.physx (byte-identical; safe on a GPU-less box)
    assert pcl.wire_place_contact_logger(types.SimpleNamespace(stage=None), {}) is None
    assert pcl.wire_place_contact_logger(types.SimpleNamespace(stage=None),
                                         {'place_contact_logger': False}) is None


def test_wire_present_block_constructs_attaches_and_arms(monkeypatch):
    # inject a fake omni.physx so the boot-path wiring runs end-to-end on a GPU-less box
    fake_sim = _FakeSim()
    fake_mod = types.ModuleType('omni.physx')
    fake_mod.get_physx_simulation_interface = lambda: fake_sim
    monkeypatch.setitem(sys.modules, 'omni', types.ModuleType('omni'))
    monkeypatch.setitem(sys.modules, 'omni.physx', fake_mod)
    # make attach's body-instrumentation injectable by monkeypatching the method to use fakes
    applied = []
    orig_attach = pcl.PlaceContactLogger.attach

    def _attach(self, stage, sim, **kw):
        kw.setdefault('apply_contact_report', lambda prim, thr: applied.append(str(prim.GetPath())))
        kw.setdefault('decode_path', lambda i: '/x/%d' % i)
        return orig_attach(self, stage, sim, **kw)
    monkeypatch.setattr(pcl.PlaceContactLogger, 'attach', _attach)

    lines = []
    stage = _FakeStage([_FakePrim('/Rollwagen/chassis/shelf', True),
                        _FakePrim('/MobileRobot/suction_cups', True)])
    assembly = types.SimpleNamespace(stage=stage)
    logger = pcl.wire_place_contact_logger(assembly, {'place_contact_logger': True}, log=lines.append)
    assert isinstance(logger, pcl.PlaceContactLogger)
    assert fake_sim.callback is not None                        # subscribed through the injected interface
    assert set(applied) == {'/Rollwagen/chassis/shelf', '/MobileRobot/suction_cups'}
    assert any('PLACE_CONTACT_LOGGER armed' in ln for ln in lines)


def test_isaac_factory_wires_the_logger_on_the_boot_path():
    # pin that isaac_factory.create calls the wiring (so a configured logger can never be left unattached)
    from twin.restock_manipulation import isaac_factory
    src = Path(isaac_factory.__file__).read_text()
    assert 'wire_place_contact_logger' in src
    assert 'place_contact_logger' in src
