from dataclasses import replace
from types import SimpleNamespace as NS
import pytest

from scripts.twin.restock_camera.freshness import frame_deadline,require_fresh


def test_acquisition_queue_age_is_not_reset_at_processing_start():
    frame=dict(rendering_time=1.,acquired_steady=10.,steady_domain='process')
    deadline=frame_deadline(frame,1.,10.4)
    assert deadline==10.5
    require_fresh(deadline,lambda:10.5)
    with pytest.raises(ValueError,match='freshness expired'):
        require_fresh(deadline,lambda:10.501)
    with pytest.raises(ValueError,match='future'):
        frame_deadline(frame,1.,9.)


def test_sequential_camera_processing_cannot_refresh_first_deadline(monkeypatch):
    from scripts.twin.restock_camera.binding import StationSensors
    from scripts.twin.restock_manipulation import isaac_sensors
    from scripts.twin.restock_manipulation.types import Sensors,Target
    clock=[100.]
    target=Target('depth','sensor',2.,'shelf',(0,0,.2,0,0,0,1),True,(.62,.42,.25))
    def adapter(camera,**kw):
        key=kw['camera_id']
        def capture(*args):
            clock[0]+=.3
            return Sensors(2.,(),target if key=='shelf' else None,True,False,2.,2.,(0.,)*6,(),2.,100.5)
        return NS(slots=(NS(location=key),),capture=capture,sensed_target_bboxes=(),evidence=())
    monkeypatch.setattr(isaac_sensors,'IsaacCartonSensors',adapter)
    sensors=StationSensors(cameras={'cart':object(),'shelf':object()},profiles={},
        pose_bridge_root='fixture',context_providers={'cart':object(),'shelf':object()},
        location_cameras={'cart':'cart','shelf':'shelf'},steady_clock=lambda:clock[0])
    with pytest.raises(ValueError,match='freshness expired'):
        sensors.capture(2.,'cart','shelf',None,None)
    assert sensors.evidence==() and sensors.sensed_target_bboxes==()


def test_require_fresh_sim_budget_holds_under_render_pacing_with_wall_backstop():
    # Render pacing: a full-res render takes seconds of WALL while the SIM/node clock barely advances.
    # The sim budget must NOT trip (the data is fresh in sim time), and a generous wall backstop that
    # is not exceeded must not trip either.
    require_fresh(100.5, lambda:100.2, wall_started=0., wall_backstop_s=60., wall_clock=lambda:3.5)
    # But a frozen sim clock with wall elapsed beyond the backstop IS a hung render -> refused.
    with pytest.raises(ValueError, match='wall backstop'):
        require_fresh(100.5, lambda:100.0, wall_started=0., wall_backstop_s=10., wall_clock=lambda:20.)
    # And genuine sim-time staleness is still refused.
    with pytest.raises(ValueError, match='freshness expired'):
        require_fresh(100.5, lambda:100.6, wall_started=0., wall_backstop_s=60., wall_clock=lambda:1.)


def test_render_paced_station_capture_is_not_misread_as_stale(monkeypatch):
    # The PR73 regression: under render pacing a full-res source render+decode takes 1.7-3.5 s WALL
    # per capture, which false-tripped the 0.5 s wall recheck. With a SIM/node steady_clock the sim
    # time barely advances across the (paused) render, so the capture stays fresh; the generous wall
    # backstop is not exceeded. Mission-representative two-camera StationSensors path.
    from scripts.twin.restock_camera.binding import StationSensors
    from scripts.twin.restock_manipulation import isaac_sensors
    from scripts.twin.restock_manipulation.types import Sensors,Target
    sim=[100.]; wall=[0.]
    target=Target('depth','sensor',2.,'shelf',(0,0,.2,0,0,0,1),True,(.62,.42,.25))
    def adapter(camera,**kw):
        key=kw['camera_id']
        def capture(*a):
            sim[0]+=0.1                      # sim advances little during the paused render
            return Sensors(2.,(),target if key=='shelf' else None,True,False,2.,2.,(0.,)*6,(),2.,100.5)
        return NS(slots=(NS(location=key),),capture=capture,sensed_target_bboxes=(),evidence=())
    monkeypatch.setattr(isaac_sensors,'IsaacCartonSensors',adapter)
    def wall_clock():
        wall[0]+=2.0                         # ~2 s WALL per read: a full-res render pace
        return wall[0]
    sensors=StationSensors(cameras={'cart':object(),'shelf':object()},profiles={},
        pose_bridge_root='fixture',context_providers={'cart':object(),'shelf':object()},
        location_cameras={'cart':'cart','shelf':'shelf'},steady_clock=lambda:sim[0],
        wall_backstop_s=60.,wall_clock=wall_clock)
    packet=sensors.capture(2.,'cart','shelf',None,None)   # several s WALL, +0.2 SIM < 0.5 -> fresh
    assert packet.target is not None and packet.valid_until_steady==100.5
    # A too-tight wall backstop would hard-stop the same render-paced capture (hung-render guard).
    sim[0]=100.; wall[0]=0.
    tight=StationSensors(cameras={'cart':object(),'shelf':object()},profiles={},
        pose_bridge_root='fixture',context_providers={'cart':object(),'shelf':object()},
        location_cameras={'cart':'cart','shelf':'shelf'},steady_clock=lambda:sim[0],
        wall_backstop_s=1.,wall_clock=wall_clock)
    with pytest.raises(ValueError, match='wall backstop'):
        tight.capture(2.,'cart','shelf',None,None)


def test_packet_delayed_after_capture_cannot_start_motion():
    from .test_restock_manipulation import make_controller,packet
    c,motion=make_controller(steady_clock=lambda:100.6)
    result=c.step(.1,replace(packet(c),valid_until_steady=100.5))
    assert result.status=='failed' and 'freshness expired' in result.reason
    assert not motion.moves and motion.stops==1


def test_pose_processing_delay_is_rechecked_immediately_before_motion():
    from .test_restock_manipulation import make_controller,packet
    clock=[100.]
    c,motion=make_controller(steady_clock=lambda:clock[0])
    def build(source,target,profile):
        clock[0]=100.6
        return c.poses
    c.pose_policy=NS(build=build)
    result=c.step(.1,replace(packet(c),valid_until_steady=100.5))
    assert result.status=='failed' and not motion.moves


def test_frame_deadline_tolerates_one_physics_step_render_lead():
    # The arm camera render leads the dt-accumulated sim clock by ~1/60 s; not "future".
    frame=dict(rendering_time=0.2333)
    d=frame_deadline(frame,0.2167,10.0)      # now < stamp by ~one physics step
    assert d>0
    with pytest.raises(ValueError,match='stale'):
        frame_deadline(dict(rendering_time=0.30),0.21,10.0)   # >1 step ahead = rejected
