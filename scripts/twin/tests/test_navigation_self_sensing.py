"""CPU closest-hit geometry fixture, not a native PhysX execution."""
import math
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from twin.mobile import MobileSpec
from twin.navigation import GridMap, Odometry
from twin.navigation.articulated import ArticulatedController,TrainState
from twin.navigation.sensors import IsaacRangeSensor,MapLocalizer,Scan,observation


class BoxQuery:
    """Exact ray/OBB intersection for supplied source-derived box colliders.

    Not arm meshes/caster dynamics. The inside-chassis counterexample does not
    depend on these omitted colliders: adding them cannot restore visibility.
    """
    def __init__(self,boxes):
        self.boxes=boxes
        self.calls=0

    def raycast_closest(self,origin,direction,limit):
        self.calls+=1
        best={'hit':False};distance=limit
        for path,centre,size,yaw in self.boxes:
            c,s=math.cos(yaw),math.sin(yaw)
            x,y,z=(origin[i]-centre[i] for i in range(3))
            dx,dy,dz=direction
            local=(c*x+s*y,-s*x+c*y,z)
            ray=(c*dx+s*dy,-s*dx+c*dy,dz)
            near,far=0.,limit
            for p,d,width in zip(local,ray,size):
                if abs(d)<1e-10:
                    if abs(p)>width/2:far=-1;break
                else:
                    a,b=(-width/2-p)/d,(width/2-p)/d
                    near=max(near,min(a,b));far=min(far,max(a,b))
            if near<=far and near<=distance:
                distance=near
                best={'hit':True,'distance':near,'collision':path,'rigidBody':path.rsplit('/',1)[0]}
        return best


def rig_boxes(state):
    """Source dimensions from mobile_assets; articulated body poses, not collinear."""
    spec=MobileSpec();robot,cart=state.bodies(spec);boxes=[]
    def add(path,body,x,y,z,size):
        px,py=body.point(x,y);boxes.append((path,(px,py,z),size,body.yaw))
    add('/Robot/base_link/collision',robot,0,0,spec.base_z,(spec.length,spec.width,spec.height))
    h=spec.arm_mount_z-spec.base_z-spec.height/2
    add('/Robot/arm_mount/collision',robot,0,0,spec.arm_mount_z-h/2,(.24,.24,h))
    add('/Cart/chassis/deck',cart,0,0,.20,(spec.cart_length,spec.cart_width,.05))
    add('/Cart/chassis/shelf',cart,0,0,.92,(spec.cart_length-.04,spec.cart_width-.04,.025))
    for i,(x,y) in enumerate((x,y) for x in (-spec.cart_length/2+.02,spec.cart_length/2-.02)
                            for y in (-spec.cart_width/2+.02,spec.cart_width/2-.02)):
        add('/Cart/chassis/post_'+str(i),cart,x,y,.20+(spec.cart_height-.20)/2,
            (.025,.025,spec.cart_height-.20))
    return boxes


def sensor(query,state=TrainState(0,0,0,0),self_paths=('/Robot','/Cart'),heights=(.15,.5,1.)):
    # Exact native planar mount x=.46; heights are world z, not frame offsets.
    robot,_=state.bodies(MobileSpec());x,y=robot.point(.46)
    return IsaacRangeSensor(lambda:(x,y,state.yaw),query=query,self_paths=self_paths,
                            sensor_to_base=(.46,0,0),heights=heights)


class SelfSensingTests(unittest.TestCase):
    def test_native_inside_chassis_vs_map_only_and_identity_counterfactual(self):
        states=[TrainState(0,0,0,0),TrainState(-2.971786,5.992829,-3.096399,-2.294942)]
        for state in states:
            with self.subTest(state=state):
                query=BoxQuery(rig_boxes(state))
                raw=sensor(query,state,self_paths=()).read(.1)
                self.assertTrue(all(r<=.001 for r in raw.ranges))
                old=observation(raw,Odometry(state.pose,.1,position_uncertainty=.10),
                                coupled=True,arm_transport=True,articulation=state.articulation)
                self.assertEqual(len(old.obstacles),360)
                filtered=sensor(query,state).read(.1)
                self.assertTrue(all(filtered.unknown))
                self.assertTrue(all(math.isnan(r) for r in filtered.ranges))
                obs=observation(filtered,Odometry(state.pose,.1,position_uncertainty=.10),
                                coupled=True,arm_transport=True,articulation=state.articulation)
                self.assertEqual(obs.obstacles,())
                self.assertEqual(obs.coverage_radius,0)
                map_only=sensor(BoxQuery([]),state).read(.1)
                self.assertFalse(any(map_only.unknown))
                self.assertEqual(observation(map_only,Odometry(state.pose,.1),coupled=True,
                                            arm_transport=True).coverage_radius,9.54)

    def test_external_behind_self_is_unknown_not_seen_through(self):
        boxes=[('/Robot/occluder',(1,0,.5),(.1,.2,.2),0),
               ('/External/obstacle',(1.06,0,.5),(.02,.2,.2),0)]
        query=BoxQuery(boxes)
        scan=sensor(query,heights=(.5,)).read(.1)
        self.assertTrue(scan.unknown[180])
        self.assertTrue(math.isnan(scan.ranges[180]))
        self.assertEqual(query.calls,360)  # no second ray through the occluder
        self.assertEqual(scan.external_returns,())
        removed=sensor(BoxQuery(boxes[1:]),heights=(.5,)).read(.1)
        self.assertFalse(removed.unknown[180])
        self.assertAlmostEqual(removed.ranges[180],.59)

    def test_other_height_external_return_retained_despite_unknown(self):
        boxes=[('/Robot/base',(0,0,.24),(1,.6,.2),0),
               ('/RobotObstacle/box',(1,0,.5),(.1,.2,.2),0)]
        scan=sensor(BoxQuery(boxes),heights=(.15,.5)).read(.1)
        self.assertTrue(scan.unknown[180])
        self.assertAlmostEqual(dict(scan.external_returns)[180],.49)
        obs=observation(scan,Odometry((0,0,0),.1),coupled=True,arm_transport=True)
        self.assertTrue(any(abs(x-.95)<1e-8 for x,y in obs.obstacles))
        self.assertEqual(obs.coverage_radius,0)
        # Losing the mask in a polar transport must fail, not regain coverage.
        plain=Scan(scan.stamp,scan.ranges,scan.angle_min,scan.angle_increment,scan.range_max)
        with self.assertRaises(ValueError):
            observation(plain,Odometry((0,0,0),.1),coupled=True,arm_transport=True)

    def test_external_only_blocks_and_hidden_scan_fails_motion_and_localization(self):
        state=TrainState(0,0,0,0)
        external=sensor(BoxQuery([('/Other/block',(1,0,.5),(.1,.2,.2),0)]),heights=(.5,)).read(.1)
        odom=Odometry(state.pose,.1,position_uncertainty=.10)
        controller=ArticulatedController(GridMap((-10,-10,10,10)))
        controller.start((4,0,0),True)
        obs=observation(external,odom,coupled=True,arm_transport=True)
        self.assertEqual(controller.gate_velocity(.1,odom,obs,.25,0).reason,'obstacle_stop')
        hidden=sensor(BoxQuery(rig_boxes(state))).read(.1)
        obs=observation(hidden,odom,coupled=True,arm_transport=True)
        self.assertEqual(controller.gate_velocity(.1,odom,obs,.25,0).reason,'insufficient_sensor_coverage')
        with self.assertRaisesRegex(ValueError,'insufficient mapped'):
            MapLocalizer(GridMap((-10,-10,10,10)),state.pose).update(odom,hidden)
        self.assertEqual(controller.gate_velocity(1.,odom,obs,.25,0).reason,'stale_sensing')

    def test_localizer_retains_visible_other_height_hits_and_floor(self):
        boxes=rig_boxes(TrainState(0,0,0,0))
        boxes.extend([('/Room/x'+str(sign),(sign*4.01,0,1),(.02,8.02,2),0)
                      for sign in (-1,1)])
        boxes.extend([('/Room/y'+str(sign),(0,sign*4.01,1),(8.02,.02,2),0)
                      for sign in (-1,1)])
        scan=sensor(BoxQuery(boxes)).read(.1)
        self.assertTrue(all(scan.unknown))
        self.assertGreater(len(scan.external_returns),12)
        localized=MapLocalizer(GridMap((-4,-4,4,4)),(0,0,0)).update(Odometry((0,0,0),.1),scan)
        self.assertEqual(localized.pose,(0.,0.,0.))
        self.assertGreaterEqual(localized.position_uncertainty,.10)
        self.assertEqual(observation(scan,localized,coupled=True,arm_transport=True).coverage_radius,0)

    def test_invalid_self_roots_and_mask_rejected(self):
        for path in ('/','relative','/Robot/../Other'):
            with self.assertRaises(ValueError):sensor(BoxQuery([]),self_paths=(path,))
            with self.assertRaises(ValueError):           # coupled_self_paths validated identically
                IsaacRangeSensor(lambda:(0,0,0),query=BoxQuery([]),self_paths=('/Robot',),
                                 coupled_self_paths=(path,))
        with self.assertRaises(ValueError):Scan(0,(1,2,3),0,1,10,unknown=(True,))

    def test_coupled_cart_is_self_only_while_coupled(self):
        # The towed cart root is rig self-geometry ONLY while coupled: masked UNKNOWN when coupled
        # (no phantom rear obstacle), a real obstacle when decoupled. The port toggles `.coupled`.
        state=TrainState(-2.971786,5.992829,-3.096399,-2.294942)
        cart_boxes=[b for b in rig_boxes(state) if b[0].startswith('/Cart/')]
        robot,_=state.bodies(MobileSpec());x,y=robot.point(.46)
        def make():
            return IsaacRangeSensor(lambda:(x,y,state.yaw),query=BoxQuery(cart_boxes),
                                    self_paths=('/Robot',),coupled_self_paths=('/Cart',),
                                    sensor_to_base=(.46,0,0),heights=(1.,))
        decoupled=make()
        self.assertFalse(decoupled.coupled)               # default: cart NOT self
        scan_d=decoupled.read(.1)
        self.assertGreater(len(scan_d.points(state.pose)),0)   # cart read as a real obstacle
        self.assertEqual(sum(scan_d.unknown),0)
        coupled=make();coupled.coupled=True
        scan_c=coupled.read(.1)
        self.assertEqual(scan_c.points(state.pose),())         # cart masked -> no phantom obstacle
        self.assertGreater(sum(scan_c.unknown),0)

    def test_cart_identity_tracks_independent_articulation(self):
        state=TrainState(-2.971786,5.992829,-3.096399,-2.294942)
        cart_boxes=[b for b in rig_boxes(state) if b[0].startswith('/Cart/')]
        raw=sensor(BoxQuery(cart_boxes),state,self_paths=('/Robot',),heights=(1.,)).read(.1)
        corrected=sensor(BoxQuery(cart_boxes),state,heights=(1.,)).read(.1)
        self.assertGreater(len(raw.points(state.pose)),0)
        self.assertEqual(corrected.points(state.pose),())
        self.assertGreater(sum(corrected.unknown),0)
        _,cart=state.bodies(MobileSpec())
        for x,y in raw.points(state.pose):
            dx,dy=x-cart.x,y-cart.y
            local_x=dx*math.cos(cart.yaw)+dy*math.sin(cart.yaw)
            local_y=-dx*math.sin(cart.yaw)+dy*math.cos(cart.yaw)
            self.assertLessEqual(abs(local_x),MobileSpec().cart_length/2+1e-8)
            self.assertLessEqual(abs(local_y),MobileSpec().cart_width/2+1e-8)


if __name__=='__main__':unittest.main()
