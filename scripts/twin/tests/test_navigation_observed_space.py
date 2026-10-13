"""Finite-resolution depth model tests; no GPU or fake native execution."""
import math
from pathlib import Path
import sys
import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from twin.navigation.observed_space import DepthMount,DepthFrame,ObservedSpace,RenderedDepthReader,RigVoxels
from twin.navigation.scenario import camera_mount_obstacles
from twin.navigation.articulated import TrainState
from twin.mobile import MobileSpec


def frame(depth=2.,**kw):
    mount=DepthMount('front','base',(0,0,0),width=3,height=3,near=.01,noise_bound=.01)
    values=dict(mount=mount,stamp=1.,pose_stamp=1.,body_pose=(0,0,0,0),
                depth=(depth,)*9,intrinsics=(10,0,1,0,10,1,0,0,1),
                map_from_optical=tuple(np.eye(4).ravel()),rendering_frame=1)
    values.update(kw)
    return DepthFrame(**values)


class DepthTests(unittest.TestCase):
    def test_image_plane_depth_and_actual_principal_point(self):
        f=frame(intrinsics=(2,0,.4,0,3,.6,0,0,1))
        origin,ray=f.world_ray(2,1)
        np.testing.assert_allclose(ray,(.8,.4/3,1))
        self.assertEqual(origin,(0,0,0))

    def test_unknown_occlusion_and_external_priority(self):
        f=frame()
        space=ObservedSpace([f],required_sensors=['front'])
        self.assertEqual(space.check_cells([(0,0,5)],1),'')
        self.assertEqual(space.check_cells([(0,0,25)],1),'unseen_swept_cell')
        self.assertEqual(space.check_cells([(0,0,20)],1,current_solid_cells=[(0,0,20)]),'occupied_swept_cell')
        hidden=ObservedSpace([replace(f,depth=(float('nan'),)*9)],required_sensors=['front'])
        self.assertEqual(hidden.check_cells([(0,0,5)],1),'unseen_swept_cell')

    def test_self_hit_does_not_clear_shadow(self):
        space=ObservedSpace([frame(self_mask=(True,)*9)],required_sensors=['front'])
        self.assertEqual(space.check_cells([(0,0,25)],1),'unseen_swept_cell')
        self.assertFalse(space.occupied)
        self.assertEqual(space.check_cells([(0,0,20)],1),'unseen_swept_cell')

    def test_missing_stale_and_duplicate(self):
        f=frame();s=ObservedSpace([f],required_sensors=['front','cart'])
        self.assertEqual(s.check_cells([],1),'missing_coverage_sensor')
        s=ObservedSpace([f],required_sensors=['front'])
        self.assertEqual(s.check_cells([],1.31),'stale_coverage_sensor')
        self.assertEqual(s.check_cells([],.9),'stale_coverage_sensor')
        s=ObservedSpace([f,f],required_sensors=['front'])
        self.assertEqual(s.check_cells([],1),'duplicate_coverage_sensor')

    def test_reader_rejects_held_render_and_wrong_depth_type(self):
        raw=dict(rendering_frame=1,rendering_time=1.,distance_to_image_plane=np.ones((3,3),np.float32))
        camera=SimpleNamespace(get_current_frame=lambda:raw,
            get_intrinsics_matrix=lambda:np.array(frame().intrinsics).reshape(3,3))
        reader=RenderedDepthReader(frame().mount)
        args=dict(map_from_optical=np.eye(4),body_pose=(0,0,0,0),pose_stamp=1.)
        reader.read(camera,**args)
        with self.assertRaisesRegex(ValueError,'no new'):reader.read(camera,**args)
        raw.update(rendering_frame=2,rendering_time=1.1)
        with self.assertRaisesRegex(ValueError,'acquisition'):reader.read(camera,**args)
        raw['distance_to_image_plane']=np.ones((3,3),np.float64)
        with self.assertRaisesRegex(ValueError,'float32'):reader.read(camera,**args)

    def test_independent_cart_optical_transform(self):
        tf=np.eye(4);tf[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];tf[:3,3]=[4,5,1]
        f=frame(map_from_optical=tuple(tf.ravel()))
        origin,ray=f.world_ray(2,1)
        np.testing.assert_allclose(origin,(4,5,1));np.testing.assert_allclose(ray,(0,.1,1))
        with self.assertRaises(ValueError):frame(pose_stamp=.8)

    def test_same_acquisition_self_ids_use_exact_subtree(self):
        ids=np.array([[1,2,3]]*3,np.uint32)
        raw=dict(rendering_frame=4,rendering_time=1.,distance_to_image_plane=np.ones((3,3),np.float32),
            instance_id_segmentation=dict(data=ids,info=dict(idToLabels={
                '1':'/Robot/base','2':'/RobotObstacle','3':'/Cart/frame'})))
        camera=SimpleNamespace(get_current_frame=lambda:raw,
            get_intrinsics_matrix=lambda:np.array(frame().intrinsics).reshape(3,3))
        reader=RenderedDepthReader(frame().mount,self_paths=('/Robot','/Cart'))
        args=dict(map_from_optical=np.eye(4),body_pose=(0,0,0,0),pose_stamp=1.)
        sensed=reader.read(camera,**args)
        self.assertEqual(sensed.self_mask,(True,False,True)*3)
        raw.update(rendering_frame=5,rendering_time=1.01)
        del raw['instance_id_segmentation']
        with self.assertRaisesRegex(ValueError,'same-acquisition'):reader.read(camera,**args)
        with self.assertRaises(ValueError):RenderedDepthReader(frame().mount,self_paths=('/',))

    def test_rig_voxels_follow_cart_articulation(self):
        rig=RigVoxels(((0,0,.2),),((-.4,.3,1.),),.05,'fixture')
        s=ObservedSpace([frame()],required_sensors=['front'],rig=rig)
        aligned=rig.cells([TrainState(0,0,0,0)],MobileSpec(),s,0,0)
        turned=rig.cells([TrainState(0,0,0,math.pi/3)],MobileSpec(),s,0,0)
        self.assertNotEqual(aligned,turned)
        self.assertEqual(s.check_train([TrainState(0,0,0,0)],MobileSpec(),.10,1),'unseen_swept_cell')
        self.assertEqual(s.check_train([TrainState(0,0,0,0)],MobileSpec(),.02,1),'invalid_depth_localization_bound')

    def test_mount_rotated_bounds_and_moving_parent(self):
        m=np.eye(4);m[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];m[:3,3]=[2,3,4]
        solid=dict(parent='map',collision=True,parent_from_solid=m.tolist(),dimensions_m=[2,4,.1])
        self.assertEqual(camera_mount_obstacles({'solids':[solid]}),[(0.,2.,4.,4.)])
        solid['parent']='cart_chassis'
        self.assertEqual(camera_mount_obstacles({'solids':[solid]}),[])
        solid['parent']='unresolved'
        with self.assertRaises(ValueError):camera_mount_obstacles({'solids':[solid]})


if __name__=='__main__':unittest.main()
