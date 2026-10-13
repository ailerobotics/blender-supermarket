"""CPU launch contract: creates no ROS node, mapper, executor or process."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from twin.navigation.mapper_launch import mapper_plan,description,SLICE_TOPIC


class MapperLaunchTests(unittest.TestCase):
    def test_installed_exports_and_real_launch_description_without_execution(self):
        try:
            from ament_index_python.packages import get_package_share_directory,PackageNotFoundError
        except ImportError:
            self.skipTest('requires existing host/plugin overlay')
        try:
            for name in ('nvblox_ros','nvblox_nav2','nvblox_examples_bringup'):
                get_package_share_directory(name)
        except PackageNotFoundError:
            self.skipTest('requires existing host/plugin overlay')
        launch=description(generation='cpu_contract',camera_count=2,transport_height=1.9)
        self.assertEqual(len(launch.entities),2)

    def test_station_independent_plan_has_no_manipulation_mask_or_frame(self):
        plan=mapper_plan(generation='mission_a',camera_count=2,transport_height=1.9)
        self.assertFalse(plan['commissioned']);self.assertFalse(plan['map_load'])
        self.assertEqual(plan['frame'],'map')
        self.assertEqual(plan['slice_topic'],SLICE_TOPIC)
        self.assertNotIn('esdf_service',plan)
        self.assertEqual(plan['forbidden_probe_services'],
                         ['/restock_sim/navigation/nvblox_node/get_esdf_and_gradient'])
        self.assertEqual(plan['parameters']['input_qos'],'SENSOR_DATA')
        self.assertGreater(plan['parameters']['static_mapper.esdf_slice_max_height'],1.9)
        self.assertNotIn('station',str(plan));self.assertNotIn('target_bbox',str(plan))
        self.assertNotIn('ur_base_link',str(plan))

    def test_plan_advertises_typed_evidence_interface_not_esdf_probe(self):
        plan=mapper_plan(generation='mission_a',camera_count=2,transport_height=1.9)
        ev=plan['evidence']
        self.assertEqual(ev['package'],'restock_mapper_evidence')
        self.assertEqual(ev['snapshot_topic'],SLICE_TOPIC.rsplit('/',1)[0]+'/evidence_snapshot')
        for name in ('empty_generation','open_admission','register_acquisition'):
            self.assertEqual(ev[name+'_service'],'/restock_sim/navigation/nvblox_node/'+name)
        # Provenance never claims full-height coverage, and the ESDF probe that
        # exits a 2D mapper is listed only as forbidden, never as an evidence source.
        self.assertFalse(ev['full_height_observation_verified'])
        self.assertNotIn(ev['snapshot_topic'],plan['forbidden_probe_services'])
        self.assertIn('/restock_sim/navigation/nvblox_node/get_esdf_and_gradient',
                      plan['forbidden_probe_services'])

    def test_generations_have_disjoint_admission_topics_and_process_names(self):
        a=mapper_plan(generation='one',camera_count=2,transport_height=1.9)
        b=mapper_plan(generation='two',camera_count=2,transport_height=1.9)
        self.assertNotEqual(a['container'],b['container'])
        ta={topic for camera in a['inputs'] for topic in camera.values()}
        tb={topic for camera in b['inputs'] for topic in camera.values()}
        self.assertTrue(ta.isdisjoint(tb))
        self.assertEqual(a['slice_topic'],b['slice_topic']) # Native output has NO generation field.

    def test_invalid_launch_arguments_fail_before_ros_import(self):
        for generation in ('','old/process','../old','1starts_digit','x'*65):
            with self.assertRaises(ValueError):
                mapper_plan(generation=generation,camera_count=1,transport_height=1.9)
        with self.assertRaises(ValueError):
            mapper_plan(generation='one',camera_count=5,transport_height=1.9)


if __name__=='__main__':unittest.main()
