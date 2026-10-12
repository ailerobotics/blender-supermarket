"""The runtime and fixture must include the same scenario-only geometry."""
import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from twin.navigation.scenario import environment_grid


class ScenarioMapTests(unittest.TestCase):
    def test_staging_support_added_without_mutating_environment(self):
        manifest={'map_bounds':[-10,10,-2,20],'static_obstacles':[]}
        scenario={'initial_staging':{'support_center_xy':[-5.73,17],
                  'support_dimensions_m':[.68,.48,.6]}}
        grid=environment_grid(manifest,scenario)
        self.assertEqual(grid.bounds,(-10,-2,10,20))
        self.assertEqual(len(grid.obstacles),1)
        for value,expected in zip(grid.obstacles[0],[-6.07,16.76,-5.39,17.24]):
            self.assertAlmostEqual(value,expected)
        self.assertEqual(manifest['static_obstacles'],[])
        self.assertEqual(len(environment_grid(manifest).obstacles),0)
