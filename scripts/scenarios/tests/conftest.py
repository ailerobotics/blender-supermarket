"""Put scripts/ and scripts/scenarios/ on sys.path so the pure modules import,
and expose the committed fixture root. These tests never import ``bpy``."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCEN = os.path.dirname(HERE)              # scripts/scenarios
SCRIPTS = os.path.dirname(SCEN)           # scripts
for p in (SCRIPTS, SCEN):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest

FIXTURES = os.path.join(HERE, "fixtures")
PLANOGRAM = os.path.join(SCRIPTS, "planogram.json")


@pytest.fixture(scope="session")
def plan():
    import catalog_geom as geom
    return geom.load_planogram(PLANOGRAM)
