"""The vendored safety core must stay byte-identical to the fleet source.

If the fleet checkout is present, assert ``messages.py`` and ``safety.py`` match it
exactly, so a future edit to either copy (which would silently fork the
safety-critical decision logic) is caught. Skips when the fleet checkout is absent.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from twin.humans.perception import FLEET_SOURCE

VENDORED = Path(__file__).resolve().parents[1] / 'perception'


@pytest.mark.parametrize('name', ['messages.py', 'safety.py'])
def test_vendored_matches_fleet(name):
    fleet = Path(FLEET_SOURCE) / name
    if not fleet.is_file():
        pytest.skip(f'fleet source not present at {fleet}')
    assert (VENDORED / name).read_bytes() == fleet.read_bytes(), (
        f'{name} has drifted from the fleet source; re-vendor or reconcile')
