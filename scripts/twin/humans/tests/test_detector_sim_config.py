"""The restock detector sim config must not loosen any safety number.

The ``safety:`` block of ``params/human_safety.restock.yaml`` (which points the
fleet detector at the mission camera topics for the GPU run) must be byte-identical
to the fleet production config, so a GPU detector result can never be flattered by a
widened radius or relaxed budget here. Skips when the fleet checkout is absent.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from twin.humans.perception import FLEET_SOURCE

RESTOCK_CFG = Path(__file__).resolve().parents[1] / 'params' / 'human_safety.restock.yaml'
FLEET_CFG = Path(FLEET_SOURCE).parent / 'params' / 'human_safety.yaml'


def _safety_block(text: str) -> str:
    """The lines from the top-level ``safety:`` key to EOF, verbatim."""
    out, collecting = [], False
    for line in text.splitlines():
        if line.startswith('safety:'):
            collecting = True
        if collecting:
            out.append(line)
    return '\n'.join(out).rstrip('\n')


def test_restock_config_exists_and_uses_cuda():
    text = RESTOCK_CFG.read_text()
    assert 'device: cuda' in text
    assert '/restock_sim/camera/rgb' in text          # points at the mission stream
    assert 'restock_camera_wrist' in text


def test_safety_block_matches_fleet_verbatim():
    if not FLEET_CFG.is_file():
        pytest.skip(f'fleet config not present at {FLEET_CFG}')
    assert _safety_block(RESTOCK_CFG.read_text()) == _safety_block(FLEET_CFG.read_text()), (
        'restock detector safety block has drifted from the fleet config')
