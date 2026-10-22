"""Offline tests for the opt-in config model."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from twin.humans.config import (DetectionConfig, HumansConfig, PersonSpec, ReactionConfig)


def test_absent_block_is_disabled():
    assert HumansConfig.from_config(None).enabled is False
    assert HumansConfig.from_config({}).enabled is False
    assert HumansConfig.from_config({'other': 1}).enabled is False


def test_enabled_without_people_is_allowed_defaults_derived_later():
    HumansConfig()                       # default, disabled, empty -> valid
    # enabled with no explicit people is valid: the default crowd is derived from
    # the mission poses at author time (resolve_people), which raises if none given.
    assert HumansConfig(enabled=True).people == ()


def test_round_trip_through_dict():
    block = {
        'enabled': True, 'backend': 'teleport', 'seed': 7,
        'people': [{'person_id': 'p1', 'role': 'shopper',
                    'waypoints': [[0, 0], [1, 0]], 'speed_mps': 1.0}],
        'detection': {'source': 'ground_truth', 'cameras': ['restock_camera_wrist']},
        'reaction': {'slow_base_scale': 0.3, 'stop_base_scale': 0.0},
    }
    cfg = HumansConfig.from_config({'humans': block})
    assert cfg.enabled and cfg.seed == 7 and len(cfg.people) == 1
    assert cfg.detection.source == 'ground_truth'
    again = HumansConfig.from_config({'humans': cfg.to_dict()})
    assert again.to_dict() == cfg.to_dict()


def test_unknown_fields_rejected():
    with pytest.raises(ValueError, match='unknown humans'):
        HumansConfig.from_config({'humans': {'enabled': True, 'bogus': 1,
                                             'people': [{'person_id': 'p', 'role': 'staff',
                                                         'waypoints': [[0, 0], [1, 1]],
                                                         'speed_mps': 1.0}]}})
    with pytest.raises(ValueError, match='unknown person'):
        PersonSpec.from_dict({'person_id': 'p', 'role': 'staff',
                              'waypoints': [[0, 0], [1, 1]], 'speed_mps': 1.0, 'x': 1})


def test_duplicate_ids_rejected():
    p = {'person_id': 'dup', 'role': 'shopper', 'waypoints': [[0, 0], [1, 0]], 'speed_mps': 1.0}
    with pytest.raises(ValueError, match='unique'):
        HumansConfig.from_config({'humans': {'enabled': True, 'people': [p, dict(p)]}})


def test_detection_source_must_be_explicit():
    with pytest.raises(ValueError, match='ground_truth'):
        DetectionConfig(source='magic')


def test_reaction_bounds():
    with pytest.raises(ValueError):
        ReactionConfig(slow_base_scale=1.5)
    with pytest.raises(ValueError):
        ReactionConfig(stop_base_scale=0.5, slow_base_scale=0.3)   # stop must not exceed slow
    ReactionConfig(slow_base_scale=0.3, stop_base_scale=0.0)       # ok
