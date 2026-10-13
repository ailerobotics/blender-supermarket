"""Recompute coverage and exclusions; hashes alone cannot validate a verdict."""
import hashlib
import json
from pathlib import Path

from .coverage import ALGORITHM


def validate_evidence(config):
    path = Path(config.get('asset_report', ''))
    if not path.is_absolute() or not path.is_file():
        raise ValueError('asset_report: absolute current coverage evidence required; rerun generate_assets')
    report = json.loads(path.read_text())
    if report.get('schema_version') != 2 or report.get('algorithm') != ALGORITHM:
        raise ValueError('asset_report: unsupported/stale coverage algorithm; rerun generate_assets')
    inputs = report.get('inputs', {})
    for key, config_key in (('manifest', 'mobile_manifest'), ('shelfbot_root', 'shelfbot_root')):
        if Path(inputs.get(key, '')).resolve() != Path(config[config_key]).resolve():
            raise ValueError('asset_report: current ' + config_key + ' differs from evidence')
    sources = report.get('source_sha256', {})
    if not sources or any(inputs.get(key) not in sources for key in ('urdf', 'manifest')):
        raise ValueError('asset_report: missing source URDF/manifest hashes')
    for filename, digest in sources.items():
        source = Path(filename)
        if not source.is_absolute() or not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise ValueError('asset_report: stale or missing source ' + filename)
    for key in ('urdf', 'srdf', 'xrdf'):
        actual = hashlib.sha256(Path(config[key]).read_bytes()).hexdigest()
        if report.get('generated_sha256', {}).get('mobile.' + key) != actual:
            raise ValueError('asset_report does not match generated ' + key)
    from .generate_assets import build_assets
    reconstructed, _ = build_assets(inputs['urdf'], inputs['manifest'], inputs['shelfbot_root'],
        pitch=report['pitch_m'], min_pitch=report['min_pitch_m'],
        acceleration=report['planner_acceleration_rad_s2'], jerk=report['planner_jerk_rad_s3'])
    if reconstructed != report:
        raise ValueError('asset_report: recomputed coverage/exclusions/verdict differ; rerun generate_assets')
    return reconstructed
