"""Recompute coverage and exclusions; hashes alone cannot validate a verdict."""
import hashlib
import json
import math
from pathlib import Path

from .coverage import ALGORITHM

# The coverage recompute (scipy ConvexHull/qhull + cKDTree reductions over summed
# float products) is numerically env-dependent: across interpreters with different
# numpy/scipy builds the retained cells, discarded counts, exclusions, the pass/fail
# verdict and every generated-artifact hash are bit-identical, but a few reduced
# floats (measured: max_sphere_penetration_m) differ at the ~1e-17 machine-epsilon
# level. The recomputed report is therefore compared to the stored one with an
# explicit tolerance on FLOAT leaves ONLY; every discrete value (ints, strings,
# bools, source and generated hashes, excluded pairs, the verdict) must still match
# EXACTLY, so stale meshes and forged exclusions are still rejected. The tolerance is
# nanometre-scale -- ~1e8x the observed noise and far below any geometrically
# meaningful change -- so a genuinely different coverage still fails.
COVERAGE_ABS_TOL = 1e-9
COVERAGE_REL_TOL = 1e-9


def _assert_equivalent(recon, stored, path='report'):
    """Deep structural equality of the recomputed vs stored report: floats within the
    documented tolerance, every other leaf exact. Raises ValueError on any difference."""
    def fail(why):
        raise ValueError('asset_report: recomputed ' + path + ' ' + why + '; rerun generate_assets')
    # bool is an int subclass; check it before the numeric branch so a verdict flip
    # (True vs False) can never be absorbed by the float tolerance.
    if isinstance(stored, bool) or isinstance(recon, bool):
        if recon is not stored:
            fail('verdict differs')
    elif isinstance(stored, dict) or isinstance(recon, dict):
        if not (isinstance(stored, dict) and isinstance(recon, dict)) or set(recon) != set(stored):
            fail('structure differs')
        for key in stored:
            _assert_equivalent(recon[key], stored[key], path + '.' + str(key))
    elif isinstance(stored, list) or isinstance(recon, list):
        if not (isinstance(stored, list) and isinstance(recon, list)) or len(recon) != len(stored):
            fail('structure differs')
        for index, (r, s) in enumerate(zip(recon, stored)):
            _assert_equivalent(r, s, path + '[' + str(index) + ']')
    elif isinstance(stored, float) or isinstance(recon, float):
        if not (isinstance(stored, (int, float)) and isinstance(recon, (int, float))):
            fail('type differs')
        a, b = float(recon), float(stored)
        if not (math.isfinite(a) and math.isfinite(b)) or \
                abs(a - b) > COVERAGE_ABS_TOL + COVERAGE_REL_TOL * max(abs(a), abs(b)):
            fail('exceeds numeric tolerance')
    elif recon != stored:
        fail('differs')


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
        acceleration=report['planner_acceleration_rad_s2'], jerk=report['planner_jerk_rad_s3'],
        collision_buffer=report.get('collision_buffers_m') or None)
    # Exact for every discrete value (counts, exclusions, verdict, hashes); env-dependent
    # float noise (~1e-17) is tolerated so the check reproduces under both the mission
    # runtime (env_isaaclab) and system Python. A real coverage change still fails.
    _assert_equivalent(reconstructed, report)
    return reconstructed
