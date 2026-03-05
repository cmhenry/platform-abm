"""Tests for the reporting pipeline (experiments/reporting.py)."""

import csv
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from experiments.reporting import (
    _classify_convergence,
    _clean_for_json,
    _nan_to_none,
    _step_log_to_dataframe,
    run_phase1_convergence,
    run_phase1_enclaves,
    run_phase2_convergence,
    run_phase2_enclaves,
    run_phase2_summary_update,
    run_config_reporting,
)


# ---------------------------------------------------------------------------
# Helpers to create synthetic test data
# ---------------------------------------------------------------------------

def _make_step_log(n_steps: int = 20, converging: bool = True) -> list[dict]:
    """Create a synthetic step_log with known convergence behavior."""
    log = []
    for t in range(1, n_steps + 1):
        if converging:
            util = 3.0 + 2.0 * (1 - math.exp(-0.3 * t))
        else:
            util = 3.0 + 0.05 * t  # still climbing
        log.append({
            'step': t,
            'avg_utility': util,
            'n_relocations': max(0, 20 - t),
            'per_governance_utilities': {'direct': util * 0.9, 'coalition': util * 1.1},
            'per_governance_community_count': {'direct': 5, 'coalition': 5},
            'per_type_utility': {'mainstream': util * 0.95, 'extremist': util * 1.05},
            'per_type_relocations': {'mainstream': max(0, 10 - t), 'extremist': max(0, 10 - t)},
        })
    return log


def _make_step_metrics(n_iters: int = 3, n_steps: int = 20, converging: bool = True) -> dict:
    """Create step_metrics dict keyed by iteration number."""
    return {str(i): _make_step_log(n_steps, converging) for i in range(n_iters)}


def _make_enclave_data(n_iters: int = 3, n_platforms: int = 2, n_steps: int = 30) -> dict:
    """Create per_iter_enclaves data with known settling behavior."""
    data = {}
    for i in range(n_iters):
        platforms = {}
        for pid in range(n_platforms):
            # Starts low, settles to high homogeneity after step 10
            series = []
            for t in range(n_steps):
                if t < 10:
                    series.append(0.5 + 0.04 * t)
                else:
                    series.append(0.95 + 0.001 * np.random.randn())
            platforms[str(pid)] = {
                'homogeneity_series': series,
                'mean_homogeneity': float(np.mean(series)),
                'fraction_enclaved': float(np.mean(np.array(series) > 0.9)),
            }
        data[str(i)] = platforms
    return data


# ---------------------------------------------------------------------------
# Tests: Helpers
# ---------------------------------------------------------------------------

class TestCleanForJson:
    def test_numpy_integer(self):
        assert _clean_for_json(np.int64(42)) == 42

    def test_numpy_float(self):
        assert _clean_for_json(np.float64(3.14)) == pytest.approx(3.14)

    def test_nan_to_none(self):
        assert _clean_for_json(float('nan')) is None

    def test_inf_to_none(self):
        assert _clean_for_json(float('inf')) is None

    def test_nested_dict(self):
        obj = {'a': np.int64(1), 'b': [np.float64(2.0), float('nan')]}
        result = _clean_for_json(obj)
        assert result == {'a': 1, 'b': [2.0, None]}

    def test_nan_to_none_helper(self):
        assert _nan_to_none(float('nan')) is None
        assert _nan_to_none(float('inf')) is None
        assert _nan_to_none(3.14) == 3.14
        assert _nan_to_none(np.float64('nan')) is None


# ---------------------------------------------------------------------------
# Tests: DataFrame construction
# ---------------------------------------------------------------------------

class TestStepLogToDataFrame:
    def test_column_names(self):
        """Verify _step_log_to_dataframe produces columns matching displacement_diagnostic."""
        log = _make_step_log(5)
        df = _step_log_to_dataframe(log)

        assert 'step' in df.columns
        assert 'per_governance_community_count_direct' in df.columns
        assert 'per_governance_community_count_coalition' in df.columns
        assert 'per_governance_utilities_direct' in df.columns
        assert 'per_governance_utilities_coalition' in df.columns
        assert 'per_type_utility_mainstream' in df.columns
        assert 'per_type_utility_extremist' in df.columns
        assert 'avg_utility' in df.columns
        assert 'n_relocations' in df.columns

    def test_row_count(self):
        log = _make_step_log(10)
        df = _step_log_to_dataframe(log)
        assert len(df) == 10

    def test_values_correct(self):
        log = [{'step': 1, 'avg_utility': 5.0, 'n_relocations': 3,
                'per_governance_utilities': {'direct': 4.0},
                'per_governance_community_count': {'direct': 7}}]
        df = _step_log_to_dataframe(log)
        assert df.iloc[0]['per_governance_utilities_direct'] == 4.0
        assert df.iloc[0]['per_governance_community_count_direct'] == 7


# ---------------------------------------------------------------------------
# Tests: Convergence classification
# ---------------------------------------------------------------------------

class TestConvergenceClassification:
    def test_converged_pattern(self):
        """Saturating utility series should classify as CONVERGED or PLATEAU."""
        util = [3.0 + 2.0 * (1 - math.exp(-0.5 * t)) for t in range(50)]
        reloc = [max(0, 20 - t) for t in range(50)]
        result = _classify_convergence(util, reloc)
        assert result['pattern'] in ('CONVERGED', 'PLATEAU')
        assert result['settling_step'] <= 50

    def test_still_climbing(self):
        """Linearly increasing utility should classify as STILL_CLIMBING."""
        util = [0.1 * t for t in range(50)]
        reloc = [10] * 50
        result = _classify_convergence(util, reloc)
        assert result['pattern'] == 'STILL_CLIMBING'

    def test_insufficient_data(self):
        result = _classify_convergence([1.0, 2.0], [1, 1])
        assert result['pattern'] == 'INSUFFICIENT_DATA'

    def test_settling_step_correct(self):
        # Constant utility — should settle at step 1
        util = [5.0] * 20
        reloc = [0] * 20
        result = _classify_convergence(util, reloc)
        assert result['settling_step'] == 1

    def test_second_half_gain(self):
        util = [float(t) for t in range(20)]
        reloc = [0] * 20
        result = _classify_convergence(util, reloc)
        assert result['second_half_gain'] == pytest.approx(util[-1] - util[10])


# ---------------------------------------------------------------------------
# Tests: Phase 1 convergence
# ---------------------------------------------------------------------------

class TestPhase1Convergence:
    def test_creates_output_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            step_metrics = _make_step_metrics(3, 20)
            result = run_phase1_convergence(config_dir, step_metrics)

            assert (config_dir / "per_iter_convergence.json").exists()
            assert len(result) == 3
            for v in result.values():
                assert 'pattern' in v
                assert 'per_governance' in v

    def test_per_governance_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            step_metrics = _make_step_metrics(2, 20)
            result = run_phase1_convergence(config_dir, step_metrics)
            first = next(iter(result.values()))
            assert 'direct' in first['per_governance']
            assert 'coalition' in first['per_governance']


# ---------------------------------------------------------------------------
# Tests: Enclave analysis
# ---------------------------------------------------------------------------

class TestPhase1Enclaves:
    def test_settling_step_detection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            enclaves = _make_enclave_data(2, 2, 30)
            result = run_phase1_enclaves(config_dir, enclaves)

            assert (config_dir / "per_iter_enclave_analysis.json").exists()
            assert len(result) == 2

            # Each iteration should have platforms with settling_step around 10
            for iter_data in result.values():
                for pdata in iter_data['platforms'].values():
                    if pdata['settling_step'] is not None:
                        assert pdata['settling_step'] <= 15

    def test_disruption_counting(self):
        """Verify disruptions after settling are counted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            # Create data with a disruption at step 20
            series = [0.95] * 15 + [0.5, 0.6, 0.7, 0.8, 0.95] + [0.95] * 10
            enclaves = {
                '0': {'0': {
                    'homogeneity_series': series,
                    'mean_homogeneity': float(np.mean(series)),
                    'fraction_enclaved': float(np.mean(np.array(series) > 0.9)),
                }}
            }
            result = run_phase1_enclaves(config_dir, enclaves)
            platform = result['0']['platforms']['0']
            assert platform['n_disruptions'] >= 1

    def test_system_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            enclaves = _make_enclave_data(2, 2, 30)
            result = run_phase1_enclaves(config_dir, enclaves)
            for iter_data in result.values():
                system = iter_data['system']
                assert 'mean_settling_step' in system
                assert 'mean_homogeneity' in system
                assert 'fraction_with_disruptions' in system


# ---------------------------------------------------------------------------
# Tests: Phase 2 aggregation
# ---------------------------------------------------------------------------

class TestPhase2Convergence:
    def test_aggregation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            step_metrics = _make_step_metrics(5, 20)
            run_phase1_convergence(config_dir, step_metrics)
            result = run_phase2_convergence(config_dir)

            assert (config_dir / "convergence_aggregate.json").exists()
            assert result['n_iterations'] == 5
            assert 'pattern_counts' in result
            assert result['final_utility_mean'] is not None


class TestPhase2Enclaves:
    def test_aggregation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            enclaves = _make_enclave_data(5, 2, 30)
            run_phase1_enclaves(config_dir, enclaves)
            result = run_phase2_enclaves(config_dir)

            assert (config_dir / "enclave_aggregate.json").exists()
            assert result['n_iterations'] == 5
            assert result['mean_homogeneity'] is not None


# ---------------------------------------------------------------------------
# Tests: Escalation filter (critical design decision)
# ---------------------------------------------------------------------------

class TestEscalationFilter:
    def test_two_burst_platforms_excluded(self):
        """Platforms with exactly 2 bursts should NOT contribute to escalation slopes."""
        from experiments.reporting import run_phase2_burst

        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)

            # Create synthetic burst data:
            # Platform A: 2 bursts (should be EXCLUDED from slopes)
            # Platform B: 4 bursts (should be INCLUDED)
            per_iter_burst = {
                '0': {
                    'A': {
                        'n_bursts': 2, 'has_bursts': True,
                        'burst_sizes': [15.0, 20.0], 'burst_steps': [10, 30],
                        'burst_intervals': [20], 'escalation_slope': 5.0,
                        'escalation_r2': 1.0,  # mechanical R²=1.0 artifact
                        'classification': 'raiding_stable',
                        'total_outflow': 100.0,
                    },
                    'B': {
                        'n_bursts': 4, 'has_bursts': True,
                        'burst_sizes': [10.0, 12.0, 14.0, 16.0],
                        'burst_steps': [5, 15, 25, 35],
                        'burst_intervals': [10, 10, 10],
                        'escalation_slope': 2.0, 'escalation_r2': 0.95,
                        'classification': 'raiding_base',
                        'total_outflow': 200.0,
                    },
                },
            }

            with open(config_dir / "per_iter_burst_analysis.json", 'w') as f:
                json.dump(per_iter_burst, f)

            result = run_phase2_burst(config_dir)

            # Only platform B's slope should be included
            assert result['escalation_n_slopes'] == 1
            assert result['escalation_mean_slope'] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Tests: Zero-event handling
# ---------------------------------------------------------------------------

class TestZeroEventHandling:
    def test_no_bursts_produces_valid_json(self):
        """Configs with no raids produce valid JSON with None stats."""
        from experiments.reporting import run_phase2_burst

        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            per_iter_burst = {
                '0': {
                    'A': {
                        'n_bursts': 0, 'has_bursts': False,
                        'burst_sizes': [], 'burst_steps': [],
                        'burst_intervals': [],
                        'classification': 'quiet',
                        'total_outflow': 2.0,
                    },
                },
            }
            with open(config_dir / "per_iter_burst_analysis.json", 'w') as f:
                json.dump(per_iter_burst, f)

            result = run_phase2_burst(config_dir)
            assert result['burst_size_mean'] is None
            assert result['escalation_n_slopes'] == 0
            assert result['escalation_mean_slope'] is None
            # JSON should be valid
            with open(config_dir / "burst_aggregate.json") as f:
                data = json.load(f)
            assert data['burst_size_mean'] is None

    def test_empty_displacement_produces_valid_json(self):
        from experiments.reporting import run_phase2_displacement

        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            per_iter_disp = {
                '0': {'n_events': 0, 'error': 'no raid events found'},
            }
            with open(config_dir / "per_iter_displacement.json", 'w') as f:
                json.dump(per_iter_disp, f)

            result = run_phase2_displacement(config_dir)
            assert result['total_events'] == 0
            assert result['mainstream_util_delta_mean'] is None


# ---------------------------------------------------------------------------
# Tests: Summary update
# ---------------------------------------------------------------------------

class TestSummaryUpdate:
    def test_appends_reporting_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)

            # Create minimal summary.csv
            summary_path = config_dir / "summary.csv"
            with open(summary_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['Measure', 'Mean', 'SD'])
                writer.writeheader()
                writer.writerow({'Measure': 'avg_utility', 'Mean': '5.0', 'SD': '0.1'})

            # Create convergence aggregate
            step_metrics = _make_step_metrics(3, 20)
            run_phase1_convergence(config_dir, step_metrics)
            run_phase2_convergence(config_dir)
            run_phase2_summary_update(config_dir)

            # Verify rows were added
            with open(summary_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            measures = [r['Measure'] for r in rows]
            assert 'avg_utility' in measures  # original preserved
            assert 'convergence_settling_step' in measures  # new metric added
            assert 'convergence_pattern_mode' in measures


# ---------------------------------------------------------------------------
# Tests: Integration (mini experiment through full pipeline)
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_exp1_config_pipeline(self):
        """Run a minimal exp1 config through the full reporting pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            step_metrics = _make_step_metrics(3, 20)

            with open(config_dir / "step_metrics.json", 'w') as f:
                json.dump(step_metrics, f)

            config = {
                'name': 'test_exp1',
                'rho_extremist': 0,  # exp1: no extremists
            }

            run_config_reporting(config_dir, config)

            # Phase 1 output
            assert (config_dir / "per_iter_convergence.json").exists()
            # Phase 2 output
            assert (config_dir / "convergence_aggregate.json").exists()
            # exp2-only outputs should NOT exist
            assert not (config_dir / "per_iter_burst_analysis.json").exists()
            assert not (config_dir / "per_iter_enclave_analysis.json").exists()

    def test_exp2_config_convergence_only(self):
        """exp2 config without dynamics files still produces convergence output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            step_metrics = _make_step_metrics(3, 20)

            with open(config_dir / "step_metrics.json", 'w') as f:
                json.dump(step_metrics, f)

            config = {
                'name': 'test_exp2',
                'rho_extremist': 0.2,  # exp2
            }

            run_config_reporting(config_dir, config)

            # Convergence should always be produced
            assert (config_dir / "per_iter_convergence.json").exists()
            assert (config_dir / "convergence_aggregate.json").exists()

    def test_missing_step_metrics_skips_gracefully(self):
        """Config without step_metrics.json should not crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            config = {'name': 'test_missing', 'rho_extremist': 0}
            run_config_reporting(config_dir, config)
            # No files should be created
            assert not (config_dir / "per_iter_convergence.json").exists()
