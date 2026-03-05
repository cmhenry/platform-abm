"""Tests for ExperimentRunner — small-scale integration tests."""

import csv
import json
import os
import tempfile
from pathlib import Path

import pytest

from experiments.configs.experiment_config import ExperimentConfig
from experiments.runner import (
    ExperimentRunner,
    _BLAS_THREAD_ENVS,
    _limit_blas_threads,
    _restore_blas_threads,
)


def _make_small_config(
    name: str = "test_cfg",
    tracking: bool = False,
    institution: str = "direct",
    rho: float = 0.0,
) -> ExperimentConfig:
    """Create a minimal config for fast testing."""
    return ExperimentConfig(
        name=name,
        experiment="test",
        n_communities=10,
        n_platforms=2,
        p_space=5,
        t_max=3,
        institution=institution,
        rho_extremist=rho,
        alpha=1.0 if rho > 0 else 0.0,
        coalitions=3,
        mutations=2,
        svd_groups=2,
        search_steps=3,
        initial_distribution="equal",
        tracking_enabled=tracking,
        n_iterations=2,
        seed_base=42,
    )


def test_single_config_run():
    """Run a single config and verify output files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = ExperimentRunner(output_dir=tmpdir)
        config = _make_small_config()
        result = runner.run_config(config)

        # Check files exist
        assert (result.config_dir / "config.json").exists()
        assert (result.config_dir / "raw.csv").exists()
        assert (result.config_dir / "summary.csv").exists()


def test_raw_csv_row_count():
    """raw.csv has correct number of rows."""
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = ExperimentRunner(output_dir=tmpdir)
        config = _make_small_config(name="test_rows")
        result = runner.run_config(config)

        with open(result.config_dir / "raw.csv") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == config.n_iterations


def test_config_json_saved():
    """config.json is saved and matches input."""
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = ExperimentRunner(output_dir=tmpdir)
        config = _make_small_config(name="test_json")
        result = runner.run_config(config)

        with open(result.config_dir / "config.json") as f:
            saved = json.load(f)
        assert saved["name"] == "test_json"
        assert saved["n_communities"] == 10


def test_skip_if_done():
    """Running the same config twice skips on second run."""
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = ExperimentRunner(output_dir=tmpdir)
        config = _make_small_config(name="test_skip")

        result1 = runner.run_config(config)
        assert len(result1.iteration_results) == 2

        result2 = runner.run_config(config)
        # Second run should be skipped (empty results)
        assert len(result2.iteration_results) == 0


def test_tracking_enabled():
    """Tracker is attached when tracking_enabled=True."""
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = ExperimentRunner(output_dir=tmpdir)
        config = _make_small_config(
            name="test_track",
            tracking=True,
            institution="mixed",
            rho=0.20,
        )
        result = runner.run_config(config)

        # Should have dynamics directory
        dynamics_dir = result.config_dir / "dynamics"
        assert dynamics_dir.exists()


def test_summary_generated():
    """Summary CSV has content."""
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = ExperimentRunner(output_dir=tmpdir)
        config = _make_small_config(name="test_summary")
        result = runner.run_config(config)

        with open(result.config_dir / "summary.csv") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) > 0
        # Check expected measures exist
        measure_names = [r["Measure"] for r in rows]
        assert "avg_utility" in measure_names
        assert "total_relocations" in measure_names


def test_run_experiment():
    """run_experiment runs multiple configs and creates experiment summary."""
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = ExperimentRunner(output_dir=tmpdir)
        configs = [
            _make_small_config(name="multi_a"),
            _make_small_config(name="multi_b"),
        ]
        exp_dir = runner.run_experiment(configs)

        assert (exp_dir / "summary.csv").exists()
        assert (exp_dir / "index.json").exists()

        # Check index has both configs
        with open(exp_dir / "index.json") as f:
            index = json.load(f)
        assert "multi_a" in index["completed"]
        assert "multi_b" in index["completed"]


# --- Parallel execution tests ---


def test_parallel_single_config():
    """Run a small config with max_workers=2, verify output files and row count."""
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = ExperimentRunner(output_dir=tmpdir, max_workers=2)
        config = _make_small_config(name="par_single")
        result = runner.run_config(config)

        assert (result.config_dir / "config.json").exists()
        assert (result.config_dir / "raw.csv").exists()
        assert (result.config_dir / "summary.csv").exists()

        with open(result.config_dir / "raw.csv") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == config.n_iterations


def test_parallel_results_deterministic():
    """Sequential and parallel runs with same seeds produce identical raw.csv rows."""
    with tempfile.TemporaryDirectory() as tmpdir_seq, \
         tempfile.TemporaryDirectory() as tmpdir_par:
        # Sequential
        runner_seq = ExperimentRunner(output_dir=tmpdir_seq)
        config_seq = _make_small_config(name="det_test")
        result_seq = runner_seq.run_config(config_seq)

        # Parallel
        runner_par = ExperimentRunner(output_dir=tmpdir_par, max_workers=2)
        config_par = _make_small_config(name="det_test")
        result_par = runner_par.run_config(config_par)

        # Compare raw CSV rows
        with open(result_seq.config_dir / "raw.csv") as f:
            rows_seq = list(csv.DictReader(f))
        with open(result_par.config_dir / "raw.csv") as f:
            rows_par = list(csv.DictReader(f))

        assert len(rows_seq) == len(rows_par)
        for r_seq, r_par in zip(rows_seq, rows_par):
            assert r_seq == r_par


def test_parallel_run_experiment():
    """run_experiment with max_workers=2 creates experiment summary and index."""
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = ExperimentRunner(output_dir=tmpdir, max_workers=2)
        configs = [
            _make_small_config(name="par_multi_a"),
            _make_small_config(name="par_multi_b"),
        ]
        exp_dir = runner.run_experiment(configs)

        assert (exp_dir / "summary.csv").exists()
        assert (exp_dir / "index.json").exists()

        with open(exp_dir / "index.json") as f:
            index = json.load(f)
        assert "par_multi_a" in index["completed"]
        assert "par_multi_b" in index["completed"]


def test_parallel_with_tracking():
    """Parallel run with tracking creates dynamics directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = ExperimentRunner(output_dir=tmpdir, max_workers=2)
        config = _make_small_config(
            name="par_track",
            tracking=True,
            institution="mixed",
            rho=0.20,
        )
        result = runner.run_config(config)

        dynamics_dir = result.config_dir / "dynamics"
        assert dynamics_dir.exists()


def test_step_metrics_json_created():
    """step_metrics.json is created even with tracking_enabled=False."""
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = ExperimentRunner(output_dir=tmpdir)
        config = _make_small_config(name="step_met", tracking=False)
        result = runner.run_config(config)

        step_metrics_path = result.config_dir / "step_metrics.json"
        assert step_metrics_path.exists()

        with open(step_metrics_path) as f:
            data = json.load(f)

        # Should have one entry per iteration
        assert len(data) == config.n_iterations
        # Each iteration should have t_max steps
        for key in data:
            steps = data[key]
            assert len(steps) == config.t_max
            for entry in steps:
                assert "step" in entry
                assert "avg_utility" in entry
                assert "n_relocations" in entry
                assert "per_governance_utilities" in entry


def test_step_metrics_json_parallel():
    """step_metrics.json is created with parallel execution (workers=2)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = ExperimentRunner(output_dir=tmpdir, max_workers=2)
        config = _make_small_config(name="step_met_par", tracking=False)
        result = runner.run_config(config)

        step_metrics_path = result.config_dir / "step_metrics.json"
        assert step_metrics_path.exists()

        with open(step_metrics_path) as f:
            data = json.load(f)

        assert len(data) == config.n_iterations
        for key in data:
            steps = data[key]
            assert len(steps) == config.t_max


def test_stepwise_csv_created():
    """stepwise.csv is created with correct row count."""
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = ExperimentRunner(output_dir=tmpdir)
        config = _make_small_config(name="stepwise_test")
        result = runner.run_config(config)

        stepwise_path = result.config_dir / "stepwise.csv"
        assert stepwise_path.exists()

        with open(stepwise_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == config.t_max
        assert "step" in rows[0]
        assert "avg_utility_mean" in rows[0]
        assert "n_relocations_mean" in rows[0]


def test_convergence_json_created():
    """convergence.json is created with required keys."""
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = ExperimentRunner(output_dir=tmpdir)
        config = _make_small_config(name="conv_test")
        result = runner.run_config(config)

        conv_path = result.config_dir / "convergence.json"
        assert conv_path.exists()

        with open(conv_path) as f:
            data = json.load(f)
        assert "pattern" in data
        assert "tail_slope" in data
        assert "util_end" in data
        assert "reloc_reduction_pct" in data


def test_stepwise_csv_parallel():
    """stepwise.csv is created with parallel execution (workers=2)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = ExperimentRunner(output_dir=tmpdir, max_workers=2)
        config = _make_small_config(name="stepwise_par")
        result = runner.run_config(config)

        stepwise_path = result.config_dir / "stepwise.csv"
        assert stepwise_path.exists()

        with open(stepwise_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == config.t_max


def test_convergence_in_experiment_summary():
    """Experiment summary CSV includes convergence_pattern column."""
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = ExperimentRunner(output_dir=tmpdir)
        configs = [
            _make_small_config(name="conv_sum_a"),
            _make_small_config(name="conv_sum_b"),
        ]
        exp_dir = runner.run_experiment(configs)

        with open(exp_dir / "summary.csv") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        for row in rows:
            assert "convergence_pattern" in row
            assert row["convergence_pattern"] != ""


def test_parallel_blas_env_restored():
    """BLAS env vars are set when pool is created and restored after shutdown."""
    # Save original env state
    orig = {k: os.environ.get(k) for k in _BLAS_THREAD_ENVS}

    with tempfile.TemporaryDirectory() as tmpdir:
        runner = ExperimentRunner(output_dir=tmpdir, max_workers=2)
        config = _make_small_config(name="blas_test")

        # Run a config (triggers pool creation which sets env vars)
        runner.run_config(config)

        # While pool is alive, env vars should be set to "1"
        for k in _BLAS_THREAD_ENVS:
            assert os.environ.get(k) == "1", f"{k} should be '1' while pool is active"

        # Shutdown should restore env vars
        runner.shutdown()

        for k in _BLAS_THREAD_ENVS:
            assert os.environ.get(k) == orig[k], (
                f"{k} should be restored to {orig[k]!r} after shutdown"
            )


# --- Burst analysis integration tests ---


def test_burst_aggregate_created():
    """Tracked config with extremists produces burst_aggregate.json."""
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = ExperimentRunner(output_dir=tmpdir)
        config = _make_small_config(
            name="burst_test",
            tracking=True,
            institution="mixed",
            rho=0.20,
        )
        result = runner.run_config(config)

        burst_agg_path = result.config_dir / "dynamics" / "burst_aggregate.json"
        assert burst_agg_path.exists(), "burst_aggregate.json should be created"

        with open(burst_agg_path) as f:
            agg = json.load(f)

        # Check expected keys
        expected_keys = [
            "n_iterations", "n_platform_iterations",
            "n_with_bursts", "n_with_escalation",
            "burst_rate", "escalation_rate",
            "mean_burst_size", "n_total_bursts",
            "escalation_n_slopes", "escalation_ttest",
            "classification_counts",
        ]
        for key in expected_keys:
            assert key in agg, f"Missing key: {key}"

        assert agg["n_iterations"] == config.n_iterations
        assert agg["n_platform_iterations"] > 0
        assert isinstance(agg["classification_counts"], dict)

        # burst_aggregate should also be on the ConfigResult
        assert result.burst_aggregate is not None


def test_burst_aggregate_parallel():
    """Tracked config with parallel workers produces burst_aggregate.json."""
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = ExperimentRunner(output_dir=tmpdir, max_workers=2)
        config = _make_small_config(
            name="burst_par",
            tracking=True,
            institution="mixed",
            rho=0.20,
        )
        result = runner.run_config(config)

        burst_agg_path = result.config_dir / "dynamics" / "burst_aggregate.json"
        assert burst_agg_path.exists(), "burst_aggregate.json should be created in parallel mode"

        with open(burst_agg_path) as f:
            agg = json.load(f)

        assert agg["n_iterations"] == config.n_iterations
        assert agg["n_platform_iterations"] > 0


def test_burst_master_csv():
    """run_experiment with tracked configs produces burst_master.csv."""
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = ExperimentRunner(output_dir=tmpdir)
        configs = [
            _make_small_config(
                name="burst_master_a",
                tracking=True,
                institution="mixed",
                rho=0.20,
            ),
            _make_small_config(
                name="burst_master_b",
                tracking=True,
                institution="direct",
                rho=0.30,
            ),
        ]
        exp_dir = runner.run_experiment(configs)

        master_path = exp_dir / "burst_master.csv"
        assert master_path.exists(), "burst_master.csv should be created"

        with open(master_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2, f"Expected 2 rows, got {len(rows)}"

        # Check expected columns
        expected_cols = [
            "config_name", "institution", "n_platforms", "rho_extremist", "alpha",
            "n_iterations", "burst_rate", "escalation_rate",
            "escalation_ttest_t", "escalation_ttest_p",
        ]
        for col in expected_cols:
            assert col in rows[0], f"Missing column: {col}"

        config_names = {r["config_name"] for r in rows}
        assert config_names == {"burst_master_a", "burst_master_b"}
