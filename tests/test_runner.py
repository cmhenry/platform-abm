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
