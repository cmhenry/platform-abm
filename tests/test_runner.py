"""Tests for ExperimentRunner — small-scale integration tests."""

import csv
import json
import tempfile
from pathlib import Path

import pytest

from experiments.configs.experiment_config import ExperimentConfig
from experiments.runner import ExperimentRunner


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
