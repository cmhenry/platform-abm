"""ExperimentRunner: execute configs, collect results, generate summaries."""

from __future__ import annotations

import csv
import json
import logging
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from experiments.configs.experiment_config import ExperimentConfig
from platform_abm.analyzer import MovementAnalyzer
from platform_abm.model import MiniTiebout
from platform_abm.reporter import IterationResult, SimulationReporter
from platform_abm.tracker import RelocationTracker

logger = logging.getLogger(__name__)

# --- BLAS thread control for parallel execution ---

_BLAS_THREAD_ENVS = [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",  # Apple Accelerate
]


def _limit_blas_threads(n: int = 1) -> dict[str, str | None]:
    """Set BLAS thread env vars, return old values for restoration."""
    old = {k: os.environ.get(k) for k in _BLAS_THREAD_ENVS}
    for k in _BLAS_THREAD_ENVS:
        os.environ[k] = str(n)
    return old


def _restore_blas_threads(old: dict[str, str | None]) -> None:
    """Restore BLAS thread env vars to previous values."""
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _init_worker() -> None:
    """ProcessPoolExecutor initializer — limit BLAS threads in each worker.

    Env vars handle 'spawn' (macOS): numpy reads them at import time.
    threadpoolctl handles 'fork' (Linux/HPC): reconfigures already-initialized BLAS.
    """
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(1)
    except ImportError:
        pass

# CSV columns for raw per-iteration output
_RAW_COLUMNS = [
    "iteration", "seed",
    "avg_utility", "avg_utility_mainstream", "avg_utility_extremist",
    "norm_utility_mainstream", "norm_utility_extremist",
    "total_relocations", "avg_relocations_per_community",
    "settling_time_90pct",
]


def _get_git_hash() -> str:
    """Capture current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _analyze_dynamics_scalar(model: MiniTiebout, config: ExperimentConfig) -> dict[str, Any]:
    """Extract scalar dynamics summaries from a single iteration."""
    scalars: dict[str, Any] = {"has_cycle": False, "mean_homogeneity": 0.0}

    if not config.tracking_enabled or model.tracker is None:
        return scalars

    tracker = model.tracker
    platform_ids = [p.id for p in model.platforms]

    try:
        analyzer = MovementAnalyzer(tracker, platform_ids)

        # Raiding cycles
        raiding = analyzer.detect_raiding_cycles(config.t_max)
        scalars["has_cycle"] = any(r["has_cycle"] for r in raiding.values())

        # Enclaves
        community_types = {c.id: c.type for c in model.communities}
        enclaves = analyzer.detect_enclaves(community_types)
        if enclaves:
            scalars["mean_homogeneity"] = float(np.mean(
                [e["mean_homogeneity"] for e in enclaves.values()]
            ))
    except Exception as e:
        logger.warning("Dynamics analysis failed for %s: %s", config.name, e)

    return scalars


def _run_iteration_worker(
    config: ExperimentConfig, iteration: int
) -> tuple[int, IterationResult, dict[str, Any]]:
    """Run a single iteration in a worker process.

    Returns (iteration_index, result, dynamics_scalar).
    Top-level function for pickle compatibility with ProcessPoolExecutor.
    """
    params = config.to_params(iteration)
    model = MiniTiebout(params)
    if config.tracking_enabled:
        model.tracker = RelocationTracker(enabled=True)
    model.run()
    result = SimulationReporter.from_model(model)
    dynamics_scalar = _analyze_dynamics_scalar(model, config)
    return iteration, result, dynamics_scalar


class ConfigResult:
    """Results from running all iterations of a single config."""

    def __init__(
        self,
        config: ExperimentConfig,
        config_dir: Path,
        iteration_results: list[IterationResult],
        dynamics_scalars: list[dict[str, Any]],
    ) -> None:
        self.config = config
        self.config_dir = config_dir
        self.iteration_results = iteration_results
        self.dynamics_scalars = dynamics_scalars


class ExperimentRunner:
    """Runs experiment configs with iteration loops, crash recovery, and reporting."""

    def __init__(self, output_dir: str = "results", max_workers: int | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.git_hash = _get_git_hash()
        self.max_workers = max_workers
        self._executor: ProcessPoolExecutor | None = None
        self._old_blas_env: dict[str, str | None] | None = None

    def _get_executor(self) -> ProcessPoolExecutor:
        """Return a shared ProcessPoolExecutor, creating it lazily."""
        if self._executor is None:
            self._old_blas_env = _limit_blas_threads(1)
            self._executor = ProcessPoolExecutor(
                max_workers=self.max_workers,
                initializer=_init_worker,
            )
        return self._executor

    def shutdown(self) -> None:
        """Shut down the shared executor and restore BLAS env vars."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        if self._old_blas_env is not None:
            _restore_blas_threads(self._old_blas_env)
            self._old_blas_env = None

    def run_config(self, config: ExperimentConfig) -> ConfigResult:
        """Run all iterations for a single config.

        Features:
        - Skip-if-done via index.json check
        - Crash recovery: resumes from last completed row in raw.csv
        - Per-iteration CSV append with flush
        - Memory-safe: processes and discards model data each iteration
        """
        experiment_dir = self.output_dir / config.experiment
        experiment_dir.mkdir(parents=True, exist_ok=True)
        config_dir = experiment_dir / config.name
        config_dir.mkdir(parents=True, exist_ok=True)

        # Skip if already done
        if self._is_config_done(experiment_dir, config.name):
            logger.info("Skipping %s (already complete)", config.name)
            return self._load_existing_result(config, config_dir)

        # Save config
        config_path = config_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config.to_dict(), f, indent=2)

        # Determine start iteration (crash recovery)
        raw_path = config_dir / "raw.csv"
        start_iter = self._count_existing_rows(raw_path)
        if start_iter > 0:
            logger.info("Resuming %s from iteration %d", config.name, start_iter)

        # Open raw CSV for appending
        write_header = start_iter == 0
        raw_file = open(raw_path, "a", newline="")
        raw_writer = csv.DictWriter(raw_file, fieldnames=_RAW_COLUMNS)
        if write_header:
            raw_writer.writeheader()
            raw_file.flush()

        iteration_results: list[IterationResult] = []
        dynamics_scalars: list[dict[str, Any]] = []
        last_model = None
        wall_start = time.time()

        if self.max_workers is not None:
            # Parallel path
            iteration_results, dynamics_scalars = self._run_iterations_parallel(
                config, start_iter,
            )

            # Write all raw rows at once
            try:
                for i, (result, dyn) in enumerate(
                    zip(iteration_results, dynamics_scalars), start=start_iter
                ):
                    row = self._make_raw_row(i, config, result)
                    raw_writer.writerow(row)
                raw_file.flush()
            finally:
                raw_file.close()

            # For detailed dynamics: re-run last iteration in main process
            if config.tracking_enabled:
                _, last_model = self._run_single_iteration(
                    config, config.n_iterations - 1
                )
        else:
            # Sequential path
            try:
                for i in range(start_iter, config.n_iterations):
                    iter_result, model = self._run_single_iteration(config, i)
                    iteration_results.append(iter_result)

                    # Extract dynamics scalars
                    dyn_scalar = _analyze_dynamics_scalar(model, config)
                    dynamics_scalars.append(dyn_scalar)

                    # Append raw row
                    row = self._make_raw_row(i, config, iter_result)
                    raw_writer.writerow(row)
                    raw_file.flush()

                    # Keep last model for detailed dynamics
                    if i == config.n_iterations - 1:
                        last_model = model

                    # Progress logging
                    elapsed = time.time() - wall_start
                    done = i - start_iter + 1
                    total = config.n_iterations - start_iter
                    if done > 0 and done % 10 == 0:
                        rate = elapsed / done
                        remaining = rate * (total - done)
                        logger.info(
                            "  %s: %d/%d iterations (%.1fs elapsed, ~%.0fs remaining)",
                            config.name, i + 1, config.n_iterations, elapsed, remaining,
                        )
            finally:
                raw_file.close()

        # Generate summary
        reporter = SimulationReporter()
        for result in iteration_results:
            reporter.add_iteration(result)
        summary = reporter.compute_summary()
        reporter.to_csv(str(config_dir / "summary.csv"))

        # Save per-step metrics (always, regardless of tracking_enabled)
        self._save_step_logs(config_dir, iteration_results)

        # Save dynamics
        if config.tracking_enabled and last_model is not None:
            self._save_dynamics(config_dir, dynamics_scalars, last_model)

        # Mark config as done
        self._update_index(experiment_dir, config.name)

        wall_total = time.time() - wall_start
        logger.info("Completed %s: %d iterations in %.1fs", config.name, config.n_iterations, wall_total)

        return ConfigResult(config, config_dir, iteration_results, dynamics_scalars)

    def run_experiment(self, configs: list[ExperimentConfig]) -> Path:
        """Run all configs for an experiment. Returns the experiment directory."""
        if not configs:
            raise ValueError("No configs to run")

        experiment_name = configs[0].experiment
        experiment_dir = self.output_dir / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Starting experiment '%s' with %d configs", experiment_name, len(configs))
        results: list[ConfigResult] = []

        try:
            for idx, config in enumerate(configs):
                logger.info("[%d/%d] Running config: %s", idx + 1, len(configs), config.name)
                result = self.run_config(config)
                results.append(result)

            # Generate experiment-level summary
            self._save_experiment_summary(experiment_dir, results)
        finally:
            self.shutdown()

        logger.info("Experiment '%s' complete. Output: %s", experiment_name, experiment_dir)
        return experiment_dir

    def _run_single_iteration(
        self, config: ExperimentConfig, iteration: int
    ) -> tuple[IterationResult, MiniTiebout]:
        """Run a single iteration and return the result + model."""
        params = config.to_params(iteration)
        model = MiniTiebout(params)

        if config.tracking_enabled:
            tracker = RelocationTracker(enabled=True)
            model.tracker = tracker

        model.run()
        result = SimulationReporter.from_model(model)
        return result, model

    def _run_iterations_parallel(
        self,
        config: ExperimentConfig,
        start_iter: int,
    ) -> tuple[list[IterationResult], list[dict[str, Any]]]:
        """Run iterations in parallel via ProcessPoolExecutor."""
        iterations = list(range(start_iter, config.n_iterations))
        iteration_results: list[IterationResult | None] = [None] * len(iterations)
        dynamics_scalars: list[dict[str, Any] | None] = [None] * len(iterations)

        wall_start = time.time()
        total = len(iterations)

        executor = self._get_executor()
        futures = {
            executor.submit(_run_iteration_worker, config, i): i
            for i in iterations
        }
        done_count = 0
        for future in as_completed(futures):
            iteration, result, dyn_scalar = future.result()
            idx = iteration - start_iter
            iteration_results[idx] = result
            dynamics_scalars[idx] = dyn_scalar
            done_count += 1
            if done_count % 10 == 0:
                elapsed = time.time() - wall_start
                logger.info(
                    "  %s: %d/%d iterations (%.1fs elapsed)",
                    config.name, done_count, total, elapsed,
                )

        return iteration_results, dynamics_scalars  # type: ignore[return-value]

    def _make_raw_row(
        self, iteration: int, config: ExperimentConfig, result: IterationResult
    ) -> dict[str, Any]:
        """Build a raw CSV row from an iteration result."""
        utilities = np.array(result.community_utilities)
        types = result.community_types
        moves = result.community_moves
        last_steps = result.community_last_move_steps

        # By-type utilities
        mainstream_mask = [t == "mainstream" for t in types]
        extremist_mask = [t == "extremist" for t in types]

        avg_u = float(np.mean(utilities))
        avg_u_main = float(np.mean(utilities[mainstream_mask])) if any(mainstream_mask) else 0.0
        avg_u_ext = float(np.mean(utilities[extremist_mask])) if any(extremist_mask) else 0.0

        # Normalized
        p_space = result.p_space
        alpha = result.alpha
        norm_main = avg_u_main / p_space if p_space > 0 else 0.0
        norm_ext = avg_u_ext / (p_space + alpha) if (p_space + alpha) > 0 else 0.0

        # Relocations
        relocations = [max(0, m - 1) for m in moves]
        total_reloc = sum(relocations)

        # Settling time 90th pct
        import math
        sorted_steps = sorted(last_steps)
        n = len(sorted_steps)
        idx = math.ceil(0.9 * n) - 1
        idx = max(0, min(idx, n - 1))

        return {
            "iteration": iteration,
            "seed": config.seed_base + iteration,
            "avg_utility": f"{avg_u:.6f}",
            "avg_utility_mainstream": f"{avg_u_main:.6f}",
            "avg_utility_extremist": f"{avg_u_ext:.6f}",
            "norm_utility_mainstream": f"{norm_main:.6f}",
            "norm_utility_extremist": f"{norm_ext:.6f}",
            "total_relocations": total_reloc,
            "avg_relocations_per_community": f"{total_reloc / result.n_comms:.6f}",
            "settling_time_90pct": sorted_steps[idx],
        }

    def _save_dynamics(
        self,
        config_dir: Path,
        dynamics_scalars: list[dict[str, Any]],
        last_model: MiniTiebout,
    ) -> None:
        """Save aggregated dynamics scalars and detailed dynamics from last iteration."""
        dynamics_dir = config_dir / "dynamics"
        dynamics_dir.mkdir(exist_ok=True)

        # Aggregated scalars
        n = len(dynamics_scalars)
        if n > 0:
            cycle_rate = sum(1 for d in dynamics_scalars if d.get("has_cycle")) / n
            homogeneities = [d.get("mean_homogeneity", 0.0) for d in dynamics_scalars]
            with open(dynamics_dir / "scalars.json", "w") as f:
                json.dump({
                    "n_iterations": n,
                    "cycle_rate": cycle_rate,
                    "mean_homogeneity": float(np.mean(homogeneities)),
                    "sd_homogeneity": float(np.std(homogeneities, ddof=1)) if n > 1 else 0.0,
                }, f, indent=2)

        # Detailed dynamics from last iteration
        if last_model.tracker is not None:
            platform_ids = [p.id for p in last_model.platforms]
            try:
                analyzer = MovementAnalyzer(last_model.tracker, platform_ids)

                # Flow matrices
                flows = analyzer.compute_flow_matrices()
                if flows:
                    np.savez_compressed(
                        str(dynamics_dir / "flow.npz"),
                        **{str(k): v for k, v in flows.items()},
                    )

                # Raiding analysis
                raiding = analyzer.detect_raiding_cycles(last_model.p.steps)
                raiding_json: dict[str, Any] = {}
                for pid, data in raiding.items():
                    raiding_json[str(pid)] = {
                        "has_cycle": data["has_cycle"],
                        "significant_lags": data["significant_lags"],
                        "outflow_series": data["outflow_series"].tolist(),
                        "acf": data["acf"].tolist(),
                    }
                with open(dynamics_dir / "raiding.json", "w") as f:
                    json.dump(raiding_json, f, indent=2)

                # Enclave analysis
                community_types = {c.id: c.type for c in last_model.communities}
                enclaves = analyzer.detect_enclaves(community_types)
                enclaves_json: dict[str, Any] = {}
                for pid, data in enclaves.items():
                    enclaves_json[str(pid)] = {
                        "mean_homogeneity": data["mean_homogeneity"],
                        "fraction_enclaved": data["fraction_enclaved"],
                        "homogeneity_series": data["homogeneity_series"].tolist(),
                    }
                with open(dynamics_dir / "enclaves.json", "w") as f:
                    json.dump(enclaves_json, f, indent=2)

                # Residence times
                community_ids = [c.id for c in last_model.communities]
                initial_assignments = {
                    c.id: c.platform.id for c in last_model.communities
                }
                # Note: initial_assignments here reflect final platform, not initial.
                # For detailed residence analysis, we'd need to track initial assignments.
                # This is best-effort from the last iteration's final state.

            except Exception as e:
                logger.warning("Detailed dynamics save failed: %s", e)

    def _save_step_logs(
        self,
        config_dir: Path,
        iteration_results: list[IterationResult],
    ) -> None:
        """Save per-step metrics from all iterations to step_metrics.json."""
        data: dict[str, list[dict]] = {}
        for i, result in enumerate(iteration_results):
            if result.step_log is not None:
                data[str(i)] = result.step_log

        if data:
            with open(config_dir / "step_metrics.json", "w") as f:
                json.dump(data, f, indent=2)

    def _count_existing_rows(self, raw_path: Path) -> int:
        """Count data rows in existing raw.csv for crash recovery."""
        if not raw_path.exists():
            return 0
        try:
            with open(raw_path, "r") as f:
                reader = csv.reader(f)
                next(reader, None)  # skip header
                return sum(1 for _ in reader)
        except Exception:
            return 0

    def _is_config_done(self, experiment_dir: Path, config_name: str) -> bool:
        """Check if a config is already marked complete in index.json."""
        index_path = experiment_dir / "index.json"
        if not index_path.exists():
            return False
        try:
            with open(index_path) as f:
                index = json.load(f)
            return config_name in index.get("completed", [])
        except Exception:
            return False

    def _update_index(self, experiment_dir: Path, config_name: str) -> None:
        """Mark a config as completed in index.json."""
        index_path = experiment_dir / "index.json"
        if index_path.exists():
            with open(index_path) as f:
                index = json.load(f)
        else:
            index = {"git_hash": self.git_hash, "completed": []}

        if config_name not in index["completed"]:
            index["completed"].append(config_name)

        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

    def _load_existing_result(self, config: ExperimentConfig, config_dir: Path) -> ConfigResult:
        """Load a stub ConfigResult for an already-completed config."""
        return ConfigResult(config, config_dir, [], [])

    def _save_experiment_summary(self, experiment_dir: Path, results: list[ConfigResult]) -> None:
        """Generate experiment-level summary CSV combining all configs."""
        summary_path = experiment_dir / "summary.csv"

        rows: list[dict[str, Any]] = []
        for result in results:
            config = result.config
            # Read config's summary.csv if it exists
            config_summary_path = result.config_dir / "summary.csv"
            if not config_summary_path.exists():
                continue

            measures: dict[str, str] = {}
            with open(config_summary_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    measures[row["Measure"]] = row["Mean"]

            row_data: dict[str, Any] = {
                "config_name": config.name,
                "institution": config.institution,
                "n_platforms": config.n_platforms,
                "rho_extremist": config.rho_extremist,
                "alpha": config.alpha,
                "p_space": config.p_space,
                "n_communities": config.n_communities,
            }
            row_data.update(measures)
            rows.append(row_data)

        if rows:
            fieldnames = list(rows[0].keys())
            # Union all keys across rows
            for row in rows[1:]:
                for key in row:
                    if key not in fieldnames:
                        fieldnames.append(key)

            with open(summary_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)

            logger.info("Experiment summary saved to %s", summary_path)
