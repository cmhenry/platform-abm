"""Smoke test: run all experiment configs at reduced scale, validate outputs."""

from __future__ import annotations

import argparse
import logging
import sys

from experiments.configs.builders import (
    build_exp1_configs,
    build_exp2_configs,
    build_interaction_configs,
    build_oat_configs,
)
from experiments.runner import ExperimentRunner
from experiments.smoke_test import build_smoke_configs, validate_smoke_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run smoke tests for all experiments")
    parser.add_argument("--output-dir", default="results/smoke", help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Print configs without running")
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Max parallel workers for iterations (default: sequential)",
    )
    args = parser.parse_args()

    # Build ALL configs from all experiments
    all_full = (
        build_exp1_configs()
        + build_exp2_configs()
        + build_oat_configs()
        + build_interaction_configs()
    )
    smoke_configs = build_smoke_configs(all_full)

    logger.info("Smoke test: %d configs (reduced from full experiments)", len(smoke_configs))

    if args.dry_run:
        for cfg in smoke_configs:
            print(f"  {cfg.name}: {cfg.n_communities}c, {cfg.n_platforms}p, "
                  f"{cfg.t_max}t, {cfg.n_iterations}i")
        return

    runner = ExperimentRunner(output_dir=args.output_dir, max_workers=args.workers)

    # Run all configs
    all_results = []
    for idx, config in enumerate(smoke_configs):
        logger.info("[%d/%d] Smoke: %s", idx + 1, len(smoke_configs), config.name)
        result = runner.run_config(config)
        all_results.append((config, result))

    # Validate all results
    total_pass = 0
    total_fail = 0
    failures_report: list[str] = []

    for config, result in all_results:
        failures = validate_smoke_results(config, result.config_dir)
        if failures:
            total_fail += 1
            for f in failures:
                failures_report.append(f"  FAIL {config.name}: {f}")
        else:
            total_pass += 1

    # Print report
    print(f"\n{'='*60}")
    print(f"SMOKE TEST RESULTS: {total_pass} passed, {total_fail} failed")
    print(f"{'='*60}")

    if failures_report:
        print("\nFailures:")
        for line in failures_report:
            print(line)

    if total_fail > 0:
        sys.exit(1)
    else:
        print("\nAll smoke tests passed!")


if __name__ == "__main__":
    main()
