"""Experiment 1: Institutional comparisons (no extremists).

6 configs x 200 iterations = 1,200 runs.
"""

from __future__ import annotations

import argparse
import logging

from experiments.configs.builders import build_exp1_configs
from experiments.runner import ExperimentRunner
from experiments.tables import format_exp1_table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment 1")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Print configs without running")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: 2 iterations, N_c=30, t_max=10")
    args = parser.parse_args()

    configs = build_exp1_configs()

    if args.smoke:
        for cfg in configs:
            cfg.n_iterations = 2
            cfg.n_communities = 30
            cfg.t_max = 10

    logger.info("Experiment 1: %d configs%s", len(configs), " (smoke)" if args.smoke else "")

    if args.dry_run:
        for cfg in configs:
            print(f"  {cfg.name}: {cfg.n_communities}c, {cfg.n_platforms}p, "
                  f"{cfg.institution}, t_max={cfg.t_max}, {cfg.n_iterations}i")
        return

    runner = ExperimentRunner(output_dir=args.output_dir)
    experiment_dir = runner.run_experiment(configs)

    # Generate LaTeX tables
    latex = format_exp1_table(experiment_dir)
    tables_path = experiment_dir / "tables.tex"
    with open(tables_path, "w") as f:
        f.write(latex)
    logger.info("LaTeX tables saved to %s", tables_path)


if __name__ == "__main__":
    main()
