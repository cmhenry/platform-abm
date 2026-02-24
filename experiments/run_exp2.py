"""Experiment 2: Mixed institutions with extremists (factorial design).

27 configs x 200 iterations = 5,400 runs.
Tracking enabled for dynamics analysis.
"""

from __future__ import annotations

import argparse
import logging

from experiments.configs.builders import build_exp2_configs
from experiments.runner import ExperimentRunner
from experiments.tables import format_exp2_tables

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment 2")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Print configs without running")
    args = parser.parse_args()

    configs = build_exp2_configs()
    logger.info("Experiment 2: %d configs", len(configs))

    if args.dry_run:
        for cfg in configs:
            print(f"  {cfg.name}: {cfg.n_communities}c, {cfg.n_platforms}p, "
                  f"rho={cfg.rho_extremist}, alpha={cfg.alpha}, {cfg.n_iterations}i")
        return

    runner = ExperimentRunner(output_dir=args.output_dir)
    experiment_dir = runner.run_experiment(configs)

    # Generate LaTeX tables
    latex = format_exp2_tables(experiment_dir)
    tables_path = experiment_dir / "tables.tex"
    with open(tables_path, "w") as f:
        f.write(latex)
    logger.info("LaTeX tables saved to %s", tables_path)


if __name__ == "__main__":
    main()
