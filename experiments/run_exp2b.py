"""Experiment 2b: rho_e disaggregation sweep.

3 configs x 200 iterations = 600 runs.
Varies frac_griefer in {0.25, 0.50, 0.75} at fixed rho_e=0.10,
N_p=9, alpha_i=2, alpha_g=10. Endpoints (f_g=0, 1) are recovered
from exp2 at analysis time.
"""

from __future__ import annotations

import argparse
import logging

from experiments.configs.builders import build_exp2b_configs
from experiments.runner import ExperimentRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment 2b")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Print configs without running")
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Max parallel workers for iterations (default: sequential)",
    )
    args = parser.parse_args()

    configs = build_exp2b_configs()
    logger.info("Experiment 2b: %d configs", len(configs))

    if args.dry_run:
        for cfg in configs:
            print(f"  {cfg.name}: {cfg.n_communities}c, {cfg.n_platforms}p, "
                  f"rho={cfg.rho_extremist}, alpha_i={cfg.alpha_ideologue}, "
                  f"alpha_g={cfg.alpha_griefer}, f_g={cfg.frac_griefer}, "
                  f"{cfg.n_iterations}i")
        return

    runner = ExperimentRunner(output_dir=args.output_dir, max_workers=args.workers)
    runner.run_experiment(configs)


if __name__ == "__main__":
    main()
