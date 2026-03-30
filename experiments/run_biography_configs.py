"""Run representative configs with per-platform detail logging for biography figures.

Re-runs four target configs with log_platform_detail=True for 50 iterations each,
outputting to results/exp2_detail/. This enables platform biography figures showing
the range of dynamics (quiet sorting, oscillatory recovery, continuous turbulence,
enclave formation).

Target configs:
- np=27, rho_e=0.15, alpha=10  (worst case / high-alpha multi-platform)
- np=9,  rho_e=0.15, alpha=10  (oscillatory regime)
- np=3,  rho_e=0.15, alpha=10  (continuous turbulence)
- np=27, rho_e=0.15, alpha=5   (moderate case)

DO NOT RUN as part of the main analysis pipeline. Run manually when biography
figures are needed:
    python -m experiments.run_biography_configs

Output: results/exp2_detail/<config_name>/platform_detail.csv
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Project root on path (when invoked as python -m experiments.run_biography_configs)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.configs.experiment_config import ExperimentConfig
from experiments.runner import ExperimentRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Shared fixed parameters (matching build_exp2_configs in builders.py)
_COMMON_FIXED: dict = {
    "coalitions": 5,
    "mutations": 3,
    "svd_groups": 10,
    "search_steps": 10,
    "initial_distribution": "equal",
    "seed_base": 42,
}


def build_biography_configs() -> list[ExperimentConfig]:
    """Build the four representative biography configs with detail logging enabled."""
    shared = dict(
        experiment="exp2",
        n_communities=900,
        p_space=10,
        t_max=100,
        institution="mixed",
        tracking_enabled=True,
        n_iterations=50,
        log_platform_detail=True,
        **_COMMON_FIXED,
    )

    return [
        # np=27, rho=0.15, alpha=10 — worst case (high-alpha, many-platform)
        ExperimentConfig(
            name="exp2_np27_rho015_alpha10",
            n_platforms=27,
            rho_extremist=0.15,
            alpha=10.0,
            **shared,
        ),
        # np=9, rho=0.15, alpha=10 — oscillatory regime
        ExperimentConfig(
            name="exp2_np9_rho015_alpha10",
            n_platforms=9,
            rho_extremist=0.15,
            alpha=10.0,
            **shared,
        ),
        # np=3, rho=0.15, alpha=10 — continuous turbulence
        ExperimentConfig(
            name="exp2_np3_rho015_alpha10",
            n_platforms=3,
            rho_extremist=0.15,
            alpha=10.0,
            **shared,
        ),
        # np=27, rho=0.15, alpha=5 — moderate case
        ExperimentConfig(
            name="exp2_np27_rho015_alpha5",
            n_platforms=27,
            rho_extremist=0.15,
            alpha=5.0,
            **shared,
        ),
    ]


def main() -> None:
    configs = build_biography_configs()

    logger.info(
        "Running %d biography configs with log_platform_detail=True, 50 iterations each",
        len(configs),
    )
    for cfg in configs:
        logger.info(
            "  %s (np=%d, rho=%.2f, alpha=%.1f)",
            cfg.name,
            cfg.n_platforms,
            cfg.rho_extremist,
            cfg.alpha,
        )

    # Output to results/exp2_detail/ to avoid overwriting main exp2 results
    runner = ExperimentRunner(output_dir="results/exp2_detail")

    try:
        for cfg in configs:
            logger.info("Starting config: %s", cfg.name)
            result = runner.run_config(cfg)
            detail_csv = Path("results/exp2_detail") / "exp2" / cfg.name / "platform_detail.csv"
            if detail_csv.exists():
                logger.info(
                    "  Wrote platform_detail.csv (%d rows)",
                    sum(1 for _ in open(detail_csv)) - 1,  # subtract header
                )
            else:
                logger.warning("  platform_detail.csv not found for %s", cfg.name)
    finally:
        runner.shutdown()

    logger.info("All biography configs complete. Output: results/exp2_detail/")


if __name__ == "__main__":
    main()
