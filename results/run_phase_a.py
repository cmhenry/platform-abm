"""Phase A analysis runner.

Runs displacement analysis (A1), enclave settling analysis (A2), ANOVA (A3),
and LaTeX factorial tables (A4) across all 27 exp2 configs using existing
reporting.py functions.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.reporting import (
    run_phase1_burst,
    run_phase1_displacement,
    run_phase1_enclaves,
    run_phase2_burst,
    run_phase2_displacement,
    run_phase2_enclaves,
    run_phase2_summary_update,
    run_phase3_anova,
    run_phase3_burst_master,
    run_phase3_displacement_master,
    run_phase3_enclave_master,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

EXP2_DIR = Path(__file__).resolve().parent / "exp2"

# Correct exp2 parameter grid
N_PLATFORMS = [3, 9, 27]
RHO_VALUES = [0.05, 0.10, 0.15]
ALPHA_VALUES = [2, 5, 10]


def discover_configs() -> list[Path]:
    """Return sorted list of all 27 exp2 config directories."""
    configs = []
    for np_val in N_PLATFORMS:
        for rho in RHO_VALUES:
            rho_str = f"{rho:.2f}".replace(".", "")
            for alpha in ALPHA_VALUES:
                name = f"exp2_np{np_val}_rho{rho_str}_alpha{alpha}"
                config_dir = EXP2_DIR / name
                if config_dir.is_dir():
                    configs.append(config_dir)
                else:
                    logger.warning("Config directory not found: %s", name)
    return configs


def _output_exists(config_dir: Path, filename: str) -> bool:
    return (config_dir / filename).exists()


def process_config(config_dir: Path, force: bool = False) -> None:
    """Run Phase 1 and Phase 2 for a single config."""
    name = config_dir.name
    dynamics_dir = config_dir / "dynamics"

    # Phase 1.1: Burst analysis
    if force or not _output_exists(config_dir, "per_iter_burst_analysis.json"):
        raiding_path = dynamics_dir / "per_iter_raiding.json"
        if raiding_path.exists():
            logger.info("[%s] Phase 1: burst analysis", name)
            with open(raiding_path) as f:
                per_iter_raiding = json.load(f)
            run_phase1_burst(config_dir, per_iter_raiding)
            del per_iter_raiding
            gc.collect()
        else:
            logger.warning("[%s] Missing per_iter_raiding.json, skipping burst", name)
    else:
        logger.info("[%s] Phase 1 burst: already exists", name)

    # Phase 1.2: Enclave analysis
    if force or not _output_exists(config_dir, "per_iter_enclave_analysis.json"):
        enclave_path = dynamics_dir / "per_iter_enclaves.json"
        if enclave_path.exists():
            logger.info("[%s] Phase 1: enclave analysis", name)
            with open(enclave_path) as f:
                per_iter_enclaves = json.load(f)
            run_phase1_enclaves(config_dir, per_iter_enclaves)
            del per_iter_enclaves
            gc.collect()
        else:
            logger.warning("[%s] Missing per_iter_enclaves.json, skipping enclaves", name)
    else:
        logger.info("[%s] Phase 1 enclaves: already exists", name)

    # Phase 1.3: Displacement analysis (needs burst + step_metrics)
    if force or not _output_exists(config_dir, "per_iter_displacement.json"):
        burst_path = config_dir / "per_iter_burst_analysis.json"
        step_path = config_dir / "step_metrics.json"
        if burst_path.exists() and step_path.exists():
            logger.info("[%s] Phase 1: displacement analysis", name)
            with open(burst_path) as f:
                per_iter_burst = json.load(f)
            with open(step_path) as f:
                step_metrics = json.load(f)
            run_phase1_displacement(config_dir, per_iter_burst, step_metrics)
            del per_iter_burst, step_metrics
            gc.collect()
        else:
            logger.warning("[%s] Missing inputs for displacement, skipping", name)
    else:
        logger.info("[%s] Phase 1 displacement: already exists", name)

    # Phase 2: Aggregation
    if _output_exists(config_dir, "per_iter_burst_analysis.json"):
        if force or not _output_exists(config_dir, "burst_aggregate.json"):
            logger.info("[%s] Phase 2: burst aggregate", name)
            run_phase2_burst(config_dir)

    if _output_exists(config_dir, "per_iter_displacement.json"):
        if force or not _output_exists(config_dir, "displacement_aggregate.json"):
            logger.info("[%s] Phase 2: displacement aggregate", name)
            run_phase2_displacement(config_dir)

    if _output_exists(config_dir, "per_iter_enclave_analysis.json"):
        if force or not _output_exists(config_dir, "enclave_aggregate.json"):
            logger.info("[%s] Phase 2: enclave aggregate", name)
            run_phase2_enclaves(config_dir)

    # Phase 2.5: Summary update
    logger.info("[%s] Phase 2: summary update", name)
    run_phase2_summary_update(config_dir)


def run_phase3(force: bool = False) -> None:
    """Run cross-config Phase 3 aggregation."""
    logger.info("Phase 3: cross-config aggregation")
    run_phase3_displacement_master(EXP2_DIR)
    run_phase3_enclave_master(EXP2_DIR)
    run_phase3_burst_master(EXP2_DIR)
    run_phase3_anova(EXP2_DIR)


# ---------------------------------------------------------------------------
# Phase A4: LaTeX factorial tables
# ---------------------------------------------------------------------------

def _read_summary(summary_path: Path) -> dict[str, dict[str, str]]:
    """Read summary.csv into measure_name -> {Mean, SD, ...}."""
    import csv
    result: dict[str, dict[str, str]] = {}
    if not summary_path.exists():
        return result
    with open(summary_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            result[row["Measure"]] = dict(row)
    return result


def _load_burst_aggregate(config_dir: Path) -> dict:
    """Load burst_aggregate.json, preferring config root (Phase 2 schema)."""
    root_path = config_dir / "burst_aggregate.json"
    dynamics_path = config_dir / "dynamics" / "burst_aggregate.json"
    if root_path.exists():
        with open(root_path) as f:
            return json.load(f)
    if dynamics_path.exists():
        with open(dynamics_path) as f:
            return json.load(f)
    return {}


def _fmt_mean_sd(mean_str: str, sd_str: str) -> str:
    """Format as 'mean (SD)' for table cells."""
    try:
        mean = float(mean_str)
        sd = float(sd_str)
        return f"{mean:.3f} ({sd:.3f})"
    except (ValueError, TypeError):
        return "---"


def _fmt_val(val, fmt: str = ".3f") -> str:
    """Format a single value."""
    if val is None:
        return "---"
    try:
        v = float(val)
        if v != v:  # NaN check
            return "---"
        return f"{v:{fmt}}"
    except (ValueError, TypeError):
        return "---"


def _fmt_fraction(numerator: float | None, denominator: float | None) -> str:
    """Format a fraction."""
    if numerator is None or denominator is None or denominator == 0:
        return "---"
    try:
        return f"{float(numerator) / float(denominator):.3f}"
    except (ValueError, TypeError, ZeroDivisionError):
        return "---"


def _fmt_slope_stars(slope, p_value) -> str:
    """Format escalation slope with significance stars."""
    if slope is None:
        return "---"
    try:
        v = float(slope)
        if v != v:  # NaN
            return "---"
        s = f"{v:.3f}"
    except (ValueError, TypeError):
        return "---"
    if p_value is not None:
        try:
            p = float(p_value)
            if p < 0.001:
                s += "***"
            elif p < 0.01:
                s += "**"
            elif p < 0.05:
                s += "*"
        except (ValueError, TypeError):
            pass
    return s


def generate_factorial_latex() -> str:
    """Generate Phase A4 factorial tables: 8 metrics x 3 rho levels = 24 tables."""
    all_tables: list[str] = []

    # Preload data for all configs
    data: dict[str, dict] = {}  # config_name -> {summary, burst_agg}
    for config_dir in discover_configs():
        name = config_dir.name
        data[name] = {
            "summary": _read_summary(config_dir / "summary.csv"),
            "burst_agg": _load_burst_aggregate(config_dir),
        }

    # Metric definitions
    metrics = [
        {
            "key": "norm_utility_mainstream",
            "label": "Norm. Mainstream Utility",
            "source": "summary",
            "formatter": lambda d, _b: _fmt_mean_sd(
                d.get("norm_utility_mainstream", {}).get("Mean", ""),
                d.get("norm_utility_mainstream", {}).get("SD", ""),
            ),
        },
        {
            "key": "avg_utility_gov_direct",
            "label": "Utility (Direct)",
            "source": "summary",
            "formatter": lambda d, _b: _fmt_mean_sd(
                d.get("avg_utility_gov_direct", {}).get("Mean", ""),
                d.get("avg_utility_gov_direct", {}).get("SD", ""),
            ),
        },
        {
            "key": "avg_utility_gov_coalition",
            "label": "Utility (Coalition)",
            "source": "summary",
            "formatter": lambda d, _b: _fmt_mean_sd(
                d.get("avg_utility_gov_coalition", {}).get("Mean", ""),
                d.get("avg_utility_gov_coalition", {}).get("SD", ""),
            ),
        },
        {
            "key": "avg_utility_gov_algorithmic",
            "label": "Utility (Algorithmic)",
            "source": "summary",
            "formatter": lambda d, _b: _fmt_mean_sd(
                d.get("avg_utility_gov_algorithmic", {}).get("Mean", ""),
                d.get("avg_utility_gov_algorithmic", {}).get("SD", ""),
            ),
        },
        {
            "key": "extremist_on_direct_frac",
            "label": "Extremist on Direct (frac.)",
            "source": "summary",
            "formatter": lambda d, _b: _compute_extremist_direct_frac(d),
        },
        {
            "key": "burst_rate",
            "label": "Burst Rate",
            "source": "burst",
            "formatter": lambda _d, b: _fmt_val(b.get("burst_rate")),
        },
        {
            "key": "burst_size_median",
            "label": "Median Burst Size",
            "source": "burst",
            "formatter": lambda _d, b: _fmt_val(
                b.get("burst_size_median", b.get("median_burst_size"))
            ),
        },
        {
            "key": "escalation_slope",
            "label": "Escalation Slope",
            "source": "burst",
            "formatter": lambda _d, b: _fmt_slope_stars(
                b.get("escalation_mean_slope"),
                b.get("escalation_p_value"),
            ),
        },
    ]

    for metric in metrics:
        for rho in RHO_VALUES:
            rho_str = f"{rho:.2f}".replace(".", "")
            rho_pct = int(rho * 100)

            lines = [
                r"\begin{table}[htbp]",
                r"\centering",
                rf"\begin{{tabular}}{{l{'c' * len(ALPHA_VALUES)}}}",
                r"\toprule",
                r" & " + " & ".join(f"$\\alpha={a}$" for a in ALPHA_VALUES) + r" \\",
                r"\midrule",
            ]

            for np_val in N_PLATFORMS:
                cells = [f"$N_p={np_val}$"]
                for alpha in ALPHA_VALUES:
                    cfg_name = f"exp2_np{np_val}_rho{rho_str}_alpha{alpha}"
                    cfg_data = data.get(cfg_name, {"summary": {}, "burst_agg": {}})
                    cell = metric["formatter"](cfg_data["summary"], cfg_data["burst_agg"])
                    cells.append(cell)
                lines.append(" & ".join(cells) + r" \\")

            lines.extend([
                r"\bottomrule",
                r"\end{tabular}",
                rf"\caption{{{metric['label']} ($\rho_e={rho_pct}\%$)}}",
                rf"\label{{tab:exp2_{metric['key']}_rho{rho_pct}}}",
                r"\end{table}",
                "",
            ])
            all_tables.append("\n".join(lines))

    return "\n\n".join(all_tables)


def _compute_extremist_direct_frac(summary: dict) -> str:
    """Compute fraction of extremists on direct platforms."""
    try:
        direct = float(summary.get("final_count_extremist_direct", {}).get("Mean", "0") or "0")
        coalition = float(summary.get("final_count_extremist_coalition", {}).get("Mean", "0") or "0")
        algorithmic = float(summary.get("final_count_extremist_algorithmic", {}).get("Mean", "0") or "0")
        total = direct + coalition + algorithmic
        if total == 0:
            return "---"
        return f"{direct / total:.3f}"
    except (ValueError, TypeError):
        return "---"


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase A analysis runner for exp2")
    parser.add_argument("--force", action="store_true", help="Recompute even if outputs exist")
    parser.add_argument("--skip-phase1", action="store_true", help="Skip Phase 1+2 per-config processing")
    parser.add_argument("--only-latex", action="store_true", help="Only generate LaTeX tables")
    args = parser.parse_args()

    if not EXP2_DIR.is_dir():
        logger.error("exp2 directory not found: %s", EXP2_DIR)
        sys.exit(1)

    configs = discover_configs()
    logger.info("Found %d configs", len(configs))

    t0 = time.time()

    if not args.only_latex:
        # Phase 1 + 2: per-config processing
        if not args.skip_phase1:
            for i, config_dir in enumerate(configs, 1):
                logger.info("=== Config %d/%d: %s ===", i, len(configs), config_dir.name)
                try:
                    process_config(config_dir, force=args.force)
                except Exception:
                    logger.exception("Failed processing %s", config_dir.name)
                gc.collect()

        # Phase 3: cross-config
        try:
            run_phase3(force=args.force)
        except Exception:
            logger.exception("Phase 3 failed")

    # Phase A4: LaTeX tables
    logger.info("Phase A4: generating factorial LaTeX tables")
    try:
        latex = generate_factorial_latex()
        output_path = EXP2_DIR / "exp2_factorial_tables.tex"
        with open(output_path, "w") as f:
            f.write(latex)
        logger.info("Wrote %s (%d bytes)", output_path, len(latex))
    except Exception:
        logger.exception("LaTeX generation failed")

    elapsed = time.time() - t0
    logger.info("Done in %.1f seconds", elapsed)


if __name__ == "__main__":
    main()
