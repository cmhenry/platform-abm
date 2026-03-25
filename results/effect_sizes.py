"""Compute effect sizes for the exp2 factorial analysis.

Produces results/effect_sizes.json with:
  1. Eta-squared for each ANOVA term (N_p, alpha, interaction) per rho level
  2. Cohen's d for key pairwise comparisons at alpha=10
  3. Bootstrap CI for diversification premium difference-in-differences

Usage:
    python results/effect_sizes.py
    (run from project root)

Inputs:
    results/exp2/exp2_anova_results.json
    results/exp2/exp2_master_summary.csv
    results/exp2/<config_name>/raw.csv   (for bootstrap CI)

Output:
    results/effect_sizes.json
    Summary table printed to stdout.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(v: str | float | None) -> float | None:
    """Convert to float or None."""
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _pooled_sd(n1: int, sd1: float, n2: int, sd2: float) -> float:
    """Pooled standard deviation for two groups."""
    num = (n1 - 1) * sd1 ** 2 + (n2 - 1) * sd2 ** 2
    denom = n1 + n2 - 2
    if denom <= 0:
        return float("nan")
    return math.sqrt(num / denom)


def _cohens_d(mean1: float, mean2: float, pooled_sd: float) -> float:
    """Cohen's d effect size."""
    if pooled_sd == 0 or math.isnan(pooled_sd):
        return float("nan")
    return (mean1 - mean2) / pooled_sd


def _bootstrap_ci(
    data1: list[float],
    data2: list[float],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap CI for difference in means (data1 - data2).

    Returns (observed_diff, ci_lower, ci_upper).
    Uses a simple module-level RNG to avoid numpy dependency for seeding.
    """
    import random
    rng = random.Random(seed)

    obs_diff = sum(data1) / len(data1) - sum(data2) / len(data2)

    boot_diffs: list[float] = []
    for _ in range(n_boot):
        s1 = [rng.choice(data1) for _ in range(len(data1))]
        s2 = [rng.choice(data2) for _ in range(len(data2))]
        boot_diffs.append(sum(s1) / len(s1) - sum(s2) / len(s2))

    boot_diffs.sort()
    lo_idx = int(math.floor(alpha / 2.0 * n_boot))
    hi_idx = int(math.ceil((1 - alpha / 2.0) * n_boot)) - 1
    return obs_diff, boot_diffs[lo_idx], boot_diffs[hi_idx]


# ---------------------------------------------------------------------------
# 1. Eta-squared from ANOVA results
# ---------------------------------------------------------------------------

def compute_eta_squared(anova_path: Path) -> dict:
    """Compute eta² for each ANOVA term per rho level.

    ANOVA JSON structure (verified):
      { "rho_005": { "N_p_F", "N_p_p", "alpha_F", "alpha_p",
                     "interaction_F", "interaction_p" }, ... }

    Uses F-to-eta² conversion:
      eta² = (F * df_effect) / (F * df_effect + df_error)

    Factorial design: N_p (3 levels), alpha (3 levels) → df_N_p=2, df_alpha=2,
    df_interaction=4. With 27 configs × 200 iterations = 5400 observations,
    df_error = 5400 - 9 = 5391.
    """
    with open(anova_path) as f:
        anova = json.load(f)

    # Degrees of freedom
    df_N_p        = 2   # 3 levels - 1
    df_alpha      = 2   # 3 levels - 1
    df_interaction = 4  # 2 * 2
    n_total        = 27 * 200  # 27 configs x 200 iterations
    df_error       = n_total - (df_N_p + df_alpha + df_interaction + 1)

    results: dict = {}
    for rho_key, stats in anova.items():
        r: dict = {}
        for term, df_eff in [("N_p", df_N_p), ("alpha", df_alpha),
                              ("interaction", df_interaction)]:
            f_key = f"{term}_F"
            F = stats.get(f_key)
            if F is None:
                r[term] = {"eta_squared": None, "F": None, "df_effect": df_eff,
                           "df_error": df_error}
                continue
            num = F * df_eff
            eta2 = num / (num + df_error)
            r[term] = {
                "eta_squared": round(eta2, 6),
                "F": round(F, 4),
                "df_effect": df_eff,
                "df_error": df_error,
                "p_value": stats.get(f"{term}_p"),
            }
        results[rho_key] = r
    return results


# ---------------------------------------------------------------------------
# 2. Cohen's d for pairwise comparisons
# ---------------------------------------------------------------------------

def compute_cohens_d(master_path: Path) -> dict:
    """Compute Cohen's d for key pairwise governance comparisons at alpha=10.

    Uses exp2_master_summary.csv which has per-config aggregated means and SDs.
    The measure used is avg_utility_mainstream.
    """
    with open(master_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Index rows by (n_platforms, alpha, governance-type is always 'mixed' in exp2)
    # We need per-governance-type utility: avg_utility_mainstream_direct,
    # avg_utility_mainstream_coalition, avg_utility_mainstream_algorithmic
    # These are aggregate means across 200 iterations per config.
    # For Cohen's d we need SD too. Use the raw.csv per-config for per-iteration data.
    # However, master_summary only has means. We'll use the per-config raw.csv files.

    # Collect per-iteration mainstream utility from raw.csv for alpha=10, rho=0.15
    # (canonical comparison slice). Governance comparison requires per-governance
    # utility per iteration. The raw.csv has avg_utility_mainstream (overall) but
    # not per-governance breakdown.
    #
    # Fall back to master_summary means + approximate SD from master_summary SD
    # for the overall avg_utility_mainstream measure, broken down by n_platforms.
    # For the governance-type comparison, we use the per-config
    # avg_utility_mainstream_{direct,coalition,algorithmic} columns as point
    # estimates (these are means over 200 iterations; SD not directly available).
    # We report the raw difference and note the limitation.

    results: dict = {}

    # Filter to alpha=10, rho_extremist=0.15 slice
    alpha10_rho15 = [
        r for r in rows
        if abs(float(r["alpha"]) - 10.0) < 0.01
        and abs(float(r["rho_extremist"]) - 0.15) < 0.001
    ]

    if not alpha10_rho15:
        return {"error": "No rows found for alpha=10, rho=0.15 in master_summary.csv"}

    # We expect one config per n_platforms value (np=3, 9, 27), all institution=mixed
    config_by_np: dict[int, dict] = {}
    for r in alpha10_rho15:
        np_val = int(r["n_platforms"])
        config_by_np[np_val] = r

    # ----- Comparison 1: Direct vs Coalition mainstream utility at alpha=10 -----
    # Average over np values (or report per np)
    comp1_rows = []
    for np_val, r in sorted(config_by_np.items()):
        direct_mean   = _safe_float(r.get("avg_utility_mainstream_direct"))
        coalition_mean = _safe_float(r.get("avg_utility_mainstream_coalition"))
        if direct_mean is None or coalition_mean is None:
            continue
        comp1_rows.append({
            "n_platforms": np_val,
            "direct_mean": direct_mean,
            "coalition_mean": coalition_mean,
            "raw_diff": direct_mean - coalition_mean,
            "note": "Cohen's d unavailable without per-iteration per-governance SD; "
                    "reporting raw mean difference",
        })

    # ----- Comparison 2: Direct vs Algorithmic mainstream utility at alpha=10 -----
    comp2_rows = []
    for np_val, r in sorted(config_by_np.items()):
        direct_mean   = _safe_float(r.get("avg_utility_mainstream_direct"))
        algo_mean     = _safe_float(r.get("avg_utility_mainstream_algorithmic"))
        if direct_mean is None or algo_mean is None:
            continue
        comp2_rows.append({
            "n_platforms": np_val,
            "direct_mean": direct_mean,
            "algorithmic_mean": algo_mean,
            "raw_diff": direct_mean - algo_mean,
            "note": "Cohen's d unavailable without per-iteration per-governance SD; "
                    "reporting raw mean difference",
        })

    # ----- Comparison 3: Low alpha (alpha=2) vs High alpha (alpha=10) in direct governance -----
    # Need configs at rho=0.15, direct governance (institution=mixed gives mixed),
    # compare alpha=2 vs alpha=10 for avg_utility_mainstream_direct
    alpha2_rho15 = [
        r for r in rows
        if abs(float(r["alpha"]) - 2.0) < 0.01
        and abs(float(r["rho_extremist"]) - 0.15) < 0.001
    ]

    comp3_rows = []
    for r2 in alpha2_rho15:
        np_val = int(r2["n_platforms"])
        r10 = config_by_np.get(np_val)
        if r10 is None:
            continue
        direct_low  = _safe_float(r2.get("avg_utility_mainstream_direct"))
        direct_high = _safe_float(r10.get("avg_utility_mainstream_direct"))
        if direct_low is None or direct_high is None:
            continue
        comp3_rows.append({
            "n_platforms": np_val,
            "alpha_low_mean": direct_low,
            "alpha_high_mean": direct_high,
            "raw_diff_low_minus_high": direct_low - direct_high,
            "note": "Direct governance mainstream utility: alpha=2 minus alpha=10",
        })

    results["direct_vs_coalition_alpha10"] = comp1_rows
    results["direct_vs_algorithmic_alpha10"] = comp2_rows
    results["low_alpha_vs_high_alpha_direct"] = comp3_rows
    results["methodology_note"] = (
        "Per-governance per-iteration SD not available in master_summary.csv. "
        "Cohen's d requires raw per-iteration per-governance utility; "
        "reporting raw mean differences. "
        "See bootstrap_diversification_premium for a full distributional comparison."
    )
    return results


# ---------------------------------------------------------------------------
# 3. Bootstrap CI for diversification premium difference-in-differences
# ---------------------------------------------------------------------------

def compute_bootstrap_diversification(
    results_dir: Path,
    n_boot: int = 10_000,
) -> dict:
    """Bootstrap CI for diversification premium difference-in-differences.

    Diversification premium = utility(np=27) - utility(np=3) at alpha=10.
    Compares this premium between governance types (direct vs mixed avg).

    Reads per-iteration avg_utility_mainstream from raw.csv for each config.
    """
    def read_mainstream_utility(config_dir: Path) -> list[float]:
        raw_path = config_dir / "raw.csv"
        if not raw_path.exists():
            return []
        vals: list[float] = []
        with open(raw_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                v = _safe_float(row.get("avg_utility_mainstream"))
                if v is not None:
                    vals.append(v)
        return vals

    # Configs at rho=0.15, alpha=10
    np27_dir = results_dir / "exp2_np27_rho015_alpha10"
    np3_dir  = results_dir / "exp2_np3_rho015_alpha10"
    np9_dir  = results_dir / "exp2_np9_rho015_alpha10"

    np27_utils = read_mainstream_utility(np27_dir)
    np3_utils  = read_mainstream_utility(np3_dir)
    np9_utils  = read_mainstream_utility(np9_dir)

    if not np27_utils or not np3_utils:
        return {
            "error": "Missing raw.csv for np=27 or np=3 at rho=0.15, alpha=10",
            "np27_n": len(np27_utils),
            "np3_n": len(np3_utils),
        }

    # Diversification premium for each iteration pair (matched by index if same length)
    # Use simpler approach: bootstrap differences from pooled premium distribution
    # Premium_i = np27_utils[i] - np3_utils[i] (if lengths match)
    # If lengths differ, compute separate bootstrap CIs and then difference of means.

    results: dict = {}

    # Premium = mean(np27) - mean(np3)
    mean_np27 = sum(np27_utils) / len(np27_utils)
    mean_np3  = sum(np3_utils)  / len(np3_utils)
    obs_premium = mean_np27 - mean_np3

    # Bootstrap CI for the premium itself (difference of means)
    obs_diff, ci_lo, ci_hi = _bootstrap_ci(np27_utils, np3_utils, n_boot=n_boot)

    results["diversification_premium"] = {
        "description": "mean(utility_mainstream, np=27) - mean(utility_mainstream, np=3) at alpha=10, rho=0.15",
        "observed_diff": round(obs_diff, 6),
        "ci_lower_95": round(ci_lo, 6),
        "ci_upper_95": round(ci_hi, 6),
        "n_bootstrap": n_boot,
        "n_np27": len(np27_utils),
        "n_np3": len(np3_utils),
        "mean_np27": round(mean_np27, 6),
        "mean_np3": round(mean_np3, 6),
    }

    # Also compute premium at alpha=5 for comparison
    np27_alpha5_dir = results_dir / "exp2_np27_rho015_alpha5"
    np3_alpha5_dir  = results_dir / "exp2_np3_rho015_alpha5"
    np27_alpha5 = read_mainstream_utility(np27_alpha5_dir)
    np3_alpha5  = read_mainstream_utility(np3_alpha5_dir)

    if np27_alpha5 and np3_alpha5:
        obs_diff_a5, ci_lo_a5, ci_hi_a5 = _bootstrap_ci(
            np27_alpha5, np3_alpha5, n_boot=n_boot
        )
        results["diversification_premium_alpha5"] = {
            "description": "mean(utility_mainstream, np=27) - mean(utility_mainstream, np=3) at alpha=5, rho=0.15",
            "observed_diff": round(obs_diff_a5, 6),
            "ci_lower_95": round(ci_lo_a5, 6),
            "ci_upper_95": round(ci_hi_a5, 6),
            "n_bootstrap": n_boot,
        }

        # Difference-in-differences: premium(alpha=10) - premium(alpha=5)
        # Bootstrap DID using paired approach: for each boot sample compute both
        # premia then difference
        import random
        rng = random.Random(42)
        did_boot: list[float] = []
        for _ in range(n_boot):
            s27_10 = [rng.choice(np27_utils) for _ in range(len(np27_utils))]
            s3_10  = [rng.choice(np3_utils)  for _ in range(len(np3_utils))]
            s27_5  = [rng.choice(np27_alpha5) for _ in range(len(np27_alpha5))]
            s3_5   = [rng.choice(np3_alpha5)  for _ in range(len(np3_alpha5))]
            prem10 = sum(s27_10) / len(s27_10) - sum(s3_10) / len(s3_10)
            prem5  = sum(s27_5)  / len(s27_5)  - sum(s3_5)  / len(s3_5)
            did_boot.append(prem10 - prem5)
        did_boot.sort()
        lo_idx = int(math.floor(0.025 * n_boot))
        hi_idx = int(math.ceil(0.975 * n_boot)) - 1
        obs_did = obs_diff - obs_diff_a5
        results["did_premium_alpha10_vs_alpha5"] = {
            "description": "DiD: (premium_alpha10) - (premium_alpha5), rho=0.15",
            "observed_did": round(obs_did, 6),
            "ci_lower_95": round(did_boot[lo_idx], 6),
            "ci_upper_95": round(did_boot[hi_idx], 6),
            "n_bootstrap": n_boot,
        }

    # np=9 premium for reference
    if np9_utils:
        mean_np9 = sum(np9_utils) / len(np9_utils)
        results["diversification_premium"]["mean_np9"] = round(mean_np9, 6)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    results_dir = Path("results/exp2")
    anova_path  = results_dir / "exp2_anova_results.json"
    master_path = results_dir / "exp2_master_summary.csv"
    output_path = Path("results/effect_sizes.json")

    # --- Check inputs ---
    for p in [anova_path, master_path]:
        if not p.exists():
            print(f"ERROR: required file not found: {p}", file=sys.stderr)
            sys.exit(1)

    output: dict = {}

    # 1. Eta-squared
    print("Computing eta-squared from ANOVA results...")
    eta2 = compute_eta_squared(anova_path)
    output["eta_squared"] = eta2

    # 2. Cohen's d pairwise
    print("Computing pairwise comparisons (Cohen's d / raw diffs)...")
    cohens = compute_cohens_d(master_path)
    output["pairwise_comparisons"] = cohens

    # 3. Bootstrap CI for diversification premium
    print("Computing bootstrap CI for diversification premium (10,000 samples)...")
    boot = compute_bootstrap_diversification(results_dir, n_boot=10_000)
    output["bootstrap_diversification"] = boot

    # Write JSON output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote effect sizes to {output_path}")

    # --- Print summary table ---
    print("\n" + "=" * 70)
    print("EFFECT SIZE SUMMARY")
    print("=" * 70)

    print("\n--- Eta-squared (ANOVA terms) ---")
    print(f"{'Rho':<10} {'Term':<15} {'F':>10} {'eta²':>10} {'p':>12}")
    print("-" * 60)
    for rho_key, terms in eta2.items():
        for term, vals in terms.items():
            if isinstance(vals, dict) and vals.get("eta_squared") is not None:
                p_val = vals.get("p_value") or float("nan")
                print(
                    f"{rho_key:<10} {term:<15} "
                    f"{vals['F']:>10.2f} {vals['eta_squared']:>10.6f} "
                    f"{p_val:>12.2e}"
                )

    print("\n--- Pairwise Comparisons (alpha=10, rho=0.15) ---")
    for key, comp_list in [
        ("direct_vs_coalition_alpha10",    output["pairwise_comparisons"].get("direct_vs_coalition_alpha10", [])),
        ("direct_vs_algorithmic_alpha10",  output["pairwise_comparisons"].get("direct_vs_algorithmic_alpha10", [])),
        ("low_alpha_vs_high_alpha_direct", output["pairwise_comparisons"].get("low_alpha_vs_high_alpha_direct", [])),
    ]:
        if not isinstance(comp_list, list):
            continue
        print(f"\n  {key}:")
        for row in comp_list:
            np_val = row.get("n_platforms", "?")
            diff   = row.get("raw_diff") or row.get("raw_diff_low_minus_high")
            if diff is not None:
                print(f"    np={np_val}: raw diff = {diff:.4f}")

    print("\n--- Diversification Premium (np=27 - np=3, alpha=10, rho=0.15) ---")
    dp = output.get("bootstrap_diversification", {}).get("diversification_premium", {})
    if isinstance(dp, dict) and "observed_diff" in dp:
        print(
            f"  Premium = {dp['observed_diff']:.4f}  "
            f"95% CI [{dp['ci_lower_95']:.4f}, {dp['ci_upper_95']:.4f}]  "
            f"(n={dp['n_bootstrap']:,} bootstrap samples)"
        )

    did = output.get("bootstrap_diversification", {}).get("did_premium_alpha10_vs_alpha5", {})
    if isinstance(did, dict) and "observed_did" in did:
        print(
            f"  DiD (alpha10 vs alpha5) = {did['observed_did']:.4f}  "
            f"95% CI [{did['ci_lower_95']:.4f}, {did['ci_upper_95']:.4f}]"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()
