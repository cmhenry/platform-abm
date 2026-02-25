"""Manuscript-ready LaTeX table formatters."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def _read_summary(summary_path: Path) -> dict[str, dict[str, str]]:
    """Read a summary.csv into measure_name -> {Mean, SD, ...}."""
    result: dict[str, dict[str, str]] = {}
    if not summary_path.exists():
        return result
    with open(summary_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            result[row["Measure"]] = dict(row)
    return result


def _fmt(mean_str: str, sd_str: str) -> str:
    """Format as 'mean (SD)' for table cells."""
    try:
        mean = float(mean_str)
        sd = float(sd_str)
        return f"{mean:.3f} ({sd:.3f})"
    except (ValueError, TypeError):
        return "---"


def format_exp1_table(experiment_dir: Path) -> str:
    """Experiment 1: rows=measures, cols=configs, cells=mean(SD).

    6 columns: Direct(9), Coalition(9), Algorithmic(9), Mixed(3), Mixed(9), Mixed(27).
    Includes mixed-specific sub-rows for per-institution utilities and proportions.
    """
    all_configs = [
        "exp1_direct_np9",
        "exp1_coalition_np9",
        "exp1_algorithmic_np9",
        "exp1_mixed_np3",
        "exp1_mixed_np9",
        "exp1_mixed_np27",
    ]
    col_labels = [
        "Direct", "Coalition", "Algorithmic",
        "Mixed (3)", "Mixed (9)", "Mixed (27)",
    ]
    mixed_configs = {"exp1_mixed_np3", "exp1_mixed_np9", "exp1_mixed_np27"}

    # Core measures (all configs)
    measures = [
        ("avg_utility", "Avg. Utility"),
        ("total_relocations", "Total Relocations"),
        ("avg_relocations_per_community", "Avg. Reloc./Community"),
        ("settling_time_90pct", "Settling Time (90\\%)"),
    ]
    # Mixed-only sub-rows (per-institution breakdown)
    mixed_measures = [
        ("avg_utility_gov_direct", "\\quad Utility (Direct plats.)"),
        ("avg_utility_gov_coalition", "\\quad Utility (Coalition plats.)"),
        ("avg_utility_gov_algorithmic", "\\quad Utility (Algo. plats.)"),
        ("final_proportion_direct", "\\quad Prop. on Direct"),
        ("final_proportion_coalition", "\\quad Prop. on Coalition"),
        ("final_proportion_algorithmic", "\\quad Prop. on Algorithmic"),
    ]

    # Load summaries
    summaries: dict[str, dict[str, dict[str, str]]] = {}
    for cfg_name in all_configs:
        summary_path = experiment_dir / cfg_name / "summary.csv"
        summaries[cfg_name] = _read_summary(summary_path)

    # Build table
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        rf"\begin{{tabular}}{{l{'c' * len(all_configs)}}}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Homogeneous ($N_p=9$)} & \multicolumn{3}{c}{Mixed Institution} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
        "Measure & " + " & ".join(col_labels) + r" \\",
        r"\midrule",
    ]

    for measure_key, label in measures:
        cells = [label]
        for cfg_name in all_configs:
            s = summaries[cfg_name]
            m = s.get(measure_key, {})
            cells.append(_fmt(m.get("Mean", ""), m.get("SD", "")))
        lines.append(" & ".join(cells) + r" \\")

    # Mixed-specific sub-rows (blank cells for homogeneous configs)
    lines.append(r"\addlinespace")
    lines.append(r"\multicolumn{7}{l}{\textit{Mixed institution breakdown}} \\")
    for measure_key, label in mixed_measures:
        cells = [label]
        for cfg_name in all_configs:
            if cfg_name in mixed_configs:
                s = summaries[cfg_name]
                m = s.get(measure_key, {})
                cells.append(_fmt(m.get("Mean", ""), m.get("SD", "")))
            else:
                cells.append("---")
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Experiment 1: Institutional Comparison}",
        r"\label{tab:exp1}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def format_exp2_tables(experiment_dir: Path) -> str:
    """Experiment 2: per-outcome 3x3 grids (N_p x alpha) for each rho level."""
    n_platforms_values = [3, 6, 9]
    rho_values = [0.05, 0.10, 0.20]
    alpha_values = [2, 5, 10]

    outcomes = [
        ("avg_utility", "Avg. Utility"),
        ("norm_utility_mainstream", "Norm. Mainstream Utility"),
        ("norm_utility_extremist", "Norm. Extremist Utility"),
        ("total_relocations", "Total Relocations"),
        ("settling_time_90pct", "Settling Time (90\\%)"),
    ]

    all_tables: list[str] = []

    for outcome_key, outcome_label in outcomes:
        for rho in rho_values:
            rho_str = f"{rho:.2f}".replace(".", "")
            lines = [
                r"\begin{table}[htbp]",
                r"\centering",
                rf"\begin{{tabular}}{{l{'c' * len(alpha_values)}}}",
                r"\toprule",
                r" & " + " & ".join(f"$\\alpha={a}$" for a in alpha_values) + r" \\",
                r"\midrule",
            ]

            for np_val in n_platforms_values:
                cells = [f"$N_p={np_val}$"]
                for alpha in alpha_values:
                    cfg_name = f"exp2_np{np_val}_rho{rho_str}_alpha{alpha}"
                    summary = _read_summary(experiment_dir / cfg_name / "summary.csv")
                    m = summary.get(outcome_key, {})
                    cells.append(_fmt(m.get("Mean", ""), m.get("SD", "")))
                lines.append(" & ".join(cells) + r" \\")

            rho_pct = int(rho * 100)
            lines.extend([
                r"\bottomrule",
                r"\end{tabular}",
                rf"\caption{{{outcome_label} ($\rho_e={rho_pct}\%$)}}",
                rf"\label{{tab:exp2_{outcome_key}_rho{rho_pct}}}",
                r"\end{table}",
                "",
            ])
            all_tables.append("\n".join(lines))

    return "\n\n".join(all_tables)


def format_oat_table(experiment_dir: Path, baseline_dir: Path) -> str:
    """OAT sensitivity table: rows=param x value, cols=outcomes, cells=% change."""
    baseline_summary = _read_summary(baseline_dir / "summary.csv")

    outcomes = [
        "avg_utility",
        "norm_utility_mainstream",
        "norm_utility_extremist",
        "total_relocations",
        "settling_time_90pct",
    ]
    outcome_labels = [
        "Avg. Util.",
        "Norm. Main.",
        "Norm. Ext.",
        "Reloc.",
        "Settle",
    ]

    # OAT configs organized by parameter
    oat_params = [
        ("$N_c$", [("50", "oat_nc50"), ("200", "oat_nc200")]),
        ("$N_p$", [("3", "oat_np3"), ("6", "oat_np6")]),
        ("$p$", [("5", "oat_pspace5"), ("20", "oat_pspace20")]),
        ("$\\rho_e$", [("5\\%", "oat_rho005"), ("20\\%", "oat_rho020")]),
        ("$\\alpha$", [("2", "oat_alpha2"), ("10", "oat_alpha10")]),
    ]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        rf"\begin{{tabular}}{{ll{'r' * len(outcomes)}}}",
        r"\toprule",
        r"Parameter & Value & " + " & ".join(outcome_labels) + r" \\",
        r"\midrule",
    ]

    for param_label, values in oat_params:
        for val_label, cfg_name in values:
            summary_path = experiment_dir / cfg_name / "summary.csv"
            test_summary = _read_summary(summary_path)

            cells = [param_label, val_label]
            for outcome in outcomes:
                baseline_mean = float(baseline_summary.get(outcome, {}).get("Mean", "0"))
                test_mean = float(test_summary.get(outcome, {}).get("Mean", "0"))

                if abs(baseline_mean) < 1e-10:
                    cells.append("---")
                else:
                    pct = ((test_mean - baseline_mean) / abs(baseline_mean)) * 100
                    cells.append(f"{pct:+.1f}\\%")

            lines.append(" & ".join(cells) + r" \\")

        # Add separator between parameter groups (except last)
        if param_label != oat_params[-1][0]:
            lines.append(r"\addlinespace")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{OAT Sensitivity Analysis (\% change from baseline)}",
        r"\label{tab:oat}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def format_interaction_table(
    rows_data: list[dict[str, Any]],
    row_param: str,
    col_param: str,
    outcome: str,
    caption: str,
    label: str,
) -> str:
    """Generic interaction table: rows=row_param values, cols=col_param values."""
    # Extract unique sorted values
    row_vals = sorted(set(r[row_param] for r in rows_data))
    col_vals = sorted(set(r[col_param] for r in rows_data))

    # Index data
    lookup: dict[tuple, dict] = {}
    for r in rows_data:
        lookup[(r[row_param], r[col_param])] = r

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\begin{{tabular}}{{l{'c' * len(col_vals)}}}",
        r"\toprule",
        f" & " + " & ".join(str(v) for v in col_vals) + r" \\",
        r"\midrule",
    ]

    for rv in row_vals:
        cells = [str(rv)]
        for cv in col_vals:
            data = lookup.get((rv, cv), {})
            mean = data.get(f"{outcome}_mean", "")
            sd = data.get(f"{outcome}_sd", "")
            if mean and sd:
                cells.append(_fmt(str(mean), str(sd)))
            else:
                cells.append("---")
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ])

    return "\n".join(lines)
