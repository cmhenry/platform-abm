#!/usr/bin/env python3
"""
visualize_exp2b.py

Figures for Experiment 2b: mixed ideologue/griefer populations.
Varies griefer fraction f_g ∈ {0, 0.25, 0.50, 0.75, 1.0} at
fixed N_p=9, rho_e=0.10.  exp2 alpha=2 / alpha=10 configs serve
as the pure-type endpoints (f_g=0, f_g=1).

Produces:
    results/figures/exp2b_governance.pdf
    results/figures/exp2b_dynamics.pdf

Usage (from project root):
    python3 results/visualize_exp2b.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

EXP2_SUMMARY   = "results/exp2/summary.csv"
EXP2B_SUMMARY  = "results/exp2b/exp2b_master_summary.csv"
EXP2B_DIR      = "results/exp2b"
FIG_DIR        = "results/figures"

os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------

GOV_COLORS = {
    "Direct":      "#e41a1c",
    "Coalition":   "#4daf4a",
    "Algorithmic": "#377eb8",
}

plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         150,
})

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

exp2  = pd.read_csv(EXP2_SUMMARY)
exp2b = pd.read_csv(EXP2B_SUMMARY)

def load_json(config_name, filename):
    path = os.path.join(EXP2B_DIR, config_name, filename)
    with open(path) as f:
        return json.load(f)

# --- exp2 baselines (N_p=9, rho=0.10) ---
row_ideologue = exp2[exp2["config_name"] == "exp2_np9_rho010_alpha2"].iloc[0]
row_griefer   = exp2[exp2["config_name"] == "exp2_np9_rho010_alpha10"].iloc[0]

# --- exp2b intermediate configs ---
fg_map = {
    "exp2b_fg025": 0.25,
    "exp2b_fg050": 0.50,
    "exp2b_fg075": 0.75,
}

# ---------------------------------------------------------------------------
# Governance utility data frame  (5-point gradient)
# ---------------------------------------------------------------------------

gov_rows = []

# Pure ideologue endpoint (f_g = 0)
gov_rows.append({
    "f_g":    0.00,
    "source": "exp2 baseline",
    "Direct":      row_ideologue["avg_utility_mainstream_direct"],
    "Coalition":   row_ideologue["avg_utility_mainstream_coalition"],
    "Algorithmic": row_ideologue["avg_utility_mainstream_algorithmic"],
})

# exp2b intermediate points
for cfg, fg in fg_map.items():
    row = exp2b[exp2b["config_name"] == cfg].iloc[0]
    gov_rows.append({
        "f_g":    fg,
        "source": "exp2b",
        "Direct":      row["avg_utility_mainstream_direct"],
        "Coalition":   row["avg_utility_mainstream_coalition"],
        "Algorithmic": row["avg_utility_mainstream_algorithmic"],
    })

# Pure griefer endpoint (f_g = 1)
gov_rows.append({
    "f_g":    1.00,
    "source": "exp2 baseline",
    "Direct":      row_griefer["avg_utility_mainstream_direct"],
    "Coalition":   row_griefer["avg_utility_mainstream_coalition"],
    "Algorithmic": row_griefer["avg_utility_mainstream_algorithmic"],
})

gov_df = pd.DataFrame(gov_rows).sort_values("f_g").reset_index(drop=True)

# Welfare loss relative to f_g=0
baseline = gov_df[gov_df["f_g"] == 0.0].iloc[0]
for col in ["Direct", "Coalition", "Algorithmic"]:
    gov_df[f"{col}_loss"] = gov_df[col] - baseline[col]

# ---------------------------------------------------------------------------
# Dynamics data  (exp2b intermediate points only)
# ---------------------------------------------------------------------------

dyn_rows = []
for cfg, fg in fg_map.items():
    burst   = load_json(cfg, "burst_aggregate.json")
    enclave = load_json(cfg, "enclave_aggregate.json")
    dyn_rows.append({
        "f_g":                fg,
        "burst_size_median":  burst["burst_size_median"],
        "burst_rate":         burst["burst_rate"],
        "escalation_slope":   burst["escalation_mean_slope"],
        "escalation_p":       burst["escalation_p_value"],
        "enclave_homogeneity":     enclave["mean_homogeneity"],
        "enclave_settling_mean":   enclave["settling_step_mean"],
        "enclave_settling_sd":     enclave["settling_step_sd"],
    })

dyn_df = pd.DataFrame(dyn_rows).sort_values("f_g").reset_index(drop=True)

# ===========================================================================
# Figure 1: exp2b_governance.pdf
#
# Left panel:  mainstream utility by governance type across f_g (5 points)
# Right panel: welfare loss relative to f_g=0 baseline
# ===========================================================================

print("Figure 1: exp2b_governance.pdf ...")

fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False)

for gov, color in GOV_COLORS.items():
    xs = gov_df["f_g"].values
    ys = gov_df[gov].values
    is_baseline = gov_df["source"] == "exp2 baseline"

    axes[0].plot(xs, ys, color=color, linewidth=1.5, label=gov, zorder=2)
    # Filled circle = exp2 baseline; open diamond = exp2b intermediate
    axes[0].scatter(
        xs[is_baseline], ys[is_baseline],
        color=color, marker="o", s=55, zorder=3, label="_nolegend_"
    )
    axes[0].scatter(
        xs[~is_baseline], ys[~is_baseline],
        facecolors="white", edgecolors=color,
        marker="D", s=50, linewidths=1.5, zorder=3, label="_nolegend_"
    )

axes[0].set_xlabel("$f_g$ (griefer fraction of extremist population)")
axes[0].set_ylabel("Mainstream utility")
axes[0].set_title("Mainstream welfare by governance type")
axes[0].set_xticks([0, 0.25, 0.50, 0.75, 1.0])
axes[0].set_xticklabels(
    ["0\n(pure\nideologue)", "0.25", "0.50", "0.75", "1.0\n(pure\ngriefer)"],
    fontsize=9
)
axes[0].legend(title="Governance", fontsize=9, title_fontsize=9)
# Annotate the large direct-platform drop
axes[0].annotate(
    "Direct: −2.03 total",
    xy=(1.0, row_griefer["avg_utility_mainstream_direct"]),
    xytext=(0.6, row_griefer["avg_utility_mainstream_direct"] - 0.35),
    arrowprops=dict(arrowstyle="->", color="#e41a1c", lw=1.0),
    color="#e41a1c", fontsize=9,
)

# Right panel: welfare loss
for gov, color in GOV_COLORS.items():
    xs   = gov_df["f_g"].values
    loss = gov_df[f"{gov}_loss"].values
    is_baseline = gov_df["source"] == "exp2 baseline"

    axes[1].plot(xs, loss, color=color, linewidth=1.5, label=gov, zorder=2)
    axes[1].scatter(xs[is_baseline], loss[is_baseline],
                    color=color, marker="o", s=55, zorder=3)
    axes[1].scatter(xs[~is_baseline], loss[~is_baseline],
                    facecolors="white", edgecolors=color,
                    marker="D", s=50, linewidths=1.5, zorder=3)

axes[1].axhline(0, color="grey", linewidth=0.6, linestyle="--")
axes[1].set_xlabel("$f_g$ (griefer fraction)")
axes[1].set_ylabel("Welfare loss relative to pure-ideologue baseline")
axes[1].set_title("Differential sensitivity by governance type")
axes[1].set_xticks([0, 0.25, 0.50, 0.75, 1.0])
axes[1].set_xticklabels(["0", "0.25", "0.50", "0.75", "1.0"], fontsize=9)
axes[1].legend(title="Governance", fontsize=9, title_fontsize=9)

# Custom legend entries for point shapes
from matplotlib.lines import Line2D
shape_legend = [
    Line2D([0], [0], marker="o", color="grey", linestyle="None",
           markersize=7, label="exp2 baseline (pure type)"),
    Line2D([0], [0], marker="D", color="grey", linestyle="None",
           markerfacecolor="white", markeredgewidth=1.5,
           markersize=7, label="exp2b (mixed)"),
]
axes[0].legend(
    handles=list(axes[0].get_legend_handles_labels()[0][:3]) + shape_legend,
    fontsize=8, title_fontsize=8,
    labels=list(GOV_COLORS.keys()) + [h.get_label() for h in shape_legend],
    title="",
)

fig.tight_layout(pad=2.0)
out = os.path.join(FIG_DIR, "exp2b_governance.pdf")
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {out}")

# ===========================================================================
# Figure 2: exp2b_dynamics.pdf
#
# Left panel:  burst size (bars) + escalation slope (line, right axis)
# Right panel: enclave settling time mean ± 1 SD (line+ribbon) +
#              enclave homogeneity (dashed, right axis)
# ===========================================================================

print("Figure 2: exp2b_dynamics.pdf ...")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

fg_vals  = dyn_df["f_g"].values
bar_w    = 0.12

# ── Left: burst dynamics ────────────────────────────────────────────────────
ax_b  = axes[0]
ax_b2 = ax_b.twinx()

bars = ax_b.bar(fg_vals, dyn_df["burst_size_median"], width=bar_w,
                color="#e41a1c", alpha=0.75, label="Burst size (median)", zorder=2)

# Escalation slope on right axis — scale to make dual-axis readable
ax_b2.plot(fg_vals, dyn_df["escalation_slope"], color="#377eb8",
           linewidth=1.5, marker="^", markersize=7,
           label="Escalation slope", zorder=3)
ax_b2.set_ylabel("Escalation slope", color="#377eb8")
ax_b2.tick_params(axis="y", labelcolor="#377eb8")
ax_b2.set_ylim(bottom=-0.05)

# Asterisks for significant escalation
for _, row in dyn_df.iterrows():
    if row["escalation_p"] < 0.05:
        y_ann = row["burst_size_median"] + 1.5
        ax_b.text(row["f_g"], y_ann, "*", ha="center", va="bottom",
                  fontsize=13, color="black")

ax_b.set_xlabel("$f_g$ (griefer fraction)")
ax_b.set_ylabel("Burst size (median)", color="#e41a1c")
ax_b.tick_params(axis="y", labelcolor="#e41a1c")
ax_b.set_xticks(fg_vals)
ax_b.set_xticklabels(["0.25", "0.50", "0.75"])
ax_b.set_ylim(0, 40)
ax_b.set_title("Burst dynamics across griefer fraction")

# Combined legend
from matplotlib.patches import Patch
legend_elems = [
    Patch(facecolor="#e41a1c", alpha=0.75, label="Burst size (median, bars)"),
    Line2D([0], [0], color="#377eb8", marker="^", linewidth=1.5,
           label="Escalation slope (line)"),
    Line2D([0], [0], color="none", label="* p < 0.05"),
]
ax_b.legend(handles=legend_elems, fontsize=8, loc="upper right")
ax_b.text(0.02, 0.97, "* = significant escalation (p < 0.05)",
          transform=ax_b.transAxes, fontsize=8, va="top", color="grey")

# ── Right: enclave dynamics ──────────────────────────────────────────────────
ax_e  = axes[1]
ax_e2 = ax_e.twinx()

settle_mean = dyn_df["enclave_settling_mean"].values
settle_sd   = dyn_df["enclave_settling_sd"].values

ax_e.fill_between(fg_vals,
                  settle_mean - settle_sd,
                  settle_mean + settle_sd,
                  color="#4daf4a", alpha=0.20, label="_nolegend_")
ax_e.plot(fg_vals, settle_mean, color="#4daf4a",
          linewidth=1.5, marker="o", markersize=7, label="Settling step (mean ± 1 SD)")
ax_e.set_ylabel("Coalition enclave settling step", color="#4daf4a")
ax_e.tick_params(axis="y", labelcolor="#4daf4a")
ax_e.set_ylim(0, 60)

# Homogeneity on right axis
ax_e2.plot(fg_vals, dyn_df["enclave_homogeneity"],
           color="grey", linewidth=1.2, marker="s", markersize=6,
           linestyle="--", label="Enclave homogeneity")
ax_e2.set_ylabel("Enclave homogeneity", color="grey")
ax_e2.tick_params(axis="y", labelcolor="grey")
# Zoom axis to show variation
ax_e2.set_ylim(0.90, 0.96)

ax_e.set_xlabel("$f_g$ (griefer fraction)")
ax_e.set_xticks(fg_vals)
ax_e.set_xticklabels(["0.25", "0.50", "0.75"])
ax_e.set_title("Coalition enclave formation across griefer fraction")

# Annotation: STILL_CLIMBING at high f_g
ax_e.axvline(0.50, color="grey", linewidth=0.6, linestyle=":", alpha=0.7)
ax_e.text(0.505, 55, "STILL_CLIMBING\n≥10% iterations", fontsize=7,
          color="grey", va="top")

legend_e = [
    Line2D([0], [0], color="#4daf4a", linewidth=1.5, marker="o",
           label="Settling step (mean ± 1 SD)"),
    Line2D([0], [0], color="grey", linewidth=1.2, marker="s",
           linestyle="--", label="Enclave homogeneity"),
]
ax_e.legend(handles=legend_e, fontsize=8, loc="upper left")

fig.tight_layout(pad=2.0)
out = os.path.join(FIG_DIR, "exp2b_dynamics.pdf")
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {out}")

print("Done.")
