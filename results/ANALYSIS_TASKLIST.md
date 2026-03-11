# Analysis & Visualization Tasklist for Claude Code

## Prerequisites

### Task 0: Complete the missing simulation config

**exp2_np27_rho015_alpha10** has no data (raw.csv is empty). This must be re-run:

```bash
# From the project root, re-run this single config
python experiments/run_exp2.py --config exp2_np27_rho015_alpha10
```

If the experiment runner doesn't support single-config reruns, the config parameters are:
- n_communities: 900, n_platforms: 27, p_space: 10, t_max: 100
- institution: mixed, rho_extremist: 0.15, alpha: 10.0
- mu: 0.05, coalitions: 5, mutations: 3, svd_groups: 10
- search_steps: 10, initial_distribution: equal, n_iterations: 200, seed_base: 42

After the run completes, the post-processing pipeline must also run on it to produce summary.csv, stepwise.csv, convergence.json, and the dynamics/ folder with burst_aggregate.json, enclaves.json, etc.

**This blocks**: The ρ_e=0.15 headline table, the ρ_e=0.15 ANOVA, and the complete escalation analysis.

---

## Phase A: Python Analysis Tasks (run in project environment)

### Task A1: Displacement analysis across all configs

Run `displacement_diagnostic.py` on each of the 26 (or 27, once Task 0 completes) exp2 configs. For each config:

**Inputs**:
- `results/exp2/{config}/stepwise.csv`
- `results/exp2/{config}/dynamics/burst_aggregate.json` (for burst event identification)
- `results/exp2/{config}/dynamics/per_iter_raiding.json` (for per-iteration burst steps/sizes)
- `results/exp2/{config}/dynamics/flow.npz` (platform-to-platform flow matrices per step, shape [N_p × N_p] per step, averaged across iterations)

**Logic**:
1. Read the burst data to identify raid events (burst steps on raiding_stable or raiding_base platforms)
2. For each raid event, build a ±5 step window around the burst
3. Use stepwise.csv to extract per-governance community counts and mainstream utility at each step in the window
4. Build a superposed epoch: align all events to t=0 (the burst step), average across events
5. Compute displacement statistics: mainstream utility delta, destination distribution (does mainstream move to algorithmic or coalition after a raid?)

**Key insight from data exploration**: The flow.npz files contain full platform-to-platform flow matrices per step. These are 27×27 integer matrices (at N_p=27). These can be used to directly compute directional flows — no need to infer from net governance counts. To use them, you need the config.json to map platform IDs to governance types.

**Outputs per config**:
- `results/exp2/{config}/dynamics/displacement.json` — per-config displacement statistics and superposed epoch data
- Structure should match the schema in REPORTING_PIPELINE_TASKLIST.md Task 1.3/2.3

**Cross-config output**:
- `results/exp2/exp2_displacement_master.csv` — one row per config with displacement summary statistics

### Task A2: Enclave settling and disruption analysis

For each exp2 config, compute enclave formation metrics from `dynamics/enclaves.json`:

**For each coalition platform**:
1. **Settling step**: First step after which homogeneity stays > 0.9 for ≥ 10 consecutive steps. If never achieved, settling_step = NaN.
2. **Disruption count**: Number of steps where homogeneity drops below 0.9 *after* the settling step.
3. **Recovery time**: Mean steps to return above 0.9 after a post-settling disruption.

**Aggregate across platforms within config**:
- mean_settling_step, sd_settling_step
- fraction_of_platforms_with_disruptions
- mean_recovery_steps

**Output**:
- `results/exp2/{config}/dynamics/enclave_detailed.json`
- `results/exp2/exp2_enclave_master.csv`

### Task A3: Re-run ANOVA at ρ_e=0.15 (after Task 0)

Once exp2_np27_rho015_alpha10 has data, re-run the two-way ANOVA. Code:

```python
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Read all raw.csv files for rho_e=0.15 configs
# Fit: avg_utility_mainstream ~ C(np) * C(alpha)
# Report F-statistics and p-values for main effects and interaction
```

Save to `results/exp2/exp2_anova_results.json`.

### Task A4: Generate factorial LaTeX tables

Produce LaTeX tables for the paper from the master data. For each of these metrics, produce a 3×3 table (rows=N_p ∈ {3,9,27}, columns=α ∈ {2,5,10}) at each ρ_e level:

1. Normalized mainstream utility (with SD in parentheses)
2. Mainstream utility by governance type (3 sub-tables: direct, coalition, algorithmic)
3. Extremist concentration on direct (fraction)
4. Burst rate
5. Median burst size
6. Escalation slope (with significance stars)

Save to `results/exp2/exp2_factorial_tables.tex`.

---

## Phase B: Visualization Tasks (R with ggplot2)

All visualizations should be publication-quality: clean white backgrounds, consistent font sizes (12pt labels, 10pt annotations), colorblind-safe palettes. Save as both PDF and PNG (300 DPI).

### Task B1: Interaction Heatmap — Normalized Mainstream Utility

**Priority: HIGH** — This is the headline figure (or a table substitute).

Three panels (one per ρ_e level), each a 3×3 heatmap:
- x-axis: α ∈ {2, 5, 10}, labeled "Parasitism intensity (α)"
- y-axis: N_p ∈ {3, 9, 27}, labeled "Number of platforms (N_p)"
- Fill: Normalized mainstream utility
- Color scale: sequential (viridis or similar), same range across all three panels
- Annotate cells with the value to 3 decimal places
- Panel titles: "ρ_e = 0.05", "ρ_e = 0.10", "ρ_e = 0.15"

**Data source**: Build from summary.csv files. The values are already computed above in the interpretation document.

```r
library(tidyverse)
library(patchwork)

# Read data from exp2 summary files
# Create heatmap panels
# Use scale_fill_viridis_c() for colorblind safety
# facet_wrap(~rho_e) or patchwork for three panels
```

**Output**: `results/viz/fig_interaction_heatmap.pdf` and `.png`

### Task B2: Escalation Slope Heatmap

**Priority: HIGH** — Key dynamics figure.

Same layout as B1 but fill = escalation mean slope. Use a diverging color scale (blue-white-red) centered at 0. Mark non-significant cells (p > 0.05) with "ns" overlay.

**Data source**: burst_aggregate.json from each config's dynamics folder.

**Output**: `results/viz/fig_escalation_heatmap.pdf` and `.png`

### Task B3: Governance Utility Divergence Plot

**Priority: HIGH** — Shows coalition firewall mechanism.

Line plot with α on x-axis (2, 5, 10) and mainstream utility on y-axis. Three lines per panel (direct, coalition, algorithmic), colored by governance type. Three panels for N_p ∈ {3, 9, 27}. Fix ρ_e = 0.15 (or use 0.10 if np27/alpha10/rho015 remains missing).

Add a horizontal dashed line at utility = 5.0 (random assignment baseline) to highlight when direct platforms go below random.

Error bars: ±1 SE from summary.csv CI data (CI width / 2 ≈ SE, or compute from SD/√200).

```r
# Three panels showing how each governance type's mainstream utility
# responds to increasing parasitism
# The story: coalition is flat, algorithmic dips slightly, direct collapses
```

**Output**: `results/viz/fig_governance_divergence.pdf` and `.png`

### Task B4: Burst Heatmap Grid

**Priority: MEDIUM** — Supplements the escalation figure.

Two 3×3 heatmaps at ρ_e=0.15 (or as a 3-panel triptych across ρ_e levels):
- Panel 1: Median burst size
- Panel 2: Burst rate (fraction of platform-iterations with bursts)

**Data source**: burst_aggregate.json.

**Output**: `results/viz/fig_burst_heatmap.pdf` and `.png`

### Task B5: Enclave Trajectory Plot

**Priority: MEDIUM** — Mechanism evidence for coalition firewall.

Single-panel line plot showing coalition homogeneity over 100 simulation steps for one representative config (suggest N_p=27, ρ_e=0.15, α=5 — we have this data).

Use the per-platform homogeneity series from enclaves.json. Plot all 9 coalition platforms as thin gray lines, with the mean as a thick colored line. Add a horizontal dashed line at homogeneity = 0.9 (enclave threshold).

```r
# Read enclaves.json for exp2_np27_rho015_alpha5
# Plot all coalition platform series + mean
# Show rapid rise to ~1.0 and sustained stability
```

**Output**: `results/viz/fig_enclave_trajectory.pdf` and `.png`

### Task B6: Superposed Epoch Plot (requires Task A1 completion)

**Priority: HIGHEST** — The paper's most impactful single figure.

Two-panel figure:
- Top panel: Community counts by governance type (y-axis) around the average raid (x-axis: relative step, -8 to +8)
  - Three lines: direct (expect sharp drop at t=0), algorithmic (expect rise at t=0), coalition (expect flat)
  - Shaded ±1 SE bands
- Bottom panel: Mainstream utility (y-axis) around the average raid
  - Show per-governance-type utility lines
  - Expect: dip on algorithmic, possible rise on direct (fewer parasites after they leave)

Show for 2-3 configs overlaid or as separate panels: suggest {N_p=9, ρ_e=0.15, α=10} and {N_p=27, ρ_e=0.15, α=5} for contrast.

**Data source**: displacement.json from Task A1.

**Output**: `results/viz/fig_superposed_epoch.pdf` and `.png`

### Task B7: Extremist Concentration Bar Chart

**Priority: LOW** — Simple supplementary figure.

Grouped bar chart showing fraction of extremists by governance type across configs. The story is simple (50% on direct everywhere) so this could also be a table.

**Output**: `results/viz/fig_extremist_concentration.pdf` and `.png`

---

## Phase C: Paper Writing Support

### Task C1: Generate results section prose

Using the completed analysis and visualization outputs, draft the results section following the structure in RESULTS_INTERPRETATION.md:
- Section 3.1: Baseline (Exp1) — ~2 pages
- Section 3.2: Extremists and system structure — ~4 pages
- Section 3.3: Raiding dynamics — ~3 pages
- Section 3.4: Sensitivity — ~1 page

### Task C2: Generate discussion section outline

Key discussion points:
1. The Tiebout extension: what works, what breaks
2. Policy implications: consolidation costs, coalition governance, system-level thinking
3. Empirical mappings: raiding cycle ↔ cross-platform extremist migration (Kiwi Farms, Gab, etc.)
4. Limitations: stylized model, binary preferences, no content moderation, no network effects
5. Future work: moving cost sensitivity, content moderation mechanisms, network topology

---

## Execution Order

1. **Task 0** (missing config) — BLOCKS everything at ρ_e=0.15
2. **Tasks A1–A2** (displacement + enclave analysis) — can run in parallel on existing 26 configs
3. **Tasks B1–B5** (visualizations from existing data) — can run immediately in parallel
4. **Task A3** (ANOVA) — after Task 0
5. **Task A4** (LaTeX tables) — after Task 0
6. **Task B6** (superposed epoch) — after Task A1
7. **Tasks C1–C2** (writing) — after all above

Tasks B1–B5 are independent of Task 0 and can proceed immediately using ρ_e=0.10 data as the primary display (with ρ_e=0.15 added once complete).
