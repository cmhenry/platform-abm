# TASKLIST: Integrate Reporting Pipeline for Experiments 1 & 2

## Context

We have three standalone analysis modules that have been validated on individual
iterations:

- `burst_analysis.py` — Detects bursty raiding patterns from extremist outflow
  series. Replaces ACF-based cycle detection. Produces per-platform burst
  statistics, classifications, and escalation slopes.

- `displacement_diagnostic.py` — Analyzes mainstream community displacement
  following extremist raids. Uses burst events + stepwise data to infer flow
  directions, compute pre/post utility changes, and build superposed epoch
  data for visualization.

- `analyze_stepwise.py` — Processes per-step metrics (utility trajectories,
  relocation rates, per-governance breakdowns). Classifies convergence patterns.

These need to be integrated into the experiment pipeline so that after all
iterations of a config complete, we produce every data artifact needed for the
results section and planned visualizations. The pipeline should also aggregate
across iterations and across configs to produce master summary files.

Below is the full task list. Each task specifies inputs, outputs, and the
analysis logic. Tasks are grouped into four phases:

  Phase 1: Per-iteration analysis (runs after each iteration completes)
  Phase 2: Per-config aggregation (runs after all 200 iterations of a config)
  Phase 3: Cross-config master summaries (runs after all configs complete)
  Phase 4: Visualization-ready data extraction (runs once at the end)

---

## Phase 1: Per-Iteration Analysis

These run immediately after each iteration's simulation loop completes.
They consume the raw simulation output and produce analysis JSON files
alongside the existing output.

### Task 1.1: Stepwise convergence analysis

**Trigger:** After each iteration completes and stepwise.csv is written.

**Input:** `{config}/{iter}/stepwise.csv`

**Logic:** Call `analyze_stepwise.py` functions:
- Load stepwise CSV into DataFrame
- Compute convergence diagnostics: settling time, pattern classification
  (CONVERGED, STILL_CLIMBING, OSCILLATING, NOISY_PLATEAU, PLATEAU)
- Compute utility trajectory statistics: final utility, utility at step 50
  (half-simulation), second-half gain
- Compute per-governance convergence: does each governance type converge
  independently?

**Output:** `{config}/{iter}/convergence.json`
```json
{
  "pattern": "CONVERGED",
  "settling_step": 12,
  "final_utility": 6.71,
  "half_utility": 6.68,
  "second_half_gain": 0.03,
  "final_relocations_per_step": 194,
  "per_governance": {
    "direct": {"pattern": "OSCILLATING", "final_utility": 10.1},
    "coalition": {"pattern": "CONVERGED", "final_utility": 6.39},
    "algorithmic": {"pattern": "CONVERGED", "final_utility": 6.30}
  }
}
```

### Task 1.2: Burst raiding analysis (Experiment 2 only)

**Trigger:** After each iteration completes and raiding.json is written.
Skip for Experiment 1 configs (ρ_e = 0).

**Input:** `{config}/{iter}/raiding.json`

**Logic:** Call `burst_analysis.analyze_all_platforms()`:
- Burst threshold = 10
- Produces per-platform burst statistics and classification
- Produces system-level summary

**Important:** The burst_analysis.json output must RETAIN the burst_steps
and burst_sizes arrays per platform (not strip them as in the summary-only
mode). These are needed by the displacement diagnostic in Task 1.3.

**Output:** `{config}/{iter}/burst_analysis.json`

### Task 1.3: Mainstream displacement analysis (Experiment 2 only)

**Trigger:** After Task 1.1 and Task 1.2 both complete for an iteration.

**Input:**
- `{config}/{iter}/stepwise.csv`
- `{config}/{iter}/burst_analysis.json`

**Logic:** Call `displacement_diagnostic.run_displacement_analysis()`:
- Extract raid events from burst data (platforms classified as raiding_stable
  or raiding_base, bursts ≥ 10)
- Build ±5 step event windows around each raid
- Infer flow directions from net governance count changes
- Build superposed epoch (±8 steps) averaging across all raid events
- Compute displacement statistics: mainstream utility delta, destination
  distribution, burst-displacement correlation

**Output:** `{config}/{iter}/displacement.json`
```json
{
  "n_events": 42,
  "n_windows": 38,
  "flow_analysis": {
    "fraction_to_algorithmic": 0.76,
    "fraction_to_coalition": 0.24,
    "mainstream_util_delta_mean": -0.034,
    "mainstream_util_delta_negative_fraction": 0.71,
    "direct_count_delta_mean": -28.3,
    "dest_util_delta_mean": -0.021,
    "burst_displacement_correlation": -0.42
  },
  "superposed_epoch": {
    "relative_steps": [-8, -7, ..., 7, 8],
    "algorithmic_count_mean": [...],
    "mainstream_util_mean": [...],
    ...
  },
  "event_details": [
    {"step": 20, "size": 59, "destination": "algorithmic", ...},
    ...
  ]
}
```

### Task 1.4: Enclave analysis (Experiment 2 only)

**Trigger:** After iteration completes and enclaves.json is written.

**Input:** `{config}/{iter}/enclaves.json`

**Logic:**
- For each coalition platform, compute:
  - Mean homogeneity, fraction of steps enclaved (homogeneity > 0.9)
  - Settling step: first step after which homogeneity stays > 0.9 for at
    least 10 consecutive steps
  - Number of disruption events: steps where homogeneity drops below 0.9
    after the settling step
  - Post-settling disruption recovery time: mean steps to return above 0.9
    after a disruption

**Output:** `{config}/{iter}/enclave_analysis.json`
```json
{
  "platforms": {
    "903": {
      "mean_homogeneity": 0.978,
      "fraction_enclaved": 0.92,
      "settling_step": 25,
      "n_disruptions": 1,
      "mean_recovery_steps": 2.0
    },
    ...
  },
  "system": {
    "mean_settling_step": 27.3,
    "mean_homogeneity": 0.955,
    "fraction_with_disruptions": 0.22
  }
}
```

---

## Phase 2: Per-Config Aggregation

These run after all 200 iterations of a given config complete. They
aggregate the per-iteration analysis JSONs into config-level summaries.

### Task 2.1: Aggregate convergence

**Input:** All 200 `convergence.json` files for the config.

**Logic:**
- Count pattern classifications across iterations (e.g., 180 CONVERGED,
  15 PLATEAU, 5 OSCILLATING)
- Compute mean and SD of settling_step, final_utility, second_half_gain
- Compute per-governance pattern distribution

**Output:** `{config}/convergence_aggregate.json`

### Task 2.2: Aggregate burst statistics

**Input:** All 200 `burst_analysis.json` files for the config.

**Logic:**
- Pool all burst sizes across all platforms across all iterations
- Pool all inter-burst intervals
- Pool all escalation slopes, FILTERING to platforms with n_bursts >= 3
  (critical: do NOT include 2-burst platforms with mechanical R²=1.0)
- Compute classification distribution across all platform-iterations
- **Escalation test:** One-sample t-test of pooled slopes against 0.
  Report mean slope, SD, t-statistic, p-value, fraction positive.
- Compute burst rate: fraction of platform-iterations with any burst activity

**Output:** `{config}/burst_aggregate.json`
```json
{
  "n_iterations": 200,
  "n_platform_iterations": 5400,
  "classification_proportions": {
    "quiet": 0.33, "raiding_stable": 0.44, "raiding_base": 0.07,
    "enclave": 0.07, "active": 0.09
  },
  "burst_size_mean": 38.7,
  "burst_size_median": 31.0,
  "burst_size_sd": 19.2,
  "interval_mean": 13.1,
  "interval_median": 9.0,
  "interval_sd": 8.4,
  "escalation_n_slopes": 1847,
  "escalation_mean_slope": 0.34,
  "escalation_sd": 12.1,
  "escalation_t_stat": 1.21,
  "escalation_p_value": 0.226,
  "escalation_fraction_positive": 0.53,
  "burst_rate": 0.67
}
```

### Task 2.3: Aggregate displacement

**Input:** All 200 `displacement.json` files for the config.

**Logic:**
- Pool event-level displacement data across iterations
- Compute mean/SD of mainstream_util_delta, fraction negative
- Compute mean destination distribution (fraction to algorithmic vs coalition)
- **Superposed epoch aggregation:** Average the per-iteration superposed
  epoch curves. Each iteration produces epoch curves aligned to t=0; average
  these to get the config-level canonical trajectory. Also compute SE bands.
- Compute burst_size → displacement correlation across pooled events

**Output:** `{config}/displacement_aggregate.json`
```json
{
  "n_iterations": 200,
  "total_events": 8420,
  "destination_algorithmic_mean": 0.74,
  "destination_coalition_mean": 0.26,
  "mainstream_util_delta_mean": -0.031,
  "mainstream_util_delta_sd": 0.018,
  "mainstream_util_delta_negative_frac": 0.69,
  "burst_displacement_corr_mean": -0.38,
  "superposed_epoch": {
    "relative_steps": [-8, ..., 8],
    "mainstream_util_mean": [...],
    "mainstream_util_se": [...],
    "algorithmic_count_mean": [...],
    "algorithmic_count_se": [...],
    "direct_count_mean": [...],
    "direct_count_se": [...],
    "coalition_count_mean": [...],
    "coalition_count_se": [...]
  }
}
```

### Task 2.4: Aggregate enclaves

**Input:** All 200 `enclave_analysis.json` files for the config.

**Logic:**
- Compute mean/SD of settling_step, mean_homogeneity, fraction_enclaved
- Compute fraction of platform-iterations with post-settling disruptions
- Compute mean recovery time from disruptions

**Output:** `{config}/enclave_aggregate.json`

### Task 2.5: Config summary (existing, updated)

**Input:** All per-iteration summary.csv rows (existing pipeline) plus the
four aggregate JSONs above.

**Logic:** Extend the existing config summary.csv row with fields from
the aggregated analyses:
- convergence_pattern_mode (most common pattern)
- burst_rate, burst_size_median, burst_interval_median
- escalation_mean_slope, escalation_p_value
- displacement_util_delta_mean, displacement_frac_negative
- enclave_mean_homogeneity, enclave_settling_step

**Output:** Updated `{config}/summary.csv` with additional columns.

---

## Phase 3: Cross-Config Master Summaries

Run once after all configs (Experiment 1 and Experiment 2) complete.

### Task 3.1: Master summary CSV

**Input:** All config-level summary.csv files.

**Logic:** Concatenate into single master file. One row per config with
all metrics. This is the primary data source for all results tables.

**Output:**
- `exp1_master_summary.csv` (6 rows)
- `exp2_master_summary.csv` (27 rows)

### Task 3.2: Burst master CSV

**Input:** All 27 `burst_aggregate.json` files from Experiment 2.

**Logic:** One row per config. Columns: config_name, n_platforms, rho_e,
alpha, burst_rate, burst_size_mean, burst_size_median, burst_size_sd,
interval_mean, interval_median, escalation_mean_slope, escalation_p_value,
escalation_fraction_positive, classification proportions.

**Output:** `exp2_burst_master.csv`

### Task 3.3: Displacement master CSV

**Input:** All 27 `displacement_aggregate.json` files from Experiment 2.

**Logic:** One row per config. Columns: config_name, n_platforms, rho_e,
alpha, total_events, dest_frac_algorithmic, dest_frac_coalition,
mainstream_util_delta_mean, mainstream_util_delta_sd, displacement_frac_negative,
burst_displacement_corr.

**Output:** `exp2_displacement_master.csv`

### Task 3.4: Enclave master CSV

**Input:** All 27 `enclave_aggregate.json` files from Experiment 2.

**Logic:** One row per config. Columns: config_name, n_platforms, rho_e,
alpha, mean_homogeneity, mean_settling_step, fraction_disrupted,
mean_recovery_steps.

**Output:** `exp2_enclave_master.csv`

### Task 3.5: Factorial interaction table

**Input:** `exp2_master_summary.csv`

**Logic:** Produce the key interaction tables for the results section.
For each of the following outcomes, produce a 3×3 table (N_p × α) at
each ρ_e level:

1. Normalized mainstream utility
2. Normalized extremist utility
3. Mainstream utility by governance type (3 sub-tables)
4. Burst rate (fraction of platforms with burst activity)
5. Median burst size
6. Mainstream displacement delta
7. Enclave mean homogeneity

Also compute the N_p × α interaction contrast (difference-in-differences)
for normalized mainstream utility. Report Δ(N_p effect at α=10) minus
Δ(N_p effect at α=2) at each ρ_e level.

**Output:** `exp2_factorial_tables.tex` — LaTeX source for all tables.

### Task 3.6: Two-way ANOVA on iteration-level data

**Input:** Iteration-level mainstream utility for all Experiment 2 configs.
This requires either: (a) retaining per-iteration utility values in the
config summaries, or (b) re-reading per-iteration summary rows.

**Logic:** At each ρ_e level, fit a two-way ANOVA:
  mainstream_utility ~ N_p + α + N_p:α
Report F-statistics and p-values for the main effects and interaction.
The interaction term tests whether the diversification premium depends
on parasitism intensity (the paper's central claim).

Implementation: use scipy.stats or statsmodels. If neither available,
compute manually from the group means and within-group variances.

**Output:** `exp2_anova_results.json`
```json
{
  "rho_005": {
    "N_p_F": 142.3, "N_p_p": 1.2e-58,
    "alpha_F": 87.1, "alpha_p": 3.4e-37,
    "interaction_F": 8.7, "interaction_p": 1.1e-6
  },
  ...
}
```

---

## Phase 4: Visualization-Ready Data Extraction

Run once after Phase 3. These tasks produce the specific data files
needed to generate the planned figures.

### Task 4.1: Superposed epoch data for visualization

**Input:** `displacement_aggregate.json` from the three highest-threat
configs: {N_p=27, ρ_e=0.15, α=10}, {N_p=9, ρ_e=0.15, α=10},
{N_p=3, ρ_e=0.15, α=10}.

**Logic:** Extract the averaged superposed epoch curves from each config.
Format as a single CSV with columns: config, relative_step,
algorithmic_count_mean, algorithmic_count_se, direct_count_mean,
direct_count_se, coalition_count_mean, coalition_count_se,
mainstream_util_mean, mainstream_util_se.

**Output:** `viz/superposed_epoch.csv`

**Target figure:** Superposed Epoch Plot (primary displacement figure).
Two-panel: community counts and mainstream utility around the average
raid, with ±1 SE bands.

### Task 4.2: Platform biography data

**Input:** Stepwise data and burst_analysis.json from one representative
iteration of {N_p=27, ρ_e=0.15, α=10}.

**Logic:**
- From burst_analysis.json, identify the direct platform with the median
  number of bursts (not the most extreme — avoid cherry-picking)
- From stepwise.csv, extract per-step metrics for that platform
- If per-platform step data isn't available in stepwise.csv (which tracks
  governance-type aggregates, not individual platforms), this task requires
  either: (a) a targeted re-run logging per-platform counts, or (b)
  approximation from the governance-level data plus burst timing.
- NOTE: If per-platform data is not available, skip this task and flag
  it as requiring a simulation code addition to log per-platform community
  counts and types at each step.

**Output:** `viz/platform_biography.csv` (if feasible) or
`viz/PLATFORM_BIOGRAPHY_NEEDS_DATA.md` (if not)

**Target figure:** Platform Biography — one direct platform's 100-step
life showing sawtooth extremist accumulation and mainstream evacuation.

### Task 4.3: Burst heatmap data

**Input:** `exp2_burst_master.csv`

**Logic:** Pivot into three 3×3 matrices (one per ρ_e level):
- Matrix 1: median burst size (rows = N_p, cols = α)
- Matrix 2: median inter-burst interval
- Matrix 3: burst rate (fraction of platforms showing burst activity)

**Output:** `viz/burst_heatmap.csv` — long-format with columns:
rho_e, n_platforms, alpha, metric, value.

**Target figure:** Burst Heatmap Grid — 3×3 heatmaps showing raiding
intensity across the factorial design.

### Task 4.4: Enclave trajectory data

**Input:** `enclave_aggregate.json` from a representative config.

**Logic:** Extract the averaged homogeneity series across coalition
platforms. Also extract one representative platform's raw homogeneity
series (the platform closest to the mean settling step).

**Output:** `viz/enclave_trajectory.csv`

**Target figure:** Enclave formation over time — one panel showing
coalition homogeneity rising to ~1.0 within 25 steps then holding.

### Task 4.5: Alluvial flow approximation data

**Input:** Stepwise data from a representative iteration of
{N_p=27, ρ_e=0.15, α=10}.

**Logic:**
- Divide 100 steps into 5 bands: 1-20, 21-40, 41-60, 61-80, 81-100
- At each band boundary, record community counts by governance type
  (from per_governance_community_count columns)
- For mainstream and extremist separately (from per-type counts if
  available by governance, or from the type × governance cross if
  available in the stepwise data)
- Compute net flows between bands: Δ(count) for each governance type
  between consecutive band boundaries
- NOTE: This is an APPROXIMATION. We infer flows from net changes, so
  we cannot distinguish "20 left and 15 arrived" from "5 net departed."
  The alluvial will show net directional flows, not gross flows.

**Output:** `viz/alluvial_flows.csv` — columns: time_band, gov_type,
community_type, count, net_flow_from_previous.

**Target figure:** Alluvial/Sankey Diagram — flows of mainstream and
extremist communities between governance types over 5 time bands.

---

## File Tree After Full Pipeline Run

```
experiment_outputs/
├── exp1/
│   ├── {config}/
│   │   ├── {iter}/
│   │   │   ├── summary.csv          (existing)
│   │   │   ├── stepwise.csv         (existing)
│   │   │   └── convergence.json     (NEW - Task 1.1)
│   │   ├── convergence_aggregate.json  (NEW - Task 2.1)
│   │   └── summary.csv             (existing, unchanged)
│   └── exp1_master_summary.csv      (Task 3.1)
│
├── exp2/
│   ├── {config}/
│   │   ├── {iter}/
│   │   │   ├── summary.csv          (existing)
│   │   │   ├── stepwise.csv         (existing)
│   │   │   ├── raiding.json         (existing)
│   │   │   ├── enclaves.json        (existing)
│   │   │   ├── convergence.json     (NEW - Task 1.1)
│   │   │   ├── burst_analysis.json  (NEW - Task 1.2)
│   │   │   ├── displacement.json    (NEW - Task 1.3)
│   │   │   └── enclave_analysis.json (NEW - Task 1.4)
│   │   ├── convergence_aggregate.json  (NEW - Task 2.1)
│   │   ├── burst_aggregate.json     (NEW - Task 2.2)
│   │   ├── displacement_aggregate.json (NEW - Task 2.3)
│   │   ├── enclave_aggregate.json   (NEW - Task 2.4)
│   │   └── summary.csv             (UPDATED - Task 2.5)
│   ├── exp2_master_summary.csv      (Task 3.1)
│   ├── exp2_burst_master.csv        (Task 3.2)
│   ├── exp2_displacement_master.csv (Task 3.3)
│   ├── exp2_enclave_master.csv      (Task 3.4)
│   ├── exp2_factorial_tables.tex    (Task 3.5)
│   └── exp2_anova_results.json      (Task 3.6)
│
└── viz/
    ├── superposed_epoch.csv         (Task 4.1)
    ├── platform_biography.csv       (Task 4.2, if feasible)
    ├── burst_heatmap.csv            (Task 4.3)
    ├── enclave_trajectory.csv       (Task 4.4)
    └── alluvial_flows.csv           (Task 4.5)
```

---

## Implementation Notes

### Dependencies
- numpy, pandas (already in use)
- scipy.stats (for t-tests in Task 2.2, ANOVA in Task 3.6)
- The three analysis modules: burst_analysis.py, displacement_diagnostic.py,
  analyze_stepwise.py — import directly, do not rewrite their internals.

### Parallelism
- Phase 1 tasks run per-iteration and are embarrassingly parallel.
  Task 1.1 is independent; Task 1.2 is independent; Task 1.3 depends on
  both 1.1 and 1.2 completing. Task 1.4 is independent.
- Phase 2 tasks run per-config after all iterations finish. They are
  sequential within a config but parallel across configs.
- Phase 3 and 4 run once, sequentially, after everything else.

### Critical filters
- Escalation slopes: ONLY aggregate from platforms with n_bursts >= 3.
  Platforms with exactly 2 bursts produce mechanical R²=1.0. This filter
  is the fix for the artifact we identified in validation.
- Displacement events: ONLY from platforms classified as raiding_stable
  or raiding_base. Do not include "quiet" or "enclave" platforms.
- Burst threshold: 10 everywhere. This is validated and consistent across
  all analysis modules.

### Error handling
- If a config has zero raid events (possible at low ρ_e and low α),
  the burst and displacement aggregations should produce valid JSON with
  zero counts and NaN statistics, not crash.
- If stepwise.csv is missing expected columns (e.g., per_type_utility
  columns absent in Experiment 1), the displacement diagnostic should
  skip gracefully.

### Backward compatibility
- Existing output files (summary.csv, stepwise.csv, raiding.json,
  enclaves.json) must not be modified. All new outputs are additional
  files alongside the existing ones.
- The updated config summary.csv (Task 2.5) adds columns but does not
  remove or rename existing ones.

### Platform biography data gap
- Task 4.2 (platform biography) requires per-platform, per-step community
  counts split by type. The current stepwise tracking records governance-type
  aggregates, not individual platform stats. If this data is not available,
  flag the gap and add a TODO for a simulation code change:
  ```
  At each step, for each platform, record:
    platform_id, step, n_mainstream, n_extremist, utility_mainstream, utility_extremist
  ```
  This is lightweight (one row per platform per step) and enables both the
  platform biography and the full network flow diagram. Recommend adding
  this as a per-step logging option gated by a config flag.
