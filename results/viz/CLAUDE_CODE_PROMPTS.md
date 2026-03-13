# Claude Code Prompts: B5 Failure Panel, B7 Redesign, Platform Biography

These are three independent tasks. Each modifies or creates R code in `results/viz/`. Run them in any order.

---

## Prompt 1: B5 Enclave Trajectory — Add Failure-Case Companion Panel

### Task

Modify the B5 section of `results/viz/generate_phase_b.R` (lines 341–383) to produce a **two-panel figure** instead of a single panel:

- **Left panel**: The current success case — N_p=27, ρ_e=0.15, α=5 (already working)
- **Right panel**: The failure case — N_p=3, ρ_e=0.15, α=10

Both panels use the same visual design: grey individual-platform traces, bold green mean line, horizontal dashed threshold at 0.9. Both share the same y-axis range (0 to 1.05) for direct comparison.

### Data

The enclave data is in `results/exp2/{config}/dynamics/enclaves.json`. Each file is a JSON object keyed by platform ID, where each platform has:
- `mean_homogeneity`: scalar
- `fraction_enclaved`: scalar
- `homogeneity_series`: array of 100 floats (one per simulation step)

For the success case (exp2_np27_rho015_alpha5), there are 9 coalition platforms.
For the failure case (exp2_np3_rho015_alpha10), there is only **1 coalition platform** (platform 902). This means the "individual traces" and "mean line" will be the same line. That's fine — the visual contrast is between one messy, low-homogeneity trace (failure) and nine stable, high-homogeneity traces (success).

### Layout

Use `patchwork` to place them side by side: `(p_success | p_failure)`.

Panel subtitles (as facet strips or plot subtitles):
- Left: `N_p = 27, α = 5` (with proper expression notation)
- Right: `N_p = 3, α = 10`

Overall subtitle: `ρ_e = 0.15`

Shared y-axis label: "Coalition platform homogeneity"
Shared x-axis label: "Simulation step"

### Expected visual contrast

- **Left (success)**: Grey traces converge to ~1.0 by step 25 and stay there, with occasional brief dips. Green mean line hugs 1.0. Almost all grey traces are above the 0.9 threshold after step 30.
- **Right (failure)**: Single trace oscillating erratically between ~0.5 and ~0.9, rarely sustaining above 0.9. The trace never settles — the enclave never forms reliably.

### Output

Save as `fig_enclave_trajectory.pdf` and `fig_enclave_trajectory.png`, width = 12, height = 5 (wider than current to accommodate two panels).

---

## Prompt 2: B7 Redesign — Simplified 3-Panel + Overrepresentation Ratio

### Task

Replace the B7 section of `results/viz/generate_phase_b67.R` (the extremist concentration bar chart, lines 195–289) with **two new figures**:

#### Figure B7a: Simplified 3-Panel Concentration

Collapse across α (average the three α values for each {N_p, ρ_e, governance} combination) and produce a grouped bar chart with:
- **3 panels** (one per ρ_e level): faceted by ρ_e
- **x-axis**: N_p ∈ {3, 9, 27}
- **y-axis**: Fraction of extremists (0% to 60%)
- **3 grouped bars** per N_p: Direct, Coalition, Algorithmic (using the same colorblind-safe palette: Direct="#D55E00", Coalition="#009E73", Algorithmic="#0072B2")
- Horizontal dashed reference line at 1/3 (equal share)
- Error bars showing the range across the three α values (min to max) to indicate how little α matters

Save as `fig_extremist_concentration_simple.{pdf,png}`, width=10, height=4.

#### Figure B7b: Overrepresentation Ratio Heatmap

Compute the **overrepresentation ratio** for each governance type: (fraction of extremists on governance type) ÷ (fraction of total population on governance type).

A ratio of 1.0 means extremists are distributed proportionally. >1.0 means overrepresented. <1.0 means underrepresented.

Produce a 3×3 heatmap (rows = governance type {Direct, Coalition, Algorithmic}, columns = N_p ∈ {3, 9, 27}) at ρ_e=0.15 (collapse across α by averaging):

- **Fill**: Overrepresentation ratio
- **Color scale**: Diverging, centered at 1.0 (white). Red for >1.0 (overrepresented), blue for <1.0 (underrepresented).
- **Cell annotations**: The ratio to 1 decimal place (e.g., "4.8×" or "0.6×")
- **x-axis**: N_p
- **y-axis**: Governance type
- **Subtitle**: ρ_e = 0.15, averaged across α

Save as `fig_overrepresentation_ratio.{pdf,png}`, width=7, height=4.

### Data

Read from the same summary.csv files already loaded in generate_phase_b67.R. The relevant measures are:
- `final_count_extremist_direct`, `final_count_extremist_coalition`, `final_count_extremist_algorithmic` (extremist counts)
- `final_count_direct`, `final_count_coalition`, `final_count_algorithmic` (total community counts)

For the overrepresentation ratio:
```
extremist_fraction_on_direct = count_extremist_direct / (count_extremist_direct + count_extremist_coalition + count_extremist_algorithmic)
population_fraction_on_direct = count_direct / (count_direct + count_coalition + count_algorithmic)
overrep_ratio_direct = extremist_fraction_on_direct / population_fraction_on_direct
```

### Expected patterns

- Direct: overrepresentation ratio ~3× at N_p=3, rising to ~4.8× at N_p=27 (more platforms = more concentration on direct)
- Coalition: ratio ~0.5–0.8× (underrepresented, protected by enclave)
- Algorithmic: ratio ~0.6–0.9× (slightly underrepresented, communities sorted elsewhere)

---

## Prompt 3: Platform Biography Figure (New — Viz 4.2)

### Task

Create a new R script `results/viz/generate_biography.R` that produces a platform biography figure: the life story of one direct platform over 100 simulation steps, showing the accumulate-raid-recover cycle.

### Data Sources

The biography combines data from two files for a single representative config and iteration:

**Config**: `exp2_np9_rho015_alpha10` (9 platforms, strong parasitism — clear raiding pattern)

**File 1**: `results/exp2/exp2_np9_rho015_alpha10/per_iter_burst_analysis.json`
- Keyed by iteration number (string). Each iteration contains per-platform data keyed by platform ID.
- For each platform: `classification`, `n_bursts`, `burst_steps` (array of step numbers), `burst_sizes` (array of community counts), `escalation_slope`, `platform_id`.
- **Selection criteria**: Pick an iteration and platform where `classification == "raiding_stable"` and `n_bursts` is between 4 and 8 (representative, not extreme). Prefer a platform with `escalation_slope > 0` to show the escalation pattern.

**File 2**: `results/exp2/exp2_np9_rho015_alpha10/dynamics/per_iter_raiding.json`
- Keyed by iteration number. Each iteration contains per-platform arrays keyed by platform ID.
- Each platform's value is a **100-element array** of net outflow per step (number of communities that LEFT the platform at each step). Positive = net departure.
- This is the "outflow series" — the raw trace from which bursts are detected.

**File 3**: `results/exp2/exp2_np9_rho015_alpha10/step_metrics.json`
- Keyed by iteration number. Each iteration contains a list of 100 step records.
- Each step record has: `step`, `avg_utility`, `per_governance_utilities` (dict with `algorithmic`, `coalition`, `direct`), `per_governance_community_count` (same dict), `per_type_utility` (dict with `mainstream`, `extremist`), `per_type_relocations`.
- Use this to overlay system-level context: governance-type community counts and utilities for the same iteration.

### Figure Layout

**Two vertically stacked panels** sharing the x-axis (simulation step 1–100):

**Top panel: Platform outflow series with burst markers**
- Plot the outflow series (100 values) as a grey step/area chart. This shows the net community departures per step.
- Overlay burst events as vertical red lines at the burst_steps, with height proportional to burst_size.
- Add small text labels showing burst size above each red line.
- y-axis: "Net community outflow"
- The visual: a spiky pattern — mostly near-zero with sharp positive spikes at burst events (the raids). If escalation is present, later spikes should be taller than earlier ones.

**Bottom panel: System-level governance community counts**
- From step_metrics for the same iteration, plot three lines showing `per_governance_community_count` for direct, coalition, and algorithmic over 100 steps.
- Use the same governance color palette (Direct="#D55E00", Coalition="#009E73", Algorithmic="#0072B2").
- Overlay the same burst event markers as faint vertical dashed red lines (connecting to the top panel visually).
- y-axis: "Community count"
- The visual: the direct line (orange) should show a sawtooth pattern — gradual rise (extremists accumulating) punctuated by sharp drops (the raids). Algorithmic (blue) should show the mirror: sharp rises at raid steps as communities arrive. Coalition (green) should be relatively flat.

**Annotations**:
- Title: `Platform biography: raiding cycle on a direct platform`
- Subtitle showing the selected iteration and platform ID, plus classification and escalation slope.
- Subtitle: `exp2_np9_rho015_alpha10 — iteration {X}, platform {Y}`

### Selection Logic

Write the selection logic in R:
1. Read per_iter_burst_analysis.json
2. For each iteration, for each platform, check: `classification == "raiding_stable"` AND `n_bursts >= 4` AND `n_bursts <= 8`
3. Among candidates, prefer platforms with `escalation_slope > 0`
4. From the filtered set, pick the **median** by n_bursts (not the most extreme) to avoid cherry-picking
5. Print the selected iteration + platform ID to console

### Output

Save as `fig_platform_biography.{pdf,png}`, width=10, height=7.

### Style

Use the same `theme_pub` as the other figures (theme_minimal, base_size=12, white background, grey90 grid). Same `save_fig` helper. Same colorblind-safe governance palette.

### Notes

- The per_iter_raiding outflow series may contain negative values (net inflow at some steps). This is expected — plot them as-is.
- The burst_steps are 0-indexed step numbers corresponding to indices in the outflow series array.
- The step_metrics steps are 1-indexed (`step` field starts at 1). Align by adding 1 to the outflow series index.
- This is a **new file** (generate_biography.R), not a modification of existing scripts. It should be self-contained with its own library imports and theme setup.
