# Claude Code Prompt: Fix Superposed Epoch Figure (B6)

## Problem

The current `fig_superposed_epoch` is flat — no visible displacement signal. The R script (`generate_phase_b67.R`) reads `displacement_aggregate.json` and plots the raw `direct_count_mean`, `coalition_count_mean`, etc. around the raid event window (t = -8 to +8). Because each iteration has a different baseline level, averaging raw counts across 200 iterations washes out the event-locked signal.

## Root Cause

The aggregation in `displacement_aggregate.json` averages raw per-governance community counts across iterations without first subtracting each iteration's pre-event baseline. The per-iteration signal is present in the data — baseline normalization reveals statistically significant displacement patterns.

## Fix: Two-Part

### Part 1: Recompute the aggregate epoch with baseline normalization

Write a Python script `results/viz/recompute_epoch.py` that:

1. For each of these configs, reads `per_iter_displacement.json`:
   - `exp2_np27_rho015_alpha5`
   - `exp2_np9_rho015_alpha10`
   - `exp2_np3_rho015_alpha10`

2. For each iteration that has events (`n_events > 0`), extracts the per-iteration `superposed_epoch` arrays: `direct_count_mean`, `coalition_count_mean`, `algorithmic_count_mean`, `mainstream_util_mean`, `extremist_util_mean`.

3. Baseline-normalizes each iteration: for each metric, subtract the mean of steps t=-8 through t=-4 (the first 5 values, indices 0–4) from all 17 values. This gives Δ (change from pre-event baseline) per iteration.

4. Averages the baselined deltas across iterations. Computes SE as `std / sqrt(n_iterations)`.

5. Writes the output to `results/viz/epoch_baselined_{config_name}.json` with this structure:

```json
{
  "config": "exp2_np9_rho015_alpha10",
  "n_iterations": 200,
  "relative_steps": [-8, -7, ..., 8],
  "direct_count_delta_mean": [...],
  "direct_count_delta_se": [...],
  "coalition_count_delta_mean": [...],
  "coalition_count_delta_se": [...],
  "algorithmic_count_delta_mean": [...],
  "algorithmic_count_delta_se": [...],
  "mainstream_util_delta_mean": [...],
  "mainstream_util_delta_se": [...],
  "extremist_util_delta_mean": [...],
  "extremist_util_delta_se": [...]
}
```

### Part 2: Rewrite the B6 figure in `generate_phase_b67.R`

Replace the current B6 section (lines 72–197 of `generate_phase_b67.R`) with code that:

1. Reads the three `epoch_baselined_*.json` files from `results/viz/`.

2. Produces a **3-row × 2-column figure** (using `patchwork`):
   - Each **row** is one config: N_p=3/α=10 (top), N_p=9/α=10 (middle), N_p=27/α=5 (bottom)
   - **Left column**: Δ community count by governance type (3 lines: direct, coalition, algorithmic)
   - **Right column**: Δ mainstream utility (1 line, black)
   - All panels share x-axis: "Relative step (t = 0 at raid)"
   - Left y-axis: "Δ community count from baseline"
   - Right y-axis: "Δ mainstream utility from baseline"

3. Visual specifications:
   - Shaded ±1 SE bands (alpha = 0.15) around each line
   - Vertical dashed line at t = 0
   - Horizontal dashed line at Δ = 0 (the baseline)
   - Color palette: Direct = "#D55E00", Coalition = "#009E73", Algorithmic = "#0072B2", Mainstream utility = "#333333"
   - Row labels as facet strip text: `N_p = 3, α = 10`, `N_p = 9, α = 10`, `N_p = 27, α = 5`
   - Overall subtitle: `ρ_e = 0.15 — baselined to pre-event mean (t = -8 to -4)`
   - Legend at bottom, shared across all panels

4. Save as `fig_superposed_epoch.pdf` and `fig_superposed_epoch.png` (300 DPI), width = 10, height = 10.

## Expected Patterns in the Corrected Figure

These patterns have been verified from the raw data — use them to validate that the code works:

**N_p=27, α=5 (bottom row — mildest case):**
- Left panel: At t=0, coalition drops ~2, algorithmic rises ~2. Post-event (t=+1 onward), direct drops ~0.6 and stays down, algorithmic rises ~0.8. Small, clean signal.
- Right panel: Utility jumps +0.02 at t=+1 and sustains at ~+0.022 through t=+8.

**N_p=9, α=10 (middle row — strong parasitism):**
- Left panel: At t=0, coalition drops ~3.8, algorithmic rises ~3.7. Direct barely moves. The raid displaces communities FROM coalition TO algorithmic.
- Right panel: Utility jumps +0.045 at t=+1, oscillates between +0.035 and +0.050 through t=+8.

**N_p=3, α=10 (top row — continuous raiding):**
- Left panel: Wild oscillation in all three governance types (±5 communities per step). Direct swings +5.8 at t=+1 then -1.5 at t=+2. No stable displacement — the system is in continuous turbulence.
- Right panel: Utility drops -0.04 at t=-1, then jumps +0.12 at t=+1, then oscillates. Large, noisy signal — this is the "continuous raiding" regime where no stable equilibrium exists between events.

## File Locations

- Input per-iteration data: `results/exp2/{config}/per_iter_displacement.json`
- Output baselined JSON: `results/viz/epoch_baselined_{config}.json`
- R script to modify: `results/viz/generate_phase_b67.R` (replace B6 section, lines 72–197; keep B7 section unchanged)
- Output figures: `results/viz/fig_superposed_epoch.{pdf,png}`

## Notes

- The B7 section of generate_phase_b67.R (extremist concentration, lines 200–292) should remain unchanged.
- The `generate_phase_b.R` script (B1–B5) is separate and should not be modified.
- Run the Python script first (`python results/viz/recompute_epoch.py`), then the R script (`Rscript results/viz/generate_phase_b67.R`).
- All paths are relative to the project root (`platform-abm/`).
