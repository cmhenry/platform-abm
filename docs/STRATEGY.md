# Strategy: Analysis Rework

**Date:** 2026-03-25
**Status:** Active planning document

---

## Diagnosis

The paper has a solid model, a complete simulation infrastructure, and a well-specified theoretical framework. But the analysis strategy was built for the wrong paper. The current pipeline produces comparative statics (factorial ANOVA, mean utility tables, heatmaps of parameter sweeps) when the model's actual contribution is in its emergent dynamics and distributional properties. Reviewer feedback and internal assessment converge on the same conclusion: the interesting results are being averaged away.

### Three core problems

**1. The averaging problem.** The factorial design runs 200 iterations per config with random governance assignments and random initial placements, then reports means and standard errors. The ANOVA interaction terms are significant because n=200 gives enormous statistical power, but effect sizes are small — the entire normalized mainstream utility range across the factorial is ~0.135. Meanwhile, the *within-config heterogeneity* — which iterations produce raiding cycles, which produce quiet sorting, which produce continuous turbulence — is the real finding, and it's invisible in the current output. The superposed epoch averages a bimodal phenomenon (some raids produce large displacement, others are absorbed) into a mush. The escalation analysis reports a mean slope when the distribution of slopes is the story.

**2. The figure set is backwards.** The current lineup leads with parametric results (interaction heatmaps, escalation heatmaps) that show "things go up when you turn the dial" — which is baked into the utility function. The emergent results (governance divergence scissors, platform biographies, displacement regimes) are treated as supporting material. The figures that would most distinguish this paper from a standard parameter sweep are the ones that are noisiest, least polished, and least prominent.

**3. The framing-results mismatch.** The introduction promises an explanation of extremist *cycling* — a dynamic, temporal claim. Most results are cross-sectional comparisons of means. The dynamic results that exist (burst analysis, superposed epoch) have small mean effects and settle into structural equilibrium quickly, as the alluvial accidentally demonstrated. The paper promises dynamics but delivers a governance comparison.

### What works

- The model itself is well-designed. The Tiebout adaptation, the three governance types, the neighborhood topology, and the extremist utility function are all well-specified.
- The governance divergence finding (B3 scissors) is genuinely emergent — nothing forces extremists to concentrate on direct platforms; it's a selection equilibrium.
- The three displacement regimes (graceful absorption at N_p=27, oscillatory recovery at N_p=9, continuous turbulence at N_p=3) are qualitatively distinct and not parametrically determined.
- The displacement paradox (raids temporarily *improving* system welfare by dispersing concentrated extremists) is novel and counterintuitive.
- The coalition enclave mechanism is a selection equilibrium, not a parameter choice.

---

## Reframing

The paper should shift from "extremists break the system" to **"the Tiebout mechanism is remarkably resilient to parasitism, except under specific structural failure conditions."** This reframing:

- Honestly reflects the small effect sizes (0.135 utility range is evidence of resilience, not catastrophic damage)
- Makes the failure conditions (low N_p x high alpha on direct platforms) the policy-relevant finding
- Turns the displacement paradox from an awkward counterpoint into evidence of self-correction
- Reframes the governance divergence: direct platforms don't "collapse" — they *specialize*, shifting from boutique governance (high utility for matched communities at low alpha) to vulnerable governance (exploited by parasites at high alpha)

The three-part argument becomes:

1. **The baseline works** — Tiebout sorting translates to platforms; diversification improves welfare. (Compressed, ~2 pages)
2. **The system is resilient but has identifiable failure modes** — parasitism costs are small on average but structurally concentrated on direct-governance platforms in diverse ecosystems. The ideologue/griefer behavioral distinction emerges from the payoff structure, not from agent types.
3. **The dynamics reveal the failure mechanism** — emergent raiding cycles, displacement cascades, and enclave formation. The heterogeneity across iterations and platforms is the finding, not the mean.

---

## Analysis Pipeline Rework

### What needs to change

The current pipeline aggregates to means too early. The per-iteration JSON files (`per_iter_burst_analysis.json`, `per_iter_displacement.json`, `per_iter_enclave_analysis.json`) contain the raw distributional data, but it's never surfaced into figures or tables. The rework needs to:

1. **Extract per-iteration and per-platform distributions** from the existing per_iter_*.json files. No new simulations needed for this — the data exists.

2. **Produce distributional figures** instead of (or alongside) mean-based ones:
   - Violin/strip plots of per-platform escalation slopes across the factorial (replacing or supplementing the escalation heatmap)
   - Spaghetti plots of individual event traces behind the superposed epoch mean
   - Distribution of burst sizes and inter-burst intervals within factorial cells (marginals for the burst heatmap)
   - Distribution of enclave settling times and disruption frequencies

3. **Produce multi-biography panels** — systematically selected platform-iterations showing the range of dynamics (quiet, stable cycling, escalating, enclave), not just one cherry-picked median.

4. **Add per-platform per-step logging** to the simulation for the biography figure. The current stepwise tracking records governance-type aggregates, not individual platform stats. This requires a code change to the model's `_record_step_log()` method.

5. **Rework the ANOVA framing.** The ANOVA is still valid but should be supplemented with effect-size measures (eta-squared, Cohen's d for pairwise comparisons). The huge F-statistics with tiny effects need to be contextualized, not celebrated.

### What to cut

- **The alluvial/Sankey.** It shows governance-level stability, which undermines the dynamics argument. Report the finding in text ("structural equilibrium is reached by step 20; the raiding cycle operates within that structure") and drop the figure.
- **The 9-panel extremist concentration bar chart.** The finding (50% on direct) is simple; the figure is busy. The simplified overrepresentation ratio version is better.

### What to keep

- **B3 (governance divergence scissors)** — the paper's strongest figure. Reframe as "boutique to vulnerable."
- **B6 (superposed epoch)** — rework as spaghetti plot with individual traces.
- **B5 (enclave trajectory)** — add failure-case companion panel.
- **B4 (burst heatmap)** — keep but add marginal distributions.
- **Platform biography** — elevate to multi-panel centerpiece showing behavioral regime variety.

### New figures needed

1. **Slope distribution violin/strip plot** — per-platform escalation slopes across the factorial. Shows that 25% of platform-iterations escalate strongly while 75% are near zero. The *fraction that escalates* is the structural finding.
2. **Spaghetti epoch** — individual event traces (light grey) with mean overlaid (bold). Shows event-to-event variability.
3. **Multi-biography panel** — 3-4 systematically selected platform-iterations from distinct behavioral regimes.
4. **Marginal distributions** for burst heatmap cells — burst size distributions within selected cells to show within-cell heterogeneity.

---

## Simulation Changes Needed

### Required: Per-platform per-step logging

Add to `_record_step_log()` (or equivalent):
```
platform_id, step, n_mainstream, n_extremist, utility_mainstream, utility_extremist
```
This is lightweight (one row per platform per step) and enables both the platform biography and network flow diagrams. Gate behind a config flag to avoid bloating output for runs that don't need it.

### Required: Re-run specific configs with new logging

Only need to re-run a handful of representative configs with the per-platform logging enabled, not the full factorial. Target configs:
- `exp2_np27_rho015_alpha10` (worst case)
- `exp2_np27_rho015_alpha5` (moderate case)
- `exp2_np9_rho015_alpha10` (oscillatory regime)
- `exp2_np3_rho015_alpha10` (continuous turbulence)

A small number of iterations (20-50) with per-platform logging should suffice for the biography figures — we're showing representative trajectories, not computing population statistics.

### Optional: Fill the factorial gaps

The ρ_e=0.20 panel is entirely empty and N_p=6 is missing. These would strengthen the paper but aren't strictly necessary if the framing focuses on the three existing ρ_e levels. Decide based on time budget.

### Not needed: Sensitivity analysis

The archived NARRATIVE.md planned nine new configs varying moving cost, preference dimensionality, coalition count, and SVD groups. This can go in an appendix or a revision response. It's not blocking the current rework.

---

## Prompt Sequence for Pipeline Rework

The following Claude Code prompts should be executed in order. Each is a self-contained task.

### Prompt 1: Distributional extraction from existing per_iter JSONs

Write a Python script `results/extract_distributions.py` that reads the existing `per_iter_burst_analysis.json`, `per_iter_displacement.json`, and `per_iter_enclave_analysis.json` for all 27 exp2 configs and produces:

- `results/distributions/escalation_slopes.csv` — one row per platform-iteration with columns: config, iteration, platform_id, n_bursts, slope, classification. Filter to platforms with n_bursts >= 3.
- `results/distributions/burst_sizes.csv` — one row per burst event: config, iteration, platform_id, burst_step, burst_size.
- `results/distributions/displacement_events.csv` — one row per displacement event: config, iteration, event_step, mainstream_util_delta, destination, burst_size.
- `results/distributions/enclave_metrics.csv` — one row per coalition platform-iteration: config, iteration, platform_id, mean_homogeneity, settling_step, n_disruptions, recovery_time.

### Prompt 2: Per-platform per-step logging code change

Modify the model's step-logging method to optionally record per-platform community counts and utility by type at each step. Add a config flag `log_platform_detail: bool = False`. When enabled, write `platform_detail.csv` alongside `stepwise.csv` with columns: step, platform_id, governance_type, n_mainstream, n_extremist, utility_mainstream, utility_extremist.

### Prompt 3: Re-run representative configs with platform detail logging

Create a run script `experiments/run_biography_configs.py` that re-runs the four target configs (listed above) with `log_platform_detail=True` for 50 iterations each. Output to `results/exp2_detail/`.

### Prompt 4: Distributional visualization suite

Write R scripts (or extend existing ones) to produce:
- Violin plot of escalation slopes across the factorial
- Spaghetti superposed epoch with individual traces
- Multi-biography panel from platform_detail.csv data
- Burst size marginal distributions for selected heatmap cells

### Prompt 5: Effect size computation

Write a Python script that computes eta-squared for each ANOVA term, Cohen's d for key pairwise comparisons (e.g., direct vs. coalition at alpha=10), and bootstrap confidence intervals for the diversification premium difference-in-differences.

---

## Archived Materials

Previous analysis notes, tasklists, visualization concepts, and Claude Code prompts have been moved to `docs/archive/`. These contain useful detail on the existing pipeline and figure specifications but reflect the old comparative-statics strategy. Reference them for technical specifics; don't follow their narrative framing.

Key archived files:
- `RESULTS_INTERPRETATION.md` — result-by-result analysis with data values (still accurate)
- `REPORTING_PIPELINE_TASKLIST.md` — full pipeline specification (technically sound, strategically outdated)
- `VIZ_EVALUATION.md` — per-figure assessment (status info still useful)
- `VISUALIZATION_CONCEPTS.md` — original viz design concepts
- `NARRATIVE.md` — framing analysis and audience mapping

## Open Questions

1. **How far to push the resilience reframing?** "Resilient with failure modes" is more honest than "extremists break everything," but the introduction currently promises a breaking story. The intro will need significant revision.

2. **How many biography panels?** Three (one per regime) is clean; four (adding an enclave case) is more complete but risks overwhelming the reader with single-iteration figures.

3. **Should the ρ_e=0.20 / N_p=6 gaps be filled?** The current three ρ_e levels tell the story, but missing cells look bad in factorial tables. Time cost: ~24 hours of compute for the full set.

4. **Where does the alluvial finding go?** "Structural equilibrium by step 20" is worth reporting. A single sentence in the dynamics subsection, or a footnote?
