# Implementation spec: staggered-relocation robustness test

Kanban: `t_aff5285a`
Reviewer gate: `needs-review:red-team`
Date: 2026-07-21

## 0. Source notes and model files

This spec is grounded in the following project notes/manuscript sections:

- Reviewer objection: `reviews/review-sim-2026-04-14/R1-identification.md`, especially lines 23-26 and 49-52. R1 flags simultaneous relocation as a potential artifact behind Result 7 and explicitly asks whether the finding survives staggered relocation.
- Meta-review synthesis: `reviews/review-sim-2026-04-14/meta-review.md`, especially lines 11-14 and 37-40. The meta-review elevates the staggered-relocation variant to the highest-priority sensitivity analysis.
- Current manuscript claim being hedged: `draft/manuscript/main.tex`, especially lines 44-50, 100-112, and 214-220. The paper currently claims that 94-100% of extremist outflow occurs in bursts and that the concentration-raid-retreat cycle is emergent rather than coordinated.
- Analysis strategy notes: `docs/STRATEGY.md`, especially lines 8-18 and 47-64, plus `docs/FEEDBACK_v2.md`, especially lines 5-11 and 17-27. These notes motivate distributional/dynamic diagnostics rather than mean-only summaries.
- Existing baseline implementation: `platform_abm/model.py`, especially lines 214-224 for step order and lines 317-344 for simultaneous two-phase relocation; `platform_abm/tracker.py`, lines 56-86 for relocation event logging; `platform_abm/burst_analysis.py`, lines 16-105 for burst metrics and lines 108-138 for platform classification.
- Experiment configuration surface: `experiments/configs/experiment_config.py`, lines 9-63 and 65-90.

## 1. Estimand / robustness target

This is an internal simulation-design robustness estimand, not an empirical causal estimand over real-world platforms.

Primary estimand:

> Among Experiment 2 mixed-institution extremist simulations, holding seeds, platform counts, extremist proportions, parasitism intensity, search rule, governance-update timing, and reporting pipeline fixed, what is the paired change in burst concentration when relocation decisions are changed from simultaneous two-phase updating to randomized staggered same-step updating?

Primary contrast:

- Treatment/perturbation: `relocation_update_order = "staggered"`.
- Baseline/control: `relocation_update_order = "simultaneous"`, which must remain the default behavior.
- Unit of paired comparison: simulation iteration seed within parameter cell `(n_platforms, rho_extremist, alpha)`.
- Main parameter cells: the 27 Experiment 2 cells: `n_platforms in {3, 9, 27}` x `rho_extremist in {0.05, 0.10, 0.15}` x `alpha in {2.0, 5.0, 10.0}`.

Primary outcome:

- `burst_concentration`: share of extremist outflow/relocations occurring in burst steps, where a burst is currently defined by `platform_abm/burst_analysis.py` as extremist outflow `>= 10` in a step.
- Report two forms:
  1. pooled cell-level concentration: `sum(burst_outflow) / sum(total_outflow)` over platform-iterations with any extremist outflow;
  2. iteration-level concentration: for each simulation iteration, `sum(burst_outflow) / sum(total_outflow)` over platforms, then aggregate mean/median and paired bootstrap confidence intervals across seeds.

Core inferential question:

> Does the 94-100% burst-concentration result remain substantively intact when same-step moves are no longer applied simultaneously?

## 2. Baseline outcome/model being challenged

The robustness test challenges Result 7's mechanism claim, not the entire ABM.

Current claim:

- Direct platforms show 94-100% of extremist outflow in burst events.
- Median burst size is approximately 31 extremist communities.
- Median inter-burst interval is approximately nine steps.
- The manuscript interprets this as an emergent concentration-raid-retreat cycle rather than coordinated strategy.

Why this is vulnerable:

- `MiniTiebout.step()` currently runs elections, then updates every community's utility/strategy, then applies all selected moves in a two-phase batch.
- In `MiniTiebout._step_relocation()`, all movers decide from the same pre-relocation state, their destinations are collected, and only then are moves executed. This means many communities can make identical leave/stay decisions before observing each other's departures.
- R1's concern is that mass bursts may partly reflect synchronous updating: if departures were processed sequentially, later communities would observe earlier departures' effects on platform composition and might not also leave.

Scope of test:

- The test isolates relocation timing. It should not change utility functions, platform governance rules, information restriction, preference generation, initialization, moving costs, stopping rule, or tracker/reporting definitions.

## 3. Treatment / perturbation definition

Add a config field:

```python
relocation_update_order: Literal["simultaneous", "staggered"] = "simultaneous"
```

The field should be available in:

- `ExperimentConfig` dataclass.
- `ExperimentConfig.to_params()` output passed into `MiniTiebout`.
- `ExperimentConfig.to_dict()` / `from_dict()` persistence.
- Any master summary/config JSON emitted by the runner.

### 3.1 Baseline: simultaneous two-phase relocation

Keep current default semantics:

1. Elections/governance update occur once at the start of the model step.
2. Governance state is tracked.
3. Every community updates utility and sets strategy against the same post-election, pre-relocation platform composition.
4. All movers choose candidate destinations.
5. All moves are executed after the move list is complete.
6. Tracker receives one relocation list for the step.

### 3.2 Robustness perturbation: randomized staggered same-step relocation

For `relocation_update_order = "staggered"`:

1. Elections/governance update still occur once at the start of the step.
2. Do not re-run elections, coalition votes, algorithmic grouping, or platform policy setting inside the relocation loop.
3. Create one seeded random permutation of communities per step using the model RNG (`self.random`), not Python's unseeded global RNG.
4. Each community gets exactly one relocation decision opportunity per step.
5. Immediately before a community's decision, recompute that community's current utility and strategy against the current platform composition, including earlier same-step moves.
6. If the community wants to move, run the existing search/destination selection logic, execute the move immediately, set `last_move_step = self.t`, and append the event to the step's move list.
7. Continue through the permutation; later communities see composition changes caused by earlier moves.
8. At the end of the loop, populate `_last_n_relocations`, `_last_relocations_by_type`, and `tracker.record_step(self.t, moves)` exactly as the baseline does.

Illustrative pseudo-code:

```python
def _step_relocation(self) -> None:
    order = getattr(self.p, "relocation_update_order", "simultaneous")
    if order == "simultaneous":
        return self._step_relocation_simultaneous()
    if order == "staggered":
        return self._step_relocation_staggered()
    raise ValueError(f"unknown relocation_update_order: {order}")

def _step_relocation_staggered(self) -> None:
    moves = []
    order = list(self.communities)
    self.random.shuffle(order)

    for community in order:
        community.update_utility()
        community.set_strategy()
        if community.strategy != Strategy.MOVE.value:
            continue

        old_platform = community.platform
        community.find_new_platform()
        new_platform = self.random.choice(community.candidates)
        if new_platform is old_platform:
            continue

        old_platform.rm_community(community)
        community.join_platform(new_platform)
        community.last_move_step = self.t
        new_platform.add_community(community)
        moves.append((community, old_platform, new_platform))

    self._record_relocation_summary(moves)
```

Implementation note: the existing `_step_update_utility()` will already have run for all communities before `_step_relocation()`. For staggered relocation this is acceptable but not sufficient; each community must be recomputed inside the staggered loop because earlier same-step moves change composition-dependent utility. Do not delete the existing pre-relocation utility update unless a broader refactor verifies that all downstream metrics and stop-condition behavior remain unchanged.

## 4. Comparison groups and time windows

Comparison groups:

- Main comparison: simultaneous versus staggered within each matched parameter cell and seed.
- Secondary grouping: direct, coalition, and algorithmic platform histories, because the challenged burst result is concentrated on direct platforms while displacement and enclave responses involve algorithmic and coalition platforms.

Time windows:

- Primary metrics use the full simulation horizon `t = 1..100`, matching Experiment 2.
- Burst detection uses the current full per-step extremist outflow series.
- For displacement/event-window diagnostics, use the existing superposed-epoch convention if already implemented; otherwise use a symmetric window of `[-5, +10]` steps around each burst. The pre-window establishes local baseline; the longer post-window captures recovery/re-sorting.
- For enclave settling, use the current full horizon plus explicit censoring at `t_max = 100`; report the share of coalition platform-iterations that never settle before the horizon.

## 5. Required covariates / simulation outputs

Minimum config fields that must be written to every run-level config JSON and master summary row:

- `relocation_update_order`
- `n_platforms`
- `rho_extremist`
- `alpha`
- `n_communities`
- `p_space`
- `mu`
- `coalitions`
- `mutations`
- `search_steps`
- `svd_groups`
- `initial_distribution`
- `tracking_enabled`
- `seed_base`
- `iteration` / realized `seed`

Minimum per-step/per-platform outputs needed:

- Relocation events: step, community id, community type, from platform id, to platform id, from institution, to institution. Current `RelocationTracker.record_step()` already supports this.
- Platform-level extremist outflow series by step. This is derivable from relocation events by grouping extremist departures by `from_platform_id` and step.
- Step metrics: total relocations, per-type relocations, average utility. Current `step_log` / `stepwise.csv` should cover this.
- Governance snapshots for coalition and algorithmic platforms. Current tracker captures coalition votes/winner and algorithmic group membership.
- Displacement diagnostics: burst-linked mainstream utility deltas and destination institution shares. If existing `per_iter_displacement.json` generation depends on baseline artifact names, extend it to include update order.
- Enclave diagnostics: mean homogeneity, share of steps above 0.90, settling step, disruption count, and censoring flag for no settlement by `t_max`.

Optional but valuable debugging output:

- `relocation_order` within step for a pilot only. Do not require this for production if it bloats outputs; the primary artifact only needs step-level event lists.

## 6. Expected diagnostics and deliverables

Primary table, one row per parameter cell:

| n_platforms | rho_extremist | alpha | burst_conc_sim | burst_conc_stag | paired_delta | 95% CI delta | class_shift | displacement_shift | enclave_shift | decision |
|---:|---:|---:|---:|---:|---:|---|---|---|---|---|

Appendix tables:

1. Burst aggregate table: burst concentration, burst rate, median burst size, median inter-burst interval, mean escalation slope, fraction positive escalation, and classification proportions by update order.
2. Displacement table: event count, destination shares, mean mainstream utility delta, negative-delta fraction, and burst-displacement correlation by update order.
3. Enclave table: mean homogeneity, share above 0.90, settling step distribution, disruption count, no-settlement fraction by update order.
4. Paired seed bootstrap table: mean paired delta and 95% percentile CI for the primary and secondary outcomes.

Expected plots:

- Paired dot/slope plot of burst concentration by cell: simultaneous and staggered connected within each cell.
- Heatmap of paired burst-concentration deltas over `n_platforms x alpha`, faceted by `rho_extremist`.
- Distribution/violin plots of iteration-level burst-concentration deltas, not platform-iteration pseudo-replication.
- Burst size and interval distributions under simultaneous versus staggered for representative cells: baseline `(9, 0.10, 5.0)`, worst case `(27, 0.15, 10.0)`, and turbulence case `(3, 0.15, 10.0)`.
- Optional event-window/spaghetti plot for displacement under both update orders.

Uncertainty rule:

- Use paired bootstrap over iteration seeds within each cell. Resample seeds, recompute simultaneous and staggered summaries for the resampled set, and take the percentile 95% interval for `staggered - simultaneous`.
- Do not use platform-iterations as the main bootstrap unit; platform histories within an iteration are coupled by the same population and same platform ecosystem.
- If baseline simultaneous outputs cannot be seed-matched, use independent bootstrap and label the comparison as unpaired/exploratory.

## 7. Production grid and compute expectations

Production staggered robustness grid:

- `experiment`: `exp2_staggered_relocation` or similarly explicit name.
- `institution`: `mixed`.
- `n_communities`: 900.
- `n_platforms`: 3, 9, 27.
- `p_space`: 10.
- `rho_extremist`: 0.05, 0.10, 0.15.
- `alpha`: 2.0, 5.0, 10.0.
- `mu`: 0.05.
- `coalitions`: 5.
- `mutations`: 3.
- `search_steps`: 10.
- `svd_groups`: 10.
- `initial_distribution`: `equal`.
- `tracking_enabled`: true.
- `relocation_update_order`: `staggered`.
- `t_max`: 100.
- `n_iterations`: 200.
- `seed_base`: same as simultaneous baseline, currently expected to be 42.

Total new production simulations: 27 cells x 200 iterations = 5,400 simulations.

Pre-production smoke requirement:

- Run 2-5 iterations for `(n_platforms=9, rho_extremist=0.10, alpha=5.0)` under both update orders.
- Verify that config JSON contains `relocation_update_order`, event logs are generated, burst analysis consumes both output directories, and paired comparison code produces a non-empty table.
- Do not use the smoke results to alter manuscript claims.

## 8. Decision rules for interpretation

Frame this as a non-inferiority/sensitivity test of the emergent-raiding signature under a plausible asynchronous update perturbation. It cannot prove empirical realism.

### Defend the current emergent-cycle language if all primary criteria pass

- Pooled staggered burst concentration remains at least 0.90.
- The paired bootstrap 95% CI lower bound for `staggered - simultaneous` is no worse than -0.05 for the main pooled/iteration-level estimate.
- At least 24 of 27 cells either have staggered burst concentration `>= 0.90` or a paired point-estimate drop no larger than 0.10.
- The baseline cell `(9, 0.10, 5.0)` does not show a substantively large attenuation.

Secondary mechanism checks should also remain directionally intact:

- Platform classifications remain mostly in raiding/enclave/absorber modes rather than shifting predominantly to quiet/active.
- Displacement diagnostics retain the same substantive direction; if current baseline indicates positive post-burst mainstream utility deltas in high-diversity cells, staggered results should not reverse that conclusion wholesale.
- Coalition enclave metrics remain high enough to support the buffering claim: mean homogeneity stays near the current 0.90 threshold region, and no-settlement/disruption rates do not erase the mechanism.

Allowed language if the test passes:

> The concentration-raid-retreat cycle is not solely an artifact of simultaneous relocation decisions; it persists under randomized staggered relocation, with burst concentration and associated displacement/enclave signatures remaining substantively similar across the tested factorial.

### Hedge the current claim if any primary failure occurs

Hedge if:

- Pooled staggered burst concentration falls below 0.90.
- The paired 95% CI permits a drop larger than five percentage points and many cells show point-estimate drops larger than ten points.
- More than three cells fall below 0.85, especially the baseline cell `(9, 0.10, 5.0)`.
- Burst classifications shift materially from raiding/enclave classes to quiet/active.
- Displacement or enclave signatures disappear even if burst concentration remains high.

Required hedged language if the test fails:

> The model can generate concentrated raiding bursts, but the strongest emergent-cycle language is conditional on the assumption that relocation decisions are effectively simultaneous within a period. Randomized staggered relocation attenuates the mechanism, so the paper should describe raiding cycles as update-order sensitive.

### Mixed outcome

If burst concentration remains high but displacement/enclave signatures weaken:

> Burst concentration is robust to staggered relocation, but the broader displacement/enclave interpretation is update-order sensitive. Defend the narrow burst result; hedge the full concentration-raid-retreat mechanism.

## 9. Assumptions and limitations

Assumptions:

- One randomized decision order per step is an adequate perturbation of simultaneity.
- Elections/governance updates remain periodic and simultaneous at the start of each step; only relocation timing is perturbed.
- Agents still have the same information restriction: destination choices use base utility rather than observed social composition.
- Matched seeds make the simultaneous and staggered runs comparable enough for paired differences, despite stochastic order draws consuming RNG streams differently after the perturbation.

Limitations:

- This robustness test only addresses synchronous relocation as an internal design dependence. It does not validate the model against empirical platform behavior.
- Randomized staggered relocation is one asynchronous rule. Other plausible rules, such as fixed activation order, Poisson clocks, limited daily move capacity, or strategic waiting, could produce different dynamics.
- Keeping governance updates fixed while staggering relocation deliberately isolates one mechanism, but it may be unrealistic for platforms whose governance state updates continuously.
- Burst threshold `>= 10` is inherited from the current analysis. If staggered relocation smooths out departures just below the threshold, include a threshold sensitivity appendix for thresholds 5, 10, and 15 before declaring the result failed.
- If `t_max = 100` censors high-alpha coalition settling under either update order, report censoring explicitly instead of treating unsettled runs as settled failures.

## 10. Missing hooks / implementation flags

The coder should verify or add the following hooks before production:

- `relocation_update_order` does not yet appear in the baseline config API on `main` unless another branch has already added it. Add it with default `"simultaneous"`.
- `_step_relocation()` currently implements simultaneous two-phase relocation only. Refactor into explicit simultaneous and staggered helpers while preserving default behavior.
- No current unit test appears to assert that same-step composition changes can alter a later mover's decision. Add one minimal test with a constructed small platform system.
- Reporting scripts must carry `relocation_update_order` into output directory names, config JSON, master summaries, and comparison tables.
- The paired bootstrap comparison script may not exist yet. Add it as a separate analysis artifact so manuscript tables are reproducible.
- If `per_iter_displacement.json` or enclave outputs are only generated for a subset of cells, extend the pipeline before making the secondary mechanism decision.

## 11. Minimal acceptance tests for implementation card

A coder should be able to implement against this checklist:

1. Default run with no new parameter produces identical behavior/schema to current simultaneous relocation.
2. Config with `relocation_update_order="staggered"` runs without errors and records the field in `config.json`.
3. Unit test demonstrates immediate same-step move application: after an early mover leaves, a later community's recomputed utility/strategy can differ from its pre-loop value.
4. Unit test demonstrates every community receives at most one relocation opportunity per step.
5. Smoke run for `(9, 0.10, 5.0)` x 2-5 iterations emits non-empty relocation events, burst aggregates, and a paired comparison table.
6. Production run produces 27 staggered cells x 200 iterations or explicitly documents any failed/censored cells.

## 12. Bottom line

The staggered-relocation robustness test is the decisive sensitivity check for the paper's strongest causal/mechanistic language. Passing it supports the claim that raiding bursts are not merely a synchronous-update artifact. Failing it does not invalidate the whole ABM, but it requires narrowing the paper's claim from "emergent structural property" to "emergent under simultaneous-period updating."