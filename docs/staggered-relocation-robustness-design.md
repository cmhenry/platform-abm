# Design memo: staggered-relocation robustness variant

Kanban: `t_ba9d4307`
Reviewer gate: `needs-review:css-pi`
Date: 2026-07-21 UTC

## 1. Robustness estimand / target

This is not a new causal estimand over real-world units. It is a design-sensitivity estimand for the simulation mechanism:

> Holding the Experiment 2 mixed-institution extremist factorial, seeds, search rule, governance timing, and reporting pipeline fixed, how much does changing relocation from simultaneous two-phase updating to randomized sequential same-step updating change the simulated concentration of extremist outflow into burst events, and do the associated displacement and enclave signatures remain directionally intact?

Primary target:

- `burst_concentration`: share of extremist outflow/relocations that occurs in burst steps. Operationally, compute per platform-iteration using the current burst threshold (`outflow >= 10`) and aggregate as both (a) pooled `sum(burst_outflow) / sum(total_outflow)` and (b) mean/median of platform-iteration `burst_fraction`.
- The paper-level robustness question is whether the reported 94-100% burst concentration is reproduced under staggered relocation closely enough to rule out a purely synchronous-update artifact.

Secondary targets:

- `displacement`: whether burst events still coincide with mainstream/community displacement and comparable destination patterns.
- `enclave_formation`: whether coalition-platform extremist enclaves still form and remain stable.

## 2. Algorithm change relative to current simultaneous Step 5

### Current baseline: simultaneous two-phase relocation

Current `MiniTiebout.step()` runs:

1. elections / governance update,
2. governance tracking snapshot,
3. utility + strategy update for every community,
4. relocation.

The current relocation implementation is intentionally order-invariant within a step:

```python
moves = []
for community in self.communities:
    if community.strategy == Strategy.MOVE.value:
        community.find_new_platform()
        new_platform = self.random.choice(community.candidates)
        moves.append((community, community.platform, new_platform))

for community, old_platform, new_platform in moves:
    old_platform.rm_community(community)
    community.join_platform(new_platform)
    community.last_move_step = self.t
    new_platform.add_community(community)
```

All communities decide from the same pre-relocation platform composition and all moves are applied only after the full move set is collected.

### Proposed robustness variant: randomized sequential same-step relocation

Add a configuration flag, e.g. `relocation_update_order`, with default `"simultaneous"` and robustness value `"staggered"`.

For `"staggered"`, replace only Step 5 relocation timing. Do not otherwise change institutional rules, search costs, preference generation, extremist parameters, initialization, or reporting.

Pseudo-code:

```python
def _step_relocation_staggered(self):
    moves = []
    order = list(self.communities)
    self.random.shuffle(order)       # seeded AgentPy/model RNG; one random permutation per step

    for community in order:
        # Recompute this community's state against current, already-updated
        # within-step platform composition.
        community.update_utility()
        community.set_strategy()

        if community.strategy == Strategy.MOVE.value:
            old_platform = community.platform
            community.find_new_platform()
            new_platform = self.random.choice(community.candidates)

            if new_platform is not old_platform:
                old_platform.rm_community(community)
                community.join_platform(new_platform)
                community.last_move_step = self.t
                new_platform.add_community(community)
                moves.append((community, old_platform, new_platform))

    self._last_n_relocations = len(moves)
    self._last_relocations_by_type = count_by_type(moves)
    if self.tracker is not None:
        self.tracker.record_step(self.t, moves)
```

Design constraints:

- Every community receives exactly one relocation decision opportunity per model step.
- Communities later in the random order see platform composition after earlier moves in the same step; earlier communities do not reconsider until the next step.
- Elections/governance updates are still run once at the start of the step. Do not re-run elections, coalition votes, algorithmic regrouping, or platform policy setting mid-relocation; otherwise the robustness test would conflate relocation timing with governance timing.
- Utility/search is recomputed immediately before each community's decision so same-step moves can affect composition-dependent utility/search terms.
- Use the existing seeded model RNG for the order permutation so results remain reproducible and seed-pairable with the simultaneous baseline.
- Keep tracker semantics as one relocation list per step. If later auditing needs update-order diagnostics, record an optional `relocation_order` field, but do not make that a requirement for the first HPC run.

Minimal implementation touch points for the follow-up card:

- Add `relocation_update_order` to `ExperimentConfig` and `to_params()` / `to_dict()`.
- Add validation/defaulting in the simulation config layer if one exists for AgentPy params.
- Dispatch in `MiniTiebout._step_relocation()` to simultaneous vs staggered helper.
- Add unit tests that prove staggered relocation recomputes strategy after an earlier same-step move and applies moves immediately, while the default remains simultaneous.
- Add a grid builder for the staggered robustness configs.

## 3. Outcome metrics and comparison table layout

Use the existing reporting pipeline wherever possible, but the robustness table should compare simultaneous and staggered runs at matched parameter cells and, ideally, matched seeds.

### Primary burst metrics

For each cell `(n_platforms, rho_e, alpha)` and update order:

- `burst_concentration_pooled`: `sum(burst_outflow) / sum(total_outflow)` across all platform-iterations with extremist outflow.
- `burst_fraction_mean` and `burst_fraction_median`: mean/median of platform-iteration burst fractions.
- `burst_rate`: current aggregate share of platform-iterations classified as non-quiet.
- `burst_size_median`, `burst_interval_median`.
- `escalation_mean_slope`, `escalation_fraction_positive`, and `escalation_p_value`, retaining the current guardrail that escalation slopes are only pooled from platforms with at least three bursts.
- Classification proportions: `class_raiding_base`, `class_raiding_stable`, `class_enclave`, `class_absorber`, `class_active`, `class_quiet`.

### Displacement metrics

For each cell and update order:

- `n_iterations_with_events` and `total_events`.
- `destination_algorithmic_mean`, `destination_coalition_mean`.
- `mainstream_util_delta_mean` around burst/displacement events.
- `mainstream_util_delta_negative_frac`.
- `burst_displacement_corr_mean`.

### Enclave metrics

For each cell and update order:

- `enclave_mean_homogeneity`.
- `enclave_settling_step`.
- `fraction_disrupted`.
- If cheap to add from existing per-platform enclave data: `fraction_enclaved` = share of coalition platform-steps or platform-iterations with homogeneity above 0.90.

### Main paper/appendix table shell

One row per parameter cell; paired deltas should be computed as staggered minus simultaneous.

| n_platforms | rho_e | alpha | C_burst sim | C_burst stag | Delta C | 95% CI Delta C | burst class shift | disp delta | enclave delta | decision |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| 3 | 0.05 | 2 |  |  |  |  |  |  |  | defend / hedge |
| ... | ... | ... |  |  |  |  |  |  |  |  |

Suggested appendix tables:

1. Burst table: all burst aggregate columns for simultaneous vs staggered.
2. Displacement table: all displacement aggregate columns for simultaneous vs staggered.
3. Enclave table: all enclave aggregate columns for simultaneous vs staggered.
4. Classification table: update-order differences in platform behavior classes.

Uncertainty:

- Prefer paired seed-level bootstrap within each cell: resample iteration seeds with replacement, recompute simultaneous/staggered difference, report percentile 95% CI.
- If paired seed outputs are unavailable for the existing baseline, use independent bootstrap by update order and mark the comparison as unpaired.
- Avoid treating platform-iterations as independent for the main uncertainty interval; bootstrap at the simulation-iteration level because platform histories within an iteration are coupled.

## 4. Parameter grid and iteration count

Run the staggered robustness grid as the same factorial as Experiment 2:

Fixed parameters:

- `experiment`: use a distinct label such as `exp2_staggered_relocation`.
- `institution`: `mixed`.
- `n_communities`: 900.
- `p_space`: 10.
- `t_max`: 100.
- `mu`: 0.05.
- `coalitions`: 5.
- `mutations`: 3.
- `svd_groups`: 10.
- `search_steps`: 10.
- `initial_distribution`: `equal`.
- `tracking_enabled`: true.
- `relocation_update_order`: `staggered`.

Factorial dimensions:

- `n_platforms`: 3, 9, 27.
- `rho_extremist`: 0.05, 0.10, 0.15.
- `alpha`: 2.0, 5.0, 10.0.

Iteration count:

- Production: 200 iterations per cell, matching Experiment 2.
- Total new runs: 27 cells x 200 = 5,400 staggered simulations.
- Use the same seed schedule as the simultaneous baseline (`seed_base + iteration`, with `seed_base = 42` unless the baseline used a different stored seed base) to enable paired comparisons.

Best-effort pilot before HPC:

- A local or short-HPC smoke grid may run 2-5 iterations for one representative cell, only to validate artifacts and schema compatibility.
- A reduced inferential pilot, if Colin wants one before the full run, should use at least 25-50 paired iterations for the baseline cell `(n_platforms=9, rho_e=0.10, alpha=5.0)`. Do not use that pilot to make the paper claim; use it only to de-risk the production run.

## 5. Decision rule for paper language

The robustness test should be framed as non-inferiority of the emergent-raiding signature under a plausible asynchronous update perturbation, not as proof that every update order yields identical dynamics.

### Defend emergent-cycle language if all are true

Primary burst concentration:

- Pooled staggered `burst_concentration` remains at least 0.90, and
- the paired pooled difference `staggered - simultaneous` has a 95% bootstrap lower bound no worse than -0.05, and
- at least 24 of 27 cells have staggered point-estimate `burst_concentration >= 0.90` or a paired drop no larger than 0.10.

Secondary mechanism signatures:

- The dominant burst classifications remain raiding/enclave/absorber rather than shifting mostly to quiet/active.
- Displacement metrics retain the same direction: burst-linked mainstream utility deltas do not flip sign toward improvement, and destination shares remain qualitatively similar rather than showing a new dominant sink.
- Enclave metrics remain consistent with coalition enclave formation: mean homogeneity stays high, settling remains present in most cells, and disruption does not increase enough to erase enclaves.

Allowed paper language if this passes:

> The emergent raiding cycle is not an artifact of simultaneous relocation decisions; it persists under randomized sequential relocation, with burst concentration and associated displacement/enclave signatures remaining substantively similar.

### Hedge emergent-cycle language if any primary failure occurs

Hedge if:

- pooled staggered `burst_concentration < 0.90`, or
- the paired 95% CI permits a drop larger than 5 percentage points and many cells show drops larger than 10 points, or
- more than 3 of 27 cells fall below 0.85, especially around the baseline cell `(9, 0.10, 5.0)`, or
- burst classifications shift materially from raiding/enclave classes to quiet/active, or
- displacement/enclave signatures disappear even if some burst concentration remains.

Hedged paper language:

> The model can generate concentrated raiding bursts, but the strongest emergent-cycle language is conditional on the timing assumption that relocation decisions are effectively simultaneous within a period. Randomized sequential relocation attenuates or qualifies the mechanism.

### Mixed outcome

If burst concentration survives but displacement or enclave formation weakens, defend only the narrow burst-concentration result and hedge the broader mechanism:

> Burst concentration is robust to sequential relocation, but the coupled displacement/enclave interpretation is update-order sensitive.

## 6. Analysis/reporting guardrails

- Keep simultaneous as the default behavior to preserve baseline reproducibility.
- Name outputs so update order cannot be confused, e.g. `exp2_staggered_np9_rho010_alpha5`.
- Store update order in every config JSON and master summary row.
- Report both absolute staggered values and paired deltas vs baseline; do not rely on side-by-side point estimates alone.
- Do not interpret this robustness check as empirical identification. It only addresses internal simulation-design dependence on synchronous relocation.
- If HPC resources force a reduced run, label it exploratory and require Colin/css-pi approval before using it to alter manuscript language.

## 7. Handoff to implementation/HPC card

The implementation/HPC card (`t_58e2a8ce`) should implement the config flag, tests, grid builder, and production run using this memo as the design contract. The output required for manuscript review is a comparison bundle with:

- `staggered` run artifacts for all 27 cells,
- updated master CSVs including update order,
- paired bootstrap CI table,
- a one-page interpretation note applying the decision rule above.
