# Results Narrative: Expected Outcomes and Metrics Mapping

## Purpose

This document maps the paper's argument structure to specific metrics, data
sources, and expected outcomes from the reporting pipeline. It is intended as
a reference for analyzing the full pipeline output and writing the results
section. It combines the framing-results analysis from collegial review with
validated single-iteration findings and cross-config patterns from the
Experiment 2 master summary.

---

## Paper Architecture

### The Argument in Three Acts

1. **The baseline works.** The Tiebout foot-voting mechanism translates to
   online platforms. Governance type shapes sorting efficiency; platform
   diversification improves welfare. (Experiment 1)

2. **Extremists break it in structurally predictable ways.** Parasitic
   communities exploit the sorting mechanism. The welfare cost depends on
   system structure (N_p), governance design, and extremist behavioral
   regime (α). The breaks are not random — they follow from the interaction
   of utility functions and institutional rules. (Experiment 2, cross-sectional)

3. **The breaks reveal something about platform system design.** The dynamics
   of extremist movement produce emergent raiding cycles that no individual
   community plans. Coalition governance creates natural firewalls. The
   direct–algorithmic interface is the system's primary vulnerability.
   (Experiment 2, dynamics)

### Target Audiences and What They Care About

- **Political science:** The Tiebout extension (does foot-voting work for
  platforms?), the formal interaction between system structure and parasitism,
  the ANOVA-backed claim about diversification premiums.
- **Political communication:** The ideologue/griefer behavioral distinction,
  the raiding cycle as a structural explanation for cross-platform extremist
  migration, the mapping to observed phenomena (Kiwi Farms → Twitter, etc.).
- **Platform studies:** Coalition enclaves as a case for community self-governance,
  the cold-start vulnerability, the policy implication that consolidation costs
  are endogenous to the threat environment.

### Plain-Text Takeaways (Ranked by Expected Impact)

1. **"The platform ecosystem is a system, not a collection of independent sites."**
   The raiding cycle is the killer illustration — a system-level emergent pattern
   that no single platform's moderation policy produces or can prevent alone.

2. **"Extremists exploit the interfaces between platform types."**
   They use small permissive platforms as staging grounds and algorithmically
   curated platforms as hunting grounds, exploiting the cold-start problem.

3. **"More platforms help everyone — but they help mainstream communities most
   when the extremist threat is worst."**
   The diversification premium grows with parasitism intensity. The cost of
   consolidation is endogenous to the threat environment.

4. **"Coalition governance works not because it's democratic, but because it
   produces enclaves."**
   Movement formation severs the parasitism channel by segregating types.

5. **"The ideologue/griefer distinction is behavioral, not typological."**
   The payoff structure makes communities behave as one or the other.
   Platform design choices that increase parasitism payoffs convert ideologues
   into griefers.

---

## Results Mapping: Metrics to Arguments

### Result 1: Governance type shapes sorting efficiency
**Section:** 3.1 (Experiment 1, compressed)
**Status:** Complete. Data in hand.

| Metric | Source | Value |
|--------|--------|-------|
| Homogeneous system utilities | exp1 summary | Algo 6.49, Coal 6.06, Dir 5.40 |
| Mixed-system per-capita utility (direct) | exp1 summary | 6.31 (N_p=3) → 7.61 (N_p=27) |
| Population shares by governance | exp1 summary | Algo 59–76%, Coal 19–20%, Dir 5–21% |
| Selection effect strength | exp1 summary | Dir holds 4.8% of communities at N_p=27 |

**Presentation:** One summary table (mixed system, by governance type and N_p).
Two pages maximum. This is the premise, not the contribution.

---

### Result 2: Platform diversification improves welfare
**Section:** 3.1 (Experiment 1, compressed)
**Status:** Complete.

| Metric | Source | Value |
|--------|--------|-------|
| System utility by N_p | exp1 summary | 6.20 → 6.42 → 6.63 |
| Utility SD by N_p | exp1 summary | 0.144 → 0.129 → 0.116 |
| Convergence pattern | convergence.json | CONVERGED within 10–15 steps |

**Presentation:** One line in text plus footnote on convergence. Sets up
the question: does this hold when some communities are parasitic?

---

### Result 3: Diversification premium grows with extremist threat
**Section:** 3.2 (Experiment 2, headline finding)
**Status:** Complete from master summary. ANOVA pending from pipeline.

| Metric | Source | Expected |
|--------|--------|----------|
| Norm mainstream utility, 3×3 table (N_p × α) at each ρ_e | exp2_master_summary.csv | In hand. Gap: 0.049 at α=2 → 0.098 at α=10 (ρ_e=0.15) |
| Difference-in-differences | Computed from table | 0.049 on normalized scale at ρ_e=0.15 |
| Two-way ANOVA: N_p × α interaction F-stat and p-value | exp2_anova_results.json | **PENDING.** Expected significant interaction (p < 0.001). This is the formal test of the paper's central claim. |
| Diversification premium across ρ_e levels | exp2_master_summary.csv | Premium should grow with ρ_e (more extremists = consolidation costs more) |

**Presentation:** The 3×3 interaction table at ρ_e=0.15 is the headline table.
Difference-in-differences in text. ANOVA in footnote. Repeat table at other
ρ_e levels in appendix or as supplementary panels.

**What could change:** If the ANOVA interaction term is not significant at
some ρ_e levels, the claim needs to be qualified ("at high extremist prevalence,
the diversification premium depends on parasitism intensity"). At ρ_e=0.05 the
interaction may be weaker because even high-α extremists are too few to produce
large effects.

---

### Result 4: Parasitism intensity produces distinct behavioral regimes
**Section:** 3.2 (Experiment 2)
**Status:** Complete from master summary.

| Metric | Source | Value |
|--------|--------|-------|
| Mainstream utility on direct platforms across α | exp2_master_summary.csv | 7.01 (α=2) → 5.65 (α=5) → 3.41 (α=10) at N_p=27, ρ_e=0.15 |
| Comparison to random baseline | Theoretical | 3.41 < 5.0 (random assignment) |
| Extremist direct/algorithmic utility ratio | exp2_master_summary.csv | 1.58 (α=2) → 1.67 (α=10) at N_p=9, ρ_e=0.10 |
| Regime boundary | Inferred from tables | Between α=5 and α=10 in most configs |

**Presentation:** The 3.41 number is the headline — mainstream communities on
direct platforms with griefer extremists are worse off than random. The
direct/algorithmic ratio table shows the growing advantage of unprotected
platforms. Frame as behavioral shift, not type difference.

**What could change:** Nothing — this result is entirely from cross-sectional
averages already in hand.

---

### Result 5: Coalition governance creates protective enclaves
**Section:** 3.2 (Experiment 2)
**Status:** Partially complete. Enclave aggregates pending from pipeline.

| Metric | Source | Expected |
|--------|--------|----------|
| Mainstream utility by governance across α | exp2_master_summary.csv | In hand. Coalition: 6.35→6.13 (Δ=0.22); Direct: 7.01→3.41 (Δ=3.60) |
| Mean coalition homogeneity | enclave_aggregate.json | **PENDING across 200 iterations.** Single iteration: 0.92–0.98 per platform |
| Fraction of steps enclaved (>0.9) | enclave_aggregate.json | **PENDING.** Single iteration: 74–92% |
| Settling step (time to stable enclave) | enclave_aggregate.json | **PENDING.** Single iteration: ~25–30 steps |
| Post-settling disruption rate | enclave_aggregate.json | **PENDING.** Single iteration: rare, quick recovery |

**Presentation:** The governance × α utility table is the cross-sectional evidence.
Enclave metrics are the mechanism evidence. One representative homogeneity time
series as a figure (Viz Task 4.4).

**What could change:** If enclave homogeneity is substantially lower when
aggregated across 200 iterations (perhaps some iterations produce messy
coalitions), the "natural firewall" framing needs to be softened to "tendency
toward enclave formation." The single-iteration data strongly suggests this
won't happen, but worth checking the distribution of mean_homogeneity across
iterations.

---

### Result 6: Extremists concentrate on direct platforms
**Section:** 3.2 (Experiment 2)
**Status:** Complete from master summary.

| Metric | Source | Value |
|--------|--------|-------|
| Fraction of extremists on direct platforms | exp2_master_summary.csv | ~50% across all 27 configs |
| Stability across ρ_e levels | exp2_master_summary.csv | 49%, 50%, 50% |
| Overrepresentation factor | Computed | 2.5–4× vs population share |

**Presentation:** Single number with breakdown showing stability. This is setup
for Result 7.

**What could change:** Nothing — purely cross-sectional, already computed.

---

### Result 7: Bursty raiding cycle
**Section:** 3.3 (Raiding Dynamics, own subsection)
**Status:** Partially complete. Single-iteration burst analysis validated.
Full pipeline aggregation pending.

| Metric | Source | Expected |
|--------|--------|----------|
| Burst detection rate | burst_aggregate.json | **PENDING.** Single iteration: 18/27 platforms (67%). ACF caught only 3/27. |
| Classification distribution | burst_aggregate.json | **PENDING.** Single iteration: 12 raiding_stable, 9 quiet, 2 raiding_base, 2 enclave, 2 active |
| Median burst size | burst_aggregate.json | **PENDING.** Single iteration: 31 |
| Median inter-burst interval | burst_aggregate.json | **PENDING.** Single iteration: 9 steps |
| Burst fraction (% of outflow in bursts) | burst_aggregate.json | **PENDING.** Single iteration: 94–100% on active platforms |
| **Escalation test** (mean slope, t-test) | burst_aggregate.json | **KEY PENDING RESULT.** If mean slope > 0 for high-α: "raids escalate." If null: "steady-state cycling." |
| Escalation fraction positive | burst_aggregate.json | **PENDING.** Determines narrative framing. |
| Burst heatmap across factorial | exp2_burst_master.csv | **PENDING.** Expect: amplitude ↑ with α, frequency ↑ with α, both modulated by N_p |

**Critical distinction — escalation determines the policy narrative:**
- If escalation is significant at high α: The raiding cycle is a runaway
  process. Policy implication: systems with griefer extremists face
  *accelerating* costs over time. Early intervention matters.
- If escalation is null: The raiding cycle is a steady-state cost of the
  system architecture. Policy implication: the damage is bounded and
  predictable, but cannot be eliminated without structural change.
- If escalation varies by config: The most interesting outcome. Map which
  configs show escalation (probably high α, low N_p) and which don't. The
  interaction between system structure and escalation would be a finding
  in its own right.

**Presentation:** Burst statistics in text. Burst heatmap as figure (Viz 4.3).
Escalation test result in a callout paragraph. ACF vs burst detection
comparison in footnote (methodological contribution for ABM modelers).

---

### Result 8 (NEW): Mainstream displacement following raids
**Section:** 3.3 (Raiding Dynamics, integrated with R7)
**Status:** Diagnostic written, not yet run on full data.

| Metric | Source | Expected |
|--------|--------|----------|
| Number of raid events per config | displacement_aggregate.json | **PENDING.** Expect dozens per iteration at high α |
| Mainstream utility delta (post-raid mean) | displacement_aggregate.json | **PENDING.** Expect negative — mainstream communities are hurt by arriving extremists |
| Fraction of events with negative delta | displacement_aggregate.json | **PENDING.** Expect >60% |
| Destination distribution (algo vs coalition) | displacement_aggregate.json | **PENDING.** Expect majority to algorithmic (it's the mass market) |
| Burst-displacement correlation | displacement_aggregate.json | **PENDING.** Expect negative (bigger raids → worse mainstream outcomes) |
| **Superposed epoch trajectory** | displacement_aggregate.json | **KEY PENDING RESULT.** The "average raid" figure. |
| Direct count drop at t=0 | superposed epoch | Expect sharp drop (extremists leaving) |
| Algorithmic count rise at t=0 | superposed epoch | Expect rise (extremists arriving) |
| Coalition count flat at t=0 | superposed epoch | Expect no change (enclave stability) |
| Mainstream utility dip at t=0 | superposed epoch | Expect dip on algorithmic, possible rise on direct |

**This result ties R5 and R7 together.** The superposed epoch shows the
full raid sequence: extremists depart direct → arrive algorithmic →
mainstream utility dips → mainstream communities exit algorithmic → some
flow to coalition (creating endogenous demand for community self-governance).
If the data shows mainstream communities flowing to coalition platforms
after raids, that closes the loop on the NARRATIVE.md suggestion.

**Presentation:** Superposed epoch plot is the primary figure (Viz 4.1).
Displacement statistics in text. This may be the single most impactful
visualization in the paper — it shows, in one figure, the entire
raiding-displacement-sorting mechanism.

---

## Sensitivity Analysis Plan

### What We Get From Existing Data

The factorial design already provides sensitivity information for three
parameters (N_p, ρ_e, α). The interaction contrasts from the factorial
tables plus the ANOVA interaction terms are the formal sensitivity tests
for the paper's central claims. No new runs needed.

### What Needs New Runs (Minimal Set)

| Parameter | Symbol | Baseline | Values | New Configs | Rationale |
|-----------|--------|----------|--------|-------------|-----------|
| Moving cost | μ | 0.05 | {0, 0.02, 0.10} | 3 | Most vulnerable to "you tuned this" criticism |
| Preference dimensionality | N_a | 10 | {5, 20} | 2 | Brackets the baseline; tests whether results depend on governance complexity |
| Number of coalitions | g | 5 | {2, 10} | 2 | Mechanism-specific: does enclave protection hold with fewer/more coalitions? |
| SVD groups | k | 10 | {5, 20} | 2 | Mechanism-specific: does algorithmic quarantine depend on grouping granularity? |

**Total: 9 new configs × 200 iterations = 1,800 runs.**

### What We Can Skip

- **t_max:** Convergence diagnostics from existing data show utility stabilizes
  by step 15. Second-half gains < 0.02. Can report this instead of re-running.
- **m (mutation rate), r (similarity radius):** Deep internals of coalition/algorithmic
  governance. No reviewer will ask.
- **Sobol indices:** Overkill for a theory-building ABM. Replace with factorial
  interaction contrasts (already computed) and ANOVA (in pipeline).
- **α × N_a interaction:** Peripheral to the paper's argument. Park in future work.

### Sensitivity Reporting Format

OAT results as a single table: for each parameter, percentage change in
mainstream utility, extremist utility, and burst rate when set to min/max.
Flag any qualitative reversals. Expect: none, but μ=0 may show different
dynamics (no friction → more churning → weaker enclaves?). That would be
an interesting finding, not a threat to robustness.

---

## Visualization Plan

### Figures for the Paper (Ranked by Priority)

1. **Superposed Epoch Plot** (Viz 4.1) — Primary displacement figure.
   Two panels: community counts and mainstream utility around the average
   raid. Data source: displacement_aggregate.json. **Highest impact-to-effort.**

2. **Burst Heatmap Grid** (Viz 4.3) — Factorial summary of raiding intensity.
   3×3 heatmaps at each ρ_e. Data source: exp2_burst_master.csv.
   **Requires full pipeline completion.**

3. **Platform Biography** (Viz 4.2) — One direct platform's 100-step life.
   **Requires per-platform step data (simulation code addition).**
   If data unavailable, substitute with annotated governance-level time series
   showing burst events overlaid on community count trajectories.

4. **Enclave Trajectory** (Viz 4.4) — Coalition homogeneity over time.
   One representative platform. Data source: enclave_aggregate.json or
   raw enclaves.json from a single iteration.

5. **Alluvial/Sankey Diagram** (Viz 4.5) — Flows between governance types.
   Approximated from net changes. Medium effort. **Consider for revision
   rather than initial submission.**

6. **Network Flow Diagram** — Aspirational. Requires simulation code change
   for directional flow logging. **Save for talks.**

### Tables for the Paper

1. Experiment 1 summary: mixed-system utility by governance type and N_p
2. Experiment 2 headline: normalized mainstream utility, N_p × α at ρ_e=0.15
3. Experiment 2 governance: mainstream utility by governance type across α
4. Experiment 2 extremist concentration: % on direct platforms
5. ANOVA results: F-statistics for N_p, α, and interaction at each ρ_e
6. Sensitivity: OAT percentage changes (appendix)

---

## Conditional Outcomes: What Changes Based on Pipeline Results

### Escalation test determines the dynamics narrative
- **Positive (mean slope > 0, p < 0.05 at high α):** "Griefer extremists
  produce escalating raids. The raiding cycle is a runaway process that
  worsens over time, implying early intervention is critical."
- **Null:** "Raids are bursty and costly but reach a steady state. The
  damage is bounded and predictable — a structural cost of the system
  architecture rather than a worsening crisis."
- **Config-dependent:** Most interesting. Map escalation significance across
  the factorial. If escalation appears only at high α and low N_p, that's
  a finding: "System diversification not only reduces raid magnitude but
  prevents raid escalation."

### Displacement direction determines the coalition narrative
- **Mainstream flows primarily to algorithmic after raids:** The mass-market
  platform absorbs displaced communities. Coalition platforms are protective
  but not a refuge. Framing: "Coalition governance prevents displacement;
  algorithmic governance absorbs it."
- **Mainstream flows to coalition after raids:** Communities seek protective
  governance after being raided. Framing: "Raids create endogenous demand
  for community self-governance." This closes the loop between R5 and R7
  and is the strongest version of the coalition finding.
- **Flows are mixed/noisy:** Report destination distribution as a statistic
  without strong narrative framing.

### Enclave aggregation determines the protection claim's strength
- **High homogeneity sustained across 200 iterations:** "Coalition platforms
  reliably form protective enclaves." Strong framing.
- **High mean but large variance:** "Coalition platforms tend toward enclave
  formation but the process is not guaranteed." Softer framing, note
  conditions under which enclaves fail.

### Burst heatmap determines whether raiding is α-driven or N_p-driven
- **α is the primary driver (burst amplitude scales with α, less with N_p):**
  Framing: "Parasitism intensity, not system structure, determines raiding
  severity." Policy: address extremist incentives.
- **N_p is the primary driver (more platforms → less intense raids):**
  Framing: "System diversification directly dampens raiding." Policy:
  prevent consolidation.
- **Both matter (α × N_p interaction in burst metrics):** Strongest result.
  Framing: "The raiding cycle's severity depends on both who the extremists
  are and how the system is structured."

---

## Data Pipeline Summary

### What exists now
- exp2_master_summary.csv (27 rows, all cross-sectional metrics)
- burst_analysis.py, displacement_diagnostic.py, analyze_stepwise.py (validated)
- Single-iteration burst analysis and enclave data for {27, 0.15, 10}
- Experiment 1 full results

### What the pipeline produces (REPORTING_PIPELINE_TASKLIST.md)
- Phase 1: Per-iteration convergence, burst, displacement, enclave JSONs
- Phase 2: Per-config aggregation of all four analysis types
- Phase 3: Master CSVs, factorial tables, ANOVA results
- Phase 4: Visualization-ready data extracts

### What needs a simulation code change
- Per-platform, per-step community counts by type (for platform biography
  and full network flow visualization). Lightweight addition gated by config
  flag. One row per platform per step: platform_id, step, n_mainstream,
  n_extremist, utility_mainstream, utility_extremist.
