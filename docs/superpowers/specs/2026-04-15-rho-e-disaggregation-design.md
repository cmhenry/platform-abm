# Design: ρ_e Disaggregation (Ideologue / Griefer Mix)

**Date:** 2026-04-15
**Status:** Approved
**Scope:** Targeted supplementary experiment + minimal model/config changes to support heterogeneous extremist α.

## Motivation

Reviewer R1 (`reviews/review-sim-2026-04-14/R1-identification.md:37`) flagged that the paper distinguishes ideologue (low α) from griefer (high α) communities theoretically, but the experimental design varies α as a single scalar applied uniformly to all extremists. The composite ρ_e parameter therefore conflates the proportion of ideologues with the proportion of griefers: at α = 2 all extremists are ideologues, at α = 10 all are griefers. R1 requests disaggregated runs varying the ideologue–griefer composition at fixed ρ_e to strengthen the behavioral-regime claim in Result 4.

This design adds a targeted supplementary experiment (`exp2b`) plus the minimal model, config, and metrics changes required to support heterogeneous α.

## Experimental Design

**Grid (single slice):**

| Parameter | Value |
|---|---|
| ρ_e | 0.10 |
| N_p | 9 |
| α_ideologue | 2.0 |
| α_griefer | 10.0 |
| institution | mixed |
| f_g (griefer fraction) | {0.25, 0.50, 0.75} |
| iterations | 200 |

**3 new configs × 200 iterations = 600 runs.**

**Endpoint reuse at analysis time:**
- f_g = 0 ≡ existing `exp2_np9_rho010_alpha2`
- f_g = 1 ≡ existing `exp2_np9_rho010_alpha10`

No new runs needed for endpoints.

**Why this slice and not a factorial.** Crossing with N_p or with alternative (α_i, α_g) pairs triples or quadruples run count for a strengthening item, not a main experiment. The goal is to establish that f_g matters independently. The (α_i = 2, α_g = 10) pair represents the regime endpoints established in Result 4. If f_g effects turn out to vary systematically with N_p or with the α pair, that motivates a follow-up paper.

## Model Changes

### 1. Utility formula (`platform_abm/utility.py`)

**Extremist gain (unchanged semantics):**
```
u_extremist = u_base + community.alpha * (n_mainstream / total)
```
Each extremist's own α scales their own gain. Heterogeneity on the gain side is already expressed because `community.alpha` is per-agent.

**Mainstream loss (attacker-weighted, new):**
```
u_mainstream = u_base - (α_i · n_ideologue + α_g · n_griefer) / total
```
Composition now shows up symmetrically on both sides of the vampirism relation. A mainstream community next to 5% ideologues + 5% griefers loses less than one next to 10% griefers alone, which matches the theoretical distinction the paper already draws.

**Reduction to current behavior.** At f_g = 0, n_griefer = 0 and α_i = α (the scalar), so the formula reduces to `u_base - α · n_extremist / total` — identical to the current implementation.

### 2. Neighbor counts (`platform_abm/neighbors.py`)

`get_neighbor_counts()` returns:
```python
{
    "n_mainstream": int,
    "n_ideologue": int,
    "n_griefer": int,
}
```
`n_extremist = n_ideologue + n_griefer` remains derivable at call sites.

Implementation: switch the counting loop to dispatch on a new `community.subtype` field (see §3). No change to the `get_neighbors()` dispatch logic.

### 3. Community subtype field (`platform_abm/agents/community.py`)

Add:
```python
subtype: str  # "", "ideologue", or "griefer"
```

Initialized to `""` in `Community.setup()` (mainstream default). Assigned during `_setup_community_types` (see §4).

**Note:** subtype is bookkeeping only — it does not duplicate α, which remains the functional parameter.

### 4. Setup (`platform_abm/model.py`)

`_setup_community_types` is extended to split the selected extremist IDs into ideologues and griefers using `random.sample` weighted by `frac_griefer`:

```python
n_ext = len(extremists)
n_griefers = int(round(n_ext * frac_griefer))
griefer_ids = set(self.random.sample(extremists, n_griefers))

for comm_id in extremists:
    comm_sel = self.communities.select(self.communities.id == comm_id)
    comm_sel.type = CommunityType.EXTREMIST.value
    if comm_id in griefer_ids:
        comm_sel.subtype = "griefer"
        comm_sel.alpha = self.p.alpha_griefer
    else:
        comm_sel.subtype = "ideologue"
        comm_sel.alpha = self.p.alpha_ideologue
    # existing preference assignment unchanged
```

**Mainstream α is no longer load-bearing.** The attacker-weighted loss formula does not reference `community.alpha` for mainstream communities, so `Community.setup()` can continue setting `self.alpha = self.p.alpha` without semantic effect. No cleanup required.

### 5. Metrics (`platform_abm/metrics.py`)

`compute_extremist_metrics` gains two keys:

```python
{
    "average_extremist_utility": ...,      # unchanged
    "average_mainstream_utility": ...,     # unchanged
    "average_ideologue_utility": ...,      # new; omitted if n_ideologue == 0
    "average_griefer_utility": ...,        # new; omitted if n_griefer == 0
}
```

Guarded to avoid empty-select errors at endpoints (f_g = 0 → no griefers; f_g = 1 → no ideologues).

**Step-series per-subtype traces are deferred.** The end-of-run metrics are sufficient for Result 4's f_g sweep; per-step subtype dynamics would require model.py and analysis plumbing that is out of scope.

## Config Changes

### `experiments/configs/experiment_config.py`

Add three fields to `ExperimentConfig`:

```python
alpha_ideologue: float | None = None
alpha_griefer: float | None = None
frac_griefer: float = 0.0
```

**Resolution in `to_params()`:**
```python
alpha_i = self.alpha_ideologue if self.alpha_ideologue is not None else self.alpha
alpha_g = self.alpha_griefer if self.alpha_griefer is not None else self.alpha
```
Both resolved values are passed through to AgentPy params as `alpha_ideologue` and `alpha_griefer`. `frac_griefer` is passed through directly.

**Semantics of the existing `alpha` field.** `alpha` remains required and acts as the scalar fallback for both subtypes when the disaggregated fields are unset. This preserves the current meaning for exp1, exp2, oat, and interactions configs without any migration.

**Serialization:** `to_dict` / `from_dict` extended for the three new fields. `from_dict` accepts dicts that omit them (backward compat with existing serialized configs).

### `experiments/configs/builders.py`

New function:

```python
def build_exp2b_configs() -> list[ExperimentConfig]:
    """Experiment 2b: ρ_e disaggregation at fixed ρ_e=0.10, N_p=9.

    Varies griefer fraction f_g ∈ {0.25, 0.50, 0.75} with α_i=2, α_g=10.
    Endpoints f_g ∈ {0, 1} reuse exp2 runs at analysis time.

    3 configs × 200 iterations = 600 runs.
    """
```

Names: `exp2b_fg025`, `exp2b_fg050`, `exp2b_fg075`. Experiment tag `"exp2b"`.

### `experiments/run_exp2b.py`

Mirror of `run_exp2.py`: argparse, logging, calls `build_exp2b_configs()`, runs through `ExperimentRunner`, writes LaTeX tables to its own output directory.

## Backward Compatibility

Default values (`frac_griefer = 0.0`, `alpha_ideologue = alpha_griefer = None`) mean every existing config (exp1, exp2, oat, interactions) produces identical model state:

- All extremists get `subtype = "ideologue"` and `alpha = α` (the scalar fallback).
- `n_griefer = 0` in every neighbor count.
- Attacker-weighted loss reduces to `α · n_extremist / total`.
- Metrics for ideologues match current extremist metrics; griefer metrics are omitted.

No migration needed for any existing experiment or serialized config.

## Testing

**Unit tests (`tests/`):**
- `get_neighbor_counts` returns correct `{n_mainstream, n_ideologue, n_griefer}` for mixed platforms.
- Utility formula regression: at f_g = 0 and f_g = 1, outputs match current formula output bit-for-bit (deterministic seed).
- `_setup_community_types` at f_g ∈ {0, 0.25, 0.5, 0.75, 1} produces expected counts per subtype (within ±1 rounding).
- `ExperimentConfig.to_params()` with only `alpha` set produces the same params dict as today (backward-compat regression).
- `compute_extremist_metrics` emits all four keys at interior f_g, and the right three keys at endpoints.

**Integration:**
- Single-iteration exp2b smoke run at each f_g ∈ {0.25, 0.50, 0.75} — verify end-of-run metrics include both subtype utilities and that reported values are finite.

## Out of Scope

The following are explicitly deferred to keep the R&R supplement focused:

- **Tracker per-subtype relocation events.** `RelocationEvent` continues to carry only `community_type ∈ {mainstream, extremist}`. Revisit only if Result 4 analysis surfaces a within-platform story that requires subtype-level event logs.
- **Step-series per-subtype utility / relocation traces.** No per-step subtype dynamics plots in this pass.
- **Factorial exp2b × N_p × ρ_e.** Single slice at N_p = 9, ρ_e = 0.10 only.
- **Alternative (α_i, α_g) pairs.** Fixed at (2, 10).
- **Mainstream α cleanup.** `Community.setup()` still assigns `self.alpha = self.p.alpha`; the value is never read for mainstream under the new formula, so no behavioral effect.

## File Touch List

- `platform_abm/utility.py` — new attacker-weighted loss
- `platform_abm/neighbors.py` — per-subtype neighbor counts
- `platform_abm/agents/community.py` — `subtype` field
- `platform_abm/model.py` — `_setup_community_types` splits by `frac_griefer`
- `platform_abm/metrics.py` — subtype utilities in `compute_extremist_metrics`
- `experiments/configs/experiment_config.py` — three new fields + serialization
- `experiments/configs/builders.py` — `build_exp2b_configs()`
- `experiments/run_exp2b.py` — new runner
- `tests/` — unit + smoke coverage per §Testing
