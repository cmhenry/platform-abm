# platform-abm

Agent-based model of community sorting across competing platform governance institutions, implemented in Python with [AgentPy](https://agentpy.readthedocs.io/).

## Overview

This repository contains the simulation code and experiment harness for a generalized Tiebout model of jurisdictional sorting applied to online platforms. Communities (modeled as unitary agents with policy preferences) choose among platforms that differ in governance institution. Three governance institutions are implemented — direct democracy, coalition-based representation, and algorithmic curation — and may coexist within a single environment. An optional extremist population introduces asymmetric externalities via a vampirism mechanism, producing raiding and enclave dynamics.

The model and experiments are designed to investigate two primary questions: (1) how the choice of governance institution affects preference–policy alignment and sorting efficiency, and (2) how the presence of extremist communities interacts with institutional design, platform supply, and vampirism intensity to produce emergent migration patterns.

## Model description

### Theoretical framework

The model extends the Tiebout (1956) hypothesis — that mobile agents sort efficiently across jurisdictions offering differentiated public-goods bundles — to the setting of platform governance. Each simulation step proceeds in four phases: election (platforms update policies), utility evaluation (communities assess current fit), search (communities identify candidate destinations), and relocation (dissatisfied communities move).

### Preference and policy space

Preferences and policies are represented as binary vectors of length `p_space`. The base utility of a community on a platform is the Hamming similarity between the community's preference vector and the platform's policy vector (i.e., the count of matching dimensions), yielding an integer on [0, `p_space`].

### Governance institutions

Three institution types govern how platforms update their policy vectors each step:

**Direct democracy.** Each policy dimension is set by majority vote among all resident communities. If the count of communities preferring 0 on dimension *i* is below ⌊0.5 * *N*⌋, the bit is flipped.

**Coalition representation.** *K* coalitions are generated as random binary policy vectors and refined via hill-climbing (a genetic search with `mutations` bit flips per iteration over `search_steps` iterations, maximizing summed community fitness). Communities vote for their nearest coalition. The plurality winner's policy vector becomes the platform policy. Ties are broken randomly.

**Algorithmic curation.** Communities are clustered via K-means (scikit-learn, *k* = min(*N*, `svd_groups`)). For each cluster, a set of `COLD_START_BUNDLE_COUNT` = 5 candidate policy bundles is generated and rated by member communities. The highest-rated bundle per cluster becomes that group's personalized policy. Unlike the other institutions, algorithmic platforms present heterogeneous policies — a community's relevant policy is determined by its cluster assignment.

**Mixed environment.** When `institution = "mixed"`, platforms are assigned institutions in round-robin order (direct, coalition, algorithmic), ensuring equal representation when the platform count is divisible by three.

### Utility with vampirism

When extremists are present, the full utility function incorporates a proportional vampirism term:

For mainstream communities:

    u = u_base - alpha * n_ext / (n_ext + n_main)

For extremist communities:

    u = u_base + alpha * n_main / (n_main + n_ext)

where `u_base` is the Hamming similarity, `alpha` is the vampirism intensity, and `n_main` / `n_ext` are counts of mainstream and extremist neighbors (excluding self). The denominator guards against division by zero. The neighbor set is institution-dependent: all residents for direct democracy, winning-coalition voters for coalition platforms, and cluster members for algorithmic platforms.

### Search and relocation

Communities evaluate their current platform with full utility (including vampirism) but can only observe base utility (Hamming similarity) on candidate destinations — modeling the information asymmetry that agents cannot observe the social composition of a platform before joining. A community relocates if the best destination's base utility exceeds its current full utility plus a moving cost of `mu * p_space`. Ties among equally attractive destinations are broken randomly.

## Agent specification

### Community agents

| Attribute | Type | Description |
|-----------|------|-------------|
| `preferences` | `NDArray[int]` | Binary preference vector of length `p_space` |
| `type` | `str` | `"mainstream"` or `"extremist"` |
| `platform` | `Platform` | Current platform reference |
| `current_utility` | `float` | Most recently computed utility |
| `strategy` | `str` | `"stay"`, `"move"`, or `""` (unset) |
| `moves` | `int` | Cumulative relocation count |
| `last_move_step` | `int` | Step of most recent relocation |
| `alpha` | `float` | Individual vampirism parameter |
| `group` | `int \| str` | Cluster assignment (algorithmic platforms) |

### Platform agents

| Attribute | Type | Description |
|-----------|------|-------------|
| `institution` | `str` | `"direct"`, `"coalition"`, or `"algorithmic"` |
| `policies` | `NDArray` | Current policy vector (or array of group policies) |
| `communities` | `list` | Resident community agents |
| `group_policies` | `dict` | Group-specific policies (algorithmic only) |
| `coalition_votes` | `list` | Per-community vote indices (coalition only) |
| `winning_coalition_index` | `int \| None` | Winning coalition (coalition only) |

## Parameters

### Simulation parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_comms` | Number of community agents | — |
| `n_plats` | Number of platform agents | — |
| `p_space` | Dimensionality of binary preference/policy space | 10 |
| `steps` | Maximum simulation steps per iteration | 50 |
| `institution` | Governance type: `direct`, `coalition`, `algorithmic`, `mixed` | — |
| `extremists` | Enable extremist subpopulation | `false` |
| `percent_extremists` | Percentage of communities assigned as extremists | 5 |
| `alpha` | Vampirism intensity | 1.0 |
| `mu` | Moving cost multiplier (cost = `mu * p_space`) | 0.05 |
| `coalitions` | Number of competing coalitions per election | 3 |
| `mutations` | Bit flips per hill-climbing mutation | 2 |
| `search_steps` | Hill-climbing iterations per coalition | 10 |
| `svd_groups` | Target cluster count for K-means (algorithmic) | 3 |
| `initial_distribution` | Starting assignment: `random` or `equal` | `random` |
| `stop_condition` | Termination: `steps` or `satisficed` (all agents stay) | `steps` |
| `seed` | Random seed for reproducibility | `None` |

### Experiment-level overrides

The experiment harness (`ExperimentConfig`) overrides several defaults for the full experiment suite:

| Parameter | Experiment value |
|-----------|-----------------|
| `coalitions` | 5 |
| `mutations` | 3 |
| `svd_groups` | 10 |
| `initial_distribution` | `equal` |
| `seed_base` | 42 |
| `n_iterations` | 200 |

### Named constants

| Constant | Value | Description |
|----------|-------|-------------|
| `MAJORITY_THRESHOLD` | 0.5 | Direct democracy vote threshold |
| `COLD_START_BUNDLE_COUNT` | 5 | Candidate policy bundles per algorithmic cluster |
| `VAMPIRISM_GAIN` | 1 | Extremist vampirism gain coefficient |
| `VAMPIRISM_LOSS` | 1 | Mainstream vampirism loss coefficient |
| `INSTITUTION_TYPE_COUNT` | 3 | Number of institution types for mixed assignment |

## Experiment design

### Experiment 1: Institutional comparison

Six configurations comparing governance institutions without extremists.

| Config | Institution | N_c | N_p | t_max |
|--------|------------|-----|-----|-------|
| 1a | Direct | 300 | 9 | 100 |
| 1b | Coalition | 300 | 9 | 100 |
| 1c | Algorithmic | 300 | 9 | 100 |
| 1d | Mixed | 300 | 3 | 100 |
| 1e | Mixed | 300 | 9 | 100 |
| 1f | Mixed | 300 | 27 | 100 |

All configurations use `rho = 0`, `alpha = 0`, tracking disabled, 200 iterations.

### Experiment 2: Factorial design with extremists

A 3 x 3 x 3 full factorial over platform count, extremist proportion, and vampirism intensity. All configurations use mixed institutions with tracking enabled.

| Factor | Levels |
|--------|--------|
| N_p | 3, 9, 27 |
| rho | 0.05, 0.10, 0.15 |
| alpha | 2.0, 5.0, 10.0 |

Fixed: N_c = 900, p_space = 10, t_max = 100, 200 iterations per configuration. Total: 27 configurations, 5,400 runs.

### Sensitivity analysis

**One-at-a-time (OAT).** Each parameter is varied independently from the Experiment 2 baseline (N_c = 900, N_p = 9, p_space = 10, rho = 0.10, alpha = 5.0, mu = 0.05):

| Parameter | Test values |
|-----------|-------------|
| `n_communities` | 50, 200 |
| `n_platforms` | 3, 6 |
| `p_space` | 5, 20 |
| `rho_extremist` | 0.05, 0.20 |
| `alpha` | 2.0, 10.0 |
| `mu` | 0.00, 0.02, 0.10 |

**Interaction analysis.** Alpha x p_space (6 configurations): alpha in {2, 5, 10} x p_space in {5, 20}, with p_space = 10 column drawn from Experiment 2 results.

All sensitivity configurations use mixed institutions with tracking enabled, 200 iterations.

## Tracking and analysis

### RelocationTracker

The tracker records per-step data when `tracking_enabled = True`:

- **Relocation events**: community ID, type, source and destination platform IDs and institution types
- **Governance snapshots**: per-platform coalition votes, winning coalition index, community ordering, and cluster membership
- **Step-level metrics**: mean utility, relocation count, per-governance utility breakdown

### MovementAnalyzer

Post-hoc analysis of tracker data:

- **Flow matrices**: per-step (N_p x N_p) transition count matrices recording inter-platform migration
- **Residence times**: fraction of simulation time each community spent on each platform
- **Raiding cycle detection**: autocorrelation function (ACF) of per-platform extremist outflow time series; significant lags identified at the 1.96/sqrt(n) threshold (using statsmodels where available, numpy fallback)
- **Enclave detection**: for coalition platforms, computes homogeneity of winning-coalition voter composition per step; reports mean homogeneity and fraction of steps exceeding the 0.9 threshold

### Burst analysis

Analysis of extremist migration bursts:

- **Burst detection**: steps where extremist outflow >= threshold (default 10.0)
- **Burst metrics**: count, sizes, intervals, fraction of total outflow attributable to bursts
- **Escalation regression**: OLS regression of burst size on burst index; escalation flagged when slope > 0, R-squared > 0.1, and at least 2 bursts observed
- **Platform classification**: each platform classified as `quiet` (low outflow), `active` (sustained outflow without burst structure), `raiding_base` (escalating bursts), `raiding_stable` (recurring non-escalating bursts), or `enclave` (burst-dominated outflow)

## Reported measures

### Per-iteration measures

Computed for each of the 200 iterations per configuration:

- Mean utility (global, by community type, by governance type, cross-tabulated type x governance)
- Normalized utility by type (scaled by theoretical maximum: `p_space` for mainstream, `p_space + alpha` for extremist)
- Total relocations and mean relocations per community
- Settling time (90th percentile of last-move step)
- Final community count and proportion by governance type
- Final community count cross-tabulated by type and governance

### Aggregation

Each per-iteration measure is aggregated across iterations into:

| Statistic | Method |
|-----------|--------|
| Mean | Arithmetic mean |
| SD | Sample standard deviation (n - 1 denominator) |
| 95% CI | Mean +/- 1.96 * SE, where SE = SD / sqrt(n) |
| Median | 50th percentile |
| Min, Max | Range |

### Step-level time series

Step-series data (utility, relocations, per-governance breakdowns) are aggregated across iterations per step, yielding mean trajectories with 95% confidence bands.

### Convergence diagnostics

Computed from the aggregated utility trajectory:

- **Tail metrics** (last 20% of steps): OLS slope, coefficient of variation, lag-1 autocorrelation of utility differences
- **Utility gains**: start, midpoint, and endpoint utility; total gain and second-half gain
- **Relocation reduction**: start and end relocation rates, percentage reduction

**Pattern classification:**

| Pattern | Criteria |
|---------|----------|
| `CONVERGED` | \|slope\| < 0.001 and CV < 0.005 |
| `STILL_CLIMBING` | slope > 0.001 |
| `OSCILLATING` | CV > 0.02 and autocorrelation < -0.3 |
| `NOISY_PLATEAU` | CV > 0.01 |
| `PLATEAU` | Otherwise |

### Dynamics measures

When tracking is enabled:

- Raiding cycles: per-platform ACF, significant lags, binary cycle flag
- Enclaves: per-platform homogeneity series, mean homogeneity, fraction enclaved
- Burst aggregates: cross-iteration burst rate, escalation rate, escalation t-test, classification distribution
- Flow matrices: per-step inter-platform transition counts

## Output structure

```
results/
+-- {experiment}/
|   +-- index.json                  # Completed configs, git commit hash
|   +-- summary.csv                 # Experiment-level aggregated measures
|   +-- burst_master.csv            # Per-config burst statistics with escalation t-test
|   +-- tables.tex                  # LaTeX tables (booktabs format)
|   +-- {config}/
|       +-- config.json             # Full configuration parameters
|       +-- raw.csv                 # Per-iteration measurements (200 rows)
|       +-- summary.csv             # Aggregated measures (mean, SD, CI, median, min, max)
|       +-- step_metrics.json       # Per-iteration step-level logs
|       +-- stepwise.csv            # Per-step aggregated time series with CI
|       +-- convergence.json        # Convergence diagnostics and pattern classification
|       +-- dynamics/               # (if tracking_enabled)
|           +-- scalars.json        # Cycle rate, mean homogeneity
|           +-- flow.npz            # Per-step flow matrices (numpy compressed)
|           +-- raiding.json        # Per-platform raiding analysis (ACF, cycles)
|           +-- enclaves.json       # Per-platform enclave metrics
|           +-- burst_aggregate.json # Cross-iteration burst statistics
```

### Key CSV schemas

**raw.csv** — one row per iteration:
`iteration, seed, avg_utility, avg_utility_mainstream, avg_utility_extremist, norm_utility_mainstream, norm_utility_extremist, total_relocations, avg_relocations_per_community, settling_time_90pct`

**summary.csv** — one row per measure:
`Measure, Mean, SD, CI_Lower, CI_Upper, Median, Min, Max, N`

**stepwise.csv** — one row per simulation step:
`step, {metric}_mean, {metric}_ci` for each tracked metric

## Reproducibility

**Deterministic seeding.** Each iteration receives seed = `seed_base` + iteration index (default base = 42), ensuring full reproducibility given identical configuration.

**BLAS thread control.** The experiment runner sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS` to 1 in worker processes to prevent nondeterministic thread scheduling in numerical libraries.

**Crash recovery.** The runner counts existing rows in `raw.csv` and resumes from the last completed iteration, appending new results with per-iteration flush.

**Skip-if-done.** Completed configurations are recorded in `index.json`. On re-execution, the runner loads prior results and skips re-computation.

**Provenance.** The current git commit hash (short form) is recorded in `index.json` at experiment creation.

## Installation and usage

### Installation

```bash
# Clone the repository
git clone https://github.com/<owner>/platform-abm.git
cd platform-abm

# Install with core dependencies
pip install -e .

# Install with development tools
pip install -e ".[dev]"

# Install with visualization support
pip install -e ".[viz]"

# Install with analysis extras (statsmodels for ACF)
pip install -e ".[analysis]"
```

Requires Python >= 3.10.

### Running experiments

```bash
# Experiment 1: institutional comparisons (6 configs x 200 iterations)
python -m experiments.run_exp1

# Experiment 2: factorial design (27 configs x 200 iterations)
python -m experiments.run_exp2

# Sensitivity analysis (requires Experiment 2 results)
python -m experiments.run_sensitivity

# Options available for all experiment scripts:
#   --output-dir DIR    Output directory (default: results)
#   --workers N         Parallel workers for iterations (default: sequential)
#   --dry-run           Print configurations without running
```

Experiment 1 also supports `--smoke` for a quick validation run (2 iterations, reduced population).

### Running tests

```bash
pytest
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| agentpy | >= 0.1.5 | Agent-based modeling framework |
| numpy | >= 1.24 | Numerical computation |
| pandas | >= 2.0 | Data structures and CSV I/O |
| scikit-learn | >= 1.3 | K-means clustering (algorithmic institution) |
| scipy | >= 1.10 | Statistical functions |
| pydantic | >= 2.0 | Configuration validation |
| pyyaml | >= 6.0 | YAML configuration support |

**Optional:**

| Group | Packages | Purpose |
|-------|----------|---------|
| dev | pytest, pytest-cov, ruff, mypy | Testing and linting |
| viz | networkx, matplotlib | Visualization |
| analysis | statsmodels >= 0.14 | ACF computation for raiding cycle detection |
