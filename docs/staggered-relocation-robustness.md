# Staggered relocation robustness implementation

Kanban: `t_58e2a8ce`

## Estimand / robustness target

The robustness target is not a new causal estimand. It is a design-sensitivity check for the emergent-raiding claim: holding the Experiment 2 mixed-institution extremist factorial fixed, does the burst/enclave/displacement pattern persist when relocation is updated in randomized same-step sequence rather than as the baseline simultaneous two-phase move set?

## Implementation

Added `relocation_update_order` with two values:

- `simultaneous` (default): existing baseline. All community strategies and destinations are computed before any relocation is applied; then all moves are applied as a batch.
- `staggered`: robustness variant. Each step randomizes community order. For each community in that order, the model recomputes utility and strategy against the current platform composition, then immediately applies any relocation before the next community decides.

The full production-grid builder is `build_staggered_relocation_configs()`. It mirrors the Experiment 2 factorial:

- `n_platforms`: 3, 9, 27
- `rho_extremist`: 0.05, 0.10, 0.15
- `alpha`: 2.0, 5.0, 10.0
- `n_iterations`: 200 per cell by default
- `tracking_enabled`: true
- `relocation_update_order`: `staggered`

Production-scale runs remain HPC-gated; this branch only defines and locally validates the variant.

## Local validation in Docker harness lane

Harness environment observed:

- Docker hostname: `b7b8d664effa`
- OS: Debian GNU/Linux 13 (trixie)
- Python: 3.11.15
- Package install: `python -m pip install -e '.[dev]'` exit 0

Commands run from `/projects/platform-abm`:

1. `pytest tests/test_relocation_update_order.py::test_staggered_relocation_recomputes_and_applies_moves_immediately -q`
   - Exit 1 before implementation, expected RED failure: staggered branch did not yet recompute/apply moves sequentially.
2. `pytest tests/test_relocation_update_order.py -q`
   - Exit 0 after implementation: 2 passed.
3. `pytest tests/test_config.py::TestSimulationConfig::test_relocation_update_order_in_agentpy_params tests/test_config.py::TestSimulationConfig::test_invalid_relocation_update_order tests/test_experiment_config.py::test_relocation_update_order_flows_to_params_and_dict -q`
   - Exit 1 before config propagation, expected RED failures.
4. Same config test command after implementation
   - Exit 0: 3 passed.
5. `pytest tests/test_experiment_config.py::test_staggered_relocation_config_grid_matches_exp2_factorial -q`
   - Exit 1 before builder implementation, expected import failure.
6. Same builder test after implementation
   - Exit 0: 1 passed.
7. `pytest tests/test_relocation_update_order.py tests/test_config.py tests/test_experiment_config.py -q`
   - Exit 0: 45 passed.
8. `ruff check platform_abm/config.py platform_abm/model.py experiments/configs tests/test_relocation_update_order.py tests/test_config.py tests/test_experiment_config.py README.md`
   - Exit 0: all checks passed.
9. `pytest -q`
   - Exit 1: 1 pre-existing/non-touched full-suite failure in `tests/test_runner.py::test_burst_master_csv`, with reporting pipeline warning `final_utility`; 287 other tests passed. I did not alter runner/reporting logic for this card.
10. Local smoke comparison with 2 iterations each for `simultaneous` and `staggered` at `n_communities=30`, `n_platforms=3`, `p_space=5`, `t_max=5`, `rho_extremist=0.10`, `alpha=5.0`, tracking enabled.
    - Exit 0.
    - Output directories:
      - `/tmp/platform_abm_staggered_smoke_t58e2a8ce/relocation_order_smoke/smoke_simultaneous_np3_rho010_alpha5`
      - `/tmp/platform_abm_staggered_smoke_t58e2a8ce/relocation_order_smoke/smoke_staggered_np3_rho010_alpha5`
    - Both produced `raw.csv`, `summary.csv`, and `stepwise.csv`.
    - Simultaneous total relocations across the 2 smoke iterations: `[82, 74]`.
    - Staggered total relocations across the 2 smoke iterations: `[86, 83]`.
    - Both produced burst aggregate keys including burst rate, burst size, burst fraction, escalation, interval, and classification summaries.

## Interpretation guardrail

The local smoke run only verifies that the staggered variant executes end-to-end and produces the expected reporting artifacts. It does not estimate whether the 94-100% burst concentration claim survives the update-order perturbation. That inference requires the gated production/HPC grid or a PI-approved reduced grid.

Decision rule for the production comparison: if burst concentration, displacement, and enclave formation are directionally stable under the staggered grid, the paper can say emergent raiding is not an artifact of simultaneous relocation. If burst concentration collapses or classifications shift materially, the paper should hedge the emergent-cycle language as conditional on synchronous decision timing.
