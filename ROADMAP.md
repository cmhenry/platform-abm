# ROADMAP.md

This document outlines the plan to modernize the platform-abm codebase from its current state (a research prototype) into a well-structured, reproducible, and extensible simulation framework.

---

## Phase 1: Project Infrastructure

### 1.1 Package Structure
- Convert flat file structure to a proper Python package:
  ```
  platform_abm/
  ├── __init__.py
  ├── agents/
  │   ├── __init__.py
  │   ├── base.py
  │   ├── community.py
  │   └── platform.py
  ├── institutions/
  │   ├── __init__.py
  │   ├── base.py
  │   ├── direct.py
  │   ├── coalition.py
  │   └── algorithmic.py
  ├── model.py
  ├── config.py
  ├── metrics.py
  └── utils.py
  experiments/
  ├── __init__.py
  ├── configs/
  │   └── *.yaml
  └── runners/
  dashboard/
  ├── __init__.py
  ├── app.py
  ├── components/
  └── callbacks/
  tests/
  ```

### 1.2 Dependency Management
- Create `pyproject.toml` with modern Python packaging (PEP 517/518)
- Pin dependency versions for reproducibility
- Add optional dependency groups: `[dev]`, `[dashboard]`, `[notebooks]`
- Core dependencies: agentpy>=0.1.5, numpy, pandas, scikit-learn
- Dashboard dependencies: streamlit (replace Dash), plotly
- Dev dependencies: pytest, ruff, mypy, pre-commit

### 1.3 Development Tooling
- Add `ruff` for linting and formatting (replaces flake8, black, isort)
- Add `mypy` for type checking
- Add `pre-commit` hooks configuration
- Create `Makefile` with common commands: `make test`, `make lint`, `make dashboard`

### 1.4 Configuration System
- Replace hardcoded parameter dictionaries with YAML/TOML config files
- Create a `Config` dataclass with validation using Pydantic
- Support config inheritance (base config + experiment-specific overrides)
- Add CLI interface using `typer` or `click` for running experiments

---

## Phase 2: Core Model Refactoring

### 2.1 Type Safety and Documentation
- Add comprehensive type hints throughout the codebase
- Replace string literals with `Enum` types:
  - `InstitutionType`: DIRECT, COALITION, ALGORITHMIC, MIXED
  - `CommunityType`: MAINSTREAM, EXTREMIST
  - `Strategy`: STAY, MOVE
- Add docstrings following NumPy/Google style
- Remove dead/commented-out code (~100 lines of commented code currently)

### 2.2 Community Agent Refactoring
Current issues to address:
- Mixed use of `''` and proper values for initialization
- Preferences generated with `random.choice` instead of `numpy.random` (breaks reproducibility with seeds)
- Utility function mixes policy alignment with "vampirism" externalities in unclear ways

Improvements:
- Create `CommunityConfig` dataclass for agent parameters
- Separate utility calculation into composable components:
  - `PolicyUtility`: Base utility from preference-policy alignment
  - `ExternalityUtility`: Utility from neighbor composition (extremist/mainstream dynamics)
- Add factory method for creating different community types
- Make preferences configurable (binary, continuous, categorical)

### 2.3 Platform Agent Refactoring
Current issues to address:
- Institution-specific logic embedded in single `Platform` class (700+ lines)
- `election()` method is a 50-line if/elif chain
- `group_policies` uses tuple `(rating, bundle)` - unclear data structure

Improvements:
- Extract institution logic into separate `Institution` classes using Strategy pattern:
  ```python
  class Institution(ABC):
      @abstractmethod
      def aggregate_preferences(self, platform: Platform) -> PolicyBundle: ...
      @abstractmethod
      def update_policies(self, platform: Platform) -> None: ...
  ```
- Create `DirectVotingInstitution`, `CoalitionInstitution`, `AlgorithmicInstitution`
- Replace magic numbers with named constants (e.g., `MAJORITY_THRESHOLD = 0.5`)
- Use proper data structures (`PolicyBundle`, `GroupPolicy` dataclasses)

### 2.4 Model Class Refactoring
Current issues:
- Setup methods do too much (200+ lines)
- Mixing of agent creation, assignment, and initialization
- `update()` method is mostly commented out

Improvements:
- Break `setup()` into focused methods with single responsibilities
- Implement proper data recording in `update()` using AgentPy's built-in mechanisms
- Add event hooks for extensibility (on_step_start, on_step_end, on_migration, etc.)
- Implement proper equilibrium detection beyond just "satisficed"

### 2.5 Random State Management
- Replace all `random.choice()` with `self.random` or `numpy.random.Generator`
- Ensure full reproducibility given a seed
- Add seed logging to experiment outputs

---

## Phase 3: New Features

### 3.1 Enhanced Utility Functions
- Implement pluggable utility functions:
  - Hamming distance (current)
  - Weighted preferences (some policies matter more)
  - Spatial/ideological distance (continuous policy space)
  - Network effects (utility depends on who else is on platform)
- Add utility function configuration in YAML

### 3.2 Platform Dynamics
- Implement platform entry/exit (platforms can be created or shut down)
- Add platform capacity constraints
- Implement platform "reputation" or quality scores
- Add switching costs for communities

### 3.3 Community Heterogeneity
- Support multiple community archetypes beyond mainstream/extremist
- Implement preference evolution over time
- Add information asymmetry (communities don't know all platform policies)
- Implement bounded rationality (satisficing vs. optimizing)

### 3.4 Network Structure
- Re-enable and improve network functionality (currently commented out)
- Implement social influence between communities
- Add geographic/spatial constraints on platform choice
- Support community-to-community connections (not just community-platform)

### 3.5 Advanced Algorithmic Institution
- Complete the SVD-based recommender system (currently stubbed out)
- Add collaborative filtering for policy recommendations
- Implement A/B testing simulation for platforms
- Add personalization vs. homogenization tradeoffs

---

## Phase 4: Dashboard

### 4.1 Replace Dash with Streamlit
The current `dashboard.py` is non-functional placeholder code. Build a real dashboard:

- Use Streamlit for simpler, more Pythonic dashboard development
- Implement as a separate runnable module: `python -m platform_abm.dashboard`

### 4.2 Dashboard Features

**Configuration Panel:**
- Parameter sliders/inputs for all model parameters
- Preset configurations (load from YAML)
- Save/load custom configurations
- Batch experiment setup

**Real-time Visualization:**
- Animated network graph showing community-platform relationships
- Color coding by community type (mainstream/extremist)
- Platform size visualization (community count)

**Time Series Charts:**
- Average utility over time (overall and by community type)
- Migration rate over time
- Platform population dynamics
- Policy convergence/divergence metrics

**Outcome Analysis:**
- Utility distribution histograms by institution type
- Community distribution across platforms
- Polarization/sorting metrics
- Gini coefficient for utility inequality

**Experiment Comparison:**
- Run multiple configurations side-by-side
- Statistical comparison of outcomes
- Parameter sensitivity analysis visualization

### 4.3 Export and Reporting
- Export simulation results to CSV/Parquet
- Generate PDF reports with key findings
- Save visualizations as images
- Export model state for later analysis

---

## Phase 5: Reproducibility and Testing

### 5.1 Test Suite
- Unit tests for utility calculations
- Unit tests for each institution type
- Integration tests for full simulation runs
- Property-based tests for model invariants (e.g., community count conservation)
- Regression tests comparing results to known baseline outputs

### 5.2 Experiment Management
- Integrate with experiment tracking (MLflow or Weights & Biases)
- Automatic logging of parameters, metrics, and artifacts
- Git commit hash recording for each run
- Results database for querying past experiments

### 5.3 Documentation
- API documentation with Sphinx or MkDocs
- Tutorial notebooks demonstrating common use cases
- Model specification document (formal description of the simulation)
- Validation documentation (how to verify model correctness)

---

## Phase 6: Performance and Scalability

### 6.1 Performance Optimization
- Profile current implementation to identify bottlenecks
- Vectorize numpy operations (avoid Python loops where possible)
- Cache expensive computations (e.g., utility calculations)
- Consider numba JIT compilation for hot paths

### 6.2 Parallel Execution
- Enable parallel experiment runs with AgentPy's `n_jobs` parameter
- Add support for running on HPC clusters (SLURM job scripts)
- Implement checkpointing for long-running simulations

---

## Migration Strategy

### Step 1: Non-breaking Infrastructure (Do First)
1. Add `pyproject.toml` and pin dependencies
2. Add linting and type checking (allow existing code to pass)
3. Add basic test scaffold
4. Create config system that wraps existing parameter dicts

### Step 2: Incremental Refactoring
1. Extract institutions one at a time (direct → coalition → algorithmic)
2. Add type hints file-by-file
3. Replace magic strings with enums
4. Clean up commented code

### Step 3: New Dashboard
1. Build Streamlit dashboard in parallel with refactoring
2. Start with read-only visualization of existing model
3. Add configuration controls incrementally

### Step 4: Feature Development
1. Add new features only after core refactoring is stable
2. Each new feature should include tests and documentation
3. Maintain backward compatibility with existing experiment configurations

---

## Success Criteria

- All existing experiments (experiment1.py, experiment2.py, experiment3.py) produce identical results after refactoring (given same seed)
- Test coverage >80% for core model logic
- Type checking passes with no errors
- Dashboard can configure and run any experiment interactively
- New researcher can understand and extend the model within 1 hour of reading documentation
