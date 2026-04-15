# ρ_e Disaggregation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Disaggregate ρ_e into an ideologue/griefer mix so `exp2b` can vary composition at fixed ρ_e with α_i=2, α_g=10.

**Architecture:** Add a per-agent `subtype` field to `Community`; replace the symmetric vampirism formula with an attacker-weighted mainstream loss that reads per-subtype neighbor counts; extend `ExperimentConfig` with three optional fields that default to current behavior; add a new `build_exp2b_configs()` builder and `run_exp2b.py` entry point.

**Tech Stack:** Python 3, AgentPy, pydantic (tests only), pytest. Spec: `docs/superpowers/specs/2026-04-15-rho-e-disaggregation-design.md`. Branch: `agent/rho-e-disaggregation`.

**Before starting:**
- `git status` on `agent/rho-e-disaggregation` should be clean with the spec committed.
- `pytest` baseline should be green. Run it once and confirm before Task 1.

---

## Task 1: Add `subtype` field to Community

**Files:**
- Modify: `platform_abm/agents/community.py`
- Test: `tests/test_community.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_community.py`:

```python
from platform_abm.config import CommunityType
from tests.conftest import make_model


class TestCommunitySubtype:
    def test_default_subtype_is_empty(self):
        """Fresh communities have empty subtype before extremist setup."""
        model = make_model({"n_comms": 4, "n_plats": 1})
        for comm in model.communities:
            assert comm.subtype == ""

    def test_mainstream_subtype_stays_empty_with_extremists(self):
        """When extremists are flagged, mainstream communities keep subtype ''."""
        model = make_model({
            "n_comms": 10, "n_plats": 1,
            "extremists": "yes", "percent_extremists": 30,
        })
        mainstream = [c for c in model.communities
                      if c.type == CommunityType.MAINSTREAM.value]
        assert mainstream
        for comm in mainstream:
            assert comm.subtype == ""
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_community.py::TestCommunitySubtype -v`
Expected: FAIL with `AttributeError: 'Community' object has no attribute 'subtype'`.

- [ ] **Step 3: Add the field to Community**

Edit `platform_abm/agents/community.py`. In the class-level type annotations block (around line 30 after `alpha: float`), add:

```python
    subtype: str
```

In `setup()` (around line 44 after `self.alpha = ...`), add:

```python
        self.subtype = ""
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_community.py::TestCommunitySubtype -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add platform_abm/agents/community.py tests/test_community.py
git commit -m "feat: add subtype field to Community agent

Defaults to '' for mainstream; will be set to 'ideologue' or
'griefer' during extremist setup in a follow-up change.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Per-subtype neighbor counts

**Files:**
- Modify: `platform_abm/neighbors.py:32-41`
- Test: `tests/test_neighbors.py`

**Context:** `get_neighbor_counts` today returns `{n_mainstream, n_extremist}`. Spec requires `{n_mainstream, n_ideologue, n_griefer}`. `n_extremist = n_ideologue + n_griefer` at call sites; today's only non-test caller is `platform_abm/utility.py` (updated in Task 3). Test file `tests/test_neighbors.py` has one test (`TestNeighborCounts.test_correct_partition`) asserting `n_mainstream + n_extremist == len(neighbors)` — we'll replace that assertion with the new partition.

- [ ] **Step 1: Write the failing tests**

Replace the existing `TestNeighborCounts` class in `tests/test_neighbors.py` with:

```python
class TestNeighborCounts:
    def test_correct_partition(self):
        """Counts partition neighbors into mainstream/ideologue/griefer."""
        model = make_model({
            "extremists": "yes", "percent_extremists": 30,
            "n_comms": 20, "n_plats": 1,
        })
        plat = model.platforms[0]
        for comm in plat.communities:
            counts = get_neighbor_counts(comm, plat)
            neighbors = get_neighbors(comm, plat)
            assert (
                counts["n_mainstream"] + counts["n_ideologue"] + counts["n_griefer"]
                == len(neighbors)
            )

    def test_zero_counts_when_solo(self):
        """Solo community has zero for all subtype counts."""
        model = make_model({"n_comms": 1, "n_plats": 1})
        comm = model.communities[0]
        plat = model.platforms[0]
        counts = get_neighbor_counts(comm, plat)
        assert counts["n_mainstream"] == 0
        assert counts["n_ideologue"] == 0
        assert counts["n_griefer"] == 0

    def test_all_mainstream(self, direct_model):
        """With no extremists, all neighbors are mainstream."""
        plat = direct_model.platforms[0]
        if not plat.communities:
            pytest.skip("Empty platform")
        comm = plat.communities[0]
        counts = get_neighbor_counts(comm, plat)
        assert counts["n_ideologue"] == 0
        assert counts["n_griefer"] == 0
        assert counts["n_mainstream"] == len(plat.communities) - 1

    def test_subtype_split(self):
        """Extremists with subtype='ideologue' counted as ideologues, 'griefer' as griefers."""
        from platform_abm.config import CommunityType
        model = make_model({"n_comms": 6, "n_plats": 1})
        plat = model.platforms[0]
        comms = list(plat.communities)
        # Leave comms[0] mainstream. Mark 2 ideologues, 3 griefers.
        for c in comms[1:3]:
            c.type = CommunityType.EXTREMIST.value
            c.subtype = "ideologue"
        for c in comms[3:6]:
            c.type = CommunityType.EXTREMIST.value
            c.subtype = "griefer"
        counts = get_neighbor_counts(comms[0], plat)
        assert counts["n_mainstream"] == 0
        assert counts["n_ideologue"] == 2
        assert counts["n_griefer"] == 3
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_neighbors.py::TestNeighborCounts -v`
Expected: FAIL — `test_correct_partition`, `test_zero_counts_when_solo`, `test_all_mainstream`, `test_subtype_split` all fail with `KeyError: 'n_ideologue'`.

- [ ] **Step 3: Update `get_neighbor_counts`**

Replace the function body in `platform_abm/neighbors.py` (lines 32-41):

```python
def get_neighbor_counts(community: Community, platform: Platform) -> dict[str, int]:
    """Return counts of mainstream, ideologue, and griefer neighbors."""
    neighbors = get_neighbors(community, platform)
    n_mainstream = 0
    n_ideologue = 0
    n_griefer = 0
    for c in neighbors:
        if c.type == CommunityType.MAINSTREAM.value:
            n_mainstream += 1
        elif getattr(c, "subtype", "") == "griefer":
            n_griefer += 1
        else:
            n_ideologue += 1
    return {
        "n_mainstream": n_mainstream,
        "n_ideologue": n_ideologue,
        "n_griefer": n_griefer,
    }
```

Why `else → ideologue`: any extremist whose `subtype` is not explicitly `"griefer"` (including legacy `""` or `"ideologue"`) counts as ideologue. This keeps backward compat for existing experiments where extremists have no subtype set: they're treated uniformly as ideologues and `community.alpha` (which holds the scalar α) drives their gain — matching current behavior.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_neighbors.py -v`
Expected: PASS for all tests in this file (including the three updated ones and the new `test_subtype_split`).

- [ ] **Step 5: Commit**

```bash
git add platform_abm/neighbors.py tests/test_neighbors.py
git commit -m "refactor: split neighbor counts by extremist subtype

get_neighbor_counts now returns {n_mainstream, n_ideologue, n_griefer}
instead of {n_mainstream, n_extremist}. Legacy extremists with no
subtype are counted as ideologues, preserving current behavior for
experiments that haven't opted into disaggregation.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Attacker-weighted mainstream loss in `compute_utility`

**Files:**
- Modify: `platform_abm/utility.py:37-60`
- Test: `tests/test_utility.py`

**Context:** Current formula (`platform_abm/utility.py:58`) is `u_base - community.alpha * (n_ext / total)`. Spec §Model Changes §1 requires `u_base - (α_i · n_ideologue + α_g · n_griefer) / total` for mainstream. Extremist gain formula is unchanged. Model reads α_i and α_g from `self.p.alpha_ideologue` and `self.p.alpha_griefer` — these params will be wired through in Task 5. For this task, we add a fallback to `community.alpha` when those params are absent, so tests and legacy callers keep working before Task 5 lands.

- [ ] **Step 1: Write the failing test (new formula regression)**

Append to `tests/test_utility.py`:

```python
class TestAttackerWeightedLoss:
    def test_mixed_subtypes_attacker_weighted(self):
        """Mainstream loss weights by attacker subtype: alpha_i*n_i + alpha_g*n_g."""
        model = make_model({
            "n_comms": 5, "n_plats": 1,
            "alpha": 2.0,
        })
        # Inject disaggregated alphas onto the model's params.
        model.p.alpha_ideologue = 2.0
        model.p.alpha_griefer = 10.0

        plat = model.platforms[0]
        comms = list(plat.communities)
        # comms[0] mainstream; 2 ideologues, 2 griefers.
        comms[0].type = CommunityType.MAINSTREAM.value
        for c in comms[1:3]:
            c.type = CommunityType.EXTREMIST.value
            c.subtype = "ideologue"
            c.alpha = 2.0
        for c in comms[3:5]:
            c.type = CommunityType.EXTREMIST.value
            c.subtype = "griefer"
            c.alpha = 10.0
        comm = comms[0]
        base = compute_base_utility(comm, plat)
        full = compute_utility(comm, plat)
        # (2.0 * 2 + 10.0 * 2) / 4 = 6.0
        assert full == pytest.approx(base - 6.0)

    def test_all_ideologues_equals_alpha_i(self):
        """With only ideologues, loss is alpha_ideologue * (n_ext / total)."""
        model = make_model({"n_comms": 4, "n_plats": 1, "alpha": 2.0})
        model.p.alpha_ideologue = 2.0
        model.p.alpha_griefer = 10.0
        plat = model.platforms[0]
        comms = list(plat.communities)
        comms[0].type = CommunityType.MAINSTREAM.value
        for c in comms[1:]:
            c.type = CommunityType.EXTREMIST.value
            c.subtype = "ideologue"
            c.alpha = 2.0
        comm = comms[0]
        base = compute_base_utility(comm, plat)
        full = compute_utility(comm, plat)
        assert full == pytest.approx(base - 2.0)

    def test_all_griefers_equals_alpha_g(self):
        """With only griefers, loss is alpha_griefer * (n_ext / total)."""
        model = make_model({"n_comms": 4, "n_plats": 1, "alpha": 10.0})
        model.p.alpha_ideologue = 2.0
        model.p.alpha_griefer = 10.0
        plat = model.platforms[0]
        comms = list(plat.communities)
        comms[0].type = CommunityType.MAINSTREAM.value
        for c in comms[1:]:
            c.type = CommunityType.EXTREMIST.value
            c.subtype = "griefer"
            c.alpha = 10.0
        comm = comms[0]
        base = compute_base_utility(comm, plat)
        full = compute_utility(comm, plat)
        assert full == pytest.approx(base - 10.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_utility.py::TestAttackerWeightedLoss -v`
Expected: FAIL — mainstream loss still equals `community.alpha * n_ext / total` (2.0 * 4/4 = 2.0 in the mixed test), not the attacker-weighted 6.0.

- [ ] **Step 3: Update `compute_utility`**

Replace `compute_utility` in `platform_abm/utility.py` (lines 37-60) with:

```python
def compute_utility(community: Community, platform: Platform) -> float:
    """Full utility with proportional vampirism.

    Mainstream: u_base - (alpha_i * n_ideologue + alpha_g * n_griefer) / total
    Extremist:  u_base + community.alpha * (n_mainstream / total)

    alpha_i and alpha_g default to community.alpha when the model does not
    expose disaggregated alpha params (legacy experiments).

    Division-by-zero guard: vampirism term is 0 when denominator is 0.
    """
    u_base = compute_base_utility(community, platform)
    counts = get_neighbor_counts(community, platform)
    n_main = counts["n_mainstream"]
    n_ideologue = counts["n_ideologue"]
    n_griefer = counts["n_griefer"]
    total = n_main + n_ideologue + n_griefer

    if total == 0:
        return float(u_base)

    if community.type == CommunityType.EXTREMIST.value:
        alpha = getattr(community, "alpha", 1.0)
        return u_base + alpha * (n_main / total)
    elif community.type == CommunityType.MAINSTREAM.value:
        model_p = community.model.p
        alpha_i = getattr(model_p, "alpha_ideologue",
                          getattr(community, "alpha", 1.0))
        alpha_g = getattr(model_p, "alpha_griefer",
                          getattr(community, "alpha", 1.0))
        return u_base - (alpha_i * n_ideologue + alpha_g * n_griefer) / total
    else:
        return float(u_base)
```

Note: `community.model.p` is how AgentPy agents reach the model's parameters. Verify in existing code: `platform_abm/agents/community.py:44` uses `self.p.alpha` — `community.p` is equivalent to `community.model.p`. Use `community.model.p` here to be explicit from within the utility module.

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `pytest tests/test_utility.py::TestAttackerWeightedLoss -v`
Expected: PASS.

- [ ] **Step 5: Run the full utility test module for regression**

Run: `pytest tests/test_utility.py -v`
Expected: PASS for all tests, including `TestMainstreamUtility` (which has no griefers — all extremists default to ideologue subtype, `alpha_i` falls back to `community.alpha`, producing the old value).

- [ ] **Step 6: Commit**

```bash
git add platform_abm/utility.py tests/test_utility.py
git commit -m "feat: attacker-weighted mainstream loss in vampirism formula

Mainstream loss now sums alpha_i*n_ideologue + alpha_g*n_griefer and
normalizes by total neighbors. Falls back to community.alpha when the
model has no disaggregated alpha params, so existing experiments keep
current behavior.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Split extremists by `frac_griefer` in setup

**Files:**
- Modify: `platform_abm/model.py:167-175`
- Test: `tests/test_integration.py` (new test class appended)

**Context:** `_setup_community_types` marks flagged communities as extremist and assigns preferences. Spec §Model Changes §4 extends it to split those communities into ideologues and griefers by `frac_griefer` and set `community.alpha` accordingly. The params `alpha_ideologue`, `alpha_griefer`, `frac_griefer` must already be on `self.p`; for this task we use `getattr(..., default)` so the model works even when the params are absent (Task 5 wires them in through ExperimentConfig). Default `frac_griefer` is 0.0, `alpha_ideologue` falls back to `self.p.alpha`, same for `alpha_griefer`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_integration.py`:

```python
from platform_abm.config import CommunityType
from tests.conftest import make_model


class TestExtremistSubtypeSplit:
    def test_all_ideologues_when_frac_griefer_zero(self):
        params = {
            "n_comms": 40, "n_plats": 1,
            "extremists": "yes", "percent_extremists": 50,
            "alpha": 3.0,
            "alpha_ideologue": 2.0, "alpha_griefer": 10.0,
            "frac_griefer": 0.0,
        }
        model = make_model(params)
        ext = [c for c in model.communities
               if c.type == CommunityType.EXTREMIST.value]
        assert ext
        assert all(c.subtype == "ideologue" for c in ext)
        assert all(c.alpha == 2.0 for c in ext)

    def test_all_griefers_when_frac_griefer_one(self):
        params = {
            "n_comms": 40, "n_plats": 1,
            "extremists": "yes", "percent_extremists": 50,
            "alpha": 3.0,
            "alpha_ideologue": 2.0, "alpha_griefer": 10.0,
            "frac_griefer": 1.0,
        }
        model = make_model(params)
        ext = [c for c in model.communities
               if c.type == CommunityType.EXTREMIST.value]
        assert ext
        assert all(c.subtype == "griefer" for c in ext)
        assert all(c.alpha == 10.0 for c in ext)

    def test_half_half_split(self):
        params = {
            "n_comms": 100, "n_plats": 1,
            "extremists": "yes", "percent_extremists": 40,
            "alpha": 3.0,
            "alpha_ideologue": 2.0, "alpha_griefer": 10.0,
            "frac_griefer": 0.5,
        }
        model = make_model(params)
        ext = [c for c in model.communities
               if c.type == CommunityType.EXTREMIST.value]
        n_griefer = sum(1 for c in ext if c.subtype == "griefer")
        n_ideologue = sum(1 for c in ext if c.subtype == "ideologue")
        assert n_griefer + n_ideologue == len(ext)
        # 40% of 100 = 40 extremists; 50% griefer = 20; allow ±1 for rounding.
        assert abs(n_griefer - 20) <= 1
        assert abs(n_ideologue - 20) <= 1

    def test_legacy_params_default_to_ideologue(self):
        """When alpha_* and frac_griefer are absent, all extremists become ideologues."""
        params = {
            "n_comms": 40, "n_plats": 1,
            "extremists": "yes", "percent_extremists": 50,
            "alpha": 3.0,
        }
        model = make_model(params)
        ext = [c for c in model.communities
               if c.type == CommunityType.EXTREMIST.value]
        assert ext
        assert all(c.subtype == "ideologue" for c in ext)
        # alpha fallback: community.alpha stays at self.p.alpha (current behavior)
        assert all(c.alpha == 3.0 for c in ext)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_integration.py::TestExtremistSubtypeSplit -v`
Expected: FAIL — `subtype` stays `""` because `_setup_community_types` doesn't set it yet.

- [ ] **Step 3: Update `_setup_community_types`**

Replace the method in `platform_abm/model.py` (lines 167-175) with:

```python
    def _setup_community_types(self, extremists: list[int]) -> None:
        """Set extremist community types, preferences, subtypes, and alpha.

        Splits extremists into ideologues and griefers by `frac_griefer`.
        Defaults (frac_griefer=0, alpha_ideologue=alpha_griefer=alpha)
        reduce to the pre-disaggregation behavior: all extremists become
        ideologues with the scalar alpha.
        """
        frac_griefer = getattr(self.p, "frac_griefer", 0.0)
        default_alpha = getattr(self.p, "alpha", 1.0)
        alpha_i = getattr(self.p, "alpha_ideologue", default_alpha)
        alpha_g = getattr(self.p, "alpha_griefer", default_alpha)

        n_griefers = int(round(len(extremists) * frac_griefer))
        griefer_ids = set(self.random.sample(extremists, n_griefers))

        for comm_id in extremists:
            comm_sel = self.communities.select(self.communities.id == comm_id)
            comm_sel.type = CommunityType.EXTREMIST.value
            if comm_id in griefer_ids:
                comm_sel.subtype = "griefer"
                comm_sel.alpha = alpha_g
            else:
                comm_sel.subtype = "ideologue"
                comm_sel.alpha = alpha_i
            if self.random.random() < 0.5:
                comm_sel.preferences = generate_zero_preferences(self.p.p_space)
            else:
                comm_sel.preferences = generate_ones_preferences(self.p.p_space)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_integration.py::TestExtremistSubtypeSplit -v`
Expected: PASS.

- [ ] **Step 5: Run the full test suite to check for regressions**

Run: `pytest -x`
Expected: PASS for all tests. Existing tests that set `comm.alpha = alpha` manually (e.g., `TestAlphaValues.test_different_alpha_scales_penalty`) still pass because they mutate α after setup.

- [ ] **Step 6: Commit**

```bash
git add platform_abm/model.py tests/test_integration.py
git commit -m "feat: split extremists into ideologues and griefers by frac_griefer

_setup_community_types now samples frac_griefer*n_extremists communities
as griefers with alpha_griefer and the rest as ideologues with
alpha_ideologue. Defaults reduce to current behavior (all ideologues,
scalar alpha).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Add heterogeneous-α fields to `ExperimentConfig`

**Files:**
- Modify: `experiments/configs/experiment_config.py`
- Test: `tests/test_experiment_config.py`

**Context:** Spec §Config Changes. Three new fields with defaults that reduce to current behavior; `to_params` resolves fallbacks; `to_dict`/`from_dict` extended but must accept legacy dicts (missing fields OK).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_experiment_config.py`:

```python
def test_new_fields_default_none_and_zero():
    """New alpha_* fields default to None; frac_griefer defaults to 0.0."""
    cfg = ExperimentConfig(
        name="test", experiment="test",
        n_communities=100, n_platforms=9, p_space=10, t_max=50,
        institution="mixed", rho_extremist=0.10, alpha=5.0,
    )
    assert cfg.alpha_ideologue is None
    assert cfg.alpha_griefer is None
    assert cfg.frac_griefer == 0.0


def test_to_params_fallback_to_alpha_when_disagg_unset():
    """When alpha_ideologue/griefer unset, both resolve to alpha."""
    cfg = ExperimentConfig(
        name="test", experiment="test",
        n_communities=100, n_platforms=9, p_space=10, t_max=50,
        institution="mixed", rho_extremist=0.10, alpha=5.0,
    )
    params = cfg.to_params(0)
    assert params["alpha_ideologue"] == 5.0
    assert params["alpha_griefer"] == 5.0
    assert params["frac_griefer"] == 0.0


def test_to_params_passes_disaggregated_alphas():
    """Explicit alpha_ideologue and alpha_griefer flow into params."""
    cfg = ExperimentConfig(
        name="test", experiment="exp2b",
        n_communities=900, n_platforms=9, p_space=10, t_max=100,
        institution="mixed", rho_extremist=0.10, alpha=2.0,
        alpha_ideologue=2.0, alpha_griefer=10.0, frac_griefer=0.5,
    )
    params = cfg.to_params(0)
    assert params["alpha_ideologue"] == 2.0
    assert params["alpha_griefer"] == 10.0
    assert params["frac_griefer"] == 0.5


def test_to_dict_includes_new_fields():
    cfg = ExperimentConfig(
        name="test", experiment="exp2b",
        n_communities=900, n_platforms=9, p_space=10, t_max=100,
        institution="mixed", rho_extremist=0.10, alpha=2.0,
        alpha_ideologue=2.0, alpha_griefer=10.0, frac_griefer=0.25,
    )
    d = cfg.to_dict()
    assert d["alpha_ideologue"] == 2.0
    assert d["alpha_griefer"] == 10.0
    assert d["frac_griefer"] == 0.25


def test_from_dict_accepts_legacy_dict_missing_new_fields():
    """Loading a pre-disaggregation dict still works; new fields get defaults."""
    legacy = {
        "name": "legacy", "experiment": "exp2",
        "n_communities": 900, "n_platforms": 9, "p_space": 10, "t_max": 100,
        "institution": "mixed", "rho_extremist": 0.10, "alpha": 5.0,
        "mu": 0.05, "coalitions": 5, "mutations": 3, "svd_groups": 10,
        "search_steps": 10, "initial_distribution": "equal",
        "tracking_enabled": True, "n_iterations": 200, "seed_base": 42,
        "log_platform_detail": False,
    }
    cfg = ExperimentConfig.from_dict(legacy)
    assert cfg.alpha_ideologue is None
    assert cfg.alpha_griefer is None
    assert cfg.frac_griefer == 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_experiment_config.py -v -k "new_fields or fallback or disaggregated or new_fields or legacy"`
Expected: FAIL with `TypeError: ExperimentConfig.__init__() got an unexpected keyword argument 'alpha_ideologue'` (first failure).

- [ ] **Step 3: Update `ExperimentConfig`**

In `experiments/configs/experiment_config.py`, after the `mu: float = 0.05` line, add:

```python
    alpha_ideologue: float | None = None
    alpha_griefer: float | None = None
    frac_griefer: float = 0.0
```

Update `to_params` (after the line computing `percent`) to include:

```python
        alpha_i = self.alpha_ideologue if self.alpha_ideologue is not None else self.alpha
        alpha_g = self.alpha_griefer if self.alpha_griefer is not None else self.alpha
```

Then in the returned dict, add three entries (place them near `"alpha": self.alpha`):

```python
            "alpha": self.alpha,
            "alpha_ideologue": alpha_i,
            "alpha_griefer": alpha_g,
            "frac_griefer": self.frac_griefer,
```

Extend `to_dict` to include the three new fields:

```python
            "alpha_ideologue": self.alpha_ideologue,
            "alpha_griefer": self.alpha_griefer,
            "frac_griefer": self.frac_griefer,
```

`from_dict` continues to use `ExperimentConfig(**d)`; because the new fields have defaults, legacy dicts missing them still construct successfully.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_experiment_config.py -v`
Expected: PASS for all tests (new and existing).

- [ ] **Step 5: Commit**

```bash
git add experiments/configs/experiment_config.py tests/test_experiment_config.py
git commit -m "feat: heterogeneous alpha fields on ExperimentConfig

Add alpha_ideologue, alpha_griefer, frac_griefer with defaults that
preserve current scalar-alpha behavior. to_params resolves unset
disaggregated fields to the scalar alpha; from_dict accepts legacy
serialized configs.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Per-subtype utilities in `compute_extremist_metrics`

**Files:**
- Modify: `platform_abm/metrics.py:62-69`
- Test: `tests/test_regression.py` (append) or new `tests/test_metrics.py` — check which exists first.

**Context:** `tests/test_regression.py` already exists; add a new test class there to avoid creating a near-empty file. Spec §Model Changes §5 adds two keys; omit each when the corresponding subtype has zero members to avoid `ZeroDivisionError` on endpoints.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_regression.py`:

```python
from platform_abm.metrics import compute_extremist_metrics
from tests.conftest import make_model


class TestExtremistMetricsSubtypes:
    def test_both_subtypes_reported_when_present(self):
        model = make_model({
            "n_comms": 100, "n_plats": 1,
            "extremists": "yes", "percent_extremists": 40,
            "alpha": 3.0,
            "alpha_ideologue": 2.0, "alpha_griefer": 10.0,
            "frac_griefer": 0.5,
        })
        # Force utility update so current_utility is populated.
        for comm in model.communities:
            comm.update_utility()
        metrics = compute_extremist_metrics(model.communities)
        assert "average_mainstream_utility" in metrics
        assert "average_extremist_utility" in metrics
        assert "average_ideologue_utility" in metrics
        assert "average_griefer_utility" in metrics

    def test_ideologue_key_omitted_when_no_ideologues(self):
        model = make_model({
            "n_comms": 100, "n_plats": 1,
            "extremists": "yes", "percent_extremists": 40,
            "alpha": 10.0,
            "alpha_ideologue": 2.0, "alpha_griefer": 10.0,
            "frac_griefer": 1.0,
        })
        for comm in model.communities:
            comm.update_utility()
        metrics = compute_extremist_metrics(model.communities)
        assert "average_griefer_utility" in metrics
        assert "average_ideologue_utility" not in metrics

    def test_griefer_key_omitted_when_no_griefers(self):
        model = make_model({
            "n_comms": 100, "n_plats": 1,
            "extremists": "yes", "percent_extremists": 40,
            "alpha": 2.0,
            "alpha_ideologue": 2.0, "alpha_griefer": 10.0,
            "frac_griefer": 0.0,
        })
        for comm in model.communities:
            comm.update_utility()
        metrics = compute_extremist_metrics(model.communities)
        assert "average_ideologue_utility" in metrics
        assert "average_griefer_utility" not in metrics
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_regression.py::TestExtremistMetricsSubtypes -v`
Expected: FAIL — new keys missing.

- [ ] **Step 3: Extend `compute_extremist_metrics`**

Replace the function body in `platform_abm/metrics.py` (lines 62-69):

```python
def compute_extremist_metrics(communities: ap.AgentList) -> dict[str, float]:
    """Per-capita utility for mainstream, extremist, and (when present)
    ideologue and griefer subgroups.

    Ideologue/griefer keys are omitted when their subgroup is empty so
    endpoints of the f_g sweep don't trigger ZeroDivisionError.
    """
    extremists = communities.select(communities.type == "extremist")
    mainstream = communities.select(communities.type == "mainstream")
    metrics: dict[str, float] = {
        "average_extremist_utility": sum(extremists.current_utility) / len(extremists),
        "average_mainstream_utility": sum(mainstream.current_utility) / len(mainstream),
    }
    ideologues = communities.select(communities.subtype == "ideologue")
    griefers = communities.select(communities.subtype == "griefer")
    if len(ideologues) > 0:
        metrics["average_ideologue_utility"] = (
            sum(ideologues.current_utility) / len(ideologues)
        )
    if len(griefers) > 0:
        metrics["average_griefer_utility"] = (
            sum(griefers.current_utility) / len(griefers)
        )
    return metrics
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_regression.py::TestExtremistMetricsSubtypes -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add platform_abm/metrics.py tests/test_regression.py
git commit -m "feat: report per-subtype utilities in compute_extremist_metrics

Adds average_ideologue_utility and average_griefer_utility to end-of-run
metrics when the respective subgroup is non-empty.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: `build_exp2b_configs()` builder

**Files:**
- Modify: `experiments/configs/builders.py`
- Test: `tests/test_experiment_config.py`

**Context:** Spec §Config Changes §builders.py. Three configs: `exp2b_fg025`, `exp2b_fg050`, `exp2b_fg075` at ρ_e=0.10, N_p=9, α_i=2.0, α_g=10.0, `tracking_enabled=True`, `n_iterations=200`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_experiment_config.py`:

```python
from experiments.configs.builders import build_exp2b_configs


def test_exp2b_config_count():
    """Experiment 2b produces 3 configs (f_g in {0.25, 0.50, 0.75})."""
    assert len(build_exp2b_configs()) == 3


def test_exp2b_configs_have_disaggregated_alpha():
    for cfg in build_exp2b_configs():
        assert cfg.experiment == "exp2b"
        assert cfg.alpha_ideologue == 2.0
        assert cfg.alpha_griefer == 10.0
        assert cfg.rho_extremist == 0.10
        assert cfg.n_platforms == 9
        assert cfg.institution == "mixed"
        assert cfg.tracking_enabled is True


def test_exp2b_frac_griefer_sweep():
    fracs = sorted(cfg.frac_griefer for cfg in build_exp2b_configs())
    assert fracs == [0.25, 0.50, 0.75]


def test_exp2b_names():
    names = sorted(cfg.name for cfg in build_exp2b_configs())
    assert names == ["exp2b_fg025", "exp2b_fg050", "exp2b_fg075"]
```

Also update `test_unique_config_names_across_builders` to include exp2b:

```python
def test_unique_config_names_across_builders():
    """All config names across all builders are unique."""
    all_configs = (
        build_exp1_configs()
        + build_exp2_configs()
        + build_exp2b_configs()
        + build_oat_configs()
        + build_interaction_configs()
    )
    names = [c.name for c in all_configs]
    assert len(names) == len(set(names)), (
        f"Duplicate names found: {[n for n in names if names.count(n) > 1]}"
    )
```

Update the builder import at the top of the test file:

```python
from experiments.configs.builders import (
    build_exp1_configs,
    build_exp2_configs,
    build_exp2b_configs,
    build_interaction_configs,
    build_oat_configs,
)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_experiment_config.py -v -k exp2b`
Expected: FAIL — `ImportError: cannot import name 'build_exp2b_configs'`.

- [ ] **Step 3: Add the builder**

Append to `experiments/configs/builders.py`:

```python
def build_exp2b_configs() -> list[ExperimentConfig]:
    """Experiment 2b: rho_e disaggregation at fixed rho_e=0.10, N_p=9.

    Varies griefer fraction f_g in {0.25, 0.50, 0.75} with alpha_i=2
    and alpha_g=10. Endpoints (f_g=0 and f_g=1) reuse exp2 runs at
    analysis time, so only 3 new configs x 200 iterations = 600 runs.
    """
    configs = []
    for fg in [0.25, 0.50, 0.75]:
        fg_str = f"{fg:.2f}".replace(".", "")
        configs.append(ExperimentConfig(
            name=f"exp2b_fg{fg_str}",
            experiment="exp2b",
            n_communities=900,
            n_platforms=9,
            p_space=10,
            t_max=100,
            institution="mixed",
            rho_extremist=0.10,
            alpha=2.0,
            alpha_ideologue=2.0,
            alpha_griefer=10.0,
            frac_griefer=fg,
            tracking_enabled=True,
            **_COMMON_FIXED,
        ))
    return configs
```

Note: `fg = 0.25` → `"025"`, `0.50` → `"050"`, `0.75` → `"075"` from `f"{fg:.2f}".replace(".", "")`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_experiment_config.py -v`
Expected: PASS for all tests.

- [ ] **Step 5: Commit**

```bash
git add experiments/configs/builders.py tests/test_experiment_config.py
git commit -m "feat: build_exp2b_configs for rho_e disaggregation sweep

Three configs at rho_e=0.10, N_p=9 with alpha_i=2, alpha_g=10 and
frac_griefer in {0.25, 0.50, 0.75}.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: `run_exp2b.py` entry point

**Files:**
- Create: `experiments/run_exp2b.py`

**Context:** Mirror of `experiments/run_exp2.py`. No LaTeX table formatter is required for this supplement; the analysis will combine exp2b with exp2 endpoints in a follow-up step outside this plan. Omit the `format_exp2_tables` call — the runner still writes per-config CSVs that downstream analysis consumes.

- [ ] **Step 1: Create the file**

Create `experiments/run_exp2b.py`:

```python
"""Experiment 2b: rho_e disaggregation sweep.

3 configs x 200 iterations = 600 runs.
Varies frac_griefer in {0.25, 0.50, 0.75} at fixed rho_e=0.10,
N_p=9, alpha_i=2, alpha_g=10. Endpoints (f_g=0, 1) are recovered
from exp2 at analysis time.
"""

from __future__ import annotations

import argparse
import logging

from experiments.configs.builders import build_exp2b_configs
from experiments.runner import ExperimentRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment 2b")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Print configs without running")
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Max parallel workers for iterations (default: sequential)",
    )
    args = parser.parse_args()

    configs = build_exp2b_configs()
    logger.info("Experiment 2b: %d configs", len(configs))

    if args.dry_run:
        for cfg in configs:
            print(f"  {cfg.name}: {cfg.n_communities}c, {cfg.n_platforms}p, "
                  f"rho={cfg.rho_extremist}, alpha_i={cfg.alpha_ideologue}, "
                  f"alpha_g={cfg.alpha_griefer}, f_g={cfg.frac_griefer}, "
                  f"{cfg.n_iterations}i")
        return

    runner = ExperimentRunner(output_dir=args.output_dir, max_workers=args.workers)
    runner.run_experiment(configs)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the dry run works**

Run: `python -m experiments.run_exp2b --dry-run`
Expected: prints three lines like:
```
  exp2b_fg025: 900c, 9p, rho=0.1, alpha_i=2.0, alpha_g=10.0, f_g=0.25, 200i
  exp2b_fg050: 900c, 9p, rho=0.1, alpha_i=2.0, alpha_g=10.0, f_g=0.5, 200i
  exp2b_fg075: 900c, 9p, rho=0.1, alpha_i=2.0, alpha_g=10.0, f_g=0.75, 200i
```

- [ ] **Step 3: Commit**

```bash
git add experiments/run_exp2b.py
git commit -m "feat: add run_exp2b entry point for rho_e disaggregation sweep

Dry-run verified; full run deferred until plan execution is complete.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Smoke test and regression verification

**Files:**
- Modify: `tests/test_regression.py`

**Context:** Final checkpoint. We want (a) a single-iteration end-to-end run per exp2b config that reaches `end()` and surfaces both subtype metrics, and (b) confirmation that exp1/exp2/oat/interactions configs produce identical `to_params` output to a frozen reference (they should, because the new fields default to `None`/`0.0` and add *new* keys — no existing keys change value). Check (b) by direct assertion on the existing exp2 configs.

- [ ] **Step 1: Write the smoke and regression tests**

Append to `tests/test_regression.py`:

```python
from experiments.configs.builders import build_exp2_configs, build_exp2b_configs
from platform_abm.metrics import compute_extremist_metrics
from platform_abm.model import MiniTiebout


class TestExp2bSmoke:
    def test_single_iteration_per_config(self):
        """Each exp2b config runs to completion and reports both subtype utilities."""
        for cfg in build_exp2b_configs():
            params = cfg.to_params(iteration=0)
            # Shrink for speed — smoke only checks end-to-end plumbing.
            params["steps"] = 3
            params["n_comms"] = 60
            model = MiniTiebout(params)
            model.run()
            metrics = compute_extremist_metrics(model.communities)
            assert "average_ideologue_utility" in metrics
            assert "average_griefer_utility" in metrics
            for v in metrics.values():
                assert v == v  # not NaN


class TestExp2ParamsUnchanged:
    def test_exp2_params_dict_shape_unchanged_for_existing_keys(self):
        """Adding new keys to to_params() must not change existing values."""
        cfg = build_exp2_configs()[0]
        params = cfg.to_params(iteration=0)
        # Spot-check the load-bearing keys.
        assert params["n_comms"] == cfg.n_communities
        assert params["n_plats"] == cfg.n_platforms
        assert params["steps"] == cfg.t_max
        assert params["alpha"] == cfg.alpha
        assert params["percent_extremists"] == int(cfg.rho_extremist * 100)
        # New keys exist and fall back to the scalar alpha.
        assert params["alpha_ideologue"] == cfg.alpha
        assert params["alpha_griefer"] == cfg.alpha
        assert params["frac_griefer"] == 0.0
```

Note on AgentPy lifecycle: the existing `conftest.make_model` calls `model.sim_setup(steps=..., seed=...)`. The smoke test does the same, then `sim_step()` runs one model step (election → utility → relocation). One step is enough to exercise the new formula without waiting for 100 steps.

- [ ] **Step 2: Run the tests**

Run: `pytest tests/test_regression.py::TestExp2bSmoke tests/test_regression.py::TestExp2ParamsUnchanged -v`
Expected: PASS.

- [ ] **Step 3: Run the full test suite**

Run: `pytest`
Expected: all tests PASS. Flag any failures — this plan should not regress any existing behavior.

- [ ] **Step 4: Lint and typecheck**

Run: `make lint && make typecheck`
Expected: both clean. Fix any issues before committing.

- [ ] **Step 5: Commit**

```bash
git add tests/test_regression.py
git commit -m "test: exp2b smoke + exp2 params-dict regression guard

Verifies each exp2b config reaches end() with both subtype utilities
reported and that adding new keys to to_params() does not change
values of existing keys for legacy configs.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Deferred / out of scope

Per spec §Out of Scope — do NOT implement in this plan:

- Tracker per-subtype relocation events (`RelocationEvent.community_subtype`)
- Step-series per-subtype utility/relocation traces in `model.step_log` or `step_series`
- Factorial exp2b × N_p × ρ_e
- Alternative (α_i, α_g) pairs
- Cleanup of mainstream `community.alpha` (unused under new formula, but leaving the assignment avoids incidental churn in unrelated tests)
- Analysis scripts combining exp2 endpoints with exp2b — lives in a follow-up

If any deferred item becomes necessary during implementation, stop and escalate rather than scope-creep.
