"""Tests for ExperimentConfig and config builders."""

from experiments.configs.experiment_config import ExperimentConfig
from experiments.configs.builders import (
    build_exp1_configs,
    build_exp2_configs,
    build_exp2b_configs,
    build_interaction_configs,
    build_oat_configs,
)


def test_to_params_maps_correctly():
    """to_params converts config fields to AgentPy param names."""
    cfg = ExperimentConfig(
        name="test", experiment="test",
        n_communities=100, n_platforms=9, p_space=10, t_max=50,
        institution="mixed", rho_extremist=0.10, alpha=5.0,
    )
    params = cfg.to_params(iteration=0)
    assert params["n_comms"] == 100
    assert params["n_plats"] == 9
    assert params["steps"] == 50
    assert params["percent_extremists"] == 10
    assert params["extremists"] == "yes"
    assert params["alpha"] == 5.0
    assert params["initial_distribution"] == "equal"


def test_seed_generation():
    """Seed = seed_base + iteration."""
    cfg = ExperimentConfig(
        name="test", experiment="test",
        n_communities=50, n_platforms=3, p_space=10, t_max=10,
        institution="direct", rho_extremist=0.0, alpha=0.0,
        seed_base=42,
    )
    assert cfg.to_params(0)["seed"] == 42
    assert cfg.to_params(5)["seed"] == 47
    assert cfg.to_params(199)["seed"] == 241


def test_zero_rho_disables_extremists():
    """rho_extremist=0.0 produces extremists='no'."""
    cfg = ExperimentConfig(
        name="test", experiment="test",
        n_communities=50, n_platforms=3, p_space=10, t_max=10,
        institution="direct", rho_extremist=0.0, alpha=0.0,
    )
    params = cfg.to_params(0)
    assert params["extremists"] == "no"
    assert params["percent_extremists"] == 0


def test_to_dict_round_trip():
    """to_dict and from_dict are inverse operations."""
    cfg = ExperimentConfig(
        name="test_rt", experiment="exp2",
        n_communities=100, n_platforms=9, p_space=10, t_max=50,
        institution="mixed", rho_extremist=0.10, alpha=5.0,
        tracking_enabled=True, n_iterations=200,
    )
    d = cfg.to_dict()
    cfg2 = ExperimentConfig.from_dict(d)
    assert cfg2.to_dict() == d


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


def test_exp1_config_count():
    """Experiment 1 produces 6 configs."""
    assert len(build_exp1_configs()) == 6


def test_exp2_config_count():
    """Experiment 2 produces 27 configs (3x3x3)."""
    assert len(build_exp2_configs()) == 27


def test_oat_config_count():
    """OAT produces 13 configs (5 params x 2 test values + 1 param x 3 test values)."""
    assert len(build_oat_configs()) == 13


def test_interaction_config_count():
    """Interaction produces 6 configs (3 alpha x 2 p_space)."""
    assert len(build_interaction_configs()) == 6


def test_exp1_no_extremists():
    """All Exp1 configs have rho=0 and alpha=0."""
    for cfg in build_exp1_configs():
        assert cfg.rho_extremist == 0.0
        assert cfg.alpha == 0.0


def test_mu_in_params():
    """mu flows through to_params() and to_dict()."""
    cfg = ExperimentConfig(
        name="test_mu", experiment="test",
        n_communities=100, n_platforms=9, p_space=10, t_max=50,
        institution="mixed", rho_extremist=0.10, alpha=5.0, mu=0.08,
    )
    params = cfg.to_params(iteration=0)
    assert params["mu"] == 0.08
    d = cfg.to_dict()
    assert d["mu"] == 0.08


def test_mu_default_in_params():
    """mu defaults to 0.05 in to_params()."""
    cfg = ExperimentConfig(
        name="test_mu_default", experiment="test",
        n_communities=100, n_platforms=9, p_space=10, t_max=50,
        institution="mixed", rho_extremist=0.0, alpha=0.0,
    )
    assert cfg.to_params(0)["mu"] == 0.05


def test_exp2_all_mixed():
    """All Exp2 configs use mixed institution."""
    for cfg in build_exp2_configs():
        assert cfg.institution == "mixed"


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
