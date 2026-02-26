"""Config builders for all experiments."""

from __future__ import annotations

from experiments.configs.experiment_config import ExperimentConfig

# Shared fixed parameters across all experiments
_COMMON_FIXED: dict = {
    "coalitions": 5,
    "mutations": 3,
    "svd_groups": 10,
    "search_steps": 10,
    "initial_distribution": "equal",
    "seed_base": 42,
}

# Baseline values for sensitivity analysis
_BASELINE = {
    "n_communities": 900,
    "n_platforms": 9,
    "p_space": 10,
    "t_max": 100,
    "rho_extremist": 0.10,
    "alpha": 5.0,
    "mu": 0.05,
}


def build_exp1_configs() -> list[ExperimentConfig]:
    """Experiment 1: institutional comparisons without extremists.

    6 configs:
      1a-1c: Homogeneous institutions (direct/coalition/algorithmic), N_p=9
      1d-1f: Mixed institution at varying platform counts (N_p=3/9/27)
    All use N_c=300, t_max=100, no extremists.
    """
    _exp1_shared = dict(
        experiment="exp1",
        n_communities=300,
        p_space=10,
        t_max=100,
        rho_extremist=0.0,
        alpha=0.0,
        tracking_enabled=False,
        **_COMMON_FIXED,
    )

    configs = [
        # 1a-1c: Homogeneous institutions, N_p=9
        ExperimentConfig(name="exp1_direct_np9", institution="direct", n_platforms=9, **_exp1_shared),
        ExperimentConfig(name="exp1_coalition_np9", institution="coalition", n_platforms=9, **_exp1_shared),
        ExperimentConfig(name="exp1_algorithmic_np9", institution="algorithmic", n_platforms=9, **_exp1_shared),
        # 1d-1f: Mixed institution, varying N_p
        ExperimentConfig(name="exp1_mixed_np9", institution="mixed", n_platforms=9, **_exp1_shared),
        ExperimentConfig(name="exp1_mixed_np3", institution="mixed", n_platforms=3, **_exp1_shared),
        ExperimentConfig(name="exp1_mixed_np27", institution="mixed", n_platforms=27, **_exp1_shared),
    ]

    return configs


def build_exp2_configs() -> list[ExperimentConfig]:
    """Experiment 2: mixed institutions with extremists.

    27 configs: N_p x rho x alpha factorial design.
    All use mixed institution type.
    """
    configs = []
    n_platforms_values = [3, 6, 9]
    rho_values = [0.05, 0.10, 0.20]
    alpha_values = [2.0, 5.0, 10.0]

    for np_val in n_platforms_values:
        for rho in rho_values:
            for alpha in alpha_values:
                rho_str = f"{rho:.2f}".replace(".", "")
                name = f"exp2_np{np_val}_rho{rho_str}_alpha{int(alpha)}"
                configs.append(ExperimentConfig(
                    name=name,
                    experiment="exp2",
                    n_communities=900,
                    n_platforms=np_val,
                    p_space=10,
                    t_max=100,
                    institution="mixed",
                    rho_extremist=rho,
                    alpha=alpha,
                    tracking_enabled=True,
                    **_COMMON_FIXED,
                ))

    return configs


def build_oat_configs() -> list[ExperimentConfig]:
    """OAT sensitivity: one parameter at a time around baseline.

    5 parameters, each with 1-3 test values (excluding baseline).
    Baseline: N_c=900, N_p=9, p_space=10, rho=0.10, alpha=5.0
    """
    configs = []

    # Parameter: n_communities (baseline=900)
    for n_c in [50, 200]:
        configs.append(ExperimentConfig(
            name=f"oat_nc{n_c}",
            experiment="oat",
            n_communities=n_c,
            n_platforms=_BASELINE["n_platforms"],
            p_space=_BASELINE["p_space"],
            t_max=_BASELINE["t_max"],
            institution="mixed",
            rho_extremist=_BASELINE["rho_extremist"],
            alpha=_BASELINE["alpha"],
            tracking_enabled=True,
            **_COMMON_FIXED,
        ))

    # Parameter: n_platforms (baseline=9)
    for np_val in [3, 6]:
        configs.append(ExperimentConfig(
            name=f"oat_np{np_val}",
            experiment="oat",
            n_communities=_BASELINE["n_communities"],
            n_platforms=np_val,
            p_space=_BASELINE["p_space"],
            t_max=_BASELINE["t_max"],
            institution="mixed",
            rho_extremist=_BASELINE["rho_extremist"],
            alpha=_BASELINE["alpha"],
            tracking_enabled=True,
            **_COMMON_FIXED,
        ))

    # Parameter: p_space (baseline=10)
    for ps in [5, 20]:
        configs.append(ExperimentConfig(
            name=f"oat_pspace{ps}",
            experiment="oat",
            n_communities=_BASELINE["n_communities"],
            n_platforms=_BASELINE["n_platforms"],
            p_space=ps,
            t_max=_BASELINE["t_max"],
            institution="mixed",
            rho_extremist=_BASELINE["rho_extremist"],
            alpha=_BASELINE["alpha"],
            tracking_enabled=True,
            **_COMMON_FIXED,
        ))

    # Parameter: rho_extremist (baseline=0.10)
    for rho in [0.05, 0.20]:
        rho_str = f"{rho:.2f}".replace(".", "")
        configs.append(ExperimentConfig(
            name=f"oat_rho{rho_str}",
            experiment="oat",
            n_communities=_BASELINE["n_communities"],
            n_platforms=_BASELINE["n_platforms"],
            p_space=_BASELINE["p_space"],
            t_max=_BASELINE["t_max"],
            institution="mixed",
            rho_extremist=rho,
            alpha=_BASELINE["alpha"],
            tracking_enabled=True,
            **_COMMON_FIXED,
        ))

    # Parameter: alpha (baseline=5.0)
    for alpha in [2.0, 10.0]:
        configs.append(ExperimentConfig(
            name=f"oat_alpha{int(alpha)}",
            experiment="oat",
            n_communities=_BASELINE["n_communities"],
            n_platforms=_BASELINE["n_platforms"],
            p_space=_BASELINE["p_space"],
            t_max=_BASELINE["t_max"],
            institution="mixed",
            rho_extremist=_BASELINE["rho_extremist"],
            alpha=alpha,
            tracking_enabled=True,
            **_COMMON_FIXED,
        ))

    # Parameter: mu (baseline=0.05)
    for mu_val in [0.0, 0.02, 0.10]:
        mu_str = f"{mu_val:.2f}".replace(".", "")
        configs.append(ExperimentConfig(
            name=f"oat_mu{mu_str}",
            experiment="oat",
            n_communities=_BASELINE["n_communities"],
            n_platforms=_BASELINE["n_platforms"],
            p_space=_BASELINE["p_space"],
            t_max=_BASELINE["t_max"],
            institution="mixed",
            rho_extremist=_BASELINE["rho_extremist"],
            alpha=_BASELINE["alpha"],
            mu=mu_val,
            tracking_enabled=True,
            **_COMMON_FIXED,
        ))

    # Note: many OAT configs overlap with Exp2 configs (e.g. oat_np3 == exp2_np3_rho010_alpha5).
    # The runner's skip-if-done mechanism handles reuse automatically.
    return configs


def build_interaction_configs() -> list[ExperimentConfig]:
    """Interaction analysis: alpha x p_space for non-baseline p_space values.

    6 configs: alpha {2, 5, 10} x p_space {5, 20}.
    (p_space=10 column comes from Exp2 results, no new runs needed.)
    """
    configs = []
    alpha_values = [2.0, 5.0, 10.0]
    p_space_values = [5, 20]

    for alpha in alpha_values:
        for ps in p_space_values:
            configs.append(ExperimentConfig(
                name=f"interact_alpha{int(alpha)}_pspace{ps}",
                experiment="interactions",
                n_communities=_BASELINE["n_communities"],
                n_platforms=_BASELINE["n_platforms"],
                p_space=ps,
                t_max=_BASELINE["t_max"],
                institution="mixed",
                rho_extremist=_BASELINE["rho_extremist"],
                alpha=alpha,
                tracking_enabled=True,
                **_COMMON_FIXED,
            ))

    return configs
