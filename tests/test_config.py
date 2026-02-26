"""Tests for config module: Pydantic validation, enums, normalization."""

import pytest

from platform_abm.config import (
    CommunityType,
    InstitutionType,
    SimulationConfig,
    StopCondition,
    Strategy,
)


class TestEnums:
    def test_institution_type_values(self):
        assert InstitutionType.DIRECT.value == "direct"
        assert InstitutionType.COALITION.value == "coalition"
        assert InstitutionType.ALGORITHMIC.value == "algorithmic"
        assert InstitutionType.MIXED.value == "mixed"

    def test_community_type_values(self):
        assert CommunityType.MAINSTREAM.value == "mainstream"
        assert CommunityType.EXTREMIST.value == "extremist"

    def test_strategy_values(self):
        assert Strategy.STAY.value == "stay"
        assert Strategy.MOVE.value == "move"
        assert Strategy.UNSET.value == ""

    def test_stop_condition_values(self):
        assert StopCondition.STEPS.value == "steps"
        assert StopCondition.SATISFICED.value == "satisficed"


class TestSimulationConfig:
    def _base_params(self, **overrides):
        params = {
            "n_comms": 100,
            "n_plats": 5,
            "p_space": 10,
            "institution": "direct",
            "extremists": "no",
        }
        params.update(overrides)
        return params

    def test_basic_config(self):
        config = SimulationConfig(**self._base_params())
        assert config.n_comms == 100
        assert config.institution == InstitutionType.DIRECT
        assert config.extremists is False

    def test_normalize_algorithm_to_algorithmic(self):
        config = SimulationConfig(**self._base_params(institution="algorithm"))
        assert config.institution == InstitutionType.ALGORITHMIC

    def test_normalize_extremists_yes_no(self):
        config_yes = SimulationConfig(**self._base_params(extremists="yes"))
        assert config_yes.extremists is True

        config_no = SimulationConfig(**self._base_params(extremists="no"))
        assert config_no.extremists is False

    def test_normalize_extremists_bool(self):
        config = SimulationConfig(**self._base_params(extremists=True))
        assert config.extremists is True

    def test_to_agentpy_params(self):
        config = SimulationConfig(**self._base_params(seed=1999))
        params = config.to_agentpy_params()
        assert params["institution"] == "direct"
        assert params["extremists"] == "no"
        assert params["seed"] == 1999

    def test_to_agentpy_params_no_seed(self):
        config = SimulationConfig(**self._base_params())
        params = config.to_agentpy_params()
        assert "seed" not in params

    def test_invalid_n_comms(self):
        with pytest.raises(ValueError):
            SimulationConfig(**self._base_params(n_comms=0))

    def test_invalid_n_plats(self):
        with pytest.raises(ValueError):
            SimulationConfig(**self._base_params(n_plats=0))

    def test_invalid_p_space(self):
        with pytest.raises(ValueError):
            SimulationConfig(**self._base_params(p_space=0))

    def test_invalid_institution(self):
        with pytest.raises(ValueError):
            SimulationConfig(**self._base_params(institution="invalid"))

    def test_defaults(self):
        config = SimulationConfig(**self._base_params())
        assert config.steps == 50
        assert config.coalitions == 3
        assert config.mutations == 2
        assert config.search_steps == 10
        assert config.svd_groups == 3
        assert config.stop_condition == StopCondition.STEPS

    def test_mu_default(self):
        config = SimulationConfig(**self._base_params())
        assert config.mu == 0.05

    def test_mu_in_agentpy_params(self):
        config = SimulationConfig(**self._base_params(mu=0.10))
        params = config.to_agentpy_params()
        assert params["mu"] == 0.10
