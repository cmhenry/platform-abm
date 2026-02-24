"""Tests for Community.last_move_step attribute."""

from tests.conftest import make_model


class TestLastMoveStep:
    def test_initial_value_is_zero(self):
        model = make_model()
        for c in model.communities:
            assert c.last_move_step == 0

    def test_updated_on_relocation(self):
        model = make_model({"steps": 3})
        model.run()
        # Any community that moved should have last_move_step > 0
        movers = [c for c in model.communities if c.moves > 1]
        for c in movers:
            assert c.last_move_step > 0

    def test_stays_zero_if_no_moves(self):
        """Communities that never relocated keep last_move_step == 0."""
        model = make_model({"steps": 3})
        model.run()
        non_movers = [c for c in model.communities if c.moves <= 1]
        for c in non_movers:
            assert c.last_move_step == 0
