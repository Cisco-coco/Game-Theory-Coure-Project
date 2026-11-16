"""Behavior checks against small-game invariants, not implementation details."""
import unittest
import numpy as np
import pyspiel
from open_spiel.python import policy
from open_spiel.python.algorithms import cfr, exploitability
from evaluation import evaluate


def check_uniform_kuhn_exact_values():
    game = pyspiel.load_game("kuhn_poker")
    strategy = policy.UniformRandomPolicy(game)
    metrics = evaluate(game, strategy)
    assert np.isclose(metrics["value_p0"], 1 / 8)
    assert np.isclose(metrics["exploitability"], 11 / 24)
    assert np.isclose(metrics["value_p0"] + metrics["value_p1"], 0)
    assert np.isclose(metrics["exploitability"], exploitability.exploitability(game, strategy))


def check_kuhn_cfr_improves_and_has_valid_information_set_policy():
    game = pyspiel.load_game("kuhn_poker")
    solver = cfr.CFRSolver(game)
    for _ in range(200):
        solver.evaluate_and_update_policy()
    average = solver.average_policy()
    assert evaluate(game, average)["exploitability"] < .02
    assert abs(evaluate(game, average)["value_p0"] + 1 / 18) < .02
    seen = {}
    def visit(state):
        if state.is_terminal():
            assert sum(state.returns()) == 0
            return
        if state.is_chance_node():
            for action, _ in state.chance_outcomes():
                visit(state.child(action))
            return
        probs = average.action_probabilities(state)
        assert np.isclose(sum(probs.values()), 1)
        assert all(p >= 0 and a in state.legal_actions() for a, p in probs.items())
        key = (state.current_player(), state.information_state_string())
        if key in seen:
            assert probs == seen[key]
        seen[key] = probs
        for action in state.legal_actions():
            visit(state.child(action))
    visit(game.new_initial_state())


def check_leduc_training_and_exact_evaluation():
    game = pyspiel.load_game("leduc_poker", {"players": 2})
    baseline = evaluate(game, policy.UniformRandomPolicy(game))
    solver = cfr.CFRSolver(game)
    for _ in range(10):
        solver.evaluate_and_update_policy()
    metrics = evaluate(game, solver.average_policy())
    assert 0 <= metrics["exploitability"] < baseline["exploitability"]
    assert np.isclose(metrics["value_p0"] + metrics["value_p1"], 0)
    assert np.isclose(metrics["exploitability"],
                      exploitability.exploitability(game, solver.average_policy()))


class EvaluationTests(unittest.TestCase):
    def test_uniform_kuhn_exact_values(self):
        check_uniform_kuhn_exact_values()

    def test_kuhn_cfr_improves_and_has_valid_information_set_policy(self):
        check_kuhn_cfr_improves_and_has_valid_information_set_policy()

    def test_leduc_training_and_exact_evaluation(self):
        check_leduc_training_and_exact_evaluation()

    def test_rejects_multiplayer_evaluation(self):
        game = pyspiel.load_game("kuhn_poker", {"players": 3})
        with self.assertRaises(ValueError):
            evaluate(game, policy.UniformRandomPolicy(game))


if __name__ == "__main__":
    unittest.main()
