"""Differential tests against pinned OpenSpiel, plus policy invariants."""
import numpy as np
import pyspiel
import unittest
from open_spiel.python.algorithms import cfr, exploitability
from tabular_cfr import TabularCFR, regret_matching


def check_matches_official(game_name, iterations, plus):
    game = pyspiel.load_game(game_name)
    ours = TabularCFR(game, plus=plus)
    reference = (cfr.CFRPlusSolver if plus else cfr.CFRSolver)(game)
    for step in range(1, iterations + 1):
        ours.evaluate_and_update_policy()
        reference.evaluate_and_update_policy()
        if step in (1, 2, 5, iterations):
            for method in ("current_policy", "average_policy"):
                actual = getattr(ours, method)().action_probability_array
                expected = getattr(reference, method)().action_probability_array
                np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-8)
            if plus:
                assert np.all(ours.regrets >= 0)
    assert abs(exploitability.nash_conv(game, ours.average_policy()) -
               exploitability.nash_conv(game, reference.average_policy())) < 1e-9


def check_simultaneous_reference_and_snapshot_independence():
    game = pyspiel.load_game("kuhn_poker")
    ours = TabularCFR(game, alternating=False)
    # Private reference API is deliberately confined to this version-pinned test.
    reference = cfr._CFRSolver(game, alternating_updates=False,
                               linear_averaging=False, regret_matching_plus=False)
    snapshot = ours.current_policy()
    original = snapshot.action_probability_array.copy()
    for _ in range(30):
        ours.evaluate_and_update_policy()
        reference.evaluate_and_update_policy()
    np.testing.assert_allclose(ours.average_policy().action_probability_array,
                               reference.average_policy().action_probability_array, atol=1e-10)
    np.testing.assert_array_equal(snapshot.action_probability_array, original)


def check_regret_matching_masks_illegal_actions_and_zero_fallback():
    result = regret_matching(np.array([[-1., 2., 6.], [-3., 0., 100.]]),
                             np.array([[True, True, True], [True, True, False]]))
    np.testing.assert_allclose(result, [[0, .25, .75], [.5, .5, 0]])


class TabularTests(unittest.TestCase):
    def test_matches_official_at_multiple_checkpoints(self):
        for game_name, iterations in (("kuhn_poker", 60), ("leduc_poker", 15)):
            for plus in (False, True):
                with self.subTest(game=game_name, plus=plus):
                    check_matches_official(game_name, iterations, plus)

    def test_simultaneous_reference_and_snapshot_independence(self):
        check_simultaneous_reference_and_snapshot_independence()

    def test_regret_matching_masks_illegal_actions_and_zero_fallback(self):
        check_regret_matching_masks_illegal_actions_and_zero_fallback()

    def test_rejects_out_of_scope_games(self):
        with self.assertRaises(ValueError):
            TabularCFR(pyspiel.load_game("tic_tac_toe"))


if __name__ == "__main__":
    unittest.main()
