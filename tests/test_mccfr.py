"""Run with tor's standard-library unittest; no new dependencies required."""
import unittest
import numpy as np
import pyspiel
from open_spiel.python import policy
from open_spiel.python.algorithms import external_sampling_mccfr as reference
from external_mccfr import ExternalSamplingMCCFR
from reservoir import ReservoirBuffer
from evaluation import evaluate


class SamplingTests(unittest.TestCase):
    def test_reference_same_random_stream(self):
        original_rng = np.random.get_state()
        try:
            for game_name in ("kuhn_poker", "leduc_poker"):
                for seed in (7, 42):
                    with self.subTest(game=game_name, seed=seed):
                        game = pyspiel.load_game(game_name)
                        local = ExternalSamplingMCCFR(game, seed)
                        official = reference.ExternalSamplingSolver(game)
                        np.random.seed(seed)
                        for i in range(100):
                            local.evaluate_and_update_policy()
                            official.iteration()
                            if i in (0, 9, 99):
                                a = policy.tabular_policy_from_callable(game, local.average_policy().action_probabilities)
                                b = policy.tabular_policy_from_callable(game, official.average_policy().action_probabilities)
                                np.testing.assert_allclose(a.action_probability_array, b.action_probability_array, atol=1e-12)
                        self.assertAlmostEqual(evaluate(game, local.average_policy())["exploitability"],
                                               evaluate(game, official.average_policy())["exploitability"], places=12)
        finally:
            np.random.set_state(original_rng)

    def test_rng_independence_snapshot_and_sparse_evaluation(self):
        game = pyspiel.load_game("leduc_poker")
        a, b = ExternalSamplingMCCFR(game, 11), ExternalSamplingMCCFR(game, 11)
        initial = a.average_policy()
        for _ in range(30):
            a.evaluate_and_update_policy()
            np.random.random(20)
            b.evaluate_and_update_policy()
        self.assertEqual(a.counters(), b.counters())
        for key in a.nodes:
            np.testing.assert_array_equal(a.nodes[key][1], b.nodes[key][1])
        visits = a.counters().copy()
        evaluate(game, a.average_policy())
        self.assertEqual(visits, a.counters())
        self.assertEqual(len(initial.nodes), 0)
        self.assertLess(len(a.nodes), len(policy.TabularPolicy(game).state_lookup))

    def test_seeded_kuhn_learns(self):
        game = pyspiel.load_game("kuhn_poker")
        solver = ExternalSamplingMCCFR(game, 42)
        for _ in range(3000):
            solver.evaluate_and_update_policy()
        self.assertLess(evaluate(game, solver.average_policy())["exploitability"], .05)

    def test_reservoir_capacity_copy_and_uniform_inclusion(self):
        buffer = ReservoirBuffer(2, 42)
        sample = {"iteration": 1, "values": [1.]}
        buffer.add(sample)
        sample["values"][0] = 99
        self.assertEqual(buffer.sample(1)[0]["values"], [1.])
        counts = np.zeros(10)
        for seed in range(2000):
            buffer = ReservoirBuffer(3, seed)
            for i in range(10):
                buffer.add(i)
            self.assertEqual((len(buffer), buffer.seen), (3, 10))
            counts[buffer.items] += 1
        # Expected inclusion=3/10; wide deterministic statistical sanity bound.
        self.assertTrue(np.all(np.abs(counts / 2000 - .3) < .05))
        with self.assertRaises(ValueError):
            ReservoirBuffer(0)


if __name__ == "__main__":
    unittest.main()
