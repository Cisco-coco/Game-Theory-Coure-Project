"""Integration checks for training cadence and the public resume contract."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from deep_cfr import DeepCFR
from train_deep_cfr import run_seed


def arguments(output, **overrides):
    values = dict(output=output, resume=None, game="kuhn_poker", device="cpu",
                  hidden=16, traversals=5, capacity=50, batch_size=16,
                  advantage_steps=2, strategy_steps=3, learning_rate=.001,
                  iterations=4, eval_every=2, keep_checkpoints=True)
    values.update(overrides)
    return SimpleNamespace(**values)


class TrainingTests(unittest.TestCase):
    def run_quietly(self, args):
        with redirect_stdout(io.StringIO()):
            return run_seed(args, 7)

    def test_evaluation_frequency_does_not_change_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.run_quietly(arguments(root / "every-one", eval_every=1))
            self.run_quietly(arguments(root / "every-two", eval_every=2))
            a, _ = DeepCFR.load(root / "every-one/seed-7/checkpoint.pt")
            b, state = DeepCFR.load(root / "every-two/seed-7/checkpoint.pt")
            np.testing.assert_array_equal(a.average_policy().action_probability_array,
                                          b.average_policy().action_probability_array)
            self.assertEqual(b.strategy_gradient_steps, 4 * 3)
            self.assertEqual(state["strategy_schedule"], "every_iteration")
            self.assertTrue((root / "every-two/seed-7/iteration-000004.pt").is_file())

    def test_cli_runner_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.run_quietly(arguments(root / "full"))
            self.run_quietly(arguments(root / "prefix", iterations=2))
            checkpoint = root / "prefix/seed-7/checkpoint.pt"
            self.run_quietly(arguments(root / "resumed", resume=checkpoint))
            full, _ = DeepCFR.load(root / "full/seed-7/checkpoint.pt")
            resumed, state = DeepCFR.load(root / "resumed/seed-7/checkpoint.pt")
            np.testing.assert_array_equal(full.average_policy().action_probability_array,
                                          resumed.average_policy().action_probability_array)
            self.assertEqual([r["iteration"] for r in state["rows"]], [1, 2, 4])
            self.assertEqual(full.node_visits, resumed.node_visits)


if __name__ == "__main__":
    unittest.main()
