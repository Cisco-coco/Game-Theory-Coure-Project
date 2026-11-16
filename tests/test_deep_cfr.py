import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from deep_cfr import Config, DeepCFR, masked_softmax, sampled_advantages, weighted_mse, validate_policy
from reservoir import ReservoirBuffer
from evaluation import evaluate


def small(**kwargs):
    values = dict(hidden=16, traversals=10, capacity=200, batch_size=32,
                  advantage_steps=4, strategy_steps=5)
    values.update(kwargs)
    return Config(**values)


class DeepTests(unittest.TestCase):
    def test_checkpoint_compatibility_guard(self):
        solver = DeepCFR(small())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed.pt"
            solver.save(path)
            payload = torch.load(path, weights_only=False)
            payload["versions"]["torch"] = "incompatible-test-version"
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "compatibility mismatch"):
                DeepCFR.load(path)
            recovered, _ = DeepCFR.load(path, allow_incompatible=True)
            self.assertEqual(recovered.iteration, 0)
            solver.save(path)
            payload = torch.load(path, weights_only=False)
            payload["source_sha256"]["deep_cfr.py"] = "changed-source"
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "deep_cfr.py"):
                DeepCFR.load(path)

    def test_sample_labels_match_reference_under_fixed_policy(self):
        from open_spiel.python.pytorch import deep_cfr as reference
        original_rng = np.random.get_state()
        try:
            for game in ("kuhn_poker", "leduc_poker"):
                local = DeepCFR(small(game=game, capacity=10000, seed=42))
                official = reference.DeepCFRSolver(local.game, policy_network_layers=(8,),
                    advantage_network_layers=(8,), memory_capacity=10000)
                def uniform(state, player=None):
                    actor = state.current_player()
                    mask = np.asarray(state.legal_actions_mask(actor), dtype=bool)
                    probs = mask.astype(float) / mask.sum()
                    return np.asarray(state.information_state_tensor(actor), dtype=np.float32), mask, probs
                local._info = uniform
                official._sample_action_from_advantage = lambda state, player: (None, uniform(state)[2])
                local.iteration = 1
                for _ in range(5):
                    local._traverse(local.game.new_initial_state(), 0)
                np.random.seed(42)
                for _ in range(5):
                    official._traverse_game_tree(local.game.new_initial_state(), 0)
                expected = list(official.advantage_buffers[0])
                self.assertEqual(len(expected), len(local.advantage_memories[0]))
                for actual, sample in zip(local.advantage_memories[0].items, expected):
                    np.testing.assert_allclose(actual["x"], sample.info_state)
                    np.testing.assert_allclose(actual["y"], sample.advantage, atol=1e-6)
                expected = list(official.strategy_buffer)
                self.assertEqual(len(expected), len(local.strategy_memory))
                for actual, sample in zip(local.strategy_memory.items, expected):
                    np.testing.assert_allclose(actual["y"], sample.strategy_action_probs, atol=1e-7)
        finally:
            np.random.set_state(original_rng)

    def test_mask_and_linear_iteration_weight(self):
        logits = torch.tensor([[10000., 0., -10000.]], requires_grad=True)
        mask = torch.tensor([[False, True, True]])
        probs = masked_softmax(logits, mask)
        np.testing.assert_allclose(probs.detach(), [[0, 1, 0]])
        prediction = torch.ones((2, 3), requires_grad=True)
        mask = torch.tensor([[True, True, False]] * 2)
        loss = weighted_mse(prediction, torch.zeros_like(prediction), mask, torch.tensor([1., 2.]), 2)
        self.assertEqual(loss.item(), .75)  # t/T, not squared iteration weights
        loss.backward()
        self.assertTrue(torch.equal(prediction.grad[:, 2], torch.zeros(2)))
        np.testing.assert_array_equal(sampled_advantages(np.array([2., -1., 999.]),
            np.array([.5, .5, 0.]), np.array([True, True, False])), [1.5, -1.5, 0.])

    def test_all_information_sets_and_sample_contract(self):
        for game in ("kuhn_poker", "leduc_poker"):
            with self.subTest(game=game):
                solver = DeepCFR(small(game=game))
                validate_policy(solver.average_policy())
                solver.step()
                solver.train_average()
                table = solver.average_policy()
                validate_policy(table)
                self.assertTrue(np.isfinite(evaluate(solver.game, table)["exploitability"]))
                for player, memory in enumerate(solver.advantage_memories):
                    for sample in memory.items:
                        self.assertEqual(sample["player"], player)
                        self.assertEqual(sample["t"], 1)
                        self.assertTrue(np.all(sample["y"][~sample["mask"]] == 0))
                for sample in solver.strategy_memory.items:
                    self.assertAlmostEqual(float(sample["y"].sum()), 1., places=6)
                table.action_probability_array *= .5
                with self.assertRaises(ValueError):
                    validate_policy(table)

    def test_optimizer_rebound_and_small_buffer_trains(self):
        solver = DeepCFR(small(batch_size=10000))
        old_params = list(solver.advantages[0].parameters())
        solver.step()
        new_params = list(solver.advantages[0].parameters())
        self.assertIsNot(old_params[0], new_params[0])
        self.assertIs(new_params[0], solver.advantage_optimizers[0].param_groups[0]["params"][0])
        self.assertEqual(solver.gradient_steps, 8)
        self.assertGreater(len(solver.advantage_optimizers[0].state), 0)

    def test_checkpoint_continuation_and_evaluation_rng_independence(self):
        original = DeepCFR(small(capacity=10))  # exercise reservoir replacement on resume
        original.step()
        original.train_average()
        before = original.average_policy().action_probability_array.copy()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            original.save(path, {"test": 7})
            restored, run_state = DeepCFR.load(path)
            self.assertEqual(run_state, {"test": 7})
            np.testing.assert_array_equal(before, restored.average_policy().action_probability_array)
            original.average_policy()  # extra evaluation must not consume training RNG
            original.step()
            restored.step()
            original.train_average()
            restored.train_average()
            np.testing.assert_array_equal(original.average_policy().action_probability_array,
                                          restored.average_policy().action_probability_array)
            self.assertEqual(original.node_visits, restored.node_visits)
            for a, b in zip(original.advantages, restored.advantages):
                for x, y in zip(a.parameters(), b.parameters()):
                    self.assertTrue(torch.equal(x, y))

    def test_fixed_batch_overfit(self):
        solver = DeepCFR(small(learning_rate=.01))
        memory = ReservoirBuffer(8)
        for i in range(8):
            memory.add({"x": np.ones(11, dtype=np.float32)*i/8,
                        "y": np.array([.3, -.2], dtype=np.float32),
                        "mask": np.ones(2, dtype=bool), "t": 1})
        solver.iteration = 1
        stats = solver._train(solver.advantages[0], solver.advantage_optimizers[0],
                              memory, solver.batch_rng[0], 80)
        self.assertLess(stats["last_loss"], stats["first_loss"] * .1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_training_and_cpu_restore(self):
        solver = DeepCFR(small(device="cuda", traversals=3, advantage_steps=2, strategy_steps=2))
        solver.step()
        solver.train_average()
        probs = solver.average_policy().action_probability_array
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cuda.pt"
            solver.save(path)
            restored, _ = DeepCFR.load(path, device="cpu")
            np.testing.assert_allclose(probs, restored.average_policy().action_probability_array, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
