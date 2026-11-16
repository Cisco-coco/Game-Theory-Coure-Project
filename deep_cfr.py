# Copyright 2019 DeepMind Technologies Limited
# Educational rewrite/adaptation: this project, 2026.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations.
"""Deep CFR for two-player Kuhn/Leduc with valid neural policies and checkpoints.

Reference: Brown et al., ICML 2019; OpenSpiel 1.6.11 python/pytorch/deep_cfr.py.
https://github.com/google-deepmind/open_spiel/blob/v1.6.11/open_spiel/python/pytorch/deep_cfr.py
Changes: masked losses/softmax, independent random streams, sparse frozen-policy
cache during traversals, configurable devices, checkpoint/resume, batched exact
evaluation. Uses project ReservoirBuffer. No official solver is invoked.
"""
from dataclasses import asdict, dataclass
import copy
import importlib.metadata as md
import hashlib
from pathlib import Path
import random
import time
import numpy as np
import pyspiel
import torch
from torch import nn
from open_spiel.python import policy
from reservoir import ReservoirBuffer


@dataclass
class Config:
    game: str = "kuhn_poker"
    seed: int = 42
    hidden: int = 64
    traversals: int = 200
    capacity: int = 30000
    batch_size: int = 256
    advantage_steps: int = 300
    strategy_steps: int = 500
    learning_rate: float = .001
    device: str = "cpu"
    threads: int = 1


def masked_softmax(logits, mask):
    if not bool(mask.any(dim=-1).all()):
        raise ValueError("Every row must have a legal action")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("Non-finite neural output")
    return torch.softmax(logits.masked_fill(~mask, -torch.inf), dim=-1)


def weighted_mse(prediction, target, mask, iteration, current_iteration):
    # Mean over legal actions per sample, then t/T weighting. 1/T is a fixed
    # normalization for the training stage, not a random batch denominator.
    error = ((prediction-target).square() * mask).sum(-1) / mask.sum(-1)
    return (error * (iteration / current_iteration)).mean()


def sampled_advantages(values, strategy, mask):
    return np.where(mask, values - np.dot(strategy, values), 0.).astype(np.float32)


def validate_policy(table):
    probs = table.action_probability_array
    mask = table.legal_actions_mask.astype(bool)
    if not np.isfinite(probs).all() or np.any(probs < 0) or np.any(probs[~mask] != 0):
        raise ValueError("Policy contains invalid/illegal action probabilities")
    if not np.allclose(probs.sum(1), 1., atol=1e-7, rtol=0):
        raise ValueError("Legal probabilities must sum to one")


class DeepCFR:
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        if config.game not in ("kuhn_poker", "leduc_poker"):
            raise ValueError("Only two-player Kuhn/Leduc supported")
        if min(config.hidden, config.traversals, config.capacity, config.batch_size,
               config.advantage_steps, config.strategy_steps, config.threads) < 1 or config.learning_rate <= 0:
            raise ValueError("Training sizes and learning rate must be positive")
        self.game = pyspiel.load_game(config.game, {"players": 2})
        self.device = torch.device(config.device)
        torch.set_num_threads(config.threads)
        self.input_size = self.game.information_state_tensor_shape()[0]
        self.actions = self.game.num_distinct_actions()
        self.rng = np.random.RandomState(config.seed)
        self.init_rng = np.random.RandomState(config.seed + 1)
        self.batch_rng = [random.Random(config.seed + 11 + i) for i in range(2)]
        self.strategy_rng = random.Random(config.seed + 20)
        self.advantage_memories = [ReservoirBuffer(config.capacity, config.seed + 30 + i) for i in range(2)]
        self.strategy_memory = ReservoirBuffer(config.capacity, config.seed + 40)
        self.advantages = [self._network() for _ in range(2)]
        self.strategy = self._network()
        self.advantage_optimizers = [self._optimizer(net) for net in self.advantages]
        self.strategy_optimizer = self._optimizer(self.strategy)
        self.iteration = 0
        self.node_visits = 0
        self.gradient_steps = 0
        self.strategy_gradient_steps = 0
        self.inference_calls = 0
        self.cache = {}
        self._table = None
        self.history = []

    def _network(self):
        # Isolate initialization from ambient torch RNG; all later computation
        # is deterministic (no dropout). Store init_rng in checkpoints.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(self.init_rng.randint(2**31-1)))
            net = nn.Sequential(nn.Linear(self.input_size, self.config.hidden), nn.ReLU(),
                                nn.Linear(self.config.hidden, self.config.hidden), nn.ReLU(),
                                nn.Linear(self.config.hidden, self.actions))
        return net.to(self.device)

    def _optimizer(self, net):
        return torch.optim.Adam(net.parameters(), lr=self.config.learning_rate)

    def _synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _info(self, state):
        actor = state.current_player()
        key = (actor, state.information_state_string(actor))
        if key not in self.cache:
            x = np.asarray(state.information_state_tensor(actor), dtype=np.float32)
            mask = np.asarray(state.legal_actions_mask(actor), dtype=bool)
            with torch.no_grad():
                raw = self.advantages[actor](torch.as_tensor(x, device=self.device)).cpu().numpy().astype(float)
            if not np.isfinite(raw).all():
                raise ValueError("Non-finite predicted advantages")
            positive = np.where(mask, np.maximum(raw, 0.), 0.)
            if positive.sum() > 0:
                probs = positive / positive.sum()
            else:
                # Matches the installed Deep CFR reference's best-action fallback.
                probs = np.zeros(self.actions)
                probs[np.argmax(np.where(mask, raw, -np.inf))] = 1.
            self.cache[key] = (x, mask, probs)
            self.inference_calls += 1
        return self.cache[key]

    def _traverse(self, state, target):
        self.node_visits += 1
        if state.is_terminal():
            return state.player_return(target)
        if state.is_chance_node():
            choices, probs = zip(*state.chance_outcomes())
            return self._traverse(state.child(int(self.rng.choice(choices, p=probs))), target)
        actor = state.current_player()
        x, mask, probs = self._info(state)
        sample = {"x": x, "mask": mask, "t": self.iteration, "player": actor}
        if actor != target:
            self.strategy_memory.add(sample | {"y": probs.astype(np.float32)})
            action = int(self.rng.choice(self.actions, p=probs))
            return self._traverse(state.child(action), target)
        values = np.zeros(self.actions)
        for action in state.legal_actions():
            values[action] = self._traverse(state.child(action), target)
        self.advantage_memories[target].add(sample | {"y": sampled_advantages(values, probs, mask)})
        return float(np.dot(probs, values))

    def _train(self, net, optimizer, memory, rng, steps, strategy=False):
        if not len(memory):
            raise RuntimeError("Training memory is empty; do not silently skip training")
        # Pack once per training phase, rather than copying Python samples per step.
        items = memory.items
        x = torch.as_tensor(np.stack([s["x"] for s in items]), device=self.device)
        y = torch.as_tensor(np.stack([s["y"] for s in items]), device=self.device)
        mask = torch.as_tensor(np.stack([s["mask"] for s in items]), device=self.device)
        t = torch.tensor([s["t"] for s in items], dtype=torch.float32, device=self.device)
        size = min(self.config.batch_size, len(items))
        losses = []
        net.train()
        for _ in range(steps):
            indices = rng.sample(range(len(items)), size)
            pred = net(x[indices])
            if strategy:
                pred = masked_softmax(pred, mask[indices])
            loss = weighted_mse(pred, y[indices], mask[indices], t[indices], self.iteration)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().item())
        net.eval()
        if strategy:
            self.strategy_gradient_steps += steps
        else:
            self.gradient_steps += steps
        return {"first_loss": losses[0], "last_loss": losses[-1], "mean_loss": float(np.mean(losses)),
                "steps": steps, "batch_size": size, "buffer_size": len(memory), "seen": memory.seen}

    def step(self):
        self.iteration += 1
        stats = {"iteration": self.iteration, "traversal_seconds": 0., "advantage_seconds": 0., "players": []}
        for player in range(2):
            self.cache.clear()
            self._synchronize()
            start = time.perf_counter()
            for _ in range(self.config.traversals):
                self._traverse(self.game.new_initial_state(), player)
            self._synchronize()
            stats["traversal_seconds"] += time.perf_counter()-start
            start = time.perf_counter()
            # Reset AFTER collection; retain all reservoir memories.
            self.advantages[player] = self._network()
            self.advantage_optimizers[player] = self._optimizer(self.advantages[player])
            stats["players"].append(self._train(self.advantages[player], self.advantage_optimizers[player],
                self.advantage_memories[player], self.batch_rng[player], self.config.advantage_steps))
            self._synchronize()
            stats["advantage_seconds"] += time.perf_counter()-start
        self.cache.clear()
        self.history.append(stats)
        return stats

    def train_average(self, steps=None):
        self._synchronize()
        start = time.perf_counter()
        result = self._train(self.strategy, self.strategy_optimizer, self.strategy_memory,
                             self.strategy_rng, steps or self.config.strategy_steps, strategy=True)
        self._synchronize()
        return result | {"seconds": time.perf_counter()-start}

    def average_policy(self):
        if self._table is None:
            self._table = policy.TabularPolicy(self.game)
        table = self._table.__copy__()
        x = np.asarray([s.information_state_tensor(s.current_player()) for s in table.states], dtype=np.float32)
        mask = table.legal_actions_mask.astype(bool)
        with torch.no_grad():
            probabilities = masked_softmax(self.strategy(torch.as_tensor(x, device=self.device)),
                                            torch.as_tensor(mask, device=self.device)).cpu().numpy().astype(float)
        probabilities /= probabilities.sum(1, keepdims=True)
        table.action_probability_array[:] = probabilities
        validate_policy(table)
        return table

    def save(self, path, run_state=None):
        """Trusted local checkpoint only; contains Python/NumPy reservoir states."""
        payload = {"format": 1, "config": asdict(self.config), "iteration": self.iteration,
            "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                              for name in ("deep_cfr.py", "reservoir.py")},
            "versions": {p: md.version(p) for p in ("torch", "open-spiel", "numpy")},
            "advantages": [net.state_dict() for net in self.advantages], "strategy": self.strategy.state_dict(),
            "advantage_optimizers": [opt.state_dict() for opt in self.advantage_optimizers],
            "strategy_optimizer": self.strategy_optimizer.state_dict(),
            "advantage_memories": [b.state_dict() for b in self.advantage_memories],
            "strategy_memory": self.strategy_memory.state_dict(),
            "rng": self.rng.get_state(), "init_rng": self.init_rng.get_state(),
            "batch_rng": [r.getstate() for r in self.batch_rng], "strategy_rng": self.strategy_rng.getstate(),
            "node_visits": self.node_visits, "inference_calls": self.inference_calls,
            "gradient_steps": self.gradient_steps, "strategy_gradient_steps": self.strategy_gradient_steps,
            "history": self.history, "run_state": run_state}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path, device=None, allow_incompatible=False):
        # Never load arbitrary third-party pickle checkpoints.
        data = torch.load(path, map_location=device or "cpu", weights_only=False)
        if data["format"] != 1:
            raise ValueError("Unsupported checkpoint")
        mismatches = [p for p, version in data["versions"].items() if md.version(p) != version]
        mismatches += [name for name, digest in data.get("source_sha256", {}).items()
                       if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != digest]
        if mismatches and not allow_incompatible:
            raise ValueError(f"Checkpoint compatibility mismatch: {mismatches}; explicit allow_incompatible=True required")
        config = Config(**data["config"])
        if device:
            config.device = device
        obj = cls(config)
        for key in ("iteration", "node_visits", "inference_calls", "gradient_steps", "strategy_gradient_steps", "history"):
            setattr(obj, key, data[key])
        for i in range(2):
            obj.advantages[i].load_state_dict(data["advantages"][i])
            obj.advantage_optimizers[i].load_state_dict(data["advantage_optimizers"][i])
            obj.advantage_memories[i].load_state_dict(data["advantage_memories"][i])
            obj.batch_rng[i].setstate(data["batch_rng"][i])
        obj.strategy.load_state_dict(data["strategy"])
        obj.strategy_optimizer.load_state_dict(data["strategy_optimizer"])
        obj.strategy_memory.load_state_dict(data["strategy_memory"])
        obj.rng.set_state(data["rng"])
        obj.init_rng.set_state(data["init_rng"])
        obj.strategy_rng.setstate(data["strategy_rng"])
        return obj, data["run_state"]
