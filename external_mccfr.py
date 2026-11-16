# Copyright 2019 DeepMind Technologies Limited
# Educational adaptation, this project, 2026.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations.
"""External-sampling MCCFR for two-player Kuhn/Leduc.

Adapted from OpenSpiel 1.6.11 external_sampling_mccfr.py and mccfr.py:
https://github.com/google-deepmind/open_spiel/tree/v1.6.11/open_spiel/python/algorithms
Preserves SIMPLE averaging and 1e-6 initialization for reference comparison.
Changes: independent RNG, sparse (player, information-set) keys, snapshots,
explicit counters. Training never enumerates the entire game tree.
"""
import copy
import numpy as np
from open_spiel.python import policy


class AveragePolicy(policy.Policy):
    def __init__(self, game, nodes):
        super().__init__(game, [0, 1])
        self.nodes = copy.deepcopy(nodes)

    def action_probabilities(self, state, player_id=None):
        player = state.current_player() if player_id is None else player_id
        actions = state.legal_actions(player)
        node = self.nodes.get((player, state.information_state_string(player)))
        if node is None:
            return {a: 1 / len(actions) for a in actions}
        saved_actions, _, total = node
        probabilities = total / total.sum()
        return dict(zip(saved_actions, map(float, probabilities)))


class ExternalSamplingMCCFR:
    def __init__(self, game, seed=42):
        if game.num_players() != 2 or game.get_type().short_name not in ("kuhn_poker", "leduc_poker"):
            raise ValueError("Only two-player Kuhn/Leduc are supported")
        self.game = game
        # RandomState intentionally matches the reference's np.random sequence,
        # without reading/modifying the global RNG used by other experiments.
        self.rng = np.random.RandomState(seed)
        self.nodes = {}
        self.iterations = 0
        self.node_visits = 0
        self.terminal_visits = 0
        self.chance_samples = 0
        self.opponent_samples = 0

    def _walk(self, state, target):
        self.node_visits += 1
        if state.is_terminal():
            self.terminal_visits += 1
            return state.player_return(target)
        if state.is_chance_node():
            actions, probabilities = zip(*state.chance_outcomes())
            action = self.rng.choice(actions, p=probabilities)
            self.chance_samples += 1
            return self._walk(state.child(int(action)), target)
        actor = state.current_player()
        key = (actor, state.information_state_string(actor))
        actions = state.legal_actions()
        if key not in self.nodes:
            self.nodes[key] = (actions, np.ones(len(actions)) / 1e6,
                               np.ones(len(actions)) / 1e6)
        _, regrets, strategy_sum = self.nodes[key]
        positive = np.maximum(regrets, 0.)
        total = positive.sum()
        strategy = positive / total if total > 0 else np.ones(len(actions)) / len(actions)
        if actor != target:
            index = self.rng.choice(np.arange(len(actions)), p=strategy)
            self.opponent_samples += 1
            value = self._walk(state.child(actions[index]), target)
            # SIMPLE averaging: visiting this opponent information set already
            # samples its reach. Do not multiply by that reach a second time.
            for index in range(len(actions)):
                strategy_sum[index] += strategy[index]
            return value
        values = np.zeros(len(actions))
        value = 0.
        for index, action in enumerate(actions):
            values[index] = self._walk(state.child(action), target)
            value += strategy[index] * values[index]
        # Sampling uses true chance/opponent probabilities, so their reach is
        # represented by sampling frequency. No extra reach multiplier here.
        for index in range(len(actions)):
            regrets[index] += values[index] - value
        return value

    def evaluate_and_update_policy(self):
        for target in range(2):
            self._walk(self.game.new_initial_state(), target)
        self.iterations += 1

    def average_policy(self):
        return AveragePolicy(self.game, self.nodes)

    def counters(self):
        return {k: getattr(self, k) for k in ("node_visits", "terminal_visits", "chance_samples", "opponent_samples")} | {"visited_information_sets": len(self.nodes)}
