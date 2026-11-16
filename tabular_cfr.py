# Copyright 2019 DeepMind Technologies Limited
# Modifications and educational rewrite: this project, 2026.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Educational CFR/CFR+ implementation for two-player Kuhn and Leduc poker.

Reference/adaptation: OpenSpiel 2.0.2, python/algorithms/cfr.py (Apache-2.0).
https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/algorithms/cfr.py
Changes: dense arrays; scalar value recursion for the updating player; explicit
pass-level regret increments; configurable synchronous/alternating updates;
snapshot policies and node-visit counters. No official solver is called here.
OpenSpiel is still responsible for rules, information sets and policy containers.
"""
import numpy as np
from open_spiel.python import policy


def regret_matching(regrets, legal_mask):
    """Normalize positive regrets, falling back to uniform LEGAL actions."""
    positive = np.where(legal_mask, np.maximum(regrets, 0.0), 0.0)
    total = positive.sum(axis=1, keepdims=True)
    uniform = legal_mask / legal_mask.sum(axis=1, keepdims=True)
    return np.divide(positive, total, out=uniform.copy(), where=total > 0)


class TabularCFR:
    def __init__(self, game, *, plus=False, alternating=True, linear_averaging=None):
        if game.get_type().short_name not in ("kuhn_poker", "leduc_poker") or game.num_players() != 2:
            raise ValueError("This educational solver supports two-player Kuhn/Leduc only")
        self.game = game
        self.plus = plus
        self.alternating = alternating
        self.linear_averaging = plus if linear_averaging is None else linear_averaging
        self.template = policy.TabularPolicy(game)
        self.legal_mask = self.template.legal_actions_mask.astype(bool)
        self.regrets = np.zeros_like(self.template.action_probability_array)
        self.strategy_sum = np.zeros_like(self.regrets)
        self.strategy = regret_matching(self.regrets, self.legal_mask)
        self.iteration = 0
        self.node_visits = 0

    def _walk(self, state, target, reach, chance_reach, increments):
        """Return target's conditional continuation value; accumulate CF regrets.

        reach contains ONLY the two players' probabilities. chance_reach is
        separate so it enters regrets, but not own-reach averaging weights.
        Strategy is frozen for this entire traversal.
        """
        self.node_visits += 1
        if state.is_terminal():
            return state.returns()[target]
        if state.is_chance_node():
            return sum(prob * self._walk(state.child(action), target, reach,
                                         chance_reach * prob, increments)
                       for action, prob in state.chance_outcomes())
        # Matches the reference's safe pruning: neither player can reach here.
        if reach[0] == 0 and reach[1] == 0:
            return 0.0
        actor = state.current_player()
        row = self.template.state_lookup[state.information_state_string(actor)]
        actions = state.legal_actions()
        values = {}
        for action in actions:
            child_reach = list(reach)
            child_reach[actor] *= self.strategy[row, action]
            values[action] = self._walk(state.child(action), target, child_reach,
                                        chance_reach, increments)
        value = sum(self.strategy[row, action] * values[action] for action in actions)
        if actor == target:
            cf_reach = chance_reach * reach[1 - target]
            weight = self.iteration if self.linear_averaging else 1
            for action in actions:
                increments[row, action] += cf_reach * (values[action] - value)
                # Aggregate each history just as the reference does. In these
                # perfect-recall games the multiplicity cancels on normalization.
                self.strategy_sum[row, action] += weight * reach[target] * self.strategy[row, action]
        return value

    def _commit(self, increments):
        self.regrets += increments
        if self.plus:
            # Do NOT clip per history: all histories of an information set
            # must contribute before the nonlinear RM+ truncation.
            np.maximum(self.regrets, 0.0, out=self.regrets)
        self.strategy = regret_matching(self.regrets, self.legal_mask)

    def evaluate_and_update_policy(self):
        self.iteration += 1
        increments = np.zeros_like(self.regrets)
        for target in range(2):
            self._walk(self.game.new_initial_state(), target, [1.0, 1.0], 1.0, increments)
            if self.alternating:
                self._commit(increments)
                increments.fill(0.0)
        if not self.alternating:
            self._commit(increments)

    def _snapshot(self, probabilities):
        result = self.template.__copy__()
        result.action_probability_array[:] = probabilities
        return result

    def current_policy(self):
        return self._snapshot(self.strategy)

    def average_policy(self):
        totals = self.strategy_sum.sum(axis=1, keepdims=True)
        uniform = self.legal_mask / self.legal_mask.sum(axis=1, keepdims=True)
        probabilities = np.divide(self.strategy_sum, totals, out=uniform.copy(), where=totals > 0)
        return self._snapshot(probabilities)
