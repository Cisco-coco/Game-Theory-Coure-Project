"""Exact policy evaluation for two-player zero-sum games using OpenSpiel."""
from open_spiel.python.algorithms import expected_game_score, exploitability
import pyspiel


def evaluate(game, strategy):
    """Return exact NashConv/2 and self-play values, in chips per hand.

    Full-tree evaluation is intended for Kuhn/Leduc, not large poker games.
    """
    if game.num_players() != 2 or game.get_type().utility != pyspiel.GameType.Utility.ZERO_SUM:
        raise ValueError("Evaluation requires a two-player zero-sum game")
    conv = float(exploitability.nash_conv(game, strategy))
    values = expected_game_score.policy_value(game.new_initial_state(), [strategy] * 2)
    return dict(nash_conv=conv, exploitability=conv / 2,
                value_p0=float(values[0]), value_p1=float(values[1]))
