"""Heuristic position evaluation: opponent's path length minus ours."""

from __future__ import annotations

from engine.pathcheck import shortest_distance
from engine.state import GOAL_ROWS, GameState, winner

WIN_SCORE = 10_000  # magnitude for a decided game; dwarfs any distance diff


def evaluate(state: GameState, player: int) -> int:
    """
    Score the position from a player's view, higher is better.
    Quoridor heuristic: we want a short path to our goal and a
    long one for our opponent, so score = opp_distance - my_distance.
    """
    won = winner(state)
    if won is not None:
        return WIN_SCORE if won == player else -WIN_SCORE

    opp = 1 - player
    my_dist = shortest_distance(state.walls, state.pawns[player], GOAL_ROWS[player])
    opp_dist = shortest_distance(state.walls, state.pawns[opp], GOAL_ROWS[opp])
    # Engine invariants keep both paths open, but stay defensive.
    if my_dist is None:
        return -WIN_SCORE
    if opp_dist is None:
        return WIN_SCORE
    return opp_dist - my_dist
