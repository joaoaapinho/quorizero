"""Alpha-beta (negamax) search over the heuristic evaluation."""

from __future__ import annotations

import math

from baseline.evaluation import evaluate
from engine.actions import encode_move, encode_wall, legal_actions, step
from engine.moves import pawn_moves, wall_placements
from engine.pathcheck import shortest_distance
from engine.state import GOAL_ROWS, GameState, winner

INF = math.inf


def _candidates(state: GameState) -> list[tuple[int, GameState]]:
    """(action, resulting_state) pairs, pruned and ordered best-first.
    Only walls that actually lengthen the opp's path are considered - a
    wall that doesn't is strategically pointless and just bloats the tree. This
    is the "try blocks first" move ordering, plus pawn advances.
    """
    player = state.to_move
    opp = 1 - player
    frm = state.pawns[player]

    cand = [
        (encode_move(frm, d), step(state, encode_move(frm, d))[0])
        for d in pawn_moves(state)
    ]

    if state.walls_left[player] > 0:
        opp_dist = shortest_distance(state.walls, state.pawns[opp], GOAL_ROWS[opp])
        for wall in wall_placements(state):
            new_dist = shortest_distance(
                state.walls | {wall}, state.pawns[opp], GOAL_ROWS[opp]
            )
            if opp_dist is not None and new_dist is not None and new_dist > opp_dist:
                a = encode_wall(wall)
                cand.append((a, step(state, a)[0]))

    cand.sort(key=lambda ac: evaluate(ac[1], player), reverse=True)
    return cand


def _negamax(state: GameState, depth: int, alpha: float, beta: float):
    if winner(state) is not None or depth == 0:
        return evaluate(state, state.to_move), None

    best, best_action = -INF, None
    for action, child in _candidates(state):
        score = -_negamax(child, depth - 1, -beta, -alpha)[0]
        if score > best:
            best, best_action = score, action
        alpha = max(alpha, best)
        if alpha >= beta:
            break
    return best, best_action


def choose_action(state: GameState, depth: int = 2) -> int:
    """Return the heuristic bot's chosen action for the player to move."""
    _, action = _negamax(state, depth, -INF, INF)
    return action if action is not None else legal_actions(state)[0]
