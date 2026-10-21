"""Tests for the heuristic baseline bot (Section 2 checklist)."""

import random

from baseline.evaluation import WIN_SCORE, evaluate
from baseline.match import baseline_agent, play_game, random_agent
from baseline.minimax import choose_action
from engine.actions import decode_action, legal_actions, step
from engine.pathcheck import shortest_distance
from engine.state import GameState, initial_state


def mk(pawns, walls=(), left=(10, 10), to_move=0):
    return GameState(
        pawns=pawns, walls=frozenset(walls), walls_left=left, to_move=to_move
    )


# --- evaluation ----------------------------------------------------------


def test_shortest_distance_open_board_and_detour():
    assert shortest_distance(frozenset(), (0, 4), 8) == 8
    # A wall in front forces a detour, lengthening the path.
    walls = frozenset({("H", 0, 3), ("H", 0, 5)})
    assert shortest_distance(walls, (0, 4), 8) == 8 + 2


def test_evaluate_prefers_being_closer():
    state = mk(((7, 4), (4, 4)))  # p0 dist 1, p1 dist 4
    assert evaluate(state, 0) == 4 - 1
    assert evaluate(state, 1) == 1 - 4


def test_evaluate_terminal_is_decisive():
    state = mk(((8, 4), (4, 4)))  # p0 already home
    assert evaluate(state, 0) == WIN_SCORE
    assert evaluate(state, 1) == -WIN_SCORE


# --- search --------------------------------------------------------------


def test_takes_immediate_winning_move():
    state = mk(((7, 4), (2, 0)))
    _, done, win = step(state, choose_action(state, depth=2))
    assert done and win == 0


def test_choose_action_is_always_legal():
    state = initial_state()
    for depth in (1, 2):
        assert choose_action(state, depth) in legal_actions(state)


def test_blocks_a_near_winning_opponent():
    # p0 is far; p1 is one step from its goal -> a good bot walls p1 off.
    state = mk(((4, 0), (1, 4)))
    action = choose_action(state, depth=2)
    kind, payload = decode_action(state, action)
    assert kind == "wall"
    before = shortest_distance(state.walls, state.pawns[1], 0)
    after = shortest_distance(state.walls | {payload}, state.pawns[1], 0)
    assert after > before


# --- sanity scripts ------------------------------------------------------


def test_baseline_vs_baseline_is_sane_and_decisive():
    # Both sides play the heuristic: no crash, and someone wins within the cap.
    assert play_game(baseline_agent(1), baseline_agent(1)) in (0, 1)


def test_baseline_dominates_random():
    rng = random.Random(0)
    baseline = baseline_agent(1)
    rnd = random_agent(rng)
    wins = 0
    games = 12
    for g in range(games):
        # Alternate who moves first to remove seat bias.
        first, second = (baseline, rnd) if g % 2 == 0 else (rnd, baseline)
        result = play_game(first, second)
        baseline_won = (result == 0) if g % 2 == 0 else (result == 1)
        wins += baseline_won
    assert wins >= 11  # heuristic should crush random play
