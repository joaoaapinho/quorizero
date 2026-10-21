"""The JAX rules must match the Python engine (the reference) on every move."""

import random

import jax
import jax.numpy as jnp
import numpy as np

from engine.actions import legal_action_mask, legal_actions, step
from engine.state import GameState, initial_state
from jaxgame import FLIP, MIRROR, from_python, init, mirror, observe
from jaxgame import step as jax_step

jit_step = jax.jit(jax_step)


def same_state(jax_state, game: GameState) -> None:
    ref = from_python(game)
    for name in ("pawns", "h_walls", "v_walls", "walls_left", "to_move", "legal"):
        np.testing.assert_array_equal(getattr(jax_state, name), getattr(ref, name))
    np.testing.assert_array_equal(ref.legal, legal_action_mask(game))


def to_python(state, like: GameState) -> GameState:
    h, v = np.asarray(state.h_walls), np.asarray(state.v_walls)
    walls = [("H", int(r), int(c)) for r, c in zip(*np.nonzero(h))]
    walls += [("V", int(r), int(c)) for r, c in zip(*np.nonzero(v))]
    pawns = tuple(tuple(int(x) for x in p) for p in np.asarray(state.pawns))
    return GameState(pawns, frozenset(walls), like.walls_left, like.to_move)


def random_games(n_games: int, pawn_bias: float, seed: int = 0):
    """Yield (state, action) along random games; pawn_bias makes pawns meet often."""
    rng = random.Random(seed)
    for _ in range(n_games):
        game = initial_state()
        for _ in range(120):
            actions = legal_actions(game)
            pawn_moves = [a for a in actions if a < 12]
            pick = pawn_moves if rng.random() < pawn_bias else actions
            action = rng.choice(pick)
            yield game, action
            game, done, _ = step(game, action)
            if done:
                break


def test_matches_python_engine_on_random_games():
    for pawn_bias in (0.0, 0.8):  # wall-heavy games, then pawn races with jumps
        for game, action in random_games(15, pawn_bias):
            state = from_python(game)
            same_state(state, game)
            nxt_game, done, win = step(game, action)
            nxt = jit_step(state, action)
            same_state(nxt._replace(legal=from_python(nxt_game).legal), nxt_game)
            assert bool(nxt.done) == done
            assert int(nxt.winner) == (-1 if win is None else win)
            if not done:
                np.testing.assert_array_equal(nxt.legal, legal_action_mask(nxt_game))


def test_jumps_match_python_engine():
    cases = [
        GameState(((4, 4), (3, 4)), frozenset(), (10, 10), 0),  # straight jump
        GameState(((4, 4), (3, 4)), frozenset({("H", 2, 3)}), (10, 10), 0),  # diagonal
        GameState(((1, 4), (0, 4)), frozenset(), (10, 10), 0),  # jump off the edge
        GameState(((0, 8), (0, 7)), frozenset(), (10, 10), 0),  # corner
    ]
    for game in cases:
        np.testing.assert_array_equal(from_python(game).legal, legal_action_mask(game))


def test_mirror_and_flip_are_symmetries():
    for game, _ in random_games(5, 0.3, seed=1):
        state = from_python(game)
        # Mirrored state is a real game state: its legal moves are the mirrored ones.
        mirrored = mirror(state)
        relegal = from_python(to_python(mirrored, game)).legal
        np.testing.assert_array_equal(mirrored.legal, relegal)
        # Flipping rows and swapping the players is the same game seen from the
        # other side, so the legal moves are the FLIP-permuted ones.
        swapped = GameState(
            (
                (8 - game.pawns[1][0], game.pawns[1][1]),
                (8 - game.pawns[0][0], game.pawns[0][1]),
            ),
            frozenset((o, 7 - r, c) for o, r, c in game.walls),
            game.walls_left[::-1],
            1 - game.to_move,
        )
        np.testing.assert_array_equal(
            np.asarray(state.legal)[FLIP], legal_action_mask(swapped)
        )
    assert sorted(FLIP) == list(range(140)) and sorted(MIRROR) == list(range(140))


def test_observe_is_canonical():
    start = init()
    planes = observe(start)
    assert planes.shape == (9, 9, 9)
    # Player 0 starts on row 0 heading down; canonically it sits on row 8 heading up.
    assert planes[8, 4, 0] == 1 and planes[0, 4, 1] == 1
    assert planes[8, 4, 6] == 8 / 16  # 8 steps to go
    # After player 0's first move, player 1 (to move) sees the mirror-image picture.
    after = jax_step(start, 1)  # player 0 steps down
    p1 = observe(after)
    assert p1[8, 4, 0] == 1 and p1[1, 4, 1] == 1
    assert jnp.all(p1[..., 8] == 1)
