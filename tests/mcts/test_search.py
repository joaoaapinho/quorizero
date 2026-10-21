"""Search finds a one-move win, with both Gumbel (self-play) and PUCT (play)."""

import jax

from engine.state import GameState
from jaxgame import from_python
from mcts.search import greedy_agent, net_agent, run_search
from model.network import create_model

# Player 0 is one step from its goal row (8); the winning move is action 1 (S).
NEAR_WIN = GameState(((7, 2), (8, 6)), frozenset(), (0, 0), 0)


def batch_of(game, n=2):
    return jax.tree.map(lambda x: x[None].repeat(n, axis=0), from_python(game))


def test_both_searches_take_the_win():
    model = create_model(num_blocks=2, channels=16)  # untrained: search must do it
    model.eval()
    states = batch_of(NEAR_WIN)
    for gumbel in (True, False):
        out = run_search(model, states, jax.random.key(0), sims=16, gumbel=gumbel)
        assert out.action.tolist() == [1, 1]
    assert net_agent(model, 16)(states, jax.random.key(1)).tolist() == [1, 1]


def test_greedy_takes_the_win():
    assert greedy_agent(batch_of(NEAR_WIN), jax.random.key(0)).tolist() == [1, 1]
