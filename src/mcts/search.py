"""
Batched mctx search on the GPU: Gumbel AlphaZero for self-play, PUCT for play.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import mctx
import numpy as np
from flax import nnx

from engine.actions import PAWN_OFFSETS
from engine.state import GOAL_ROWS, GameState
from jaxgame import from_python, observe, orient_policy, path_lengths, step
from jaxgame.rules import distance_map, open_edges, wall_path_lengths

NEG = jnp.finfo(jnp.float32).min


def evaluate(model, states):
    """Prior logits (masked, board action order) and value for the side to move."""
    logits, value, _ = model(jax.vmap(observe)(states))
    logits = jax.vmap(orient_policy)(logits, states.to_move)
    logits = logits - logits.max(axis=-1, keepdims=True)
    return jnp.where(states.legal, logits, NEG), value


def _recurrent_fn(graphdef):
    # mctx's MuZero-style hook (learned-model API), fed the real rules instead
    def recurrent_fn(params, key, action, states):
        nxt = jax.vmap(step)(states, action)
        logits, value = evaluate(nnx.merge(graphdef, params), nxt)
        won = (nxt.winner == states.to_move) & ~states.done
        output = mctx.RecurrentFnOutput(
            reward=won.astype(jnp.float32),  # for the player who just moved
            discount=jnp.where(nxt.done, 0.0, -1.0),  # zero-sum: flip perspective
            prior_logits=logits,
            value=jnp.where(nxt.done, 0.0, value),
        )
        return output, nxt

    return recurrent_fn


def run_search(model, states, key, sims: int, gumbel: bool):
    """One search per game in the batch. The model must be in eval mode."""
    graphdef, params = nnx.split(model)
    logits, value = evaluate(model, states)
    search = dict(
        params=params,
        rng_key=key,
        root=mctx.RootFnOutput(prior_logits=logits, value=value, embedding=states),
        recurrent_fn=_recurrent_fn(graphdef),
        num_simulations=sims,
        invalid_actions=~states.legal,
    )
    if gumbel:  # Gumbel AlphaZero: sampled root moves, sequential halving, few sims
        return mctx.gumbel_muzero_policy(**search)
    # PUCT (AlphaZero): prior-weighted exploration bonus, no root noise
    return mctx.muzero_policy(**search, dirichlet_fraction=0.0, temperature=0.0)


@partial(nnx.jit, static_argnames="sims")
def _play_actions(model, states, key, sims: int):
    return run_search(model, states, key, sims, gumbel=False).action


def net_agent(model, sims: int):
    """Batched agent (states, key) -> actions: PUCT with `sims` simulations."""
    model.eval()
    return lambda states, key: _play_actions(model, states, key, sims)


@jax.jit
def greedy_agent(states, key):
    """
    Depth-1 shortest-path bot (minimax baseline at depth 1), random tie-breaks.
    A fixed, model-free anchor for Elo: maximise opponent path - own path.
    """
    goals = jnp.array(GOAL_ROWS)

    def scores(s):
        me, opp = s.to_move, 1 - s.to_move
        my_map = distance_map(goals[me], *open_edges(s.h_walls, s.v_walls))
        dest = jnp.clip(s.pawns[me] + jnp.array(PAWN_OFFSETS), 0, 8)
        wins = 1000 * (dest[:, 0] == goals[me])
        pawn = path_lengths(s)[opp] - my_map[dest[:, 0], dest[:, 1]] + wins
        walls = wall_path_lengths(s)
        return jnp.concatenate([pawn, walls[:, opp] - walls[:, me]])

    noise = jax.random.uniform(key, states.legal.shape, maxval=0.5)
    score = jax.vmap(scores)(states) + noise
    return jnp.argmax(jnp.where(states.legal, score, -jnp.inf), axis=-1)


def python_agent(agent, seed: int = 0):
    """Wrap a batched agent for engine.GameState arenas (e.g. vs minimax)."""
    rng = np.random.default_rng(seed)

    def act(game: GameState) -> int:
        state = jax.tree.map(lambda x: x[None], from_python(game))
        return int(agent(state, jax.random.key(rng.integers(2**31)))[0])

    return act
