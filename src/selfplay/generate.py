"""
Self-play on the GPU: K games at once in one jit-compiled scan (Gumbel search)
A finished game is replaced by a fresh one in its lane, so the batch stays full
Samples: state, improved policy, outcome. Unfinished games are dropped
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jaxgame import State, init, orient_policy, path_lengths, step
from mcts.search import run_search


class Sample(NamedTuple):
    state: State  # the position the move was chosen in
    policy: jax.Array  # (140,) improved search policy, mover's canonical frame
    value: jax.Array  # outcome for the mover: 1 win, -1 loss, 0 draw
    margin: jax.Array  # loser's distance left at the end, for the mover (NaN: unknown)
    heuristic: jax.Array  # tanh(path advantage / 2): warm-start value prior


def _for_mover(pair: jax.Array, mover: jax.Array, mine: bool = True) -> jax.Array:
    """Pick column 'mover' (or the opponent's) from a (B, 2) per-player array."""
    idx = mover if mine else 1 - mover
    return jnp.take_along_axis(pair, idx[:, None], axis=1)[:, 0]


def _heuristic(states: State) -> jax.Array:
    lengths = jax.vmap(path_lengths)(states)
    mover = states.to_move
    advantage = _for_mover(lengths, mover, False) - _for_mover(lengths, mover)
    return jnp.tanh(advantage / 2)


@partial(nnx.jit, static_argnames=("lanes", "steps", "sims"))
def _play(model, key, lanes: int, steps: int, sims: int):
    fresh = jax.vmap(lambda _: init())(jnp.arange(lanes))

    def move(states, key):
        out = run_search(model, states, key, sims, gumbel=True)
        nxt = jax.vmap(step)(states, out.action)
        mover = states.to_move
        won = nxt.winner == mover
        loser_left = _for_mover(jax.vmap(path_lengths)(nxt), mover, False)
        record = dict(
            state=states,
            policy=jax.vmap(orient_policy)(out.action_weights, mover),
            heuristic=_heuristic(states),
            won=won.astype(jnp.float32),
            margin=jnp.where(won, loser_left, 0).astype(jnp.float32),
            ended=nxt.done,
            length=jnp.where(nxt.done, nxt.ply, 0),
            p0_won=nxt.winner == 0,
        )
        reset = jax.tree.map(
            lambda f, n: jnp.where(nxt.done.reshape((-1,) + (1,) * (n.ndim - 1)), f, n),
            fresh,
            nxt,
        )
        return reset, record

    _, rec = jax.lax.scan(move, fresh, jax.random.split(key, steps))

    # Outcomes flow backwards: a game's last move sets it, each earlier ply flips it.
    def back(carry, move):
        value, margin, known = carry
        ended, won, final_margin = move
        value = jnp.where(ended, won, -value)
        margin = jnp.where(ended, final_margin, -margin)
        known = ended | known
        return (value, margin, known), (value, margin, known)

    zeros = jnp.zeros(lanes)
    moves = (rec["ended"], rec["won"], rec["margin"])
    init_carry = (zeros, zeros, jnp.zeros(lanes, bool))
    _, (value, margin, known) = jax.lax.scan(back, init_carry, moves, reverse=True)
    samples = Sample(rec["state"], rec["policy"], value, margin, rec["heuristic"])
    games = rec["ended"].sum()
    stats = dict(
        games=games,
        draws=(rec["ended"] & (rec["won"] == 0)).sum(),
        p0_wins=rec["p0_won"].sum(),
        avg_length=rec["length"].sum() / jnp.maximum(games, 1),
    )
    return samples, known, stats


def selfplay(model, key, lanes: int, steps: int, sims: int):
    """Play 'steps' moves in each of 'lanes' games -> (Sample on host, stats)."""
    model.eval()
    samples, known, stats = _play(model, key, lanes, steps, sims)
    keep = np.asarray(known).reshape(-1)
    flat = jax.tree.map(
        lambda x: np.asarray(x).reshape(-1, *x.shape[2:])[keep], samples
    )
    return flat, {k: float(v) for k, v in stats.items()}


@partial(nnx.jit, static_argnames="sims")
def _reanalyze(model, states, key, sims: int) -> Sample:
    out = run_search(model, states, key, sims, gumbel=True)
    nan = jnp.full(states.to_move.shape, jnp.nan)
    return Sample(
        state=states,
        policy=jax.vmap(orient_policy)(out.action_weights, states.to_move),
        value=out.search_tree.summary().value,
        margin=nan,
        heuristic=_heuristic(states),
    )


def reanalyze(model, states: State, key, sims: int, batch: int = 512) -> Sample:
    """
    Fresh search targets for positions from outside self-play (MuZero Reanalyze).
    Policy and value come from searching each position with the current network,
    so human mistakes shape which positions are learned from, not the targets.
    """
    model.eval()
    parts = []
    for i in range(0, len(states.to_move), batch):
        chunk = jax.tree.map(lambda x: x[i : i + batch], states)
        key, sub = jax.random.split(key)
        parts.append(jax.tree.map(np.asarray, _reanalyze(model, chunk, sub, sims)))
    return jax.tree.map(lambda *xs: np.concatenate(xs), *parts)
