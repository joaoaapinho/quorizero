"""
Quoridor rules in JAX, one game per call (batch with jax.vmap).
engine/ twin, checked move by move in tests/jaxgame.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from engine.actions import ACTION_SIZE, N_PAWN, N_WALL, PAWN_OFFSETS
from engine.state import DIRS, GOAL_ROWS, STARTS, WALLS_PER_PLAYER, GameState

MAX_PLY = 200  # a game this long is a draw - stops endless wall-shuffling games
FAR = 99  # path length of a cell that can't reach the goal

_OFFSETS = jnp.array(PAWN_OFFSETS, jnp.int32)
_GOALS = jnp.array(GOAL_ROWS, jnp.int32)
_ROWS = jnp.arange(9)[:, None] + jnp.zeros((1, 9), jnp.int32)
_ONE_WALL = jnp.eye(N_WALL, dtype=bool).reshape(N_WALL, 8, 8)  # wall k alone
_BITS = jnp.uint32(1) << jnp.arange(9, dtype=jnp.uint32)


class State(NamedTuple):
    pawns: jax.Array  # int32 (2, 2): (row, col) per player
    h_walls: jax.Array  # bool (8, 8): anchors as engine's ('H', r, c)
    v_walls: jax.Array  # bool (8, 8)
    walls_left: jax.Array  # int32 (2,)
    to_move: jax.Array  # int32 ()
    ply: jax.Array  # int32 ()
    winner: jax.Array  # int32 (): -1 until someone wins (stays -1 on a draw)
    done: jax.Array  # bool ()
    legal: jax.Array  # bool (140,): cached, so search never recomputes it


def open_edges(h_walls: jax.Array, v_walls: jax.Array) -> tuple[jax.Array, jax.Array]:
    """
    right[r, c]: (r,c)<->(r,c+1) is open (9, 8); down[r, c]: (r,c)<->(r+1,c) (8, 9).
    A V wall at (r, c) blocks rows r and r+1; an H wall at (r, c) cols c and c+1.
    """
    v_block = jnp.pad(v_walls, ((0, 1), (0, 0))) | jnp.pad(v_walls, ((1, 0), (0, 0)))
    h_block = jnp.pad(h_walls, ((0, 0), (0, 1))) | jnp.pad(h_walls, ((0, 0), (1, 0)))
    return ~v_block, ~h_block


def _spread(cells: jax.Array, right: jax.Array, down: jax.Array) -> jax.Array:
    """Cells plus every neighbour reachable in one step."""
    return (
        cells
        | jnp.pad(cells[:, :-1] & right, ((0, 0), (1, 0)))
        | jnp.pad(cells[:, 1:] & right, ((0, 0), (0, 1)))
        | jnp.pad(cells[:-1] & down, ((1, 0), (0, 0)))
        | jnp.pad(cells[1:] & down, ((0, 1), (0, 0)))
    )


def distance_map(goal_row, right: jax.Array, down: jax.Array) -> jax.Array:
    """Steps from every cell to goal_row (FAR if walled off): BFS from the goal row."""

    def grow(carry):
        reached, dist, _, d = carry
        grown = _spread(reached, right, down)
        dist = jnp.where(grown & ~reached, d + 1, dist)
        return grown, dist, jnp.any(grown != reached), d + 1

    reached = _ROWS == goal_row
    dist = jnp.where(reached, 0, FAR).astype(jnp.int32)
    init = (reached, dist, jnp.array(True), jnp.int32(0))
    return jax.lax.while_loop(lambda c: c[2], grow, init)[1]


def path_lengths(state: State) -> jax.Array:
    """Each player's shortest path length to their goal, int32 (2,)."""
    right, down = open_edges(state.h_walls, state.v_walls)
    return jnp.stack(
        [distance_map(_GOALS[p], right, down)[tuple(state.pawns[p])] for p in (0, 1)]
    )


def wall_path_lengths(state: State) -> jax.Array:
    """Both path lengths after each of the 128 candidate walls, int32 (128, 2)."""
    empty = jnp.zeros_like(_ONE_WALL)
    with_h = state.h_walls | jnp.concatenate([_ONE_WALL, empty])
    with_v = state.v_walls | jnp.concatenate([empty, _ONE_WALL])
    return jax.vmap(lambda h, v: path_lengths(state._replace(h_walls=h, v_walls=v)))(
        with_h, with_v
    )


def _reaches(pawn: jax.Array, goal_row: int, right: jax.Array, down: jax.Array):
    """Bitboard flood fill (rows as bits): can the pawn still reach goal_row?"""

    def grow(carry):
        reach, _ = carry
        grown = (
            reach
            | ((reach & right) << 1)
            | ((reach >> 1) & right)
            | jnp.pad(reach[:-1] & down, (1, 0))
            | jnp.pad(reach[1:] & down, (0, 1))
        )
        return grown, jnp.any(grown != reach) & (grown[goal_row] == 0)

    start = jnp.zeros(9, jnp.uint32).at[pawn[0]].set(_BITS[pawn[1]])
    reach, _ = jax.lax.while_loop(lambda c: c[1], grow, (start, jnp.array(True)))
    return reach[goal_row] != 0


# Bitboard flood fill
def _keeps_paths(state: State) -> jax.Array:
    """bool (128,): each candidate wall leaves both players a route to their goal."""
    right, down = open_edges(state.h_walls, state.v_walls)
    right = (right * _BITS[:8]).sum(-1).astype(jnp.uint32)
    down = (down * _BITS).sum(-1).astype(jnp.uint32)

    def both_reach(right, down):
        return _reaches(state.pawns[0], GOAL_ROWS[0], right, down) & _reaches(
            state.pawns[1], GOAL_ROWS[1], right, down
        )

    rows, cols = jnp.arange(8).repeat(8), jnp.tile(jnp.arange(8), 8)
    h = jax.vmap(  # an H wall at (r, c) cuts the down-edges of cols c, c+1 in row r
        lambda r, c: both_reach(right, down.at[r].set(down[r] & ~(_BITS[c] * 3)))
    )(rows, cols)
    v = jax.vmap(  # a V wall at (r, c) cuts the right-edge of col c in rows r, r+1
        lambda r, c: both_reach(
            right.at[r]
            .set(right[r] & ~_BITS[c])
            .at[r + 1]
            .set(right[r + 1] & ~_BITS[c]),
            down,
        )
    )(rows, cols)
    return jnp.concatenate([h, v])


def _can_step(right: jax.Array, down: jax.Array, cell: jax.Array, d) -> jax.Array:
    """True if a pawn on `cell` can step in direction d (static): board + walls."""
    r, c = cell[0], cell[1]
    if d == (-1, 0):
        return (r > 0) & down[jnp.maximum(r - 1, 0), c]
    if d == (1, 0):
        return (r < 8) & down[jnp.minimum(r, 7), c]
    if d == (0, 1):
        return (c < 8) & right[r, jnp.minimum(c, 7)]
    return (c > 0) & right[r, jnp.maximum(c - 1, 0)]


def _pawn_mask(state: State, right: jax.Array, down: jax.Array) -> jax.Array:
    """Steps, straight jumps, and diagonal jumps when the straight one is blocked."""
    me, opp = state.pawns[state.to_move], state.pawns[1 - state.to_move]
    mask = jnp.zeros(N_PAWN, bool)
    for i, d in enumerate(DIRS):  # DIRS order == PAWN_OFFSETS[0:4], jumps at [4:8]
        nxt = jnp.clip(me + jnp.array(d), 0, 8)
        can = _can_step(right, down, me, d)
        meets = can & jnp.all(nxt == opp)
        straight = _can_step(right, down, nxt, d)
        mask = mask.at[i].set(can & ~meets).at[4 + i].set(meets & straight)
        sides = ((0, 1), (0, -1)) if d[1] == 0 else ((1, 0), (-1, 0))
        for side in sides:
            k = PAWN_OFFSETS.index((d[0] + side[0], d[1] + side[1]))
            diagonal = meets & ~straight & _can_step(right, down, nxt, side)
            mask = mask.at[k].set(mask[k] | diagonal)
    return mask


def legal_mask(state: State) -> jax.Array:
    """bool (140,) in engine.actions order: pawn moves, then H walls, then V walls."""
    h, v = state.h_walls, state.v_walls
    right, down = open_edges(h, v)
    # A slot is free if no wall overlaps it end-to-end or crosses it at the centre.
    h_free = ~(
        h
        | v
        | jnp.pad(h[:, :-1], ((0, 0), (1, 0)))
        | jnp.pad(h[:, 1:], ((0, 0), (0, 1)))
    )
    v_free = ~(
        v | h | jnp.pad(v[:-1], ((1, 0), (0, 0))) | jnp.pad(v[1:], ((0, 1), (0, 0)))
    )
    keeps_paths = _keeps_paths(state)
    has_walls = state.walls_left[state.to_move] > 0
    walls = jnp.concatenate([h_free.ravel(), v_free.ravel()]) & keeps_paths & has_walls
    return jnp.concatenate([_pawn_mask(state, right, down), walls])


def init() -> State:
    state = State(
        pawns=jnp.array(STARTS, jnp.int32),
        h_walls=jnp.zeros((8, 8), bool),
        v_walls=jnp.zeros((8, 8), bool),
        walls_left=jnp.full(2, WALLS_PER_PLAYER, jnp.int32),
        to_move=jnp.array(0, jnp.int32),
        ply=jnp.array(0, jnp.int32),
        winner=jnp.array(-1, jnp.int32),
        done=jnp.array(False),
        legal=jnp.zeros(ACTION_SIZE, bool),
    )
    return state._replace(legal=legal_mask(state))


def step(state: State, action) -> State:
    """Apply a legal action. A finished game is left unchanged."""
    action = jnp.asarray(action)
    player = state.to_move
    me = state.pawns[player]
    is_pawn = action < N_PAWN
    dest = me + _OFFSETS[jnp.minimum(action, N_PAWN - 1)]
    slot = jnp.maximum(action - N_PAWN, 0)
    wall = _ONE_WALL[slot % N_WALL] & ~is_pawn
    won = is_pawn & (dest[0] == _GOALS[player])
    nxt = State(
        pawns=state.pawns.at[player].set(jnp.where(is_pawn, dest, me)),
        h_walls=state.h_walls | (wall & (slot < N_WALL)),
        v_walls=state.v_walls | (wall & (slot >= N_WALL)),
        walls_left=state.walls_left.at[player].add(jnp.where(is_pawn, 0, -1)),
        to_move=1 - player,
        ply=state.ply + 1,
        winner=jnp.where(won, player, -1),
        done=won | (state.ply + 1 >= MAX_PLY),
        legal=state.legal,
    )
    nxt = nxt._replace(legal=legal_mask(nxt) & ~nxt.done)
    return jax.tree.map(lambda old, new: jnp.where(state.done, old, new), state, nxt)


def from_python(game: GameState, ply: int = 0) -> State:
    """Convert an engine GameState (e.g. from a human game or the minimax arena)."""
    walls = np.zeros((2, 8, 8), bool)
    for orient, r, c in game.walls:
        walls[int(orient == "V"), r, c] = True
    state = State(
        pawns=jnp.array(game.pawns, jnp.int32),
        h_walls=jnp.array(walls[0]),
        v_walls=jnp.array(walls[1]),
        walls_left=jnp.array(game.walls_left, jnp.int32),
        to_move=jnp.array(game.to_move, jnp.int32),
        ply=jnp.array(ply, jnp.int32),
        winner=jnp.array(-1, jnp.int32),
        done=jnp.array(False),
        legal=jnp.zeros(ACTION_SIZE, bool),
    )
    return state._replace(legal=_legal_jit(state))


_legal_jit = jax.jit(legal_mask)
