"""
Network input: canonical planes (mover always heads to row 0), mirror symmetry.
Planes: my/opp pawn, H/V walls, my/opp walls left, my/opp distance-to-goal, ones.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from engine.actions import N_PAWN, PAWN_OFFSETS
from engine.state import WALLS_PER_PLAYER
from jaxgame.rules import State, distance_map, open_edges

N_PLANES = 9


def _action_permutation(flip_rows: bool) -> np.ndarray:
    """Where each of the 140 actions goes under a row flip or a column mirror."""
    perm = [
        PAWN_OFFSETS.index((-dr, dc) if flip_rows else (dr, -dc))
        for dr, dc in PAWN_OFFSETS
    ]
    for orient in range(2):
        for r in range(8):
            for c in range(8):
                rr, cc = (7 - r, c) if flip_rows else (r, 7 - c)
                perm.append(N_PAWN + orient * 64 + rr * 8 + cc)
    return np.array(perm)


FLIP = _action_permutation(flip_rows=True)  # both are their own inverse
MIRROR = _action_permutation(flip_rows=False)


def flip(state: State) -> State:
    """Rows reversed. Only for features: the goal rows are not swapped."""
    return state._replace(
        pawns=state.pawns.at[:, 0].set(8 - state.pawns[:, 0]),
        h_walls=state.h_walls[::-1],
        v_walls=state.v_walls[::-1],
        legal=state.legal[FLIP],
    )


# Mirror augmentation: left-right symmetry, free extra data.
def mirror(state: State) -> State:
    """Columns reversed: an equally valid game state (left-right symmetry)."""
    return state._replace(
        pawns=state.pawns.at[:, 1].set(8 - state.pawns[:, 1]),
        h_walls=state.h_walls[:, ::-1],
        v_walls=state.v_walls[:, ::-1],
        legal=state.legal[MIRROR],
    )


def orient_policy(policy: jax.Array, to_move) -> jax.Array:
    """Swap a 140-vector between the board's frame and the mover's canonical frame."""
    return jnp.where(to_move == 0, policy[FLIP], policy)


def observe(state: State) -> jax.Array:
    """Canonical planes (9, 9, N_PLANES) float32 for one state."""
    s = jax.tree.map(
        lambda a, b: jnp.where(state.to_move == 0, a, b), flip(state), state
    )
    me, opp = s.to_move, 1 - s.to_move
    right, down = open_edges(s.h_walls, s.v_walls)

    def pawn(cell):
        return jnp.zeros((9, 9)).at[cell[0], cell[1]].set(1.0)

    def walls(w):
        return jnp.pad(w, ((0, 1), (0, 1))).astype(jnp.float32)

    def dist(goal_row):
        return jnp.minimum(distance_map(goal_row, right, down), 32) / 16.0

    def const(x):
        return jnp.full((9, 9), x, jnp.float32)

    planes = [
        pawn(s.pawns[me]),
        pawn(s.pawns[opp]),
        walls(s.h_walls),
        walls(s.v_walls),
        const(s.walls_left[me] / WALLS_PER_PLAYER),
        const(s.walls_left[opp] / WALLS_PER_PLAYER),
        dist(0),  # KataGo-style game-specific input: shortest-path maps
        dist(8),
        const(1.0),
    ]
    return jnp.stack(planes, axis=-1)
