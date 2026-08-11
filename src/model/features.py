"""State -> stacked-planes encoding: the tensor the network actually sees.

The board is 9x9. We stack N_PLANES feature planes (channels-LAST, so the shape
is (9, 9, N_PLANES) for one position — Flax convolutions expect channels last).

Everything is written from the CURRENT player's point of view (plane 0 is always
"my" pawn, plane 4 is always "my" walls left). That way one network handles both
sides; the only thing that differs per side is which goal row is "forward", which
plane 6 tells the network.

Plane layout
------------
    0  my pawn            (one-hot on the 9x9 grid)
    1  opponent pawn      (one-hot)
    2  horizontal walls   (1.0 at each H anchor (r, c); rows/cols 0..7 used)
    3  vertical walls      (1.0 at each V anchor)
    4  my walls left       (constant plane = count / 10)
    5  opponent walls left (constant plane = count / 10)
    6  player to move      (constant plane = 0.0 for player 0, 1.0 for player 1)
"""

from __future__ import annotations

import jax.numpy as jnp

from engine.state import WALLS_PER_PLAYER, GameState  # noqa: F401 (used in TODO 4/5)

N_PLANES = 7


def encode_state(state: GameState) -> jnp.ndarray:
    """Encode one GameState as a (9, 9, N_PLANES) float32 array.

    Reminder: JAX arrays are immutable. You don't write `planes[r, c, k] = v`;
    you write `planes = planes.at[r, c, k].set(v)`, which returns a NEW array.
    Reassign every time, or your change is lost.
    """
    cur = state.to_move
    opp = 1 - cur
    planes = jnp.zeros((9, 9, N_PLANES), dtype=jnp.float32)

    # Plane 0: my pawn
    r, c = state.pawns[cur]
    planes = planes.at[r, c, 0].set(1.0)

    # Plane 1: opponent pawn
    r, c = state.pawns[opp]
    planes = planes.at[r, c, 1].set(1.0)

    # Plane 2: horizontal walls
    for orient, r, c in state.walls:
        if orient == "H":
            planes = planes.at[r, c, 2].set(1.0)

    # Plane 3: vertical walls
    for orient, r, c in state.walls:
        if orient == "V":
            planes = planes.at[r, c, 3].set(1.0)

    # Plane 4: my walls left, as a CONSTANT plane
    value = state.walls_left[cur] / WALLS_PER_PLAYER
    planes = planes.at[:, :, 4].set(value)  # ':' fills the whole plane

    # Plane 5: opponent walls left
    value = state.walls_left[opp] / WALLS_PER_PLAYER
    planes = planes.at[:, :, 5].set(value)

    # Plane 6: player-to-move flag, constant plane = float(cur)
    planes = planes.at[:, :, 6].set(float(cur))

    return planes


def encode_batch(states: list[GameState]) -> jnp.ndarray:
    """Stack several encoded states into one (B, 9, 9, N_PLANES) batch."""
    return jnp.stack([encode_state(s) for s in states])
