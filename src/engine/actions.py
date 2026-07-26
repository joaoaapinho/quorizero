"""Fixed integer action space + the core step / legal-action API.

Layout (ACTION_SIZE = 140):
  0..11    pawn moves, encoded by relative offset (see PAWN_OFFSETS)
  12..75   horizontal walls, index 12 + r*8 + c
  76..139  vertical walls,   index 76 + r*8 + c
"""

from __future__ import annotations

from engine.moves import pawn_moves, wall_placements
from engine.state import GOAL_ROWS, GameState, Wall

# Ordered: N S E W | straight jumps NN SS EE WW | diagonal jumps NE NW SE SW.
# fmt: off
PAWN_OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, 0), (1, 0), (0, 1), (0, -1),
    (-2, 0), (2, 0), (0, 2), (0, -2),
    (-1, 1), (-1, -1), (1, 1), (1, -1),
)
# fmt: on
_OFFSET_INDEX = {off: i for i, off in enumerate(PAWN_OFFSETS)}

N_PAWN = len(PAWN_OFFSETS)  # 12
N_WALL = 64  # 8x8 per orientation
ACTION_SIZE = N_PAWN + 2 * N_WALL  # 140


def encode_move(frm: tuple[int, int], to: tuple[int, int]) -> int:
    return _OFFSET_INDEX[(to[0] - frm[0], to[1] - frm[1])]


def encode_wall(wall: Wall) -> int:
    orient, r, c = wall
    base = N_PAWN if orient == "H" else N_PAWN + N_WALL
    return base + r * 8 + c


def decode_action(state: GameState, action: int):
    """Return ('move', (r, c)) or ('wall', (orient, r, c))."""
    if action < N_PAWN:
        dr, dc = PAWN_OFFSETS[action]
        r, c = state.pawns[state.to_move]
        return "move", (r + dr, c + dc)
    idx = action - N_PAWN
    orient = "H" if idx < N_WALL else "V"
    idx %= N_WALL
    return "wall", (orient, idx // 8, idx % 8)


def legal_actions(state: GameState) -> list[int]:
    frm = state.pawns[state.to_move]
    actions = [encode_move(frm, dest) for dest in pawn_moves(state)]
    actions += [encode_wall(w) for w in wall_placements(state)]
    return actions


def legal_action_mask(state: GameState) -> list[bool]:
    mask = [False] * ACTION_SIZE
    for a in legal_actions(state):
        mask[a] = True
    return mask


def step(state: GameState, action: int) -> tuple[GameState, bool, int | None]:
    """Apply a legal action, returning (next_state, done, winner)."""
    kind, payload = decode_action(state, action)
    player = state.to_move
    if kind == "move":
        nxt = state.move_pawn(payload)
        if payload[0] == GOAL_ROWS[player]:
            return nxt, True, player
        return nxt, False, None
    return state.place_wall(payload), False, None
