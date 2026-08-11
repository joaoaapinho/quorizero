"""Move generation: wall-blocking test, pawn moves (incl. jumps), wall placements."""

from __future__ import annotations

from engine.state import DIRS, GameState, Wall, on_board


def edge_blocked(walls: frozenset[Wall], r: int, c: int, nr: int, nc: int) -> bool:
    """True if a wall blocks the step between orthogonally adjacent cells.

    Out-of-range wall anchors are absent from 'walls', so membership
    tests handle board edges without extra bounds checks.
    """
    if nr == r - 1:  # up: wall in the groove below row r-1
        return ("H", r - 1, c) in walls or ("H", r - 1, c - 1) in walls
    if nr == r + 1:  # down: wall in the groove below row r
        return ("H", r, c) in walls or ("H", r, c - 1) in walls
    if nc == c + 1:  # right: wall in the groove right of col c
        return ("V", r, c) in walls or ("V", r - 1, c) in walls
    # left: wall in the groove right of col c-1
    return ("V", r, c - 1) in walls or ("V", r - 1, c - 1) in walls


def _perpendicular(dr: int, dc: int) -> tuple[tuple[int, int], tuple[int, int]]:
    return ((0, 1), (0, -1)) if dc == 0 else ((1, 0), (-1, 0))


def pawn_moves(state: GameState) -> list[tuple[int, int]]:
    """All legal pawn destinations for the player to move, including jumps."""
    walls = state.walls
    r, c = state.pawns[state.to_move]
    opp = state.pawns[1 - state.to_move]
    dests: list[tuple[int, int]] = []

    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if not on_board(nr, nc) or edge_blocked(walls, r, c, nr, nc):
            continue
        if (nr, nc) != opp:
            dests.append((nr, nc))
            continue
        # Opponent is adjacent: try a straight jump over it.
        jr, jc = nr + dr, nc + dc
        if on_board(jr, jc) and not edge_blocked(walls, nr, nc, jr, jc):
            dests.append((jr, jc))
        else:
            # Straight jump blocked (wall or edge): fall back to diagonals.
            for pdr, pdc in _perpendicular(dr, dc):
                sr, sc = nr + pdr, nc + pdc
                if on_board(sr, sc) and not edge_blocked(walls, nr, nc, sr, sc):
                    dests.append((sr, sc))
    return dests


def _slot_free(walls: frozenset[Wall], orient: str, r: int, c: int) -> bool:
    """True if an H/V wall at (r, c) neither overlaps nor crosses an existing one."""
    if orient == "H":
        return (
            ("H", r, c) not in walls
            and ("H", r, c - 1) not in walls
            and ("H", r, c + 1) not in walls
            and ("V", r, c) not in walls
        )
    return (
        ("V", r, c) not in walls
        and ("V", r - 1, c) not in walls
        and ("V", r + 1, c) not in walls
        and ("H", r, c) not in walls
    )


def wall_placements(state: GameState) -> list[Wall]:
    """All legal wall placements for the player to move."""
    if state.walls_left[state.to_move] == 0:
        return []
    from engine.pathcheck import wall_keeps_paths

    walls = state.walls
    placements: list[Wall] = []
    for r in range(8):
        for c in range(8):
            for orient in ("H", "V"):
                wall = (orient, r, c)
                if _slot_free(walls, orient, r, c) and wall_keeps_paths(state, wall):
                    placements.append(wall)
    return placements
