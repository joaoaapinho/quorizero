"""Wall-legality guard: every player must always retain a path to their goal row.

Isolated on purpose — this is the trickiest correctness piece in the engine.
"""

from __future__ import annotations

from collections import deque

from engine.moves import edge_blocked
from engine.state import DIRS, GameState, Wall, on_board


def shortest_distance(
    walls: frozenset[Wall], start: tuple[int, int], goal_row: int
) -> int | None:
    """Fewest steps from start to any cell in goal_row, or None if unreachable.

    Ignores opponent pawns, same as has_path — used by the heuristic evaluation.
    """
    seen = {start}
    queue = deque([(start, 0)])
    while queue:
        (r, c), dist = queue.popleft()
        if r == goal_row:
            return dist
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if (
                on_board(nr, nc)
                and (nr, nc) not in seen
                and not edge_blocked(walls, r, c, nr, nc)
            ):
                seen.add((nr, nc))
                queue.append(((nr, nc), dist + 1))
    return None


def has_path(walls: frozenset[Wall], start: tuple[int, int], goal_row: int) -> bool:
    """BFS from start to any cell in goal_row, respecting wall blocks.

    Opponent pawns are ignored: they move and can be jumped, so they never
    permanently block a path.
    """
    seen = {start}
    frontier = [start]
    while frontier:
        r, c = frontier.pop()
        if r == goal_row:
            return True
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if (
                on_board(nr, nc)
                and (nr, nc) not in seen
                and not edge_blocked(walls, r, c, nr, nc)
            ):
                seen.add((nr, nc))
                frontier.append((nr, nc))
    return False


def wall_keeps_paths(state: GameState, wall: Wall) -> bool:
    """True if placing `wall` still leaves both players a route to their goal."""
    from engine.state import GOAL_ROWS

    walls = state.walls | {wall}
    return has_path(walls, state.pawns[0], GOAL_ROWS[0]) and has_path(
        walls, state.pawns[1], GOAL_ROWS[1]
    )
