"""Wall-legality guard: every player must always retain a path to their goal row."""

from __future__ import annotations

from collections import deque

import numpy as np

from engine.moves import edge_blocked
from engine.state import BOARD_SIZE, DIRS, GameState, Wall, on_board


def shortest_distance(
    walls: frozenset[Wall], start: tuple[int, int], goal_row: int
) -> int | None:
    """
    Fewest steps from start to any cell in goal_row, or None if unreachable.
    Ignores opponent pawns, same as has_path - used by the heuristic evaluation.
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


def shortest_path_edges(
    walls: frozenset[Wall], start: tuple[int, int], goal_row: int
) -> set[frozenset]:
    """Edges on one shortest path from start to goal_row (as {cell, cell} pairs).

    Used to prune wall legality: a wall that blocks no edge on a player's current
    shortest path can't disconnect them, so it never needs a full path re-check.
    """
    parent = {start: None}
    queue = deque([start])
    goal = None
    while queue:
        r, c = queue.popleft()
        if r == goal_row:
            goal = (r, c)
            break
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if (
                on_board(nr, nc)
                and (nr, nc) not in parent
                and not edge_blocked(walls, r, c, nr, nc)
            ):
                parent[(nr, nc)] = (r, c)
                queue.append((nr, nc))
    edges: set[frozenset] = set()
    node = goal
    while node is not None and parent[node] is not None:
        edges.add(frozenset((node, parent[node])))
        node = parent[node]
    return edges


def distance_map(walls: frozenset[Wall], goal_row: int) -> np.ndarray:
    """Distance from every cell to goal_row, respecting walls (-1 if unreachable).

    Multi-source BFS out from the goal row. Fed to the network as an input plane
    so it can *see* the effect of walls on how far each square is from a goal —
    the signal that makes goal-seeking learnable from the very first games.
    """
    dist = np.full((BOARD_SIZE, BOARD_SIZE), -1, dtype=np.int32)
    queue = deque()
    for c in range(BOARD_SIZE):
        dist[goal_row, c] = 0
        queue.append((goal_row, c))
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if (
                on_board(nr, nc)
                and dist[nr, nc] == -1
                and not edge_blocked(walls, r, c, nr, nc)
            ):
                dist[nr, nc] = dist[r, c] + 1
                queue.append((nr, nc))
    return dist


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
    """True if placing 'wall' still leaves both players a route to their goal."""
    from engine.state import GOAL_ROWS

    walls = state.walls | {wall}
    return has_path(walls, state.pawns[0], GOAL_ROWS[0]) and has_path(
        walls, state.pawns[1], GOAL_ROWS[1]
    )
