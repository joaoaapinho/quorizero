"""Quoridor state board snapshot + win/terminal checks."""

from __future__ import annotations

from dataclasses import dataclass, replace

BOARD_SIZE = 9  # 9x9 board
WALL_SLOTS = BOARD_SIZE - 1  # 8x8 slots
WALLS_PER_PLAYER = 10

# P0 starts at the top and races to the bottom row
# P1 does the reverse
STARTS = ((0, 4), (8, 4))
GOAL_ROWS = (8, 0)

# Step directions: up, down, right, left
DIRS = ((-1, 0), (1, 0), (0, 1), (0, -1))

# A wall is ('H'|'V', r, c)
# 'H' sits under r, from c and c+1
# 'V' sits right c, from r and r+1
Wall = tuple[str, int, int]


def on_board(r: int, c: int) -> bool:
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE


@dataclass(frozen=True)
class GameState:
    pawns: tuple[tuple[int, int], tuple[int, int]]
    walls: frozenset[Wall]
    walls_left: tuple[int, int]
    to_move: int  # 0 or 1

    def move_pawn(self, dest: tuple[int, int]) -> GameState:
        pawns = list(self.pawns)
        pawns[self.to_move] = dest
        return replace(self, pawns=(pawns[0], pawns[1]), to_move=1 - self.to_move)

    def place_wall(self, wall: Wall) -> GameState:
        left = list(self.walls_left)
        left[self.to_move] -= 1
        return replace(
            self,
            walls=self.walls | {wall},
            walls_left=(left[0], left[1]),
            to_move=1 - self.to_move,
        )


def initial_state() -> GameState:
    return GameState(
        pawns=STARTS,
        walls=frozenset(),
        walls_left=(WALLS_PER_PLAYER, WALLS_PER_PLAYER),
        to_move=0,
    )


def winner(state: GameState) -> int | None:
    if state.pawns[0][0] == GOAL_ROWS[0]:
        return 0
    if state.pawns[1][0] == GOAL_ROWS[1]:
        return 1
    return None


def is_terminal(state: GameState) -> bool:
    return winner(state) is not None
