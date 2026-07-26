"""Minimal text-board CLI to play Quoridor against the engine.

Run:
    python -m engine.play            # hotseat: two humans
    python -m engine.play --random   # you are player 0, opponent plays randomly

Commands each turn:
    r c        move your pawn to cell (row col), e.g.  1 4
    h r c      place a Horizontal wall at anchor (r c), 0..7
    v r c      place a Vertical wall at anchor (r c),   0..7
    moves      list your legal pawn destinations
    q          quit
"""

from __future__ import annotations

import random
import sys

from engine.actions import encode_move, encode_wall, legal_actions, step
from engine.moves import edge_blocked, pawn_moves
from engine.state import initial_state


def render(state) -> str:
    walls = state.walls
    out = ["    " + "   ".join(str(c) for c in range(9))]
    for r in range(9):
        cells = []
        for c in range(9):
            if state.pawns[0] == (r, c):
                ch = "0"
            elif state.pawns[1] == (r, c):
                ch = "1"
            else:
                ch = "."
            cells.append(ch)
            if c < 8:
                cells.append(" | " if edge_blocked(walls, r, c, r, c + 1) else "   ")
        out.append(f"{r}   " + "".join(cells))
        if r < 8:  # between-row line shows horizontal walls
            seps = []
            for c in range(9):
                seps.append("---" if edge_blocked(walls, r, c, r + 1, c) else "   ")
                if c < 8:
                    seps.append(" ")
            out.append("    " + "".join(seps))
    return "\n".join(out)


def parse(tokens, state):
    """Turn user tokens into a legal action int, or None if invalid."""
    frm = state.pawns[state.to_move]
    try:
        if len(tokens) == 2:  # pawn move: "r c"
            dest = (int(tokens[0]), int(tokens[1]))
            return encode_move(frm, dest)
        if len(tokens) == 3 and tokens[0] in ("h", "v"):  # wall
            orient = tokens[0].upper()
            return encode_wall((orient, int(tokens[1]), int(tokens[2])))
    except (ValueError, KeyError):
        return None
    return None


def main() -> None:
    random_opponent = "--random" in sys.argv
    rng = random.Random()
    state = initial_state()

    while True:
        print("\n" + render(state))
        p = state.to_move
        print(f"\nplayer {p} to move | walls left: {state.walls_left}")

        if random_opponent and p == 1:
            action = rng.choice(legal_actions(state))
            state, done, win = step(state, action)
            if done:
                print(f"\n{render(state)}\n\nplayer {win} wins!")
                return
            continue

        raw = input("> ").strip().split()
        if not raw:
            continue
        if raw[0] == "q":
            return
        if raw[0] == "moves":
            print("legal pawn moves:", sorted(pawn_moves(state)))
            continue

        action = parse(raw, state)
        if action is None or action not in set(legal_actions(state)):
            print("illegal — try again ('moves' lists pawn options)")
            continue

        state, done, win = step(state, action)
        if done:
            print(f"\n{render(state)}\n\nplayer {win} wins!")
            return


if __name__ == "__main__":
    main()
