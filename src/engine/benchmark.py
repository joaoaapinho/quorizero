"""
Baseline speed check for step() / legal_actions()
python -m engine.benchmark
"""

from __future__ import annotations

import random
import time

from engine.actions import legal_actions, step
from engine.state import initial_state, is_terminal


def _play_random_game(rng: random.Random) -> tuple[int, int]:
    state = initial_state()
    steps = legal_calls = 0
    for _ in range(400):  # cap to avoid non-terminating games
        actions = legal_actions(state)
        legal_calls += 1
        state, done, _ = step(state, rng.choice(actions))
        steps += 1
        if done or is_terminal(state):
            break
    return steps, legal_calls


def main(games: int = 200) -> None:
    rng = random.Random(0)
    total_steps = total_legal = 0
    t0 = time.perf_counter()
    for _ in range(games):
        s, lc = _play_random_game(rng)
        total_steps += s
        total_legal += lc
    dt = time.perf_counter() - t0

    print(f"{games} games in {dt:.2f}s")
    print(f"  step():          {total_steps / dt:,.0f} calls/s ({total_steps} total)")
    print(f"  legal_actions(): {total_legal / dt:,.0f} calls/s ({total_legal} total)")


if __name__ == "__main__":
    main()
