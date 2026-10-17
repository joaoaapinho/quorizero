"""Play agents against each other; CLI runs Step 2 sanity checks.

python -m baseline.match                 # both sanity checks (tally only)
python -m baseline.match --games 20 --depth 2
python -m baseline.match --view          # ONE bot game, move by move
"""

from __future__ import annotations

import argparse
import random
import time
from typing import Callable

from baseline.minimax import choose_action
from engine.actions import decode_action, legal_actions, step
from engine.play import render
from engine.state import GameState, initial_state

Agent = Callable[[GameState], int]


def _describe(state: GameState, action: int) -> str:
    kind, payload = decode_action(state, action)
    return f"move to {payload}" if kind == "move" else f"wall {payload}"


def random_agent(rng: random.Random) -> Agent:
    return lambda state: rng.choice(legal_actions(state))


def baseline_agent(depth: int = 2) -> Agent:
    return lambda state: choose_action(state, depth)


def play_game(
    agent0: Agent,
    agent1: Agent,
    max_moves: int = 400,
    view: bool = False,
    delay: float = 0.0,
    state: GameState | None = None,
) -> int | None:
    """Play one game (from `state`, default the start); return the winner or None
    if the move cap is hit. With view=True, print the board after every ply.
    """
    state = state or initial_state()
    agents = (agent0, agent1)
    if view:
        print(render(state))
    for ply in range(max_moves):
        player = state.to_move
        action = agents[player](state)
        if view:
            print(f"\nply {ply + 1} - player {player}: {_describe(state, action)}")
        state, done, win = step(state, action)
        if view:
            print(render(state))
            time.sleep(delay)
        if done:
            return win
    return None


def _series(agent0: Agent, agent1: Agent, games: int) -> tuple[int, int, int]:
    """Return (agent0_wins, agent1_wins, draws), alternating who starts."""
    wins = [0, 0]
    draws = 0
    for g in range(games):
        # Alternate seats so neither agent keeps the first-move advantage.
        first, second = (agent0, agent1) if g % 2 == 0 else (agent1, agent0)
        result = play_game(first, second)
        if result is None:
            draws += 1
        else:
            winner = result if g % 2 == 0 else 1 - result
            wins[winner] += 1
    return wins[0], wins[1], draws


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline bot sanity checks")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--view", action="store_true", help="watch one bot game move by move"
    )
    parser.add_argument(
        "--delay", type=float, default=0.4, help="seconds between moves when viewing"
    )
    args = parser.parse_args()
    rng = random.Random(args.seed)

    if args.view:
        print(f"watching one baseline vs baseline game (depth={args.depth}):")
        win = play_game(
            baseline_agent(args.depth),
            baseline_agent(args.depth),
            view=True,
            delay=args.delay,
        )
        print(f"\nplayer {win} wins!")
        return

    bvb_games = max(2, args.games // 5)  # slow at depth 2; only a crash check
    print(f"baseline vs baseline ({bvb_games} games - sane, non-crashing, decisive):")
    b_wins, w_wins, draws = _series(
        baseline_agent(args.depth), baseline_agent(args.depth), bvb_games
    )
    print(f"  player0 {b_wins} | player1 {w_wins} | draws {draws}")

    print(
        f"baseline(depth={args.depth}) vs random ({args.games} games, should win most):"
    )
    b_wins, r_wins, draws = _series(
        baseline_agent(args.depth), random_agent(rng), args.games
    )
    total = b_wins + r_wins + draws
    print(f"  baseline {b_wins}/{total} | random {r_wins} | draws {draws}")


if __name__ == "__main__":
    main()
