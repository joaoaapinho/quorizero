"""
Benchmark a checkpoint vs greedy, minimax and another checkpoint (Elo, 95% CI)
python -m evaluation.run_eval runs/main/iter_0100
python -m evaluation.run_eval runs/main/latest --against runs/main/iter_0050
"""

from __future__ import annotations

import argparse

import jax

from baseline.match import baseline_agent
from evaluation.arena import match_report, play_match, win_rate
from evaluation.elo import elo_from_winrate, elo_interval
from mcts.search import greedy_agent, net_agent, python_agent
from model.checkpoint import load_model


def show(name: str, score: float, games: int) -> None:
    low, high = elo_interval(score, games)
    elo = elo_from_winrate(score)
    print(f"{name:<28} score {score:5.2f}  elo {elo:+6.0f}  [{low:+.0f}, {high:+.0f}]")


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark a checkpoint")
    p.add_argument("checkpoint")
    p.add_argument("--against", help="another checkpoint to play")
    p.add_argument("--depths", type=int, nargs="*", default=[1, 2])
    p.add_argument("--games", type=int, default=40, help="games per minimax depth")
    p.add_argument("--batch-games", type=int, default=512, help="games vs greedy/ckpt")
    p.add_argument("--sims", type=int, default=128)
    args = p.parse_args()

    agent = net_agent(load_model(args.checkpoint), args.sims)
    key_greedy, key_ckpt = jax.random.split(jax.random.key(0))
    report = match_report(agent, greedy_agent, key_greedy, args.batch_games)
    show("greedy (depth 1)", report["score"], args.batch_games)
    if args.against:
        other = net_agent(load_model(args.against), args.sims)
        report = match_report(agent, other, key_ckpt, args.batch_games)
        show(args.against, report["score"], args.batch_games)
    for depth in args.depths:
        result = play_match(python_agent(agent), baseline_agent(depth), args.games)
        show(f"minimax depth {depth}", win_rate(*result), args.games)


if __name__ == "__main__":
    main()
