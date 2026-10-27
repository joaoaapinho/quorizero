"""
Benchmark against SigmaQuoridor, Qudor AI and gorisanson's AI, in one table
python -m evaluation.benchmark runs/pilot-final/iter_0560 --games 100
python -m evaluation.benchmark runs/pilot-final/iter_0560 --opponents goris --jobs 4
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

OPPONENTS = {
    "sigma": "SigmaQuoridor (github.com/bartolomeo3000/SigmaQuoridor)",
    "qudor": "Qudor AI gen85 (github.com/ktoto1300/Qudor-AI)",
    "goris": "gorisanson Strong, 60k rollouts (github.com/gorisanson/quoridor-ai)",
}
MAX_PLIES = 200  # after the opening; a game this long is a draw
_worker: dict = {}  # each process builds its agents once


def opening(seed: int, pair: int) -> list[int]:
    from evaluation.arena import opening_moves

    return opening_moves(random.Random(f"{seed}-{pair}"))


def init_worker(checkpoint: str, opponent: str, sims: int, rollouts=60_000) -> None:
    import torch

    from mcts.puct import agent, network_evaluator
    from model.checkpoint import load_model

    torch.set_num_threads(1)
    ours = agent(network_evaluator(load_model(checkpoint)), sims)
    _worker["ours"] = lambda game, history: ours(game)
    if opponent == "sigma":
        from evaluation.vs_sigma import SIGMA_PATH, sigma_evaluator

        theirs = agent(sigma_evaluator(SIGMA_PATH), sims)
        _worker["theirs"] = lambda game, history: theirs(game)
    elif opponent == "qudor":
        from evaluation.vs_qudor import QUDOR_PATH, qudor_evaluator

        clock = [0]  # their network reads the game's ply
        theirs = agent(qudor_evaluator(QUDOR_PATH, clock), sims)

        def qudor_move(game, history):
            clock[0] = len(history)
            return theirs(game)

        _worker["theirs"] = qudor_move
    else:
        from evaluation.vs_goris import GorisBot

        bot = GorisBot(rollouts=rollouts)
        _worker["theirs"] = lambda game, history: bot.move(history)


def play_game(seed: int, pair: int, our_seat: int) -> dict:
    """One game from the pair's opening, QuoriZero in our_seat."""
    from engine.actions import step
    from engine.state import initial_state

    start = opening(seed, pair)
    game, history, done, winner = initial_state(), list(start), False, None
    for action in start:
        game = step(game, action)[0]
    while not done and len(history) - len(start) < MAX_PLIES:
        side = "ours" if game.to_move == our_seat else "theirs"
        action = _worker[side](game, history)
        history.append(action)
        game, done, winner = step(game, action)
    outcome = "draw" if winner is None else ("win" if winner == our_seat else "loss")
    return dict(pair=pair, our_seat=our_seat, outcome=outcome, moves=history)


def play_pair(seed: int, pair: int) -> list[dict]:
    """Both games of one opening: QuoriZero first, then second."""
    return [play_game(seed, pair, our_seat) for our_seat in (0, 1)]


def summarize(games: list[dict]) -> dict:
    from evaluation.arena import win_rate
    from evaluation.elo import elo_from_winrate, elo_interval

    n = len(games)
    wins = sum(g["outcome"] == "win" for g in games)
    draws = sum(g["outcome"] == "draw" for g in games)
    score = win_rate(wins, n - wins - draws, draws)
    low, high = elo_interval(score, n)
    first = [g for g in games if g["our_seat"] == 0]
    second = [g for g in games if g["our_seat"] == 1]
    return dict(
        games=n,
        wins=wins,
        draws=draws,
        losses=n - wins - draws,
        score=score,
        elo=elo_from_winrate(score),
        elo_low=low,
        elo_high=high,
        wins_moving_first=sum(g["outcome"] == "win" for g in first),
        wins_moving_second=sum(g["outcome"] == "win" for g in second),
        avg_length=sum(len(g["moves"]) for g in games) / max(n, 1),
    )


def run(checkpoint: str, opponent: str, games: int, sims: int, seed: int, jobs: int):
    pairs = games // 2
    print(
        f"\nours vs {OPPONENTS[opponent]}: {pairs * 2} games, {jobs} in parallel",
        flush=True,
    )
    played, tic = [], time.time()
    with ProcessPoolExecutor(
        jobs, get_context("spawn"), init_worker, (checkpoint, opponent, sims)
    ) as pool:
        futures = [pool.submit(play_pair, seed, p) for p in range(pairs)]
        for future in as_completed(futures):
            played += future.result()
            s = summarize(played)
            print(
                f"  {s['games']:>3}/{pairs * 2}: {s['wins']}W {s['draws']}D "
                f"{s['losses']}L | score {s['score']:.2f} | {time.time() - tic:.0f}s",
                flush=True,
            )
    return summarize(played), played


def main() -> None:
    p = argparse.ArgumentParser(description="QuoriZero vs the public Quoridor engines")
    p.add_argument("checkpoint", help="ours, e.g. runs/pilot-final/iter_0560")
    p.add_argument("--opponents", nargs="+", default=list(OPPONENTS), choices=OPPONENTS)
    p.add_argument("--games", type=int, default=100, help="per opponent, even")
    p.add_argument("--sims", type=int, default=64, help="for both networks")
    p.add_argument("--seed", type=int, default=1, help="which openings")
    p.add_argument("--jobs", type=int, default=6, help="games in parallel")
    p.add_argument(
        "--save-losses", help="folder for our lost games (jsonl per opponent)"
    )
    args = p.parse_args()
    os.environ["JAX_PLATFORMS"] = "cpu"  # inherited by every worker: none takes the GPU

    table = {}
    for opponent in args.opponents:
        summary, played = run(
            args.checkpoint, opponent, args.games, args.sims, args.seed, args.jobs
        )
        table[opponent] = summary
        if args.save_losses:
            out = Path(args.save_losses)
            out.mkdir(parents=True, exist_ok=True)
            with open(out / f"lost_to_{opponent}.jsonl", "w") as f:
                for g in played:
                    if g["outcome"] == "loss":
                        f.write(json.dumps(g) + "\n")
        log = Path("runs/benchmarks.jsonl")
        log.parent.mkdir(exist_ok=True)
        entry = dict(time=time.time(), checkpoint=args.checkpoint, opponent=opponent)
        entry.update(sims=args.sims, seed=args.seed, **summary)
        with log.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    print("\n| Opponent | Games | Won (first / second) | Score | Elo gap (95% CI) |")
    print("|---|---|---|---|---|")
    for opponent, s in table.items():
        print(
            f"| {OPPONENTS[opponent]} | {s['games']} | {s['wins']} "
            f"({s['wins_moving_first']} / {s['wins_moving_second']}) "
            f"| {s['score']:.0%} | {s['elo']:+.0f} "
            f"[{s['elo_low']:+.0f}, {s['elo_high']:+.0f}] |"
        )


if __name__ == "__main__":
    main()
