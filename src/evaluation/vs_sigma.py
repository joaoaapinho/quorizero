"""
Benchmark against SigmaQuoridor (MIT): github.com/bartolomeo3000/SigmaQuoridor
python -m evaluation.vs_sigma runs/pilot-final/iter_0560 --check
python -m evaluation.vs_sigma runs/pilot-final/iter_0560 --games 100 --sims 64
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort

from baseline.match import play_game
from engine.actions import PAWN_OFFSETS, step
from engine.pathcheck import distance_map
from engine.state import GOAL_ROWS, GameState, initial_state
from evaluation.arena import opening_moves, win_rate
from evaluation.elo import elo_from_winrate, elo_interval
from mcts.puct import agent, network_evaluator, softmax_over
from mcts.search import greedy_agent, python_agent
from model.checkpoint import load_model

SIGMA_URL = (
    "https://github.com/bartolomeo3000/SigmaQuoridor/raw/main/"
    "docs/models_9x9/best.onnx"
)
SIGMA_PATH = Path("data/sigma/best.onnx")
C_INIT, C_BASE = 1.25, 19652

# Their encoding: (x, y) is our (col, row), 8 planes and 136 actions
# (8 pawn directions, 64 H and 64 V walls), flipped for player 2

SIGMA_PAWN_DIRS = ((0, 1), (0, -1), (-1, 0), (1, 0), (-1, 1), (1, 1), (-1, -1), (1, -1))


def _ours_to_sigma() -> np.ndarray:
    """Our 140 action indices -> their 136 (real board frame)."""
    index = []
    for dr, dc in PAWN_OFFSETS:
        if abs(dr) == 2 or abs(dc) == 2:  # straight jump = their plain step
            dr, dc = dr // 2, dc // 2
        index.append(SIGMA_PAWN_DIRS.index((dc, dr)))
    return np.array(index + [8 + k for k in range(128)])


def _sigma_vflip() -> np.ndarray:
    """Their vertical-flip permutation (vert_policy_permutation in game.py)."""
    pawn = [1, 0, 2, 3, 6, 7, 4, 5]
    walls = [
        8 + o * 64 + (7 - y) * 8 + x for o in (0, 1) for y in range(8) for x in range(8)
    ]
    return np.array(pawn + walls)


OURS_TO_SIGMA = _ours_to_sigma()
OURS_TO_SIGMA_P2 = _sigma_vflip()[OURS_TO_SIGMA]  # player 2 reads the flipped frame


def sigma_planes(game: GameState) -> np.ndarray:
    """(1, 8, 9, 9) input exactly as their State.to_nn_input()."""
    me, opp = game.to_move, 1 - game.to_move
    hw, vw = np.zeros((9, 9), np.float32), np.zeros((9, 9), np.float32)
    for orient, r, c in game.walls:  # their wall planes mark both covered cells
        if orient == "H":
            hw[r, c] = hw[r, c + 1] = 1
        else:
            vw[r, c] = vw[r + 1, c] = 1
    dist = [
        distance_map(game.walls, GOAL_ROWS[p]).astype(np.float32) for p in (me, opp)
    ]
    my_pos, opp_pos = game.pawns[me], game.pawns[opp]
    if me == 1:  # their player 2: flip so the mover heads to row 8
        hw[:8] = hw[:8][::-1].copy()
        vw, dist = vw[::-1], [d[::-1] for d in dist]
        my_pos, opp_pos = (8 - my_pos[0], my_pos[1]), (8 - opp_pos[0], opp_pos[1])
    planes = np.zeros((8, 9, 9), np.float32)
    planes[0][my_pos] = planes[1][opp_pos] = 1
    planes[2], planes[3] = hw, vw
    planes[4], planes[5] = game.walls_left[me] / 10, game.walls_left[opp] / 10
    for k, d in enumerate(dist):
        planes[6 + k] = np.where(d >= 0, d / 80, 1.0)
    return planes[None]


# Evaluator: (game, legal actions) -> (priors over our 140, value)


def sigma_evaluator(path: Path):
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def evaluate_sigma(game: GameState, legal: list[int]):
        logits, value = session.run(
            ["policy_logits", "value"], {"input": sigma_planes(game)}
        )
        index = OURS_TO_SIGMA if game.to_move == 0 else OURS_TO_SIGMA_P2
        return softmax_over(logits[0][index], legal), float(value.reshape(-1)[0])

    return evaluate_sigma


# Match


def recorded(agent, moves: list[int]):
    """The same agent, appending each move it plays to `moves`."""

    def act(game: GameState) -> int:
        moves.append(agent(game))
        return moves[-1]

    return act


def match(
    name_a, agent_a, name_b, agent_b, games: int, seed: int = 0, lost=None
) -> dict:
    """
    Paired openings, seats swapped, prints every game as it finishes
    lost: if a list, a's lost games are appended, stored as the website does
    """
    rng, points, wins, draws = random.Random(seed), 0.0, 0, 0
    tic = time.time()
    for g in range(games):
        if g % 2 == 0:
            opening = opening_moves(rng)
            start = initial_state()
            for action in opening:
                start = step(start, action)[0]
        a_seat = g % 2
        moves = list(opening)
        seats = [recorded(agent_a, moves), recorded(agent_b, moves)]
        seats = seats if a_seat == 0 else seats[::-1]
        result = play_game(*seats, max_moves=200, state=start)
        outcome = "draw" if result is None else (name_a if result == a_seat else name_b)
        if lost is not None and outcome == name_b:
            lost.append(
                dict(
                    id=f"{name_b}-{seed}-{g}",
                    created_at=len(lost) + 1,
                    status="completed",
                    difficulty="hard",  # a full-strength loss: weight 1
                    model_version=name_a,
                    human_side=1 - a_seat,  # the opponent's seat, as for a player
                    winner=result,
                    moves=moves,
                    move_ms=[0] * len(moves),
                )
            )
        wins += outcome == name_a
        draws += result is None
        points += 1.0 if outcome == name_a else 0.5 if result is None else 0.0
        print(
            f"  game {g + 1:>3}/{games}: {name_a} as P{a_seat + 1} -> {outcome:<6} "
            f"| {name_a} score {points / (g + 1):.2f} | {time.time() - tic:.0f}s",
            flush=True,
        )
    score = win_rate(wins, games - wins - draws, draws)
    low, high = elo_interval(score, games)
    return dict(
        score=score,
        wins=wins,
        draws=draws,
        games=games,
        elo=elo_from_winrate(score),
        elo_low=low,
        elo_high=high,
    )


def report(title: str, result: dict) -> None:
    print(
        f"{title}: score {result['score']:.2f} ({result['wins']}W {result['draws']}D "
        f"of {result['games']}) | Elo {result['elo']:+.0f} "
        f"[{result['elo_low']:+.0f}, {result['elo_high']:+.0f}]\n",
        flush=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark against SigmaQuoridor")
    p.add_argument("checkpoint", help="our checkpoint, e.g. runs/pilot-final/iter_0560")
    p.add_argument("--sigma", default=str(SIGMA_PATH), help="their ONNX model")
    p.add_argument("--games", type=int, default=40, help="even number")
    p.add_argument("--sims", type=int, default=64, help="0 = raw policy")
    p.add_argument("--check", action="store_true", help="sanity checks vs greedy")
    p.add_argument("--save-losses", help="write our lost games here (jsonl)")
    p.add_argument("--seed", type=int, default=0, help="which random openings")
    args = p.parse_args()

    sigma_path = Path(args.sigma)
    if not sigma_path.exists():
        sigma_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {SIGMA_URL}")
        urllib.request.urlretrieve(SIGMA_URL, sigma_path)
    ours = network_evaluator(load_model(args.checkpoint))
    sigma = sigma_evaluator(sigma_path)

    results = {}
    # Both raw policies must crush the greedy bot, or the adapter is wrong
    if args.check:
        greedy = python_agent(greedy_agent)
        for name, fn in (("sigma", sigma), ("ours", ours)):
            print(f"[check] {name} raw policy vs greedy")
            results[f"{name}_vs_greedy"] = match(
                name, agent(fn, 0), "greedy", greedy, 20
            )
            report(f"[check] {name} vs greedy", results[f"{name}_vs_greedy"])
    label = f"{args.sims} sims" if args.sims else "raw policy"
    print(f"ours vs sigma ({label}, {args.games} games)")
    lost = [] if args.save_losses else None
    results["ours_vs_sigma"] = match(
        "ours",
        agent(ours, args.sims),
        "sigma",
        agent(sigma, args.sims),
        args.games,
        seed=args.seed,
        lost=lost,
    )
    report(f"ours vs sigma ({label})", results["ours_vs_sigma"])
    if lost is not None:
        Path(args.save_losses).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_losses, "w") as f:
            f.writelines(json.dumps(game) + "\n" for game in lost)
        print(f"{len(lost)} lost games saved to {args.save_losses}")

    log = Path("runs/benchmarks.jsonl")
    log.parent.mkdir(exist_ok=True)
    entry = dict(
        time=time.time(),
        checkpoint=args.checkpoint,
        sigma=str(sigma_path),
        sims=args.sims,
        seed=args.seed,
        **results,
    )
    with log.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"logged to {log}")


if __name__ == "__main__":
    main()
