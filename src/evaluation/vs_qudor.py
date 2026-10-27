"""
Benchmark against Qudor AI (MIT): github.com/ktoto1300/Qudor-AI
python -m evaluation.vs_qudor runs/pilot-final/iter_0560 --check
python -m evaluation.vs_qudor runs/pilot-final/iter_0560 --games 100 --sims 64
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from baseline.match import play_game
from engine.actions import ACTION_SIZE, decode_action, step
from engine.pathcheck import distance_map
from engine.state import GOAL_ROWS, GameState, initial_state
from evaluation.arena import opening_moves, win_rate
from evaluation.elo import elo_from_winrate, elo_interval
from mcts.puct import agent, network_evaluator, softmax_over
from mcts.search import greedy_agent, python_agent
from model.checkpoint import load_model

QUDOR_URL = (
    "https://media.githubusercontent.com/media/ktoto1300/Qudor-AI/master/"
    "checkpoints/gen85_best.pt"
)
QUDOR_PATH = Path("data/qudor/gen85_best.pt")
QUDOR_SHA256 = "2126e3c2937916972d633efee237f0176cc4e3b8bb23fb86de86189a8712e170"
DNORM, MAX_PLIES = 24.0, 220  # their distance clip and ply scale

# Their network (model.py), rebuilt so the weights load without their code


class SE(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        hidden = max(8, c // 4)
        self.fc = nn.Sequential(nn.Linear(c, hidden), nn.SiLU(), nn.Linear(hidden, c))

    def forward(self, x):
        return x * torch.sigmoid(self.fc(x.mean((2, 3))))[:, :, None, None]


class ResBlock(nn.Module):
    def __init__(self, c: int, se: bool):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
        )
        self.se = SE(c) if se else None

    def forward(self, x):
        y = self.net(x)
        return F.silu(x + (self.se(y) if self.se is not None else y))


class QudorNet(nn.Module):
    def __init__(self, channels: int, blocks: int, planes: int, se: bool):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(planes, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
        )
        self.body = nn.Sequential(*[ResBlock(channels, se) for _ in range(blocks)])
        self.policy = nn.Sequential(
            nn.Conv2d(channels, 8, 1), nn.SiLU(), nn.Flatten(), nn.Linear(8 * 81, 209)
        )
        self.value = nn.Sequential(
            nn.Conv2d(channels, 4, 1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(4 * 81, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        z = self.body(self.stem(x))
        return self.policy(z), self.value(z).squeeze(1)


def load_qudor(path: Path) -> QudorNet:
    # weights_only: plain tensors, never code from the file
    sd = torch.load(path, map_location="cpu", weights_only=True)["model"]
    channels, planes = sd["stem.0.weight"].shape[:2]
    blocks = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("body."))
    se = any(k.startswith("body.0.se.") for k in sd)
    net = QudorNet(int(channels), blocks, int(planes), se)
    net.load_state_dict(sd)
    return net.eval()


# Their encoding (core/encoding.py, v3), from our GameState: 16 planes and 209
# actions (81 pawn cells, 64 H and 64 V walls), with the mover's goal at row 0.
# For our player 0 a cell (r, c) is (8-r, c) and a wall slot (7-r, c).


def _dist_plane(walls, goal_row: int) -> np.ndarray:
    d = distance_map(walls, goal_row).astype(np.float32)
    d[d < 0] = DNORM  # unreachable pinned at 1.0
    return np.minimum(d, DNORM) / DNORM


def qudor_planes(game: GameState, ply: int) -> np.ndarray:
    """(16, 9, 9) as their encode_v3: the mover's goal at row 0."""
    me, opp = game.to_move, 1 - game.to_move
    flip = me == 0
    x = np.zeros((16, 9, 9), np.float32)
    for plane, (r, c) in ((0, game.pawns[me]), (1, game.pawns[opp])):
        x[plane, 8 - r if flip else r, c] = 1
    for orient, r, c in game.walls:  # each wall painted as a 2x2 block on its slot
        rs = 7 - r if flip else r
        x[2 if orient == "H" else 3, rs : rs + 2, c : c + 2] = 1
        x[4, rs : rs + 2, c : c + 2] = 1
    mine, theirs = (_dist_plane(game.walls, GOAL_ROWS[p]) for p in (me, opp))
    my_d, their_d = mine[game.pawns[me]], theirs[game.pawns[opp]]
    x[5], x[6] = (mine[::-1], theirs[::-1]) if flip else (mine, theirs)
    x[7], x[8] = game.walls_left[me] / 10, game.walls_left[opp] / 10
    x[9], x[10] = my_d, their_d
    x[11] = np.clip(0.5 + (their_d - my_d) * 2.0, 0.0, 1.0)
    x[12] = min(1.0, ply / MAX_PLIES)
    x[13] = 1.0
    x[14, 0] = 1
    x[15] = float(game.walls_left[me] > 0)
    return x


def qudor_index(game: GameState, action: int) -> int:
    """Our action -> their 209 index, in the mover's frame."""
    flip = game.to_move == 0
    kind, payload = decode_action(game, action)
    if kind == "move":
        r, c = payload
        return (8 - r if flip else r) * 9 + c
    orient, r, c = payload
    return (81 if orient == "H" else 145) + (7 - r if flip else r) * 8 + c


def qudor_evaluator(path: Path, clock: list[int]):
    """clock[0]: the game's ply at the root (their plane 12 scales it by 220)."""
    net = load_qudor(path)

    def evaluate_qudor(game: GameState, legal: list[int]):
        planes = torch.from_numpy(qudor_planes(game, clock[0]))[None]
        with torch.no_grad():
            logits, value = net(planes)
        ours = np.zeros(ACTION_SIZE, np.float64)
        ours[legal] = logits[0].numpy()[[qudor_index(game, a) for a in legal]]
        return softmax_over(ours, legal), float(value[0])

    return evaluate_qudor


# Match


def match(name_a, agent_a, name_b, agent_b, games: int, seed=0, clock=None) -> dict:
    """Paired openings, seats swapped, prints every game as it finishes."""
    rng, points, wins, draws = random.Random(seed), 0.0, 0, 0
    tic = time.time()
    for g in range(games):
        if g % 2 == 0:
            opening = opening_moves(rng)
            start = initial_state()
            for action in opening:
                start = step(start, action)[0]
        a_seat, moves = g % 2, list(opening)

        def seat(chosen):
            def act(game: GameState) -> int:
                if clock is not None:
                    clock[0] = len(moves)
                moves.append(chosen(game))
                return moves[-1]

            return act

        seats = [seat(agent_a), seat(agent_b)]
        result = play_game(*(seats if a_seat == 0 else seats[::-1]), 200, state=start)
        outcome = "draw" if result is None else (name_a if result == a_seat else name_b)
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


def report(title: str, r: dict) -> None:
    print(
        f"{title}: score {r['score']:.2f} ({r['wins']}W {r['draws']}D of {r['games']}) "
        f"| Elo {r['elo']:+.0f} [{r['elo_low']:+.0f}, {r['elo_high']:+.0f}]\n",
        flush=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark against Qudor AI")
    p.add_argument("checkpoint", help="our checkpoint, e.g. runs/pilot-final/iter_0560")
    p.add_argument("--qudor", default=str(QUDOR_PATH), help="their .pt checkpoint")
    p.add_argument("--games", type=int, default=40, help="even number")
    p.add_argument("--sims", type=int, default=64, help="0 = raw policy")
    p.add_argument("--seed", type=int, default=0, help="which random openings")
    p.add_argument("--check", action="store_true", help="sanity checks vs greedy")
    args = p.parse_args()

    qudor_path = Path(args.qudor)
    if not qudor_path.exists():
        qudor_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {QUDOR_URL}")
        urllib.request.urlretrieve(QUDOR_URL, qudor_path)
    if qudor_path == QUDOR_PATH:  # the published file: refuse anything else
        import hashlib

        digest = hashlib.sha256(qudor_path.read_bytes()).hexdigest()
        assert digest == QUDOR_SHA256, f"unexpected checkpoint {digest}"
    clock = [0]
    ours = network_evaluator(load_model(args.checkpoint))
    qudor = qudor_evaluator(qudor_path, clock)

    results = {}
    # Both raw policies must crush the greedy bot, or the adapter is wrong
    if args.check:
        greedy = python_agent(greedy_agent)
        for name, fn in (("qudor", qudor), ("ours", ours)):
            print(f"[check] {name} raw policy vs greedy")
            results[f"{name}_vs_greedy"] = match(
                name, agent(fn, 0), "greedy", greedy, 20, args.seed, clock
            )
            report(f"[check] {name} vs greedy", results[f"{name}_vs_greedy"])
    label = f"{args.sims} sims" if args.sims else "raw policy"
    print(f"ours vs qudor ({label}, {args.games} games)")
    results["ours_vs_qudor"] = match(
        "ours",
        agent(ours, args.sims),
        "qudor",
        agent(qudor, args.sims),
        args.games,
        args.seed,
        clock,
    )
    report(f"ours vs qudor ({label})", results["ours_vs_qudor"])

    log = Path("runs/benchmarks.jsonl")
    log.parent.mkdir(exist_ok=True)
    entry = dict(time=time.time(), checkpoint=args.checkpoint, qudor=str(qudor_path))
    entry.update(sims=args.sims, seed=args.seed, **results)
    with log.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"logged to {log}")


if __name__ == "__main__":
    main()
