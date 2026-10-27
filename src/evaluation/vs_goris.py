"""
Benchmark against gorisanson's Quoridor AI (MIT): github.com/gorisanson/quoridor-ai
python -m evaluation.vs_goris runs/pilot-final/iter_0560 --games 100 --sims 64
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

from engine.actions import decode_action, encode_move, encode_wall, legal_actions, step
from engine.state import GameState, initial_state

COMMIT = "e15d6e7825e6f7c2f3a41de0ecef1af7aa290f1e"
RAW = f"https://raw.githubusercontent.com/gorisanson/quoridor-ai/{COMMIT}/src/js/"
GORIS_DIR = Path("data/goris")
STRONG = dict(rollouts=60_000, uct=0.2)  # their website's "Strong" level

# Their game.js and ai.js run unchanged in Node, loaded into one context; each stdin
# line is a move history, answered with their move.
BRIDGE = r"""
const fs = require("fs"), vm = require("vm"), readline = require("readline");
const dir = process.argv[1];
console.log = () => {};  // their debug prints must not reach stdout
vm.runInThisContext(fs.readFileSync(dir + "/game.js", "utf8") + "\n" +
                    fs.readFileSync(dir + "/ai.js", "utf8") +
                    "\nglobalThis.QGame = Game; globalThis.QAI = AI;");
const out = (x) => process.stdout.write(JSON.stringify(x) + "\n");
readline.createInterface({ input: process.stdin }).on("line", (line) => {
  const req = JSON.parse(line);
  const game = new QGame(false);  // their pawn 0 starts at row 0, as ours
  for (const move of req.moves) game.doMove(move, false);
  out({ move: new QAI(req.rollouts, req.uct, false, false).chooseNextMove(game) });
});
"""


def fetch_goris(directory: Path = GORIS_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("game.js", "ai.js"):
        if not (directory / name).exists():
            urllib.request.urlretrieve(RAW + name, directory / name)
    return directory


# Their board matches ours with Game(false): wall (r, c) covers the same cells, and a
# move is [pawn target, horizontal wall, vertical wall]
def to_goris(game: GameState, action: int) -> list:
    """Our action -> their [pawn, horizontal wall, vertical wall] triple."""
    kind, payload = decode_action(game, action)
    if kind == "move":
        return [list(payload), None, None]
    orient, r, c = payload
    return [None, [r, c], None] if orient == "H" else [None, None, [r, c]]


def from_goris(game: GameState, move: list) -> int:
    """Their triple -> our action, which must be legal here."""
    pawn, h, v = move
    if pawn is not None:
        action = encode_move(game.pawns[game.to_move], tuple(pawn))
    else:
        action = encode_wall(("H", *h) if h is not None else ("V", *v))
    if action not in legal_actions(game):
        raise ValueError(f"goris played {move}, illegal under our rules")
    return action


class GorisBot:
    """One Node process; agent(history) -> our action for the position after history."""

    def __init__(
        self, rollouts=STRONG["rollouts"], uct=STRONG["uct"], directory=GORIS_DIR
    ):
        self.rollouts, self.uct = rollouts, uct
        self.proc = subprocess.Popen(
            ["node", "-e", BRIDGE, str(fetch_goris(directory).resolve())],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )

    def move(self, history: list[int]) -> int:
        game, moves = initial_state(), []
        for action in history:
            moves.append(to_goris(game, action))
            game = step(game, action)[0]
        request = dict(moves=moves, rollouts=self.rollouts, uct=self.uct)
        self.proc.stdin.write(json.dumps(request) + "\n")
        self.proc.stdin.flush()
        reply = self.proc.stdout.readline()
        if not reply:
            raise RuntimeError("the goris process stopped")
        return from_goris(game, json.loads(reply)["move"])

    def close(self) -> None:
        self.proc.kill()


def main() -> None:
    from evaluation.benchmark import init_worker, play_game, summarize

    p = argparse.ArgumentParser(description="Benchmark against gorisanson's AI")
    p.add_argument("checkpoint", help="ours, e.g. runs/pilot-final/iter_0560")
    p.add_argument("--games", type=int, default=100, help="even number")
    p.add_argument("--sims", type=int, default=64, help="ours; theirs is --rollouts")
    p.add_argument(
        "--rollouts", type=int, default=STRONG["rollouts"], help="60k: Strong"
    )
    p.add_argument("--seed", type=int, default=0, help="which random openings")
    p.add_argument(
        "--jobs", type=int, default=4, help="games in parallel, ~1.4 GB each"
    )
    args = p.parse_args()
    os.environ["JAX_PLATFORMS"] = "cpu"  # inherited by every worker: none takes the GPU

    label = f"{args.sims} sims vs {args.rollouts // 1000}k rollouts"
    print(f"ours vs goris ({label}, {args.games} games, {args.jobs} in parallel)")
    played, points, tic = [], 0.0, time.time()
    init = (args.checkpoint, "goris", args.sims, args.rollouts)
    with ProcessPoolExecutor(
        args.jobs, get_context("spawn"), init_worker, init
    ) as pool:
        # One task per game, not per pair: each result prints as soon as it ends.
        futures = [
            pool.submit(play_game, args.seed, k, seat)
            for k in range(args.games // 2)
            for seat in (0, 1)
        ]
        for future in as_completed(futures):
            g = future.result()
            played.append(g)
            points += {"win": 1.0, "draw": 0.5}.get(g["outcome"], 0.0)
            winner = {"win": "ours", "loss": "goris"}.get(g["outcome"], "draw")
            print(
                f"  game {len(played):>3}/{args.games}: "
                f"ours as P{g['our_seat'] + 1} -> {winner:<6}"
                f"| ours score {points / len(played):.2f} "
                f"| {time.time() - tic:.0f}s",
                flush=True,
            )
    s = summarize(played)
    result = {k: s[k] for k in ("score", "wins", "draws", "games", "elo")}
    result.update(elo_low=s["elo_low"], elo_high=s["elo_high"])
    print(
        f"ours vs goris ({label}): score {s['score']:.2f} ({s['wins']}W {s['draws']}D "
        f"of {s['games']}) | Elo {s['elo']:+.0f} "
        f"[{s['elo_low']:+.0f}, {s['elo_high']:+.0f}]"
    )
    log = Path("runs/benchmarks.jsonl")
    log.parent.mkdir(exist_ok=True)
    entry = dict(time=time.time(), checkpoint=args.checkpoint, goris=COMMIT)
    entry.update(sims=args.sims, rollouts=args.rollouts, seed=args.seed)
    entry["ours_vs_goris"] = result
    with log.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"logged to {log}")


if __name__ == "__main__":
    main()
