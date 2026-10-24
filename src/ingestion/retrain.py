"""
Retrain on the games QuoriZero lost, once the website's weighted loss count is due
QZ_EXPORT_TOKEN=... python -m ingestion.retrain --run main --api https://<worker-url>
Every loss so far (newest first) -> deep search of every position -> self-play seeded
from the critical ones -> fine-tune -> three-part gate -> export. Recording the run
restarts the count.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from evaluation.arena import match_report
from evaluation.registry import ModelRegistry
from ingestion.games import fetch, record_retrain, replay, retrain_status
from jaxgame import from_python
from mcts.search import evaluate, net_agent
from model.checkpoint import load_model, save_checkpoint
from model.export import export
from selfplay.generate import Sample, reanalyze, selfplay
from train.tracking import MetricsLogger
from train.trainer import make_optimizer, train

# Levels whose losses count, as in worker/src/index.js (Easy loses on purpose).
LOSS_WEIGHT = {"hard": 1.0, "challenge": 1.0, "medium": 0.5}
NOT_DUE = 10  # --check exit code
HOLD_OUT_FROM = 20  # with this many losses, every 5th older one becomes a test game


def lost_games(games: list[dict], until: int) -> list[dict]:
    """The losses the count is made of: finished, a real model, a level that counts."""
    return [
        g
        for g in games
        if g["created_at"] <= until
        and g.get("status") == "completed"
        and g.get("winner") == g.get("human_side")
        and g.get("difficulty") in LOSS_WEIGHT
        and g.get("model_version") != "heuristic"
    ]


def split_games(losses: list[dict], since: int, max_games: int):
    """
    (new, older trained on, held out) from every loss so far, oldest first.
    New losses are always trained on. Once there are enough, every 5th older one is
    held out as a test, and training keeps the newest max_games.
    """
    new = [g for g in losses if g["created_at"] > since]
    older = [g for g in losses if g["created_at"] <= since]
    held = older[::5] if len(losses) >= HOLD_OUT_FROM else []
    kept = [g for g in older if g not in held]
    kept = kept[len(kept) - max(0, max_games - len(new)) :]
    return new, kept, held


def positions_of(games: list[dict]):
    """Valid games -> (every position as one batched State, its game's index)."""
    states, owner = [], []
    for i, record in enumerate(games):
        for ply, game in enumerate(replay(record) or []):
            states.append(from_python(game, ply))
            owner.append(i)
    if not states:
        return None, None
    return jax.tree.map(lambda *xs: jnp.stack(xs), *states), np.array(owner)


@nnx.jit
def _raw_value(model, states):
    return evaluate(model, states)[1]


def critical(model, deep: Sample, k: int, batch: int = 512) -> np.ndarray:
    """The k positions where deep search disagrees most with the raw network."""
    model.eval()
    n = len(deep.value)
    raw = np.concatenate(
        [
            np.asarray(
                _raw_value(model, jax.tree.map(lambda x: x[i : i + batch], deep.state))
            )
            for i in range(0, n, batch)
        ]
    )
    return np.argsort(-np.abs(deep.value - raw))[:k]


def lines_closed(candidate, best, states, owner, games, key) -> float:
    """
    Share of games whose line no longer works: somewhere on QuoriZero's turn the
    candidate picks another move than the model that lost (each game at its level).
    """
    closed = np.zeros(len(games), bool)
    human = np.array([g["human_side"] for g in games])[owner]
    sims = np.array([g.get("sims") or 64 for g in games])[owner]
    our_turn = np.asarray(states.to_move) != human
    for s in np.unique(sims[our_turn]):
        idx = np.flatnonzero(our_turn & (sims == s))
        batch = jax.tree.map(lambda x: x[idx], states)
        new = np.asarray(net_agent(candidate, int(s))(batch, key))
        old = np.asarray(net_agent(best, int(s))(batch, key))
        np.logical_or.at(closed, owner[idx], new != old)
    return float(closed.mean())


def main() -> None:
    p = argparse.ArgumentParser(description="Retrain on the games QuoriZero lost")
    p.add_argument("--run", default="main", help="training run under runs/")
    p.add_argument("--api", help="games API base URL")
    p.add_argument("--games-file", help="local test: games from a jsonl, no API")
    p.add_argument(
        "--check", action="store_true", help=f"exit 0 if due, {NOT_DUE} if not"
    )
    p.add_argument("--force", action="store_true", help="run even if not due (tests)")
    p.add_argument("--max-games", type=int, default=100, help="newest losses used")
    p.add_argument(
        "--deep-sims", type=int, default=800, help="search of lost positions"
    )
    p.add_argument("--critical", type=int, default=64, help="seed positions")
    p.add_argument("--human-share", type=float, default=0.25, help="lost positions")
    p.add_argument("--seeded-share", type=float, default=0.25, help="seeded self-play")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument(
        "--lr", type=float, default=1e-4, help="no higher than training ended"
    )
    p.add_argument("--sims", type=int, default=64, help="self-play and gate")
    p.add_argument("--lanes", type=int, default=1024)
    p.add_argument("--games", type=int, default=400, help="gate games")
    p.add_argument(
        "--threshold", type=float, default=0.5, help="gate 1: not worse overall"
    )
    p.add_argument(
        "--fix-threshold", type=float, default=0.55, help="gate 2: better at seeds"
    )
    p.add_argument(
        "--lines", type=float, default=1.0, help="gate 3: share of lines closed"
    )
    p.add_argument("--export", default="frontend/model")
    p.add_argument("--wandb", action="store_true")
    args = p.parse_args()

    if args.games_file:  # a local test: every saved loss is new, the count untouched
        token = None
        saved = Path(args.games_file).read_text().splitlines()
        saved = [json.loads(line) for line in saved if line.strip()]
        until = max((g["created_at"] for g in saved), default=0)
        losses = lost_games(saved, until)
        weight = sum(LOSS_WEIGHT[g["difficulty"]] for g in losses)
        status = dict(since=0, until=until, losses=weight, due=False)
        print(f"{len(losses)} lost games of {len(saved)} in {args.games_file}")
    else:
        if not args.api:
            p.error("needs --api or --games-file")
        token = os.environ["QZ_EXPORT_TOKEN"]
        status = retrain_status(args.api, token)
        print(
            f"{status['losses']:g} weighted losses in {status['games']} games "
            f"(retrain at {status['retrain_at']:g})"
        )
        if args.check:
            sys.exit(0 if status["due"] else NOT_DUE)
        if not (status["due"] or args.force):
            print("not due")
            return
        losses = lost_games(fetch(args.api, token, 0), status["until"])  # every one

    def record(promoted=False, gate_score=None, model_version=None):
        if status["due"]:  # a forced test run leaves the count alone
            record_retrain(
                args.api,
                token,
                dict(
                    games_until=status["until"],
                    losses=status["losses"],
                    promoted=promoted,
                    gate_score=gate_score,
                    model_version=model_version,
                ),
            )

    new, kept, held = split_games(losses, status["since"], args.max_games)
    used = new + kept + held
    stamp = int(time.time())
    data_dir = Path("data/human")
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(data_dir / f"lost_{stamp}.jsonl", "w") as f:
        f.writelines(json.dumps(g) + "\n" for g in used)
    states, owner = positions_of(used)
    if states is None or not (owner < len(new)).any():
        print(f"{len(new)} new lost games, none valid: nothing to learn")
        record()
        return
    is_new = owner < len(new)
    is_trained = owner < len(new) + len(kept)
    print(f"{len(new)} new, {len(kept)} older and {len(held)} held-out lost games")

    run_dir = Path("runs") / args.run
    registry = ModelRegistry(run_dir)
    best = load_model(registry.best_path() or run_dir / "latest")
    key, rng = jax.random.key(stamp), np.random.default_rng(stamp)
    k_deep, k_seed, k_self, k_gate, k_fix, k_held, k_lines = jax.random.split(key, 7)

    # MuZero Reanalyze with a deep search: targets for every lost position come
    # from the search, never from the moves played.
    deep = reanalyze(best, states, k_deep, args.deep_sims)

    def part(mask):
        return jax.tree.map(lambda x: x[mask], deep)

    def seeds_of(sample, k):
        return jax.tree.map(lambda x: x[critical(best, sample, k)], sample.state)

    # Critical positions seed self-play, mostly from the new losses.
    new_seeds = seeds_of(part(is_new), args.critical)
    seeds = new_seeds
    older = is_trained & ~is_new
    if older.any():
        old_seeds = seeds_of(part(older), args.critical // 2)
        seeds = jax.tree.map(lambda a, b: np.concatenate([a, b]), seeds, old_seeds)
    lost = part(is_trained)
    seeded, _ = selfplay(best, k_seed, args.lanes, 256, args.sims, seeds=seeds)
    own, _ = selfplay(best, k_self, args.lanes, 256, args.sims)  # keeps what it knew

    sources = [(lost, args.human_share), (seeded, args.seeded_share)]
    sources = [(s, round(args.batch * share)) for s, share in sources if len(s.value)]
    sources.append((own, args.batch - sum(n for _, n in sources)))

    def sample_batch():
        parts = []
        for source, n in sources:
            idx = rng.integers(0, len(source.value), n)
            parts.append(jax.tree.map(lambda x: x[idx], source))
        return jax.tree.map(lambda *xs: np.concatenate(xs), *parts)

    candidate = nnx.clone(best)
    optimizer = make_optimizer(candidate, args.lr)
    metrics = train(candidate, optimizer, sample_batch, args.steps, rng)

    # Promotion gate, against the model that lost:
    # 1. not worse overall: paired games from random openings
    # 2. better where it lost: paired games from the new critical positions
    # 3. the losing lines are closed: it no longer repeats the moves that lost
    me, them = net_agent(candidate, args.sims), net_agent(best, args.sims)
    gate = match_report(me, them, k_gate, args.games)
    fix = match_report(me, them, k_fix, starts=new_seeds)
    in_new = np.flatnonzero(is_new)
    closed = lines_closed(
        candidate,
        best,
        jax.tree.map(lambda x: x[in_new], deep.state),
        owner[in_new],
        new,
        k_lines,
    )
    held_out = None
    if held:  # an honest test on positions it never trained on (reported, not gated)
        held_out = match_report(
            me, them, k_held, starts=seeds_of(part(~is_trained), args.critical)
        )
    promoted = (
        gate["score"] >= args.threshold
        and fix["score"] >= args.fix_threshold
        and closed >= args.lines
    )
    version = None
    if promoted:
        path = run_dir / f"human_{stamp}"
        save_checkpoint(path, candidate)
        registry.promote(path)
        version = f"{args.run}/{path.name}"
        export(candidate, args.export, version)
    record(promoted, gate["score"], version)

    logger = MetricsLogger(
        run_dir / "logs", use_wandb=args.wandb, name=f"retrain-{stamp}"
    )
    logger.log(
        "retrain",
        stamp,
        {
            "games_new": len(new),
            "games_older": len(kept),
            "games_held": len(held),
            "losses": status["losses"],
            "positions": int(is_trained.sum()),
            "seeds": len(seeds.to_move),
            "seeded_samples": len(seeded.value),
            **metrics,
            **{f"gate/{k}": v for k, v in gate.items()},
            **{f"fix/{k}": v for k, v in fix.items()},
            "lines_closed": closed,
            **({f"held/{k}": v for k, v in held_out.items()} if held_out else {}),
            "promoted": int(promoted),
        },
    )
    logger.close()
    held_text = f", held-out {held_out['score']:.2f}" if held_out else ""
    print(
        f"gate: overall {gate['score']:.2f} (need {args.threshold:.2f}), "
        f"critical positions {fix['score']:.2f} (need {args.fix_threshold:.2f}), "
        f"lines closed {closed:.0%} (need {args.lines:.0%}){held_text}"
    )
    print(f"PROMOTED and exported {version}" if promoted else "kept the current best")


if __name__ == "__main__":
    main()
