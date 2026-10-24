"""
Retrain on the games QuoriZero lost, once the website's weighted loss count is due
QZ_EXPORT_TOKEN=... python -m ingestion.retrain --run main --api https://<worker-url>
Lost games -> deep search of every position -> self-play seeded from the critical ones
-> fine-tune -> gate -> export. Recording the run restarts the count.
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


def positions_of(games: list[dict]):
    """Valid games -> one batched jaxgame.State holding every position."""
    states = [
        from_python(game, ply)
        for record in games
        for ply, game in enumerate(replay(record) or [])
    ]
    return jax.tree.map(lambda *xs: jnp.stack(xs), *states) if states else None


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


def main() -> None:
    p = argparse.ArgumentParser(description="Retrain on the games QuoriZero lost")
    p.add_argument("--run", default="main", help="training run under runs/")
    p.add_argument("--api", required=True, help="games API base URL")
    p.add_argument(
        "--check", action="store_true", help=f"exit 0 if due, {NOT_DUE} if not"
    )
    p.add_argument("--force", action="store_true", help="run even if not due (tests)")
    p.add_argument(
        "--deep-sims", type=int, default=800, help="search of lost positions"
    )
    p.add_argument("--critical", type=int, default=64, help="seed positions")
    p.add_argument("--human-share", type=float, default=0.25, help="lost positions")
    p.add_argument("--seeded-share", type=float, default=0.25, help="seeded self-play")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--sims", type=int, default=64, help="self-play and gate")
    p.add_argument("--lanes", type=int, default=1024)
    p.add_argument("--games", type=int, default=400, help="gate games")
    p.add_argument("--threshold", type=float, default=0.55)
    p.add_argument("--export", default="frontend/model")
    p.add_argument("--wandb", action="store_true")
    args = p.parse_args()

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

    games = lost_games(fetch(args.api, token, status["since"]), status["until"])
    stamp = int(time.time())
    data_dir = Path("data/human")
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(data_dir / f"lost_{stamp}.jsonl", "w") as f:
        f.writelines(json.dumps(g) + "\n" for g in games)
    states = positions_of(games)
    if states is None:
        print(f"{len(games)} lost games, none valid: nothing to learn")
        record()
        return

    run_dir = Path("runs") / args.run
    registry = ModelRegistry(run_dir)
    best = load_model(registry.best_path() or run_dir / "latest")
    key, rng = jax.random.key(stamp), np.random.default_rng(stamp)
    k_deep, k_seed, k_self, k_gate = jax.random.split(key, 4)

    # MuZero Reanalyze with a deep search: targets for every lost position come
    # from the search, never from the moves played.
    deep = reanalyze(best, states, k_deep, args.deep_sims)
    # Critical positions seed self-play, so it explores the lines it misjudged.
    seeds = jax.tree.map(lambda x: x[critical(best, deep, args.critical)], deep.state)
    seeded, _ = selfplay(best, k_seed, args.lanes, 256, args.sims, seeds=seeds)
    own, _ = selfplay(best, k_self, args.lanes, 256, args.sims)  # keeps what it knew

    sources = [(deep, args.human_share), (seeded, args.seeded_share)]
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
    # Promotion gate: candidate vs best over paired games.
    gate = match_report(
        net_agent(candidate, args.sims), net_agent(best, args.sims), k_gate, args.games
    )
    promoted = gate["score"] >= args.threshold
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
            "games": len(games),
            "losses": status["losses"],
            "positions": len(deep.value),
            "seeds": len(seeds.to_move),
            "seeded_samples": len(seeded.value),
            **metrics,
            **{f"gate/{k}": v for k, v in gate.items()},
            "promoted": int(promoted),
        },
    )
    logger.close()
    print(
        f"{len(games)} lost games, {len(deep.value)} positions, "
        f"{len(seeds.to_move)} seeds, gate {gate['score']:.2f}"
    )
    print(f"PROMOTED and exported {version}" if promoted else "kept the current best")


if __name__ == "__main__":
    main()
