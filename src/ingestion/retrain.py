"""
Retrain on website games: reanalyse, fine-tune, gate, export
QZ_EXPORT_TOKEN=... python -m ingestion.retrain --run main --api https://<worker-url>
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from evaluation.arena import match_report
from evaluation.registry import ModelRegistry
from ingestion.games import fetch, replay
from jaxgame import from_python
from mcts.search import net_agent
from model.checkpoint import load_model, save_checkpoint
from model.export import export
from selfplay.generate import reanalyze, selfplay
from selfplay.replay_buffer import ReplayBuffer
from train.tracking import MetricsLogger
from train.trainer import make_optimizer, train


def positions_of(games: list[dict]):
    """Valid games -> one batched jaxgame.State holding every position."""
    states = [
        from_python(game, ply)
        for record in games
        for ply, game in enumerate(replay(record) or [])
    ]
    return jax.tree.map(lambda *xs: jnp.stack(xs), *states) if states else None


def main() -> None:
    p = argparse.ArgumentParser(description="Retrain on website games")
    p.add_argument("--run", default="main", help="training run under runs/")
    p.add_argument("--api", required=True, help="games API base URL")
    p.add_argument("--human-share", type=float, default=0.25)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--sims", type=int, default=64)
    p.add_argument("--lanes", type=int, default=1024)
    p.add_argument("--games", type=int, default=400, help="gate games")
    p.add_argument("--threshold", type=float, default=0.55)
    p.add_argument("--export", default="frontend/model")
    p.add_argument("--wandb", action="store_true")
    args = p.parse_args()

    run_dir = Path("runs") / args.run
    data_dir = Path("data/human")
    data_dir.mkdir(parents=True, exist_ok=True)
    cursor = data_dir / "last_fetch.json"
    since = json.loads(cursor.read_text())["since"] if cursor.exists() else 0

    games = fetch(args.api, os.environ["QZ_EXPORT_TOKEN"], since)
    if not games:
        print("no new games")
        return
    stamp = int(time.time())
    with open(data_dir / f"games_{stamp}.jsonl", "w") as f:
        f.writelines(json.dumps(g) + "\n" for g in games)
    states = positions_of(games)
    cursor.write_text(json.dumps({"since": games[-1]["created_at"]}))
    if states is None:
        print(f"{len(games)} new games, none valid")
        return

    registry = ModelRegistry(run_dir)
    best_path = registry.best_path() or run_dir / "latest"
    best = load_model(best_path)
    key, rng = jax.random.key(stamp), np.random.default_rng(stamp)
    k_human, k_self, k_gate = jax.random.split(key, 3)
    # MuZero reanalyze: human positions, targets from fresh search (never human moves).
    human = reanalyze(best, states, k_human, args.sims)
    own = ReplayBuffer(10**7)
    own.add(selfplay(best, k_self, args.lanes, 256, args.sims)[0])

    n_human = round(args.batch * args.human_share)

    def sample_batch():
        idx = rng.integers(0, len(human.value), n_human)
        mixed = (
            jax.tree.map(lambda x: x[idx], human),
            own.sample(rng, args.batch - n_human),
        )
        return jax.tree.map(lambda a, b: np.concatenate([a, b]), *mixed)

    candidate = nnx.clone(best)
    optimizer = make_optimizer(candidate, args.lr)
    metrics = train(candidate, optimizer, sample_batch, args.steps, rng)
    # Promotion gate: candidate vs best over paired games.
    gate = match_report(
        net_agent(candidate, args.sims), net_agent(best, args.sims), k_gate, args.games
    )
    promoted = gate["score"] >= args.threshold
    if promoted:
        path = run_dir / f"human_{stamp}"
        save_checkpoint(path, candidate)
        registry.promote(path)
        export(candidate, args.export, f"{args.run}/{path.name}")

    logger = MetricsLogger(
        run_dir / "logs", use_wandb=args.wandb, name=f"retrain-{stamp}"
    )
    logger.log(
        "retrain",
        stamp,
        {
            "games": len(games),
            "positions": len(human.value),
            **metrics,
            **{f"gate/{k}": v for k, v in gate.items()},
            "promoted": int(promoted),
        },
    )
    logger.close()
    print(f"{len(games)} games, {len(human.value)} positions, gate {gate['score']:.2f}")
    print("PROMOTED and exported" if promoted else "kept the current best")


if __name__ == "__main__":
    main()
