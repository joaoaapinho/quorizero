"""Gumbel AlphaZero loop: self-play -> replay buffer -> train -> evaluate, repeat.

    python -m train.run --name overnight --wandb   # see scripts/train_cloud.sh
    python -m train.run --name smoke --lanes 32 --steps 24 --sims 8 --iterations 2

Everything lives in runs/<name>/: config, logs, `latest` weights and `optimizer`
state (every iteration) and `iter_NNNN` snapshots (every --eval-every).
Re-running the same command resumes exactly from there. With --wandb the whole
directory is also backed up to W&B every --backup-every iterations (artifact
run-<name>; each version stores ~16 MB of new weights, so every iteration would
fill the free 5 GB in ~300 iterations). Restore with:
    wandb artifact get <entity>/quorizero/run-<name>:latest --root runs/<name>
Why each choice: docs/training.md.
"""

# ruff: noqa: E402  (GPU env must be set before JAX is imported below)
from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np
from flax import nnx

from evaluation.arena import match_report
from mcts.search import greedy_agent, net_agent
from model.checkpoint import load_checkpoint, save_checkpoint
from model.network import create_model
from selfplay.generate import selfplay
from selfplay.replay_buffer import ReplayBuffer
from train.tracking import MetricsLogger
from train.trainer import make_optimizer, train

MODEL_KEYS = ("blocks", "channels")  # fixed for a run; everything else may change


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gumbel AlphaZero self-play training")
    p.add_argument("--name", default="main", help="run directory under runs/")
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--lanes", type=int, default=1024, help="games played at once")
    p.add_argument("--steps", type=int, default=256, help="moves per lane per iter")
    p.add_argument("--sims", type=int, default=32, help="Gumbel simulations / move")
    p.add_argument("--blocks", type=int, default=8)
    p.add_argument("--channels", type=int, default=96)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--buffer", type=int, default=1_500_000, help="samples kept")
    p.add_argument("--reuse", type=float, default=2.0, help="times each sample is seen")
    p.add_argument("--warm-mix", type=float, default=0.5, help="heuristic value share")
    p.add_argument("--warm-iters", type=int, default=20, help="iters to fade it out")
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--backup-every", type=int, default=10, help="W&B backup interval")
    p.add_argument("--eval-games", type=int, default=256)
    p.add_argument("--eval-sims", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path("runs") / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    config_file, progress_file = run_dir / "config.json", run_dir / "progress.json"
    if config_file.exists():  # resuming: the architecture must match the weights
        saved = json.loads(config_file.read_text())
        for k in MODEL_KEYS:
            setattr(args, k, saved[k])
    config_file.write_text(json.dumps(vars(args), indent=2))

    model = create_model(args.seed, args.blocks, args.channels)
    optimizer = make_optimizer(model, args.lr)
    progress = {"iteration": 0, "elo": 0.0}
    if progress_file.exists():
        model = load_checkpoint(run_dir / "latest", model)
        optimizer = make_optimizer(model, args.lr)  # new lr applies on resume
        optimizer = load_checkpoint(run_dir / "optimizer", optimizer)
        progress = json.loads(progress_file.read_text())
        print(f"resuming {run_dir} at iteration {progress['iteration']}")
    print("JAX devices:", jax.devices())

    buffer = ReplayBuffer(args.buffer)  # not saved: refilled by the next self-play
    logger = MetricsLogger(
        run_dir / "logs",
        use_wandb=args.wandb,
        config=vars(args),
        name=args.name,
        run_id=f"quorizero-{args.name}",
    )
    start = progress["iteration"]
    rng = np.random.default_rng(args.seed + start)
    key = jax.random.key(args.seed + start)
    previous = nnx.clone(model)  # last evaluated weights: the Elo chain's anchor

    for it in range(start + 1, args.iterations + 1):
        tic = time.time()
        key, play_key, eval_key = jax.random.split(key, 3)
        samples, stats = selfplay(model, play_key, args.lanes, args.steps, args.sims)
        buffer.add(samples)
        new = len(samples.value)
        stats.update(samples=new, buffer=len(buffer), seconds=time.time() - tic)
        logger.log("selfplay", it, stats)

        if new == 0:  # no game finished: nothing to learn from yet
            print(f"iter {it}: no game finished in {args.steps} moves, skipped")
            continue
        mix = args.warm_mix * max(0.0, 1 - (it - 1) / args.warm_iters)
        steps = max(1, round(args.reuse * new / args.batch))
        metrics = train(
            model, optimizer, lambda: buffer.sample(rng, args.batch), steps, rng, mix
        )
        logger.log("train", it, {**metrics, "value_mix": mix, "steps": steps})

        if it % args.eval_every == 0:
            current = net_agent(model, args.eval_sims)
            key_a, key_b = jax.random.split(eval_key)
            greedy = match_report(current, greedy_agent, key_a, args.eval_games)
            prev = net_agent(previous, args.eval_sims)
            chain = match_report(current, prev, key_b, args.eval_games)
            progress["elo"] += chain["elo"]
            logger.log(
                "eval",
                it,
                {
                    **{f"vs_greedy/{k}": v for k, v in greedy.items()},
                    **{f"vs_previous/{k}": v for k, v in chain.items()},
                    "elo": progress["elo"],  # cumulative chain of head-to-heads
                },
            )
            save_checkpoint(run_dir / f"iter_{it:04d}", model)
            previous = nnx.clone(model)
            print(
                f"iter {it}: vs greedy {greedy['score']:.2f} | "
                f"vs previous {chain['score']:.2f} | elo {progress['elo']:+.0f}"
            )

        save_checkpoint(run_dir / "latest", model)
        save_checkpoint(run_dir / "optimizer", optimizer)
        progress["iteration"] = it
        progress_file.write_text(json.dumps(progress))
        if it % args.backup_every == 0:
            logger.backup(f"run-{args.name}", run_dir, ["latest", f"iter_{it:04d}"])
        print(
            f"iter {it}: {stats['games']:.0f} games, {new} samples, "
            f"len {stats['avg_length']:.0f}, draws {stats['draws']:.0f}, "
            f"loss {metrics['loss']:.3f}, {time.time() - tic:.0f}s"
        )

    logger.close()


if __name__ == "__main__":
    main()
