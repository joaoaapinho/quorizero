"""Training lowers the loss on a fixed batch; the full loop runs and resumes."""

import json
import sys

import jax
import numpy as np
from flax import nnx

from model.checkpoint import load_checkpoint, save_checkpoint
from model.network import create_model
from selfplay.generate import selfplay
from train import run
from train.trainer import make_optimizer, train


def test_loss_goes_down_on_a_fixed_batch():
    model = create_model(num_blocks=2, channels=16)
    samples, _ = selfplay(model, jax.random.key(0), lanes=8, steps=96, sims=4)
    rng = np.random.default_rng(0)
    batch = jax.tree.map(lambda x: x[:64], samples)
    optimizer = make_optimizer(model, 1e-3)
    first = train(model, optimizer, lambda: batch, 1, rng)["loss"]
    last = train(model, optimizer, lambda: batch, 60, rng)["loss"]
    assert last < first


def test_optimizer_state_survives_a_restart(tmp_path):
    model = create_model(num_blocks=2, channels=16)
    samples, _ = selfplay(model, jax.random.key(0), lanes=8, steps=96, sims=4)
    batch = jax.tree.map(lambda x: x[:32], samples)
    optimizer = make_optimizer(model, 1e-3)
    train(model, optimizer, lambda: batch, 3, np.random.default_rng(0))
    save_checkpoint(tmp_path / "optimizer", optimizer)
    restored = load_checkpoint(tmp_path / "optimizer", make_optimizer(model, 1e-3))
    assert int(restored.step[...]) == 3
    for a, b in zip(
        jax.tree.leaves(nnx.state(optimizer)), jax.tree.leaves(nnx.state(restored))
    ):
        np.testing.assert_array_equal(a, b)


def test_loop_runs_and_resumes(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WANDB_MODE", "offline")  # exercise the backup path, no network
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_SILENT", "true")
    tiny = "--name t --lanes 8 --steps 96 --sims 4 --batch 32 --blocks 2 --channels 16"
    tiny += " --eval-every 1 --eval-games 4 --eval-sims 4 --wandb --backup-every 1"
    for iterations in (1, 2):
        monkeypatch.setattr(
            sys, "argv", ["run", *tiny.split(), "--iterations", str(iterations)]
        )
        run.main()
    assert json.loads((tmp_path / "runs/t/progress.json").read_text())["iteration"] == 2
    assert (tmp_path / "runs/t/iter_0002").exists()
    assert (tmp_path / "runs/t/logs/eval.jsonl").read_text().count("\n") == 2
    assert (tmp_path / "runs/t/optimizer").exists()
    out = capsys.readouterr().out
    assert (
        "resuming" in out and "backup failed" not in out and "wandb disabled" not in out
    )
