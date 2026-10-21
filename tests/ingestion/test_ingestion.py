"""Website games are replayed before use; the retrain gate decides promotion."""

import json
import random
import sys

from engine.actions import legal_actions, step
from engine.state import initial_state
from ingestion import retrain
from ingestion.games import replay
from model.checkpoint import save_checkpoint
from model.network import create_model


def played_game(seed: int = 0) -> dict:
    """A legal random game, recorded the way the website stores it."""
    rng = random.Random(seed)
    state, moves, done, win = initial_state(), [], False, None
    while not done:
        pawn = [a for a in legal_actions(state) if a < 12]
        action = rng.choice(pawn if rng.random() < 0.7 else legal_actions(state))
        moves.append(action)
        state, done, win = step(state, action)
    return {
        "status": "completed",
        "winner": win,
        "moves": moves,
        "created_at": seed + 1,
    }


def test_replay_keeps_only_valid_games():
    game = played_game()
    assert len(replay(game)) == len(game["moves"])
    assert replay({**game, "status": "abandoned"}) is None
    assert replay({**game, "winner": 1 - game["winner"]}) is None
    assert replay({**game, "moves": [130] * 3 + game["moves"]}) is None  # illegal
    assert replay({**game, "moves": game["moves"][:-1]}) is None  # unfinished


def test_retrain_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "runs/t"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"blocks": 2, "channels": 16}))
    save_checkpoint(run_dir / "latest", create_model(num_blocks=2, channels=16))
    games = [played_game(seed) for seed in range(3)]
    monkeypatch.setattr(retrain, "fetch", lambda api, token, since: games)
    monkeypatch.setenv("QZ_EXPORT_TOKEN", "x")
    args = "--run t --api http://x --steps 5 --batch 32 --sims 4 --lanes 4 --games 4"
    # threshold 0: always promote, so export runs too
    monkeypatch.setattr(sys, "argv", ["retrain", *args.split(), "--threshold", "0"])
    retrain.main()
    assert (tmp_path / "frontend/model/model.onnx").exists()
    assert "human_" in (run_dir / "best.txt").read_text()
    assert (
        json.loads((tmp_path / "data/human/last_fetch.json").read_text())["since"] == 3
    )
