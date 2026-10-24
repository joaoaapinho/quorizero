"""Website games are replayed before use; only QuoriZero's losses are retrained on."""

import random
import sys

import pytest

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
        "human_side": win,  # the player won: a QuoriZero loss
        "difficulty": "hard",
        "model_version": "t/latest",
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


def test_only_counted_losses_are_used():
    loss = played_game()
    games = [
        loss,
        {**loss, "human_side": 1 - loss["winner"]},  # QuoriZero won
        {**loss, "difficulty": "easy"},  # Easy loses on purpose
        {**loss, "status": "resigned"},
        {**loss, "model_version": "heuristic"},  # no model was loaded
        {**loss, "created_at": 99},  # after the status snapshot
    ]
    assert retrain.lost_games(games, until=10) == [loss]


def status(due: bool) -> dict:
    return dict(since=0, until=3, losses=5.0, games=3, retrain_at=5, due=due)


def test_check_exits_by_status(monkeypatch):
    monkeypatch.setenv("QZ_EXPORT_TOKEN", "x")
    monkeypatch.setattr(sys, "argv", ["retrain", "--api", "http://x", "--check"])
    for due, code in ((True, 0), (False, retrain.NOT_DUE)):
        monkeypatch.setattr(retrain, "retrain_status", lambda api, token: status(due))
        with pytest.raises(SystemExit) as stop:
            retrain.main()
        assert stop.value.code == code


def test_retrain_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "runs/t"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text('{"blocks": 2, "channels": 16}')
    save_checkpoint(run_dir / "latest", create_model(num_blocks=2, channels=16))
    games = [played_game(seed) for seed in range(3)]
    recorded = []
    monkeypatch.setattr(retrain, "retrain_status", lambda api, token: status(True))
    monkeypatch.setattr(retrain, "fetch", lambda api, token, since: games)
    monkeypatch.setattr(
        retrain, "record_retrain", lambda a, t, run: recorded.append(run)
    )
    monkeypatch.setenv("QZ_EXPORT_TOKEN", "x")
    args = "--run t --api http://x --steps 5 --batch 32 --sims 4 --lanes 4 --games 4"
    small = "--deep-sims 4 --critical 3"
    # threshold 0: always promote, so export runs too
    argv = ["retrain", *args.split(), *small.split(), "--threshold", "0"]
    monkeypatch.setattr(sys, "argv", argv)
    retrain.main()
    assert (tmp_path / "frontend/model/model.onnx").exists()
    assert "human_" in (run_dir / "best.txt").read_text()
    # the run is recorded so the count restarts after the newest game it covered
    assert len(recorded) == 1 and recorded[0]["promoted"] is True
    assert recorded[0]["games_until"] == 3 and recorded[0]["losses"] == 5.0
    assert recorded[0]["model_version"].startswith("t/human_")
