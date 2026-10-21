"""The browser bot (frontend/src) must play by exactly the same rules and inputs."""

import json
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from engine.actions import legal_action_mask, legal_actions, step
from engine.state import initial_state
from jaxgame import from_python, observe

ROOT = Path(__file__).parents[2]


def random_game(rng: random.Random, pawn_bias: float) -> dict:
    game, actions, legal, planes, win = initial_state(), [], [], {}, -1
    for ply in range(120):
        moves = legal_actions(game)
        pawn = [a for a in moves if a < 12]
        action = rng.choice(pawn if rng.random() < pawn_bias else moves)
        legal.append([int(x) for x in legal_action_mask(game)])
        if ply % 5 == 0:
            planes[ply] = np.asarray(observe(from_python(game))).ravel().tolist()
        actions.append(action)
        game, done, winner = step(game, action)
        if done:
            win = winner
            break
    return {"actions": actions, "legal": legal, "planes": planes, "winner": win}


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_browser_code_matches_python(tmp_path):
    rng = random.Random(0)
    games = [random_game(rng, bias) for bias in (0.0, 0.8) for _ in range(6)]
    for g in games:  # JSON object keys are strings; index planes by ply instead
        g["planes"] = [g["planes"].get(i) for i in range(len(g["actions"]))]
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps({"games": games}))
    result = subprocess.run(
        ["node", str(ROOT / "frontend/test/check.mjs"), str(fixture)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_games_api_worker():
    result = subprocess.run(
        ["node", "--no-warnings", str(ROOT / "worker/test.mjs")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout
