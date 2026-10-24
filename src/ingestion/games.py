"""
Website games: download, validate by replaying, turn into positions.
Kept only if completed, legal under the Python engine and with the stored winner.
Also the retraining count: GET /retrain-status, POST /retrains (owner only).
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request

from engine.actions import legal_actions, step
from engine.state import GameState, initial_state

MIN_PLIES = 10  # shorter games are resignations or noise
PAGE = 1000  # games are downloaded in pages of x


def _call(api_url: str, token: str, path: str, body: dict | None = None) -> dict:
    """One owner request to the games API: GET, or POST when there is a body."""
    request = urllib.request.Request(
        f"{api_url.rstrip('/')}{path}",
        data=None if body is None else json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def fetch(api_url: str, token: str, since_ms: int) -> list[dict]:
    """All games stored after `since_ms` (epoch ms), oldest first."""
    games: list[dict] = []
    while True:
        page = _call(api_url, token, f"/games?since={since_ms}&limit={PAGE}")["games"]
        games += page
        if len(page) < PAGE:
            return games
        since_ms = page[-1]["created_at"]


def fetch_game(api_url: str, token: str, game_id: str) -> dict:
    """One stored game by id (owner only), to review a challenge claim."""
    return _call(api_url, token, f"/games?id={urllib.parse.quote(game_id)}")["games"][0]


def retrain_status(api_url: str, token: str) -> dict:
    """{since, until, losses, games, retrain_at, due}: losses since the last run."""
    return _call(api_url, token, "/retrain-status")


def record_retrain(api_url: str, token: str, run: dict) -> dict:
    """Record a run (restarts the count) and return the new status."""
    return _call(api_url, token, "/retrains", run)


def replay(record: dict) -> list[GameState] | None:
    """The position before every move, or None if the game is invalid."""
    moves = record.get("moves") or []
    if record.get("status") != "completed" or len(moves) < MIN_PLIES:
        return None
    state, positions, done, win = initial_state(), [], False, None
    for action in moves:
        if done or action not in legal_actions(state):
            return None
        positions.append(state)
        state, done, win = step(state, action)
    return positions if done and win == record.get("winner") else None
