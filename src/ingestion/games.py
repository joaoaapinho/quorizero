"""Website games: download, validate by replaying, turn into positions.

Each record (worker/schema.sql) stores moves as the engine's 140-action indices.
A game is kept only if it was completed, every move is legal under the Python
reference engine, and the replay produces the stored winner. Anything a tampered
browser sends that breaks the rules is dropped here, not trusted.
"""

from __future__ import annotations

import json
import urllib.request

from engine.actions import legal_actions, step
from engine.state import GameState, initial_state

MIN_PLIES = 10  # shorter "games" are resignations or noise
PAGE = 1000


def fetch(api_url: str, token: str, since_ms: int) -> list[dict]:
    """All games stored after `since_ms` (epoch ms), oldest first."""
    games: list[dict] = []
    while True:
        request = urllib.request.Request(
            f"{api_url.rstrip('/')}/games?since={since_ms}&limit={PAGE}",
            headers={"Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(request) as response:
            page = json.load(response)["games"]
        games += page
        if len(page) < PAGE:
            return games
        since_ms = page[-1]["created_at"]


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
