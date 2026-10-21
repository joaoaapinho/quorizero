"""
Website games: download, validate by replaying, turn into positions.
Kept only if completed, legal under the Python engine and with the stored winner.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request

from engine.actions import legal_actions, step
from engine.state import GameState, initial_state

MIN_PLIES = 10  # shorter games are resignations or noise
PAGE = 1000  # games are downloaded in pages of x


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


def fetch_game(api_url: str, token: str, game_id: str) -> dict:
    """One stored game by id (owner only), to review a challenge claim."""
    request = urllib.request.Request(
        f"{api_url.rstrip('/')}/games?id={urllib.parse.quote(game_id)}",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)["games"][0]


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
