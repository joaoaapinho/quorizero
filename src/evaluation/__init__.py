"""Evaluation: arena matches, Elo (with confidence intervals), model registry."""

from evaluation.arena import match_report, play_batch, play_match, win_rate
from evaluation.elo import elo_from_winrate, elo_interval, expected_score, update_elo
from evaluation.registry import ModelRegistry

__all__ = [
    "ModelRegistry",
    "elo_from_winrate",
    "elo_interval",
    "expected_score",
    "match_report",
    "play_batch",
    "play_match",
    "update_elo",
    "win_rate",
]
