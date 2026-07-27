"""Heuristic (non-learned) Quoridor bot: shortest-path eval + alpha-beta search."""

from baseline.evaluation import WIN_SCORE, evaluate
from baseline.minimax import choose_action

__all__ = ["WIN_SCORE", "choose_action", "evaluate"]
