"""Elo maths: expected score, incremental update, win-rate -> Elo gap (+ its 95% CI)."""

from __future__ import annotations

import math


def expected_score(rating_a: float, rating_b: float) -> float:
    """Probability that A beats B under the Elo model."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def update_elo(
    rating_a: float, rating_b: float, score_a: float, k: float = 32.0
) -> tuple[float, float]:
    """Nudge both ratings after a result. score_a in [0, 1] (A's win-rate)."""
    expected_a = expected_score(rating_a, rating_b)
    rating_a += k * (score_a - expected_a)
    rating_b += k * ((1 - score_a) - (1 - expected_a))
    return rating_a, rating_b


def elo_from_winrate(win_rate: float, anchor: float = 0.0) -> float:
    """Elo relative to an opponent fixed at `anchor`, inferred from a win-rate.

    Handy for the headline chart: pin the baseline at 0, and each generation's
    win-rate against it becomes "Elo above baseline".
    """
    win_rate = min(max(win_rate, 1e-4), 1 - 1e-4)
    return anchor + 400 * math.log10(win_rate / (1 - win_rate))


def elo_interval(score: float, games: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval of the Elo gap behind a match score (normal approximation).

    Draws only shrink the variance, so p(1-p) is a safe upper bound.
    """
    half = z * math.sqrt(max(score * (1 - score), 1e-4) / games)
    return elo_from_winrate(score - half), elo_from_winrate(score + half)
