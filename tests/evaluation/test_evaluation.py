"""Arena, Elo and registry tests."""

import random

import jax
import jax.numpy as jnp

from baseline.match import baseline_agent, random_agent
from evaluation.arena import match_report, play_batch, play_match, win_rate
from evaluation.elo import elo_from_winrate, elo_interval, expected_score, update_elo
from evaluation.registry import ModelRegistry
from mcts.search import greedy_agent


def random_jax_agent(states, key):
    keys = jax.random.split(key, states.legal.shape[0])
    logits = jnp.where(states.legal, 0.0, -jnp.inf)
    return jax.vmap(jax.random.categorical)(keys, logits)


# --- arena ---------------------------------------------------------------


def test_play_match_counts_sum_to_games():
    rng = random.Random(0)
    a, b, d = play_match(baseline_agent(1), random_agent(rng), num_games=6)
    assert a + b + d == 6
    assert a > b  # the baseline should beat a random mover most of the time


def test_batched_arena_greedy_beats_random():
    a, b, d = play_batch(greedy_agent, random_jax_agent, jax.random.key(0), 16)
    assert a + b + d == 16 and a > b
    report = match_report(greedy_agent, random_jax_agent, jax.random.key(1), 16)
    assert report["elo_low"] < report["elo"] < report["elo_high"]


def test_win_rate_scoring():
    assert win_rate(6, 0, 0) == 1.0
    assert win_rate(0, 6, 0) == 0.0
    assert win_rate(2, 2, 0) == 0.5
    assert win_rate(0, 0, 4) == 0.5  # all draws


# --- elo -----------------------------------------------------------------


def test_expected_score_is_symmetric():
    assert expected_score(1500, 1500) == 0.5
    assert expected_score(1600, 1400) > 0.5
    assert abs(expected_score(1600, 1400) + expected_score(1400, 1600) - 1.0) < 1e-9


def test_update_rewards_the_winner():
    r_a, r_b = update_elo(1500, 1500, score_a=1.0)
    assert r_a > 1500 and r_b < 1500


def test_elo_interval_narrows_with_games():
    low, high = elo_interval(0.6, 100)
    assert low < elo_from_winrate(0.6) < high
    low2, high2 = elo_interval(0.6, 1000)
    assert high2 - low2 < high - low


def test_elo_from_winrate_is_monotonic():
    assert elo_from_winrate(0.5) == 0.0
    assert elo_from_winrate(0.75) > 0
    assert elo_from_winrate(0.25) < 0


# --- registry ------------------------------------------------------------


def test_registry_promote_and_read(tmp_path):
    reg = ModelRegistry(tmp_path / "registry")
    assert reg.best_path() is None
    reg.promote(tmp_path / "gen_0001")
    assert reg.best_path().endswith("gen_0001")
