"""
Matches between agents, from paired random openings with seats swapped
play_match: engine.GameState agents (slow, e.g. minimax)
play_batch: batched JAX agents, all games at once
"""

from __future__ import annotations

import random
from typing import Callable

import jax
import jax.numpy as jnp

from baseline.match import play_game
from engine.actions import N_PAWN, legal_actions, step
from engine.state import GameState, initial_state
from evaluation.elo import elo_from_winrate, elo_interval
from jaxgame import init
from jaxgame import step as jax_step

Agent = Callable[[GameState], int]
OPENING_PLIES = 4

_vstep = jax.jit(jax.vmap(jax_step))


def opening_moves(rng: random.Random) -> list[int]:
    """OPENING_PLIES random pawn moves from the start."""
    state, moves = initial_state(), []
    for _ in range(OPENING_PLIES):
        pawn_moves = [a for a in legal_actions(state) if a < N_PAWN]
        moves.append(rng.choice(pawn_moves))
        state = step(state, moves[-1])[0]
    return moves


def random_opening(rng: random.Random) -> GameState:
    """The start position after OPENING_PLIES random pawn moves."""
    state = initial_state()
    for action in opening_moves(rng):
        state = step(state, action)[0]
    return state


def play_match(
    agent_a: Agent, agent_b: Agent, num_games: int = 20, max_moves: int = 200, seed=0
) -> tuple[int, int, int]:
    """Return (a_wins, b_wins, draws)."""
    rng = random.Random(seed)
    a_wins = b_wins = draws = 0
    for g in range(num_games):
        if g % 2 == 0:
            start = random_opening(rng)
        a_seat = g % 2
        first, second = (agent_a, agent_b) if a_seat == 0 else (agent_b, agent_a)
        result = play_game(first, second, max_moves, state=start)
        if result is None:
            draws += 1
        elif result == a_seat:
            a_wins += 1
        else:
            b_wins += 1
    return a_wins, b_wins, draws


def play_batch(
    agent_a, agent_b, key, num_games: int = 256, starts=None
) -> tuple[int, int, int]:
    """
    Return (a_wins, b_wins, draws) over num_games (even) played in parallel.
    starts: positions to play from instead of random openings, each one twice with
    seats swapped (num_games is then 2 x the number of positions).
    """
    if starts is not None:
        states = jax.tree.map(lambda x: jnp.repeat(jnp.asarray(x), 2, axis=0), starts)
        num_games = states.to_move.shape[0]
    a_seat = jnp.arange(num_games) % 2
    if starts is None:
        states = jax.vmap(lambda _: init())(jnp.arange(num_games))
        for _ in range(OPENING_PLIES):
            key, sub = jax.random.split(key)
            pair_keys = jnp.repeat(jax.random.split(sub, num_games // 2), 2, axis=0)
            legal_pawn = states.legal.at[:, N_PAWN:].set(False)
            pawn_only = jnp.where(legal_pawn, 0.0, -jnp.inf)
            moves = jax.vmap(jax.random.categorical)(pair_keys, pawn_only)
            states = _vstep(states, moves)
    while not bool(states.done.all()):
        key, key_a, key_b = jax.random.split(key, 3)
        a_moves, b_moves = agent_a(states, key_a), agent_b(states, key_b)
        states = _vstep(states, jnp.where(states.to_move == a_seat, a_moves, b_moves))
    a_wins = int((states.winner == a_seat).sum())
    b_wins = int((states.winner == 1 - a_seat).sum())
    return a_wins, b_wins, num_games - a_wins - b_wins


def win_rate(a_wins: int, b_wins: int, draws: int) -> float:
    """Score for agent_a: a win is 1, a draw is 0.5."""
    total = a_wins + b_wins + draws
    return (a_wins + 0.5 * draws) / total if total else 0.0


def match_report(agent_a, agent_b, key, num_games: int = 256, starts=None) -> dict:
    """play_batch summarised for logs: agent_a's score and Elo gap (+ 95% CI).

    The Elo uses (points + 1/2) / (games + 1), so a clean sweep reads as a large
    but finite gap set by the sample size, not as infinity.
    """
    a_wins, b_wins, draws = play_batch(agent_a, agent_b, key, num_games, starts)
    num_games = a_wins + b_wins + draws
    score = win_rate(a_wins, b_wins, draws)
    smoothed = (score * num_games + 0.5) / (num_games + 1)
    low, high = elo_interval(smoothed, num_games)
    elo = elo_from_winrate(smoothed)
    return dict(score=score, elo=elo, elo_low=low, elo_high=high, draws=draws)
