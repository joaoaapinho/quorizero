"""
frontend/src/search.js in Python, move for move (PUCT)
Re-checks website games and benchmarks other networks
"""

from __future__ import annotations

from typing import Callable

import jax
import numpy as np
from flax import nnx

from engine.actions import ACTION_SIZE, legal_actions, step
from engine.state import GameState, winner
from jaxgame import from_python
from mcts.search import evaluate

C_INIT, C_BASE = 1.25, 19652

# (game, legal actions) -> (priors over the 140 actions, value for the side to move)
Evaluator = Callable[[GameState, list[int]], tuple[np.ndarray, float]]


def softmax_over(logits: np.ndarray, legal: list[int]) -> np.ndarray:
    priors = np.zeros(ACTION_SIZE, np.float64)
    x = logits[legal] - logits[legal].max()
    priors[legal] = np.exp(x) / np.exp(x).sum()
    return priors


def network_evaluator(model) -> Evaluator:
    model.eval()
    forward = nnx.jit(evaluate)

    def evaluate_game(game: GameState, legal: list[int]):
        state = jax.tree.map(lambda x: x[None], from_python(game))
        logits, value = forward(model, state)
        return softmax_over(np.asarray(logits[0], np.float64), legal), float(value[0])

    return evaluate_game


class Node:
    __slots__ = ("game", "done", "legal", "priors", "visits", "value_sum", "children")

    def __init__(self, game: GameState, evaluate_fn: Evaluator):
        self.game, self.visits, self.children = game, 1, {}
        self.done = winner(game) is not None
        self.legal, self.priors, self.value_sum = [], None, 0.0  # finished: worth 0
        if not self.done:
            self.legal = legal_actions(game)
            self.priors, self.value_sum = evaluate_fn(game, self.legal)


def _q(parent: Node, child: Node) -> float:
    reward = float(winner(child.game) == parent.game.to_move)
    return reward + (0.0 if child.done else -1.0) * child.value_sum / child.visits


def _select(node: Node) -> int:
    qs = {a: _q(node, c) for a, c in node.children.items()}
    value = node.value_sum / node.visits
    lo, hi = min([value, *qs.values()]), max([value, *qs.values()])
    scale = max(hi - lo, 1e-8)
    # PUCT (AlphaZero): the prior's bonus fades as a child gets visits
    explore = np.sqrt(node.visits) * (
        C_INIT + np.log((node.visits + C_BASE + 1) / C_BASE)
    )

    def score(a):
        child = node.children.get(a)
        q = (qs[a] - lo) / scale if child else 0.0
        return q + explore * node.priors[a] / ((child.visits if child else 0) + 1)

    return max(node.legal, key=score)


def _simulate(root: Node, evaluate_fn: Evaluator) -> None:
    path, node, value = [root], root, 0.0
    while not node.done:
        action = _select(node)
        child = node.children.get(action)
        if child is None:
            child = Node(step(node.game, action)[0], evaluate_fn)
            node.children[action] = child
            path.append(child)
            value = child.value_sum
            break
        path.append(node := child)
    if node.done:
        node.visits += 1  # revisited a finished game
    for parent, child in zip(reversed(path[:-1]), reversed(path[1:])):
        reward = float(winner(child.game) == parent.game.to_move)
        value = reward + (0.0 if child.done else -1.0) * value
        parent.value_sum += value
        parent.visits += 1


def agent(evaluate_fn: Evaluator, sims: int) -> Callable[[GameState], int]:
    """GameState -> action: PUCT with `sims` simulations, or the raw policy if 0."""

    def act(game: GameState) -> int:
        root = Node(game, evaluate_fn)
        if sims == 0 or len(root.legal) == 1:
            return max(root.legal, key=lambda a: root.priors[a])
        for _ in range(sims):
            _simulate(root, evaluate_fn)
        visits = {a: c.visits for a, c in root.children.items()}
        return max(root.legal, key=lambda a: (visits.get(a, 0), root.priors[a]))

    return act
