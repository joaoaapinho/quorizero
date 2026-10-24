"""AlphaZero loss + KataGo-style margin term."""

from __future__ import annotations

import jax.numpy as jnp
import optax

MARGIN_WEIGHT = 0.1  # KataGo-style aux target: small, shapes features only
MARGIN_SCALE = 10.0  # margins are in steps; bring them to ~[-1, 1]


def az_loss(logits, value, margin, target_policy, target_value, target_margin):
    """
    Return (policy_loss, value_loss, margin_loss).
    Policy: cross-entropy to the search's improved policy (zero on illegal moves).
    Value: squared error to the outcome. Margin: squared error where it's known
    (NaN marks samples without one, e.g. reanalysed human positions).
    """
    policy_loss = optax.softmax_cross_entropy(logits, target_policy).mean()
    value_loss = jnp.mean((value - target_value) ** 2)
    known = ~jnp.isnan(target_margin)
    target = jnp.nan_to_num(target_margin) / MARGIN_SCALE
    margin_loss = jnp.sum(known * (margin - target) ** 2) / jnp.maximum(known.sum(), 1)
    return policy_loss, value_loss, margin_loss
