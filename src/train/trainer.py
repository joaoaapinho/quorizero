"""Gradient steps on replay samples, with mirror augmentation and value warm start."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from jaxgame import MIRROR, mirror, observe
from model.network import QuoridorNet
from selfplay.generate import Sample
from train.loss import MARGIN_WEIGHT, az_loss


def make_optimizer(model: QuoridorNet, lr: float, weight_decay: float = 1e-4):
    return nnx.Optimizer(
        model, optax.adamw(lr, weight_decay=weight_decay), wrt=nnx.Param
    )


@nnx.jit
def train_step(model, optimizer, batch: Sample, flip: jax.Array, value_mix: jax.Array):
    # Mirror a random half of the batch: an exact symmetry, so free extra data.
    mirrored = jax.vmap(mirror)(batch.state)
    states = jax.tree.map(
        lambda m, s: jnp.where(flip.reshape((-1,) + (1,) * (s.ndim - 1)), m, s),
        mirrored,
        batch.state,
    )
    policy = jnp.where(flip[:, None], batch.policy[:, MIRROR], batch.policy)
    # Warm start: early outcomes are mostly noise, so blend in the path heuristic,
    # fading to the pure outcome (value_mix -> 0) - the final target is unbiased.
    target_value = (1 - value_mix) * batch.value + value_mix * batch.heuristic
    obs = jax.vmap(observe)(states)

    def loss_fn(model):
        logits, value, margin = model(obs)
        p, v, m = az_loss(logits, value, margin, policy, target_value, batch.margin)
        return p + v + MARGIN_WEIGHT * m, (p, v, m, logits)

    (loss, (p, v, m, logits)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    logp = jax.nn.log_softmax(logits)
    return dict(
        loss=loss,
        policy_loss=p,
        value_loss=v,
        margin_loss=m,
        grad_norm=optax.tree.norm(grads),
        policy_entropy=-(jnp.exp(logp) * logp).sum(-1).mean(),  # collapse detector
    )


def train(
    model: QuoridorNet,
    optimizer,
    sample_batch: Callable[[], Sample],
    steps: int,
    rng: np.random.Generator,
    value_mix: float = 0.0,
) -> dict:
    """Take `steps` gradient steps; return the mean of each metric."""
    model.train()
    history = []
    for _ in range(steps):
        batch = sample_batch()
        flip = jnp.asarray(rng.random(len(batch.value)) < 0.5)
        history.append(
            train_step(model, optimizer, batch, flip, jnp.float32(value_mix))
        )
    return {k: float(np.mean([h[k] for h in history])) for k in history[0]}
