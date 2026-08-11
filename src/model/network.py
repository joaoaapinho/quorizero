"""AlphaZero-style network in Flax NNX: a small ResNet trunk + policy & value heads.

Small on purpose (~6 blocks, 64 channels). In NNX a module owns its parameters:
you declare layers in __init__ and use them in __call__. `nnx.Rngs` seeds the
initialisation. Convolutions are channels-last, matching features.encode_state.
"""

from __future__ import annotations

import jax.numpy as jnp
from flax import nnx

from engine.actions import ACTION_SIZE
from model.features import N_PLANES

MASK_FILL = -1e9  # additive "-infinity" for illegal actions before softmax


class ResidualBlock(nnx.Module):
    """conv -> norm -> relu -> conv -> norm, then add the input and relu."""

    def __init__(self, channels: int, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(channels, channels, (3, 3), padding="SAME", rngs=rngs)
        self.norm1 = nnx.BatchNorm(channels, rngs=rngs)
        self.conv2 = nnx.Conv(channels, channels, (3, 3), padding="SAME", rngs=rngs)
        self.norm2 = nnx.BatchNorm(channels, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # implement the residual block:
        residual = x  # remember the input
        y = self.conv1(x)
        y = self.norm1(y)
        y = nnx.relu(y)
        y = self.conv2(y)
        y = self.norm2(y)
        return nnx.relu(y + residual)  # the skip connection
        raise NotImplementedError("ResidualBlock.__call__")


class QuoridorNet(nnx.Module):
    """Shared trunk feeding a policy head (logits) and a value head (scalar)."""

    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        num_blocks: int = 6,
        channels: int = 64,
    ):
        # Trunk: lift N_PLANES input planes to 'channels', then N residual blocks.
        self.stem_conv = nnx.Conv(N_PLANES, channels, (3, 3), padding="SAME", rngs=rngs)
        self.stem_norm = nnx.BatchNorm(channels, rngs=rngs)
        # nnx.List so Flax tracks these submodules (a plain list won't work).
        self.blocks = nnx.List(
            [ResidualBlock(channels, rngs=rngs) for _ in range(num_blocks)]
        )

        # Policy head: 1x1 conv down to 2 channels, flatten, dense to ACTION_SIZE.
        self.policy_conv = nnx.Conv(channels, 2, (1, 1), rngs=rngs)
        self.policy_norm = nnx.BatchNorm(2, rngs=rngs)
        self.policy_dense = nnx.Linear(2 * 9 * 9, ACTION_SIZE, rngs=rngs)

        # Value head: 1x1 conv to 1 channel, flatten, dense -> relu -> dense -> tanh.
        self.value_conv = nnx.Conv(channels, 1, (1, 1), rngs=rngs)
        self.value_norm = nnx.BatchNorm(1, rngs=rngs)
        self.value_dense1 = nnx.Linear(1 * 9 * 9, channels, rngs=rngs)
        self.value_dense2 = nnx.Linear(channels, 1, rngs=rngs)

    def __call__(
        self, x: jnp.ndarray, legal_mask: jnp.ndarray | None = None
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """x: (B, 9, 9, N_PLANES). Returns (policy_logits (B, ACTION_SIZE), value (B,)).

        If legal_mask (B, ACTION_SIZE) bool is given, illegal logits are pushed to
        MASK_FILL so they vanish under a later softmax.
        """
        # trunk:
        x = nnx.relu(self.stem_norm(self.stem_conv(x)))
        for block in self.blocks:
            x = block(x)

        # policy head:
        p = nnx.relu(self.policy_norm(self.policy_conv(x)))  # (B,9,9,2)
        p = p.reshape((p.shape[0], -1))  # -> (B,162)
        logits = self.policy_dense(p)  # (B, ACTION_SIZE)
        if legal_mask is not None:
            logits = jnp.where(legal_mask, logits, MASK_FILL)

        # value head:
        v = nnx.relu(self.value_norm(self.value_conv(x)))  # (B,9,9,1)
        v = v.reshape((v.shape[0], -1))  # (B,81)
        v = nnx.relu(self.value_dense1(v))
        v = jnp.tanh(self.value_dense2(v))  # (B,1) in [-1,1]
        v = v.squeeze(-1)  # (B,)

        return logits, v


def create_model(seed: int = 0, num_blocks: int = 6, channels: int = 64) -> QuoridorNet:
    """Build a freshly-initialised network."""
    return QuoridorNet(rngs=nnx.Rngs(seed), num_blocks=num_blocks, channels=channels)
