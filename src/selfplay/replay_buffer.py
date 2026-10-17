"""Replay buffer: the most recent N samples in host memory, sampled uniformly.

Stores compact game states (~300 bytes) instead of network input planes (~3 KB);
planes are rebuilt on the GPU per batch, which also makes mirroring free.
"""

from __future__ import annotations

import jax
import numpy as np

from selfplay.generate import Sample


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data: Sample | None = None
        self.size = 0
        self.cursor = 0

    def __len__(self) -> int:
        return self.size

    def add(self, batch: Sample) -> None:
        batch = jax.tree.map(lambda x: x[-self.capacity :], batch)
        n = len(batch.value)
        if self.data is None:
            self.data = jax.tree.map(
                lambda x: np.zeros((self.capacity, *x.shape[1:]), x.dtype), batch
            )
        idx = (self.cursor + np.arange(n)) % self.capacity
        for buf, x in zip(jax.tree.leaves(self.data), jax.tree.leaves(batch)):
            buf[idx] = x
        self.cursor = (self.cursor + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, rng: np.random.Generator, n: int) -> Sample:
        idx = rng.integers(0, self.size, n)
        return jax.tree.map(lambda x: x[idx], self.data)
