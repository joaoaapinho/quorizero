"""Network shapes, output ranges, and checkpoint round-trips."""

import json

import jax
import jax.numpy as jnp
import numpy as np

from engine.actions import ACTION_SIZE
from jaxgame import init, observe
from model.checkpoint import load_model, save_checkpoint
from model.network import create_model


def test_forward_shapes_and_ranges():
    model = create_model(num_blocks=3, channels=16)
    model.eval()
    planes = jnp.stack([observe(init())] * 4)
    logits, value, margin = model(planes)
    assert logits.shape == (4, ACTION_SIZE)
    assert value.shape == margin.shape == (4,)
    assert jnp.all(jnp.abs(value) <= 1)


def test_checkpoint_roundtrip(tmp_path):
    model = create_model(seed=1, num_blocks=3, channels=16)
    model.eval()
    (tmp_path / "config.json").write_text(json.dumps({"blocks": 3, "channels": 16}))
    save_checkpoint(tmp_path / "latest", model)
    loaded = load_model(tmp_path / "latest")
    loaded.eval()
    planes = observe(init())[None]
    for a, b in zip(model(planes), loaded(planes)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-6)
    assert jax.tree.structure(model) == jax.tree.structure(loaded)
