"""Save / load network weights with Orbax (the standard JAX checkpointer).

The NNX idea: a module splits into (graphdef, state) - graphdef is the static
structure, state is the pytree of parameters. Orbax saves the *state*; to load,
you rebuild an identically-shaped model as a template, restore the state into its
shape, and merge them back together.

Note: Orbax needs an ABSOLUTE path and won't overwrite an existing directory -
use a fresh path per checkpoint (e.g. model_registry/gen_0007).
"""

from __future__ import annotations

from pathlib import Path

import orbax.checkpoint as ocp
from flax import nnx

from model.network import QuoridorNet


def save_checkpoint(path: str | Path, model: QuoridorNet) -> None:
    """WORKED EXAMPLE - load_checkpoint is the mirror image of this."""
    path = Path(path).absolute()
    _, state = nnx.split(model)  # (graphdef, state); we only persist the state
    ckptr = ocp.StandardCheckpointer()
    ckptr.save(path, state)
    ckptr.wait_until_finished()  # saving is async; block until it's on disk


def load_checkpoint(path: str | Path, template: QuoridorNet) -> QuoridorNet:
    """Restore weights into a template model built with the same architecture."""
    path = Path(path).absolute()
    # mirror save_checkpoint to rebuild a live model:
    graphdef, abstract_state = nnx.split(template)
    restored = ocp.StandardCheckpointer().restore(path, abstract_state)
    return nnx.merge(graphdef, restored)
    raise NotImplementedError("load_checkpoint")
