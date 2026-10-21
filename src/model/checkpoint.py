"""
Save / load NNX objects (weights, optimizer moments) with Orbax
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import orbax.checkpoint as ocp
from flax import nnx

from model.network import QuoridorNet, create_model


def save_checkpoint(path: str | Path, obj) -> None:
    """Persist an NNX object's state to `path`.

    Written to a temporary directory first and then swapped in, so a crash in
    the middle of a save never destroys the previous checkpoint.
    """
    path = Path(path).absolute()
    tmp, old = path.with_name(path.name + ".tmp"), path.with_name(path.name + ".old")
    for leftover in (tmp, old):
        shutil.rmtree(leftover, ignore_errors=True)
    ckptr = ocp.StandardCheckpointer()
    ckptr.save(tmp, nnx.state(obj))
    ckptr.wait_until_finished()  # saving is async; block until it's on disk
    if path.exists():
        path.rename(old)
    tmp.rename(path)
    shutil.rmtree(old, ignore_errors=True)


def load_checkpoint(path: str | Path, template):
    """Restore a saved state into a template built the same way (same shapes)."""
    path = Path(path).absolute()
    graphdef, abstract_state = nnx.split(template)
    restored = ocp.StandardCheckpointer().restore(path, abstract_state)
    return nnx.merge(graphdef, restored)


def load_model(path: str | Path) -> QuoridorNet:
    """Load a run's checkpoint; the architecture comes from <run>/config.json."""
    config = json.loads((Path(path).parent / "config.json").read_text())
    template = create_model(num_blocks=config["blocks"], channels=config["channels"])
    return load_checkpoint(path, template)
