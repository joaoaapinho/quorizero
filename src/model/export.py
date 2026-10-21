"""
Export a checkpoint to ONNX for the browser (onnxruntime-web), checked against JAX
python -m model.export runs/main/iter_0100 frontend/model
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort
from jax2onnx import to_onnx

from jaxgame import N_PLANES, init, observe, step
from model.checkpoint import load_model


def sample_planes(n: int = 16, seed: int = 0) -> np.ndarray:
    """Planes from a short random game, to compare ONNX against JAX."""
    rng = np.random.default_rng(seed)
    state, planes = init(), []
    for _ in range(n):
        planes.append(observe(state))
        state = step(state, int(rng.choice(np.flatnonzero(state.legal))))
    return np.stack(planes)


def export(model, out_dir: str | Path, version: str) -> Path:
    model.eval()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "model.onnx"
    to_onnx(
        lambda planes: model(planes),
        [("B", 9, 9, N_PLANES)],
        opset=17,
        export_mode="web",
        return_mode="file",
        output_path=str(path),
        input_names=["planes"],
        output_names=["logits", "value", "margin"],
    )
    # jax2onnx annotates some NHWC tensors with NCHW shapes; onnxruntime-web
    # rejects that, and the annotations are optional, so drop them.
    graph = onnx.load(path)
    del graph.graph.value_info[:]
    onnx.save(graph, path)
    planes = sample_planes()
    with jax.default_matmul_precision("highest"):  # GPUs default to TF32
        expected = jax.tree.map(np.asarray, model(jnp.asarray(planes)))
    got = ort.InferenceSession(str(path)).run(None, {"planes": planes})
    for want, have in zip(expected, got):
        np.testing.assert_allclose(have, want, rtol=1e-4, atol=1e-4)
    (out_dir / "version.json").write_text(
        json.dumps({"version": version, "exported_at": int(time.time())})
    )
    return path


def main() -> None:
    p = argparse.ArgumentParser(description="Export a checkpoint to ONNX")
    p.add_argument("checkpoint")
    p.add_argument("out_dir", nargs="?", default="frontend/model")
    p.add_argument("--version", help="label shown in game records (default: path)")
    args = p.parse_args()
    ckpt = Path(args.checkpoint)
    version = args.version or f"{ckpt.parent.name}/{ckpt.name}"
    print("exported", export(load_model(ckpt), args.out_dir, version), version)


if __name__ == "__main__":
    main()
