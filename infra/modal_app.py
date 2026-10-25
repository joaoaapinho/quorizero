"""
Retraining on Modal: a daily check on a small CPU container, and the GPU retrain only
when the website's weighted loss count is due.
modal deploy infra/modal_app.py                      # schedules the daily check
modal run infra/modal_app.py::retrain --force        # one test run, count untouched
The run lives on the quorizero-runs volume, secrets in the quorizero secret.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
RUN = "pilot-final"  # the run retraining improves, under /app/runs on the volume
API = "https://quorizero-games.joaoapinho.workers.dev"
PAGES_PROJECT = "quorizero"

app = modal.App("quorizero-retrain")
runs = modal.Volume.from_name("quorizero-runs", create_if_missing=True)
# QZ_EXPORT_TOKEN, WANDB_API_KEY, CLOUDFLARE_API_TOKEN, CLOUDFLARE_ACCOUNT_ID
secret = modal.Secret.from_name("quorizero")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "ca-certificates")
    .run_commands(  # Node, only for wrangler: redeploys the page after a promotion
        "curl -fsSL https://deb.nodesource.com/setup_22.x | bash -",
        "apt-get install -y nodejs",
        "npm install -g wrangler@4.152.0",
    )
    .pip_install_from_pyproject(
        str(ROOT / "pyproject.toml"), optional_dependencies=["ml"]
    )
    .pip_install("jax[cuda12]==0.10.2")
    .env({"PYTHONPATH": "/app/src"})
    .workdir("/app")
    .add_local_dir(
        ROOT / "src", "/app/src", ignore=["**/__pycache__", "evaluation/vs_sigma.py"]
    )
    .add_local_dir(ROOT / "scripts", "/app/scripts", ignore=["**/__pycache__"])
)


@app.function(schedule=modal.Cron("0 5 * * *"), secrets=[secret], timeout=120)
def check() -> None:
    """Daily, on CPU: start the GPU retrain only when one is due."""
    request = urllib.request.Request(
        f"{API}/retrain-status",
        headers={"Authorization": f"Bearer {os.environ['QZ_EXPORT_TOKEN']}"},
    )
    with urllib.request.urlopen(request) as response:
        status = json.load(response)
    print(status)
    if status["due"]:
        retrain.spawn()


@app.function(
    image=image,
    gpu="L4",
    timeout=3 * 3600,
    max_containers=1,  # never two retrains at once
    volumes={"/app/runs": runs},
    secrets=[secret],
)
def retrain(force: bool = False) -> None:
    """Retrain on the losses, then redeploy the page if the new model is promoted."""
    command = ["python", "-m", "ingestion.retrain", "--run", RUN, "--api", API]
    subprocess.run(command + ["--wandb"] + (["--force"] if force else []), check=True)
    runs.commit()  # the checkpoint and logs persist even if the deploy fails

    model = Path("/app/frontend/model")
    if not (model / "model.onnx").exists():  # exported only when promoted
        print("kept the current model, the site is unchanged")
        return
    # the page code from GitHub main, so a promotion never brings back old page code
    page = tempfile.mkdtemp()
    subprocess.run(["python", "scripts/fetch_page.py", page], check=True)
    shutil.copytree(model, Path(page) / "model")
    deploy = ["wrangler", "pages", "deploy", page, "--project-name", PAGES_PROJECT]
    subprocess.run(deploy + ["--branch", "main"], check=True)
