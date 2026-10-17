#!/usr/bin/env bash
# Overnight training on a rented GPU (Vast.ai / RunPod; any CUDA 12 box, 4090-class).
#
#   git clone <this repo> quorizero && cd quorizero
#   WANDB_API_KEY=... bash scripts/train_cloud.sh            # setup, then train in tmux
#   WANDB_API_KEY=... bash scripts/train_cloud.sh --sims 64  # extra args go to train.run
#   tmux attach -t train                                     # watch (Ctrl-b d detaches)
#
# Re-running resumes from runs/$RUN, so preemptions and crashes cost one iteration.
# With WANDB_API_KEY set, runs/$RUN is backed up to W&B every iteration, and a
# fresh box pulls that backup first - losing the whole machine loses one iteration.
# Copy the run home in the morning:  rsync -av <box>:quorizero/runs/$RUN runs/
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=${RUN:-overnight}

command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux)
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
[ -d .venv ] || uv venv -p 3.11 .venv
uv pip install -p .venv -q -e ".[ml,dev]" "jax[cuda12]==0.10.2"
.venv/bin/python -c "import jax; d = jax.devices(); assert d[0].platform == 'gpu', d; print(d)"

WANDB=""
if [ -n "${WANDB_API_KEY:-}" ]; then
  WANDB="--wandb"
  if [ ! -d "runs/$RUN" ]; then  # new machine: continue from the last backup, if any
    ENTITY=${WANDB_ENTITY:-$(.venv/bin/python -c "import wandb; print(wandb.Api().default_entity)")}
    .venv/bin/wandb artifact get "$ENTITY/quorizero/run-$RUN:latest" --root "runs/$RUN" \
      || echo "no W&B backup of '$RUN' - starting fresh"
  fi
fi
mkdir -p runs
tmux new-session -d -s train \
  "until .venv/bin/python -m train.run --name $RUN $WANDB $*; do echo 'crashed - resuming in 30s'; sleep 30; done 2>&1 | tee -a runs/$RUN.log"
echo "training '$RUN' in tmux session 'train': tmux attach -t train"
