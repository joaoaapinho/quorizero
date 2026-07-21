# Compute Plan

## Local (dev / debug)
- Laptop with RTX 3070 8GB.
- Used for: writing and debugging engine/MCTS/model code, unit tests, small-scale sanity-check training runs.

## Cloud (real training runs)
- Provider: TODO — pick one (e.g. Lambda Labs, RunPod, vast.ai, Paperspace).
- Instance type: TODO — GPU tier and VRAM needed for full self-play training.
- Trigger: burst to cloud once a training run needs more than a sanity check (e.g. full self-play generations, longer training horizons, or anything that won't fit/finish reasonably on the 3070).
- Budget / cost ceiling: TODO.
- Data/checkpoint sync between local and cloud: TODO (e.g. object storage bucket, `model_registry/`).
