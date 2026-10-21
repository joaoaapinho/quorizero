"""
Metrics logging: JSONL files (source of truth), mirrored to W&B if enabled
"""

from __future__ import annotations

import json
import time
from pathlib import Path


class MetricsLogger:
    def __init__(
        self,
        logdir: str | Path,
        use_wandb: bool = False,
        project: str = "quorizero",
        config: dict | None = None,
        name: str | None = None,
        run_id: str | None = None,
    ):
        self.dir = Path(logdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._files: dict[str, object] = {}
        self.run = None
        if use_wandb:
            try:
                import wandb

                # A fixed run_id makes a restarted job continue the same W&B run.
                self.run = wandb.init(
                    project=project, name=name, id=run_id, resume="allow"
                )
                self.run.config.update(config or {}, allow_val_change=True)
            except Exception as exc:  # not installed / not logged in — keep going
                print(f"[wandb disabled: {exc}]")

    def log(self, section: str, step: int, metrics: dict) -> None:
        # section groups related rows: "train", "selfplay", "generations".
        record = {"step": step, "time": time.time(), **metrics}
        handle = self._files.get(section)
        if handle is None:
            handle = open(self.dir / f"{section}.jsonl", "a")
            self._files[section] = handle
        handle.write(json.dumps(record) + "\n")
        handle.flush()  # day-one durability: survive a mid-run crash
        if self.run is not None:
            # Don't pass step= — sections log at different cadences (train steps
            # vs generation numbers) and a shared step must be monotonic. Log the
            # step as a field instead and let W&B keep its own counter.
            payload = {f"{section}/{k}": v for k, v in metrics.items()}
            payload[f"{section}/step"] = step
            self.run.log(payload)

    def backup(self, name: str, directory: str | Path, aliases: list[str]) -> None:
        """Upload `directory` as a new version of the W&B artifact `name`.

        W&B stores files by content, so each version only uploads what changed.
        A failed upload is reported but never stops training.
        """
        if self.run is None:
            return
        try:
            import wandb

            artifact = wandb.Artifact(name, type="run")
            artifact.add_dir(str(directory))
            self.run.log_artifact(artifact, aliases=aliases)
        except Exception as exc:
            print(f"[wandb backup failed: {exc}]")

    def close(self) -> None:
        for handle in self._files.values():
            handle.close()
        if self.run is not None:
            self.run.finish()
