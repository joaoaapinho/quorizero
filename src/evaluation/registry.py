"""Model registry: a tiny pointer to the current 'best' checkpoint."""

from __future__ import annotations

from pathlib import Path


class ModelRegistry:
    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.pointer = self.dir / "best.txt"

    def best_path(self) -> str | None:
        # None means nothing promoted yet — callers fall back to the baseline.
        return self.pointer.read_text().strip() if self.pointer.exists() else None

    def promote(self, checkpoint_path: str | Path) -> None:
        self.pointer.write_text(str(Path(checkpoint_path).absolute()))
