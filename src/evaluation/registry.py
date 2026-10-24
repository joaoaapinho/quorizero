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
        if not self.pointer.exists():
            return None
        return str(self.dir / self.pointer.read_text().strip())

    def promote(self, checkpoint_path: str | Path) -> None:
        # relative to the run, so the run can move between laptop, bucket and cloud
        path = Path(checkpoint_path).absolute()
        if path.is_relative_to(self.dir.absolute()):
            path = path.relative_to(self.dir.absolute())
        self.pointer.write_text(str(path))
