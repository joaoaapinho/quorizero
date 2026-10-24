"""
Copy a training run between this machine and Cloud Storage
python scripts/sync_run.py down gs://<bucket>/runs/pilot-final runs/pilot-final
python scripts/sync_run.py up runs/pilot-final gs://<bucket>/runs/pilot-final
Up sends only new or resized files, so each checkpoint uploads once
"""

from __future__ import annotations

import sys
from pathlib import Path

from google.cloud import storage


def split(url: str) -> tuple[str, str]:
    bucket, _, prefix = url.removeprefix("gs://").partition("/")
    return bucket, prefix.strip("/")


def down(src: str, dst: str) -> None:
    bucket, prefix = split(src)
    blobs = list(storage.Client().list_blobs(bucket, prefix=f"{prefix}/"))
    if not blobs:
        sys.exit(f"nothing at {src}")
    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        path = Path(dst) / blob.name[len(prefix) + 1 :]
        path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(path)
    print(f"{len(blobs)} files from {src}")


def up(src: str, dst: str) -> None:
    bucket_name, prefix = split(dst)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    remote = {
        b.name: b.size for b in client.list_blobs(bucket_name, prefix=f"{prefix}/")
    }
    sent = 0
    for path in sorted(Path(src).rglob("*")):
        name = f"{prefix}/{path.relative_to(src).as_posix()}"
        if path.is_file() and remote.get(name) != path.stat().st_size:
            bucket.blob(name).upload_from_filename(path)
            sent += 1
    print(f"{sent} files to {dst}")


if __name__ == "__main__":
    direction, src, dst = sys.argv[1:4]
    {"down": down, "up": up}[direction](src, dst)
