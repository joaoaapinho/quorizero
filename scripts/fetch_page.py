"""
The page code from GitHub, so the retrain job never deploys an older copy of it
python scripts/fetch_page.py <out_dir> [owner/repo] [branch]
"""

from __future__ import annotations

import io
import sys
import tarfile
import urllib.request
from pathlib import Path


def fetch_page(out: str, repo: str = "joaoaapinho/quorizero", branch: str = "main"):
    url = f"https://codeload.github.com/{repo}/tar.gz/refs/heads/{branch}"
    with urllib.request.urlopen(url) as response:
        archive = tarfile.open(fileobj=io.BytesIO(response.read()), mode="r:gz")
    top = archive.getmembers()[0].name.split("/")[0]  # <repo>-<branch>
    prefix, files = f"{top}/frontend/", 0
    for member in archive.getmembers():
        name = member.name.removeprefix(prefix)
        if not member.isfile() or name == member.name or ".." in name.split("/"):
            continue
        path = Path(out) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(archive.extractfile(member).read())
        files += 1
    if not files:
        sys.exit(f"no frontend/ in {repo}@{branch}")
    print(f"{files} page files from {repo}@{branch}")


if __name__ == "__main__":
    fetch_page(*sys.argv[1:4])
