import logging as lg
from pathlib import Path

import requests


def ensure_url(url: str, path: Path, force: bool = False) -> Path:
    path.mkdir(parents=True, exist_ok=True)

    if path.exists() and not force:
        lg.debug(f"File already exists at {path}. Skipping download.")
        return path

    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    lg.debug(f"Downloaded {url} to {path}")
    return path
