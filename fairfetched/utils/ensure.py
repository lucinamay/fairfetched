import logging as lg
from pathlib import Path

import requests
from rich.progress import track


def ensure_url(url: str, path: Path, force: bool = False) -> Path:
    """Downloads url to path if not already existing. Makes path dirs if not existing"""
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not force:
        lg.debug(f"File already exists at {path}. Skipping download.")
        return path

    response = requests.get(url)
    response.raise_for_status()
    with open(path, "wb") as f:
        total = int(response.headers.get("content-length", 0))
        chunk_size = 8192
        for chunk in track(
            response.iter_content(chunk_size=chunk_size),
            total=(total // chunk_size) + int(total % chunk_size != 0),
            description=f"downloading {url.split('/')[-1]}",
        ):
            f.write(chunk)
    lg.debug(f"Downloaded {url} to {path}")
    return path
