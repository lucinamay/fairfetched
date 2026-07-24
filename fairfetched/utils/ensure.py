import logging as lg
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from ._track import track

_lg = lg.getLogger(__name__)


def ensure_url(url: str, path: Path | str, force: bool = False) -> Path:
    """Downloads url to path if not already existing. Makes path dirs if not existing"""
    if isinstance(path, str):
        path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not force:
        _lg.debug(f"File already exists at {path}. Skipping download.")
        return path

    req = urllib.request.Request(url)
    # urllib will raise HTTPError for non-2xx responses
    with urllib.request.urlopen(req) as resp:
        total_hdr = resp.getheader("Content-Length")
        total = int(total_hdr) if total_hdr and total_hdr.isdigit() else 0
        chunk_size = 8192

        def _iter_resp():
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                yield chunk

        with open(path, "wb") as f:
            for chunk in track(
                _iter_resp(),
                total=(total // chunk_size) + int(total % chunk_size != 0),
                desc=f"downloading {url.split('/')[-1]}",
            ):
                f.write(chunk)
    _lg.info(f"Downloaded {url} to {path} on {datetime.now()}")  # ruff: ignore[DTZ005]
    return path
