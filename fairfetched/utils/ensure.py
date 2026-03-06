import datetime
import logging as lg
import urllib.error
import urllib.request
from pathlib import Path

from ._track import track


def ensure_url(url: str, path: Path, force: bool = False) -> Path:
    """Downloads url to path if not already existing. Makes path dirs if not existing"""
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not force:
        lg.debug(f"File already exists at {path}. Skipping download.")
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
    lg.info(f"Downloaded {url} to {path} on {datetime.datetime.now()}")
    return path
