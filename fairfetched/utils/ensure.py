import errno
import logging as lg
import shutil
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from ._track import track

_lg = lg.getLogger(__name__)

# share of the disk that should stay free after a download, as in papyrus_scripts
_DISK_MARGIN = 0.1


def check_disk_space(directory: Path, nbytes: int, what: str) -> None:
    """Raise ``ENOSPC`` if ``nbytes`` do not fit on ``directory``'s disk; warn if
    writing them leaves under ``_DISK_MARGIN`` of its capacity free."""
    disk = shutil.disk_usage(directory)
    if nbytes > disk.free:
        raise OSError(
            errno.ENOSPC,
            f"{what} ({nbytes / 1e9:.1f} GB) does not fit in the "
            f"{disk.free / 1e9:.1f} GB free on the disk of {directory}",
        )
    if disk.free - nbytes < _DISK_MARGIN * disk.total:
        _lg.warning(
            f"{what} ({nbytes / 1e9:.1f} GB) leaves "
            f"{(disk.free - nbytes) / 1e9:.1f} GB free on the disk of {directory}, "
            f"under {_DISK_MARGIN:.0%} of its capacity"
        )


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

        check_disk_space(path.parent, total, f"downloading {url.split('/')[-1]}")

        def _iter_resp():
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                yield chunk

        # staged so an interrupted download is not mistaken for a complete file
        part = path.with_name(path.name + ".part")
        received = 0
        with open(part, "wb") as f:
            for chunk in track(
                _iter_resp(),
                total=(total // chunk_size) + int(total % chunk_size != 0),
                desc=f"downloading {url.split('/')[-1]}",
            ):
                f.write(chunk)
                received += len(chunk)
        if total and received != total:
            part.unlink(missing_ok=True)
            raise OSError(
                f"download of {url} ended after {received} bytes; expected {total} bytes"
            )
        part.replace(path)
    _lg.info(f"Downloaded {url} to {path} on {datetime.now()}")  # ruff: ignore[DTZ005]
    return path
