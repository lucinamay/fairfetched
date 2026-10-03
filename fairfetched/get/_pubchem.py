"""PubChem's rolling builds, labelled by a file's ``Last-Modified`` date.
A local manifest pins each acquired build for offline reuse. Unpinned legacy
caches must match the currently served date and upstream checksums before adoption.
"""

import tempfile
import urllib.request
from email.utils import parsedate_to_datetime
from pathlib import Path

from fairfetched.utils import ensure_url, manifest


def snapshot_date(url: str) -> str:
    """``Last-Modified`` of ``url`` as ``YYYYMMDD``, from a HEAD request."""
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as resp:
        return parsedate_to_datetime(resp.headers["Last-Modified"]).strftime("%Y%m%d")


def available_versions(version_url: str) -> tuple[str, ...]:
    """Only the current upstream build is available to download."""
    return (snapshot_date(version_url),)


def upstream_md5s(md5_urls: tuple[str, ...]) -> dict[str, str]:
    """Upstream file name -> md5, parsed from PubChem's ``md5sum``-format listings."""
    md5s = {}
    for url in md5_urls:
        with urllib.request.urlopen(url) as resp:
            lines = resp.read().decode().splitlines()
            md5s.update(line.split()[::-1] for line in lines)
    return md5s


def ensure_snapshot(
    version: str,
    raw_dir: Path | str | None = None,
    force: bool = False,
    *,
    version_url: str,
    urls: dict[str, str],
    md5_urls: tuple[str, ...],
    root_dir: Path,
) -> dict[str, Path]:
    """Acquire PubChem's rolling build or reopen its verified local pin offline."""
    version = str(version)
    raw_dir = Path(raw_dir or root_dir / version / "raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: raw_dir / f"{name}.tsv.gz" for name in urls}
    pin = raw_dir / "_manifest.json"
    pinned = pin.exists()
    todo = [name for name, path in paths.items() if force or not path.is_file()]
    if (todo or not pinned) and snapshot_date(version_url) != version:
        raise ValueError(f"PubChem no longer serves snapshot {version}.")
    # Once pinned, SHA256 is authoritative, including for repairs/forced downloads.
    checksums = {} if pinned else upstream_md5s(md5_urls)
    expected = (
        pin
        if pinned
        else {
            "version": version,
            "files": {
                name: {"md5": checksums[url.rsplit("/", 1)[-1]]}
                for name, url in urls.items()
            },
        }
    )
    with tempfile.TemporaryDirectory(dir=raw_dir) as temp:
        staged = dict(paths)
        for name in todo:
            staged[name] = Path(temp) / paths[name].name
            ensure_url(urls[name], staged[name])
        manifest.verify(
            staged,
            expected,
            "restore the pinned bytes; do not re-pin this snapshot.",
            strict=True,
            algorithm="sha256" if pinned else "md5",
            version=version,
        )
        if not pinned and (
            snapshot_date(version_url) != version
            or upstream_md5s(md5_urls) != checksums
        ):
            raise ValueError(
                "PubChem changed during acquisition; retry the current build."
            )
        for name in todo:
            staged[name].replace(paths[name])
    if not pinned:
        manifest.write(paths, pin, version=version)
    return paths
