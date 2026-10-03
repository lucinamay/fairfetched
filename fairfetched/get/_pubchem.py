"""PubChem's rolling builds, labelled by a file's ``Last-Modified`` date.
A local manifest pins each acquired build for offline reuse. Unpinned legacy
caches must match the currently served date and upstream checksums before adoption.
"""

import gzip
import shutil
import tempfile
import urllib.request
from email.utils import parsedate_to_datetime
from pathlib import Path

import polars as pl

from fairfetched.utils import ensure_url, manifest, raw
from fairfetched.utils.ensure import check_disk_space


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


def _scan(path: Path, schema: dict[str, type[pl.DataType]]) -> pl.LazyFrame:
    """Tab-separated scan with quote parsing off (assay names carry unescaped
    double quotes). Raises if the file's header is not ``schema``'s names:
    polars applies a full schema by position, so a renamed or reordered
    upstream column would otherwise be read under the old name."""
    with (gzip.open if path.suffix == ".gz" else open)(path, "rt") as fh:
        found = fh.readline().rstrip("\n").split("\t")
    if found != list(schema):
        raise ValueError(
            f"{path.name}: upstream columns changed. Expected {list(schema)}, found {found}"
        )
    return pl.scan_csv(path, separator="\t", quote_char=None, schema=schema)


def ensure_parquet_tables(
    raw_paths: dict[str, Path],
    table_dir: Path | str | None = None,
    *,
    schemas: dict[str, dict[str, type[pl.DataType]]],
    headerless: bool = False,
    decompress_first: dict[str, float] | None = None,
) -> dict[str, Path]:
    """:func:`raw.ensure_parquet_tables` with each table's ``schemas`` entry.

    polars holds a ``.gz`` in memory whole, so ``decompress_first`` tables are
    gunzipped to a system temp directory (not ``BASE_DIR``), one at a time, and
    removed once all tables are written. ``decompress_first`` maps each such
    table to its uncompressed size as a multiple of the ``.gz`` size: gzip does
    not record sizes above 4 GB, so the disk check before decompression
    (:func:`fairfetched.utils.ensure.check_disk_space`) uses that estimate.
    ``headerless`` files take their column names from the schema."""
    decompress_first = decompress_first or {}
    with tempfile.TemporaryDirectory() as tmp:

        def scanner(name: str, path: Path) -> pl.LazyFrame:
            if name in decompress_first:
                for previous in Path(tmp).iterdir():
                    previous.unlink()
                check_disk_space(
                    Path(tmp),
                    int(path.stat().st_size * decompress_first[name]),
                    f"decompressing {path.name}",
                )
                plain = Path(tmp) / f"{name}.tsv"
                with gzip.open(path, "rb") as f_in, plain.open("wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                path = plain
            if headerless:
                return pl.scan_csv(
                    path,
                    separator="\t",
                    quote_char=None,
                    has_header=False,
                    schema=schemas[name],
                )
            return _scan(path, schemas[name])

        return raw.ensure_parquet_tables(raw_paths, table_dir, scanner=scanner)
