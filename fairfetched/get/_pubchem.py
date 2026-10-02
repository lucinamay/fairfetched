"""Snapshot handling shared by the PubChem sources.

PubChem keeps no release history: every URL serves the current build, rewritten
every one to two weeks. A snapshot is therefore named by the ``Last-Modified``
date of one of its files. One already on disk can be reopened by that date; an
older one cannot be downloaded again.
"""

import gzip
import hashlib
import shutil
import tempfile
import urllib.request
from email.utils import parsedate_to_datetime
from pathlib import Path

import polars as pl

from fairfetched.utils import ensure_url, raw
from fairfetched.utils.ensure import check_disk_space


def snapshot_date(url: str) -> str:
    """``Last-Modified`` of ``url`` as ``YYYYMMDD``, from a HEAD request."""
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as resp:
        return parsedate_to_datetime(resp.headers["Last-Modified"]).strftime("%Y%m%d")


def upstream_md5s(md5_urls: tuple[str, ...]) -> dict[str, str]:
    """Upstream file name -> md5, parsed from PubChem's ``md5sum``-format listings."""
    md5s = {}
    for url in md5_urls:
        with urllib.request.urlopen(url) as resp:
            for line in resp.read().decode().splitlines():
                md5, filename = line.split()
                md5s[filename] = md5
    return md5s


def _md5(path: Path) -> str:
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "md5").hexdigest()


def ensure_snapshot(
    version: str,
    version_url: str,
    urls: dict[str, str],
    md5_urls: tuple[str, ...],
    raw_dir: Path,
    force: bool = False,
) -> dict[str, Path]:
    """Download each of ``urls`` to ``raw_dir/<name>.tsv.gz`` and check it against
    :func:`upstream_md5s`. A file that differs is deleted and raises.

    A complete ``raw_dir`` is returned without a request. Otherwise ``version``
    must be the :func:`snapshot_date` of ``version_url``, and raises if not."""
    paths = {name: raw_dir / f"{name}.tsv.gz" for name in urls}
    todo = [name for name, path in paths.items() if force or not path.exists()]
    if not todo:
        return paths
    if (served := snapshot_date(version_url)) != version:
        raise ValueError(
            f"PubChem serves snapshot {served} only. Snapshot {version} is not "
            f"complete in {raw_dir} and cannot be downloaded again."
        )
    for name in todo:
        ensure_url(urls[name], paths[name], force=force)
    md5s = upstream_md5s(md5_urls)
    corrupt = [n for n in todo if _md5(paths[n]) != md5s[urls[n].split("/")[-1]]]
    for name in corrupt:
        paths[name].unlink()
    if corrupt:
        raise ValueError(
            f"{corrupt} did not match upstream's md5 and were deleted: a broken "
            f"transfer, or PubChem rebuilt the files mid-download. Retry."
        )
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
