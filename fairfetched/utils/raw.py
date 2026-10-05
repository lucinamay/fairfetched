"""Shared raw-file -> Parquet consolidation for the flat-file sources."""

import gzip
import json
import logging as lg
import lzma
import shutil
import tempfile
import zipfile
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path

import polars as pl

from fairfetched.utils import manifest as pins
from fairfetched.utils.ensure import check_disk_space

_lg = lg.getLogger(__name__)

_SEPARATORS = {".tsv": "\t", ".txt": "\t", ".csv": ","}
MANIFEST = "_tables.json"


def _files(table_dir: Path, tables: dict[str, Path]) -> dict[str, Path]:
    """Every file in ``tables`` (a directory table contributes its part-files),
    keyed by path relative to ``table_dir``."""
    return {
        str(f.relative_to(table_dir)): f
        for path in tables.values()
        for f in sorted(path.rglob("*") if path.is_dir() else [path])
        if f.is_file()
    }


def read_manifest(
    table_dir: Path | str,
    *,
    hash_contents: bool = False,
    only_size_presence: bool = False,
) -> dict[str, Path] | None:
    """Tables recorded by :func:`write_manifest`. None if there is no manifest or
    a table or part-file is missing or extra (rebuild); ``partial`` instead returns
    the recorded tables still present (``{}`` without a manifest). Raises
    ``ValueError`` if a present file's size changed (truncation, overwrite) or, with
    ``hash_contents``, its sha256 (bit rot), even when another table is missing.
    Sizes cost a ``stat``; hashing reads every byte (ChEMBL's 1.65 GB of parquet:
    4.4 s), so loads check sizes only."""
    table_dir = Path(table_dir)
    path = table_dir / MANIFEST
    if not path.exists():
        return {} if only_size_presence else None
    recorded = json.loads(path.read_text())
    tables = {t: table_dir / rel for t, rel in recorded["tables"].items()}
    present = {t: p for t, p in tables.items() if p.exists()}
    files = _files(table_dir, present)
    for rel, f in files.items():
        if (
            rel in recorded["files"]
            and f.stat().st_size != recorded["files"][rel]["bytes"]
        ):
            raise ValueError(
                f"{f} has a different size than recorded in {path}; delete {path} "
                f"to adopt the files as they are, or the table to rebuild it."
            )
    if hash_contents:
        pins.verify(
            {rel: f for rel, f in files.items() if rel in recorded["files"]},
            path,
            strict=True,
            drift_hint=f"delete {path} to adopt the files as they are, or the table to rebuild it.",
        )
    if len(present) < len(tables):
        return present if only_size_presence else None
    if files.keys() != recorded["files"].keys():
        return None
    return tables


def write_manifest(table_dir: Path | str, tables: dict[str, Path]) -> dict[str, Path]:
    """Record each table file's size and sha256. Call after every table is in place,
    before deleting any uncompressed intermediate: the manifest is what marks the
    table set complete and what later loads verify against."""
    table_dir = Path(table_dir)
    path = table_dir / MANIFEST
    part = path.with_name(path.name + ".part")
    pins.write(
        _files(table_dir, tables),
        part,
        tables={t: str(p.relative_to(table_dir)) for t, p in tables.items()},
    )
    part.replace(path)
    return tables


def _decompress(path: Path, out: Path) -> Path:
    with (
        (gzip.open if path.suffix == ".gz" else lzma.open)(path, "rb") as f_in,
        out.open("wb") as f_out,
    ):
        shutil.copyfileobj(f_in, f_out)
    return out


def _extract_zip_member(path: Path, substrings: tuple[str, ...], out_dir: Path) -> Path:
    """Extract the one member of zip ``path`` whose name contains every substring;
    raises on 0 or >1 matches (member names carry stamps that vary between releases)."""
    with zipfile.ZipFile(path) as zf:
        hits = [n for n in zf.namelist() if all(s in n for s in substrings)]
        if len(hits) != 1:
            raise ValueError(
                f"expected 1 member of {path.name} with {substrings}, got {hits}"
            )
        check_disk_space(
            out_dir, zf.getinfo(hits[0]).file_size, f"extracting {hits[0]}"
        )
        out = out_dir / Path(hits[0]).name
        with zf.open(hits[0]) as src, out.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    return out


def scan_raw(path: Path | str, **scan_kwargs) -> pl.LazyFrame:
    """Lazy frame over one tabular file, reader chosen by suffix.

    ``.xz`` is decompressed beside the archive first; ``.gz`` is read by polars
    directly. The inner suffix picks the reader: ``.tsv``/``.txt`` (tab) and
    ``.csv`` (comma) via ``scan_csv``, ``.xlsx`` via ``read_excel`` (not
    streamable, for metadata-sized files).
    An explicit ``separator`` also selects ``scan_csv`` for unrecognized suffixes;
    other ``scan_kwargs`` go to the selected reader.
    """
    path = Path(path)
    if path.suffix == ".xz":
        return scan_raw(_decompress(path, path.with_suffix("")), **scan_kwargs)
    inner = (Path(path.stem) if path.suffix == ".gz" else path).suffix.lower()
    if inner == ".xlsx":
        return pl.read_excel(path, **scan_kwargs).lazy()
    if inner in _SEPARATORS or "separator" in scan_kwargs:
        if inner in _SEPARATORS:
            scan_kwargs.setdefault("separator", _SEPARATORS[inner])
        scan_kwargs.setdefault("infer_schema_length", 10_000)
        return pl.scan_csv(path, **scan_kwargs)
    raise ValueError(f"cannot infer how to read {path.name!r}")


def ensure_parquet_tables(
    raw_paths: dict[str, Path],
    table_dir: Path | str | None = None,
    scan_kwargs: dict[str, dict] | None = None,
    scanner: Callable[..., pl.LazyFrame] | None = None,
    table_names: dict[str, str] | None = None,
    decompress_first: dict[str, float] | None = None,
    members: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, Path]:
    """Stream each raw file into ``<table_dir>/<table>.parquet``, untouched; skip
    tables that exist and the ``readme`` entry. A table is written to ``.part`` and renamed, so an
    interrupted run leaves no partial table. Finished tables are pinned in
    ``_tables.json`` (:func:`write_manifest`); every later call checks their sizes.

    ``scan_kwargs`` (per raw name) go to :func:`scan_raw` or to
    ``scanner(name, path, **kwargs)`` when supplied. ``scanner`` replaces
    :func:`scan_raw` for sources needing extra checks or parsing.
    ``decompress_first`` maps selected gzip tables to an estimated uncompressed
    size / archive size; each is checked and staged in system temp for its sink.
    ``members`` maps zip tables to name substrings selecting the one member to
    extract (checked, staged in system temp) and scan in place of the archive.
    ``table_names`` renames raw name -> table name. ``table_dir`` defaults to
    ``<raw dir>/../parquet``.
    """
    table_dir = Path(
        table_dir or next(iter(raw_paths.values())).parent.parent / "parquet"
    )
    table_dir.mkdir(exist_ok=True, parents=True)

    decompress_first = decompress_first or {}
    members = members or {}
    recorded = read_manifest(table_dir, only_size_presence=True)
    out: dict[str, Path] = {}
    try:
        for name, path_ in raw_paths.items():
            if name == "readme":
                continue
            table = (table_names or {}).get(name, name)
            dest = out[table] = table_dir / f"{table}.parquet"
            if dest.exists():
                continue
            _lg.info(f"parsing {path_} -> {dest}")
            part = dest.with_name(dest.name + ".part")
            try:
                with ExitStack() as stack:
                    path = Path(path_)
                    if name in members:
                        tmp = Path(stack.enter_context(tempfile.TemporaryDirectory()))
                        path = _extract_zip_member(path, members[name], tmp)
                    elif name in decompress_first and path.suffix == ".gz":
                        tmp = Path(stack.enter_context(tempfile.TemporaryDirectory()))
                        check_disk_space(
                            tmp,
                            int(path.stat().st_size * decompress_first[name]),
                            f"decompressing {path.name}",
                        )
                        path = _decompress(path, tmp / path.with_suffix("").name)
                    kwargs = (scan_kwargs or {}).get(name, {})
                    lf = (
                        scanner(name, path, **kwargs)
                        if scanner
                        else scan_raw(path, **kwargs)
                    )
                    lf.sink_parquet(part)
                    part.replace(dest)
            finally:
                part.unlink(missing_ok=True)
        if out.keys() - recorded.keys():
            write_manifest(table_dir, {**recorded, **out})
    finally:  # after write_manifest on success; also on a failed scan
        for path_ in raw_paths.values():
            if Path(path_).suffix == ".xz":  # the decompressed copy made by scan_raw
                Path(path_).with_suffix("").unlink(missing_ok=True)
    return out


def scan_parquets(
    parquet_paths: dict[str, Path],
    clean: Callable[[pl.LazyFrame], pl.LazyFrame] = lambda lf: lf,
) -> dict[str, pl.LazyFrame]:
    """Scan every Parquet table, applying ``clean`` lazily."""
    return {name: clean(pl.scan_parquet(p)) for name, p in parquet_paths.items()}
