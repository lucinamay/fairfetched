"""Shared raw-file -> Parquet consolidation for the flat-file sources."""

import logging as lg
import lzma
import shutil
from collections.abc import Callable
from pathlib import Path

import polars as pl

_lg = lg.getLogger(__name__)

_SEPARATORS = {".tsv": "\t", ".txt": "\t", ".csv": ","}


def decompress_xz(path: Path) -> Path:
    """Decompress ``<name>.xz`` to ``<name>`` beside the archive, overwriting any
    leftover from an interrupted run. polars cannot stream a compressed ``.xz``
    without buffering it whole; the caller removes the file after the sink."""
    out = path.with_suffix("")
    with lzma.open(path, "rb") as f_in, out.open("wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return out


def scan_raw(path: Path | str, **scan_kwargs) -> pl.LazyFrame:
    """Lazy frame over one tabular file, reader chosen by suffix.

    ``.xz`` is decompressed beside the archive first; ``.gz`` is read by polars
    directly. The inner suffix picks the reader: ``.tsv``/``.txt`` (tab) and
    ``.csv`` (comma) via ``scan_csv``, ``.xlsx`` via ``read_excel`` (not
    streamable, for metadata-sized files).
    ``scan_kwargs`` go to that reader and override the inferred separator.
    """
    path = Path(path)
    if path.suffix == ".xz":
        return scan_raw(decompress_xz(path), **scan_kwargs)
    inner = (Path(path.stem) if path.suffix == ".gz" else path).suffix.lower()
    if inner == ".xlsx":
        return pl.read_excel(path, **scan_kwargs).lazy()
    if inner in _SEPARATORS:
        scan_kwargs.setdefault("separator", _SEPARATORS[inner])
        scan_kwargs.setdefault("infer_schema_length", 10_000)
        return pl.scan_csv(path, **scan_kwargs)
    raise ValueError(f"cannot infer how to read {path.name!r}")


def ensure_parquet_tables(
    raw_paths: dict[str, Path],
    table_dir: Path | str | None = None,
    scan_kwargs: dict[str, dict] | None = None,
    scanner: Callable[[str, Path], pl.LazyFrame] | None = None,
    table_names: dict[str, str] | None = None,
) -> dict[str, Path]:
    """Stream each raw file into ``<table_dir>/<table>.parquet``, untouched; skip
    tables that exist and the ``readme`` entry. A table is written to ``.part`` and renamed, so an
    interrupted run leaves no partial table.

    ``scan_kwargs`` (per raw name) go to :func:`scan_raw`; ``scanner(name, path)``
    replaces :func:`scan_raw` for sources that need more (zip members,
    casts). ``table_names`` renames raw name -> table name. ``table_dir``
    defaults to ``<raw dir>/../parquet``.
    """
    table_dir = Path(
        table_dir or next(iter(raw_paths.values())).parent.parent / "parquet"
    )
    table_dir.mkdir(exist_ok=True, parents=True)

    out: dict[str, Path] = {}
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
            lf = (
                scanner(name, Path(path_))
                if scanner
                else scan_raw(path_, **(scan_kwargs or {}).get(name, {}))
            )
            lf.sink_parquet(part)
            part.replace(dest)
        finally:
            if Path(path_).suffix == ".xz":
                Path(path_).with_suffix("").unlink(missing_ok=True)
    return out


def scan_parquets(
    parquet_paths: dict[str, Path],
    clean: Callable[[pl.LazyFrame], pl.LazyFrame] = lambda lf: lf,
) -> dict[str, pl.LazyFrame]:
    """Scan every Parquet table, applying ``clean`` lazily."""
    return {name: clean(pl.scan_parquet(p)) for name, p in parquet_paths.items()}
