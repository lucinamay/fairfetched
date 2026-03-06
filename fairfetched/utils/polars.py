import logging
import lzma
import os
import sqlite3
import tempfile
from pathlib import Path

import polars as pl

from fairfetched.utils._track import track


def scan_tsvxz(path: str | Path, **kwargs) -> pl.LazyFrame:
    logging.debug(f"scanning {path}")
    print(f"scanning {path}")
    with lzma.open(str(path), "rb") as f_lzma:
        return pl.scan_csv(f_lzma, **kwargs)


def __tmpfile(true_file: Path) -> Path:
    """Create the temp file in the same directory as the target file so
    os.replace(tmp, true_file) can perform an atomic rename without
    raising "Invalid cross-device link"."""
    tmp = tempfile.NamedTemporaryFile(
        prefix=true_file.stem + ".",
        suffix=true_file.suffix,
        dir=str(true_file.parent),
        delete=False,
    )
    pth = Path(tmp.name)
    tmp.close()
    return pth


def overwrite_scanned_lf(lf: pl.LazyFrame, path_: Path) -> None:
    tmpfile = __tmpfile(path_)
    lf.sink_parquet(tmpfile)
    os.replace(tmpfile, path_)


def lowercase_columns(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.select(pl.all().name.to_lowercase())


def _sqlite_tables(db_path: str | Path) -> list[str]:
    with sqlite3.connect(db_path) as conn:
        return (
            pl.read_database(
                """
                SELECT name
                FROM sqlite_master
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """,
                conn,
            )
            .get_column("name")
            .to_list()
        )


def ensure_sqlite_db_to_parquets(
    db_path: str | Path, cache_dir: str | Path | None = None, force: bool = False
) -> dict[str, Path]:
    """extracts a sqlite database into its tabular data as parquet files"""
    db_path = Path(db_path)
    if not cache_dir:
        cache_dir = db_path.parent
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True, parents=True)

    tables = _sqlite_tables(db_path)
    conn = sqlite3.connect(db_path)

    # get user tables
    tables = (
        pl.read_database(
            """
            SELECT name
            FROM sqlite_master
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """,
            conn,
        )
        .get_column("name")
        .to_list()
    )

    # load tables
    out = {}
    for t in track(tables, desc="exctracting tables from sqlite"):
        path_out = Path(cache_dir) / f"{t}.parquet"
        if path_out.exists() and not force:
            continue
        out[t] = path_out
        if not path_out.exists():
            pl.read_database(
                f'SELECT * FROM "{t}"', conn, infer_schema_length=None
            ).write_parquet(path_out)

    conn.close()  # 5m37s
    return out
