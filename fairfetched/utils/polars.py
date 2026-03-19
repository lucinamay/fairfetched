import logging
import lzma
import os
import sqlite3
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Callable, Hashable, Iterable

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
        out[t] = path_out
        if path_out.exists() and not force:
            continue
        pl.read_database(
            f'SELECT * FROM "{t}"', conn, infer_schema_length=None
        ).write_parquet(path_out)

    conn.close()  # 5m37s
    return out


def map_elements_pooled_cached(
    df: pl.LazyFrame,
    function: Callable[[Any], Any],
    col: str,
    alias: str | None,
    return_dtype: pl.DataType | pl.DataTypeExpr | None = None,
    suffix: str = "_right",
    **kwargs,
) -> pl.LazyFrame:

    fnx = lru_cache(1024)(partial(function, **kwargs))
    df = df.with_row_index("tmp_index").sort(col, "tmp_index")
    with ThreadPoolExecutor(max_workers=round((os.cpu_count() or 1) * 0.9)) as executor:
        res = list(executor.map(fnx, df.select(col).collect().get_column(col)))
    return df.with_columns(
        pl.Series(name=alias, values=res, return_dtype=return_dtype)
        .sort("tmp_index")
        .drop("tmp_index")
    )


def map_elements_cached(
    df: pl.LazyFrame,
    function: Callable[[Any], Any],
    col: str,
    alias: str | None,
    return_dtype: pl.DataType | pl.DataTypeExpr | None = None,
    suffix: str = "_right",
    **kwargs,
) -> pl.LazyFrame:
    if not isinstance(df.select(col).collect_schema().dtypes()[0], Hashable):
        raise ValueError(f"can only map cached for hashable items, not {type(first)}")
    fnx = lru_cache(1)(cached_fnx)
    return (
        df.with_row_index("tmp_index")
        .sort(col)
        .with_columns(
            pl.col(col)
            .map_elements(fnx, **kwargs, return_dtype=return_dtype)
            .alias(alias)
        )
        .sort("tmp_index")
        .drop("tmp_index")
    )


def map_elements_unique(
    df: pl.LazyFrame,
    function: Callable[[Any], Any],
    col: str,
    alias: str | None,
    return_dtype: pl.DataType | pl.DataTypeExpr | None = None,
    suffix: str = "_right",
    **kwargs,
) -> pl.LazyFrame:
    tmp_col = col
    if not alias or alias == col:
        tmp_col = f"{col}_tmp"
        alias = col
    return df.join(
        df.select(col)
        .unique()
        .rename({col: tmp_col})
        .with_columns(
            pl.col(tmp_col)
            .map_elements(function, return_dtype=return_dtype, **kwargs)
            .alias(alias)
        ),
        left_on=col,
        right_on=tmp_col,
        how="left",
        maintain_order="left",
        suffix=suffix,
    )


def map_elements_unique_pooled(
    df: pl.LazyFrame,
    function: Callable[[Any], Any],
    col: str,
    alias: str | None,
    return_dtype: pl.DataType | None = None,
    **kwargs,
) -> pl.LazyFrame:
    tmp_col = col
    if not alias or alias == col:
        tmp_col = f"{col}_tmp"
        alias = col
    unique = (
        df.select(pl.col(col)).unique().collect().get_column(col)  # ty: ignore[unresolved-attribute]
    )
    function = partial(function, **kwargs)
    with ThreadPoolExecutor(max_workers=round((os.cpu_count() or 1) * 0.9)) as executor:
        res = list(executor.map(function, unique))
        mapped_df = pl.DataFrame(
            [
                pl.Series(
                    unique, dtype=df.select(col).collect_schema().dtypes()[0]
                ).alias(tmp_col),
                pl.Series(res, dtype=return_dtype).alias(alias),
            ]
        )
    return df.join(
        mapped_df.lazy(),
        left_on=col,
        right_on=tmp_col,
        how="left",
        maintain_order="left",
    )


def map_batches_unique(
    df: pl.DataFrame,
    function: Callable[[Iterable[Any]], pl.Series],
    col: str,
    alias: str | None = None,
    return_dtype: pl.DataType | pl.DataTypeExpr | None = None,
    **kwargs,
) -> pl.DataFrame:
    tmp_col = col
    if not alias or alias == col:
        tmp_col = f"{col}_tmp"
        alias = col

    return df.join(
        df.select(col)
        .unique()
        .rename({col: tmp_col})
        .with_columns(
            pl.col(tmp_col)
            .map_batches(function, return_dtype=return_dtype, **kwargs)
            .alias(alias)
        ),
        left_on=col,
        right_on=tmp_col,
        how="left",
        maintain_order="left",
    )


def apply_to_unique(
    lf: pl.LazyFrame,
    fn: Callable,
    from_col: str,
    to_col: str,
    parallel=True,
    chunks_per_worker: int = 8,
    return_dtype: pl.DataType | pl.DataTypeExpr | None = pl.Object,  # ty:ignore [invalid-parameter-default]
) -> pl.LazyFrame:
    """
    Best if used on easily hashable columns, not pl.Object columns.

    for ~1M mols: 1h07m for fully linear one, 15m for this impl. (now actually <1m)
    chunksize is calculated as max(1, len(unique_molstrings) // (workers * chunks_per_worker))
    so for smaller chunks (good for larger operations), you would need more chunks per worker (800 or so)
    """
    if parallel:
        unique_molstrings = (
            lf.select(pl.col(from_col)).unique().collect().get_column(from_col)  # ty: ignore[unresolved-attribute]
        )
        workers = round(os.cpu_count() or 1 * 0.8)
        chunksize = max(1, len(unique_molstrings) // (workers * chunks_per_worker))
        with ProcessPoolExecutor(max_workers=workers) as p:
            res = track(
                p.map(fn, unique_molstrings, chunksize=chunksize),
                total=len(unique_molstrings),
            )

            mols_df = pl.DataFrame(
                [
                    pl.Series(unique_molstrings, dtype=pl.Utf8).alias(from_col),
                    pl.Series(list(res), dtype=return_dtype).alias(to_col),  # ty:ignore [invalid-argument-type]
                ]
            )
        return lf.join(
            mols_df.lazy(),
            on=from_col,
            how="left",
            maintain_order="left",
        )

    return lf.join(
        lf.select(pl.col(from_col))
        .unique()
        .with_columns(
            pl.col(from_col).map_elements(fn, return_dtype=return_dtype).alias(to_col)
        ),
        on=from_col,
        how="left",
        maintain_order="left",
    )


def map_batches_pooled(
    series: pl.Series,
    fn: Callable[[Any], Any],
    return_dtype: pl.DataType | pl.DataTypeExpr | None,
    parallel=False,
    chunks_per_worker: int = 8,
    desc=None,
) -> pl.Series:
    """ """
    if not parallel:
        return pl.Series(values=map(fn, series), dtype=return_dtype)  # ty: ignore[invalid-argument-type]
    workers = round((os.cpu_count() or 1) * 0.2)
    chunksize = max(1, len(series) // (workers * chunks_per_worker))
    with ProcessPoolExecutor(max_workers=workers) as p:
        res = list(
            track(p.map(fn, series, chunksize=chunksize), total=len(series), desc=desc)
        )
        return pl.Series(res, dtype=return_dtype)  # ty: ignore[invalid-argument-type]


def map_batches_wrap(
    series: pl.Series,
    fn: Callable[[Iterable[Any]], Iterable[Any]],
    return_dtype: pl.DataType | pl.DataTypeExpr | None,
    **kwargs,
) -> pl.Series:
    return pl.Series(values=fn(series, **kwargs), dtype=return_dtype)  # ty: ignore[invalid-argument-type]


def map_batches_wrap_multi(
    series: pl.Series,  # this will be a Struct series
    fn: Callable[[Iterable[Iterable[Any]]], Iterable[Any]],
    fields: list[str],
    return_dtype: pl.DataType | pl.DataTypeExpr | None,
    **kwargs,
) -> pl.Series:
    # Extract each field from the struct series
    extracted = [series.struct.field(f) for f in fields]
    return pl.Series(values=fn(*extracted, **kwargs), dtype=return_dtype)  # ty: ignore[invalid-argument-type]
