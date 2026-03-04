import logging as lg
import lzma
import os
import re
import sqlite3
import tarfile
import tempfile
from pathlib import Path
from typing import Any, NotRequired, TypedDict

import polars as pl


def _get_fairfetched_home_dir() -> Path:
    if fairfetched_name := os.environ.get("FAIRFETCHED_NAME", None):
        lg.debug(f"FAIRFETCHED_NAME as base directory: {fairfetched_name}")
        return Path(fairfetched_name)
    elif pystow_name := os.environ.get("PYSTOW_NAME", None):
        lg.debug(f"using PYSTOW_NAME as base directory: {pystow_name}")
        return Path(pystow_name)
    return Path.home() / ".data"


BASE_DIR = _get_fairfetched_home_dir()


class ComposedLFDict(TypedDict):
    bioactivity: pl.LazyFrame
    compounds: pl.LazyFrame
    full: NotRequired[pl.LazyFrame]
    proteins: NotRequired[pl.LazyFrame]
    components: NotRequired[pl.LazyFrame]




# ComposedDict = TypedDict(
#     "ComposedDict",
#     {
#         "bioactivity": pl.LazyFrame,
#         "compounds": pl.LazyFrame,
#         "proteins": NotRequired[pl.LazyFrame],
#     },
#     total=False,
# )


# ==== import and basic cleaning ====
def scan_tsvxz(path, **kwargs) -> pl.LazyFrame:
    with lzma.open(str(path), "rb") as f_lzma:
        return pl.scan_csv(f_lzma, **kwargs)


def lowercase_columns(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.select(pl.all().name.to_lowercase())


def file_suffix_from_url(url: str) -> str:
    match_ = re.search(
        r"\.\w{1,5}(?:\.\w{1,5})?$", url.split("?download=1")[0].split("/")[-1]
    )
    if match_ is None:
        raise RuntimeError(f"Could not extract suffix from url: {url}")
    return match_.group(0)


def untar_sqlite(tar_gz_path: str | Path) -> Path:
    files: list[Path] = []
    with tarfile.open(tar_gz_path, mode="r", encoding="utf-8") as tar_file:
        for tar_subfile in tar_file:
            if not tar_subfile.name.endswith(".db"):
                continue
            targetpath = Path(tar_gz_path).parent / tar_subfile.name.split("/")[-1]
            lg.info(f"unarchiving {tar_gz_path} to {targetpath}")
            files.append(targetpath)
            if not targetpath.exists():
                tar_file._extract_member(tar_subfile, str(targetpath))
    assert len(files) == 1
    return files[0]


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


def sqlite_db_to_parquets(
    db_path: str | Path, cache_dir: str | Path | None = None
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
    for t in tables:
        path_out = Path(cache_dir) / f"{t}.parquet"
        out[t] = path_out
        if not path_out.exists():
            pl.read_database(
                f'SELECT * FROM "{t}"', conn, infer_schema_length=None
            ).write_parquet(path_out)

    conn.close()  # 5m37s
    return out


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


def __is_pystow_module(obj):
    """Dynamically check for pystow.Module by name"""
    return (
        obj.__class__.__module__ == "pystow.impl" and obj.__class__.__name__ == "Module"
    )


def _to_path(obj: Path | str | Any) -> Path:
    if isinstance(obj, Path):
        return obj
    if isinstance(obj, str):
        return Path(obj)
    if __is_pystow_module(obj):
        return obj.base

    try:
        return Path(obj)
    except Exception:
        raise TypeError(f"Cannot convert {type(obj)} to Path")
