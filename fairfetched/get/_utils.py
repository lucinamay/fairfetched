import lzma
import re
import sqlite3
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import polars as pl

is_lazy = partial(isinstance, pl.LazyFrame)


@dataclass(frozen=True)
class Database:
    name: str
    version: str
    sources: dict[str, str]

    @property
    def file_names(self) -> list[str]:
        return list(self.sources.keys())

    @property
    def file_urls(self) -> list[str]:
        return list(self.sources.values())


# ==== import and basic cleaning ====
def scan_tsvxz(path, **kwargs):
    with lzma.open(str(path), "rb") as f_lzma:
        return pl.scan_csv(f_lzma, **kwargs)


def lowercase_columns(df):
    return df.select(pl.all().name.to_lowercase())


def file_suffix_from_url(url: str) -> str:
    match_ = re.search(
        r"\.\w{1,5}(?:\.\w{1,5})?$", url.split("?download=1")[0].split("/")[-1]
    )
    if match_ is None:
        raise RuntimeError(f"Could not extract suffix from url: {url}")
    return match_.group(0)


def _sqlite_tables(db_path: str | Path) -> list[str]:
    with sqlite3.connect(db_path):
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


def sqlite_db_to_parquets(db_path: str | Path) -> dict[str, Path]:
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
        path_out = Path(db_path).parent / f"{t}.parquet"
        out[t] = path_out
        if not path_out.exists():
            pl.read_database(
                f'SELECT * FROM "{t}"', conn, infer_schema_length=None
            ).write_parquet(path_out)

    conn.close()  # 5m37s
    return out
