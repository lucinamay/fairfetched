from dataclasses import dataclass
from pathlib import Path

import chembl_downloader
import polars as pl
import pystow

from fairfetched.get._ensure import ensure_url

from ._utils import Database, file_suffix_from_url, sqlite_db_to_parquets

CHEMBL36 = Database(
    name="chembl",
    version="36",
    sources={
        "db": "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_36/chembl_36_sqlite.tar.gz"
    },
)  # @TODO: implement like this


@dataclass(frozen=True)
class SQL_db:
    path: Path

    @property
    def tables(self):
        return sqlite_db_to_parquets(self.path)


def _version_formatter(version: str | float | int | None) -> str:
    if version in [None, "latest"]:
        version = "36"  # @TODO: automate
    if isinstance(version, int | float):
        version = str(version)
    if not isinstance(version, str):
        raise ValueError(f"invalid version: {version}")

    version = version.lstrip("0")
    # for versions 22.1 and 24.1, it's important to canonicalize the version number
    # for versions < 10 it's important to left pad with a zero
    return version.replace(".", "_").zfill(2)


def _version_picker(version: str | float | int) -> Database:
    version = _version_formatter(version)
    chembl_host = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases"
    version_base = f"{chembl_host}/chembl_{version}"
    return Database(
        name="chembl",
        version=version,
        sources={
            "sql_db": f"{version_base}_sqlite.tar.gz",
        },
    )


def ensure_raw_sql(
    sources: dict[str, str] = CHEMBL36.sources,
    folder: Path | str = Path.cwd(),
) -> dict[str, Path]:
    return {
        k: ensure_url(url=v, path=Path(folder) / v.split("/")[-1])
        for k, v in sources.items()
    }


def ensure_raw(
    sources: dict[str, str] = CHEMBL36.sources,
    folder: Path | str = pystow.module("chembl").join("raw"),
    return_only_parquets: bool = True,
) -> dict[str, Path]:
    """returns a dictionary with the paths (in the same folder) of both the raw sql database file
    and the underlying tables in parquet form"""
    raw_paths = ensure_raw_sql(sources, folder)
    raw_paths.update(sqlite_db_to_parquets(raw_paths["sql_db"]))
    if return_only_parquets:
        raw_paths.pop("sql_db")
    return raw_paths


def ensure_clean(raw_parquets: dict[str, Path]) -> Path:
    raise NotImplementedError


def ensure_lfs(clean_paths: dict[str, Path]) -> dict[str, pl.LazyFrame]:
    return {k: pl.scan_parquet(v) for k, v in clean_paths.items()}


def ensure_raw_and_lfs(
    version: str,
    raw_folder: str | Path = Path.cwd(),
    parquet_folder: str | Path = Path.cwd(),
):
    chembl: Database = _version_picker(version)
    raw_paths = ensure_raw(sources=chembl.sources, folder=raw_folder)
    raw_parquets = ensure_raw_parquets(raw_paths)

    return raw_filepaths, ensure_lfs(ensure_raw_parquets)


def activity(lfs) -> pl.LazyFrame:
    return (
        # dfs["activity_properties"]
        # .join(dfs["activities"], on="activity_id", how="left")
        # .join(dfs["action_type"], on="action_type", how="left")
        # .join
        lfs["activities"]
        .join(lfs["action_type"], on="action_type", how="left", suffix="_action_type")
        .join(
            lfs["assays"], on="assay_id", how="left", suffix="_assay"
        )  # doc_ids to this
        .join(lfs["assay_type"], on="assay_type", how="left", suffix="_assay_type")
        .join(lfs, on="bao_format")
        # .join(dfs[])
    )


def components(dfs) -> pl.LazyFrame:
    return (
        dfs["component_sequences"]
        .join(
            dfs["component_class"],
            on="component_id",
            how="left",
            suffix="_class",
            validate="1:m",
        )
        .join(
            dfs["component_domains"].join(
                dfs["domains"],
                on="domain_id",
                how="left",
                suffix="_domains",
                validate="m:1",
            ),
            on="component_id",
            how="left",
            suffix="_domains",
            validate="1:m",
        )
        # .join(
        #     dfs["component_synonyms"],
        #     on="component_id",
        #     how="left",
        #     suffix="_synonyms",
        #     validate="m:m",
        # )
    )
