from pathlib import Path
from typing import Any

import polars as pl

from fairfetched.utils import (
    BASE_DIR,
    ensure_sqlite_db_to_parquets,
    ensure_untarred_sqlite,
    ensure_url,
    file_suffix_from_url,
    lowercase_columns,
)
from fairfetched.utils.typing import ComposedLFDict

CHEMBL_DIR = BASE_DIR / "chembl"


def _version_formatter(version: int | float | str) -> str:
    if isinstance(version, int | float):
        version = str(version)
    if not isinstance(version, str):
        try:
            version = str(version)
        finally:
            raise TypeError(f"invalid version type: {type(version)}")

    version = version.lstrip("0")
    # for canonicalize the version number 22.1 and 24.1 and left pad with a zero if needed
    return version.replace(".", "_").zfill(2)


def _version_to_url(version: str):
    base = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases"
    return f"{base}/chembl_{version}/chembl_{version}_sqlite.tar.gz"


CHEMBL_VERSIONS: dict[str, dict[str, str]] = {
    version: {"sql_db": _version_to_url(version)}
    for version in sorted(
        map(_version_formatter, list(range(1, 37)) + ["24_1", "22_1"])
    )
}


def available_versions() -> tuple[str, ...]:
    return tuple(CHEMBL_VERSIONS.keys())


def latest() -> str:
    return available_versions()[-1]


def get_sources(version: str) -> dict[str, str]:
    return CHEMBL_VERSIONS[str(version)]


def ensure_raw(version: str, raw_dir: Path | str | None = None) -> dict[str, Path]:
    """downloads the original sql database, with its original name and compression"""
    if raw_dir is None:
        raw_dir = CHEMBL_DIR / version
    raw_dir = Path(raw_dir)
    return {
        name: ensure_url(url=url, path=raw_dir / f"{file_suffix_from_url(url)}")
        for name, url in get_sources(version).items()
    }


def ensure_consolidated(
    raw_paths: dict[str, Path], consolidated_dir: Path | str | Any | None = None
) -> dict[str, Path]:
    sql_tar_gz_path = raw_paths["sql_db"]
    if consolidated_dir is None:
        consolidated_dir = Path(sql_tar_gz_path).parent / "extracted"
    consolidated_dir = Path(consolidated_dir)
    consolidated_dir.mkdir(exist_ok=True, parents=True)

    raw_sql = ensure_untarred_sqlite(sql_tar_gz_path)
    # the untarred should also stay so that we have access.....
    # #@TODO: perhaps make tables deterministic for chembl to circumvent
    parquets = ensure_sqlite_db_to_parquets(
        raw_sql, cache_dir=consolidated_dir, force=False
    )

    return parquets


def clean(extracted_sqlite_parquet_paths: dict[str, Path]) -> dict[str, pl.LazyFrame]:
    return {
        name: pl.scan_parquet(path_)
        .pipe(lowercase_columns)
        .fill_nan(None)
        .with_columns(
            pl.col(pl.String).replace({"": None}),
        )
        for name, path_ in extracted_sqlite_parquet_paths.items()
    }


def compose(lfs: dict[str, pl.LazyFrame]) -> ComposedLFDict:
    """Join/combine lazy frames. Optional, returns single LF."""
    return {
        "bioactivity": _bioactivities(lfs),
        "compounds": _compounds(lfs),
        "proteins": lfs["protein"],
        "components": _components(lfs),
    }


def _bioactivities(lfs: dict[str, pl.LazyFrame]) -> pl.LazyFrame:
    return (
        # dfs["activity_properties"]
        # .join(dfs["activities"], on="activity_id", how="left")
        # .join(dfs["action_type"], on="action_type", how="left")
        # .join
        lfs["bioactivity"]
        .join(
            lfs["protein"],
            on="target_id",
            how="left",
            maintain_order="left",
            validate="m:1",  # one unique protein only from right, can reoccur within compounds.
        )
        .join(lfs["action_type"], on="action_type", how="left", suffix="_action_type")
        .join(
            lfs["assays"], on="assay_id", how="left", suffix="_assay"
        )  # doc_ids to this
        .join(lfs["assay_type"], on="assay_type", how="left", suffix="_assay_type")
        # .join(lfs, on="bao_format")
        # .join(dfs[])
    )


def _components(lfs: dict[str, pl.LazyFrame]) -> pl.LazyFrame:
    return (
        lfs["component_sequences"]
        .join(
            lfs["component_class"],
            on="component_id",
            how="left",
            suffix="_class",
            validate="1:m",
        )
        .join(
            lfs["component_domains"].join(
                lfs["domains"],
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


def _compounds(lfs) -> pl.LazyFrame:
    return (
        lfs["molecule_dictionary"]
        .join(
            lfs["compound_properties"],
            on="molregno",
            how="left",
            suffix="_compound_properties",
            validate="1:1",
        )
        .join(
            lfs["compound_structures"],
            on="molregno",
            how="left",
            suffix="_compound_structures",
            validate="1:1",
        )
        .join(
            lfs["compound_records"].join(
                lfs["docs"], on="doc_id", how="left", suffix="_doc", validate="m:1"
            ),
            on="molregno",
            how="left",
            suffix="_compound_records",
            validate="1:m",
        )
        .join(
            lfs["compound_structural_alerts"],
            #     .join(
            #     dfs["docs"], on="doc_id", how="left", suffix="_doc",validate="m:1"
            # )
            on="molregno",
            how="left",
            suffix="_compound_structural_alerts",
            validate="1:m",
        )
    )
