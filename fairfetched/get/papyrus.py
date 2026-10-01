"""Papyrus dataset utilities for downloading, cleaning, and joining bioactivity and protein data.

This module provides functions to ensure the presence of raw and cleaned Papyrus dataset files,
and defines the Papyrus_57 database configuration.
"""

from functools import partial
from pathlib import Path
from typing import Any

import polars as pl

from fairfetched.utils import (
    BASE_DIR,
    ensure_url,
    file_suffix_from_url,
    lowercase_columns,
    tables,
)
from fairfetched.utils.tables import scan_raw
from fairfetched.utils.typing import BioactivityDBViews

PAPYRUS_VERSIONS: dict[str, dict[str, str]] = {
    "05.6": {
        "bioactivity": "https://zenodo.org/records/7373214/files/05.6_combined_set_with_stereochemistry.tsv.xz",
        # "bioactivity_nostereochemistry": "https://zenodo.org/records/7373214/files/05.6_combined_set_without_stereochemistry.tsv.xz",
        "readme": "https://zenodo.org/records/7373214/files/README.txt",
        "protein": "https://zenodo.org/records/7373214/files/05.6_combined_set_protein_targets.tsv.xz",
    },
    "05.7": {
        "bioactivity": "https://zenodo.org/records/13987985/files/05.7_combined_set_with_stereochemistry.tsv.xz",
        # "bioactivity_nostereochemistry": "https://zenodo.org/records/13987985/files/05.7_combined_set_without_stereochemistry.tsv.xz",
        "readme": "https://zenodo.org/records/13987985/files/README.txt",
        "protein": "https://zenodo.org/records/13987985/files/05.7_combined_set_protein_targets.tsv.xz",
    },
}


def available_versions() -> tuple[str, ...]:
    return tuple(PAPYRUS_VERSIONS.keys())


def latest() -> str:
    return available_versions()[-1]


def source_urls(version: str) -> dict[str, str]:
    return PAPYRUS_VERSIONS[str(version)]


def ensure_raw_files(
    version: str, raw_dir: Path | str | Any | None = None
) -> dict[str, Path]:
    """Download if missing, return path to raw file."""
    if raw_dir is None:
        raw_dir = BASE_DIR / "papyrus" / version / "raw"
    raw_dir = Path(raw_dir)

    return {
        name: ensure_url(url=url, path=raw_dir / f"{name}{file_suffix_from_url(url)}")
        for name, url in source_urls(version).items()
    }


def _scan_raw(name: str, path: Path) -> pl.LazyFrame:
    lf = scan_raw(
        path,
        infer_schema=False,
        schema_overrides={
            "Year": pl.Int32,
            "pchembl_value_Mean": pl.Float64,
            "pchembl_value_StdDev": pl.Float64,
            "pchembl_value_SEM": pl.Float64,
            # because of stray floats in pchembl_value_N, we do float -> int
            "pchembl_value_N": pl.Float64,
            "pchembl_value_Median": pl.Float64,
            "pchembl_value_MAD": pl.Float64,
        },
        null_values=["NA", ""],
    )
    if name == "bioactivity":
        lf = lf.cast({"pchembl_value_N": pl.Int64})
    return lf


# the table_dir default and README skip live in tables.ensure_parquet_tables
ensure_parquet_tables = partial(tables.ensure_parquet_tables, scanner=_scan_raw)


def _clean(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Lowercase columns & protein table's ``uniprotid`` => ``uniprot_id``"""
    return lf.pipe(lowercase_columns).rename({"uniprotid": "uniprot_id"}, strict=False)


cleanly_scan_parquet_tables = partial(tables.scan_parquets, clean=_clean)


def build_views(parquet_paths: dict[str, Path]) -> BioactivityDBViews:
    """Build joined domain views from the scanned source tables."""
    lfs = cleanly_scan_parquet_tables(parquet_paths)
    return {
        "bioactivity": lfs["bioactivity"],
        "compounds": lfs["bioactivity"]
        .drop(
            "activity_id",
        )
        .unique(("inchikey", "inchi")),  # no 'connectivity' for with-stereo-papyrus
        "full": lfs["bioactivity"].join(
            lfs["protein"],
            on="target_id",
            how="left",
            maintain_order="left",
            validate="m:1",  # one unique protein only from right, can reoccur within compounds.
        ),
        "proteins": lfs["protein"],
    }


def help() -> None:
    """prints out example usage"""
    print("""
        Example usage:
        ```
        raw = ensure_raw_files("05.7")
        tables = cleanly_scan_parquet_tables(raw)
        views = build_views(tables)
        views["full"].sink_parquet("output.parquet")
        ```
        """)
