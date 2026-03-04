"""Papyrus dataset utilities for downloading, cleaning, and joining bioactivity and protein data.

This module provides functions to ensure the presence of raw and cleaned Papyrus dataset files,
and defines the Papyrus_57 database configuration.
"""

from pathlib import Path
from typing import Any

import polars as pl

from fairfetched.utils import (
    BASE_DIR,
    ensure_url,
    file_suffix_from_url,
    lowercase_columns,
    scan_tsvxz,
)
from fairfetched.utils.typing import ComposedLFDict

PAPYRUS_VERSIONS: dict[str, dict[str, str]] = {
    "05.7": {
        "bioactivity": "https://zenodo.org/records/13987985/files/05.7_combined_set_without_stereochemistry.tsv.xz?download=1",
        "readme": "https://zenodo.org/records/13987985/files/README.txt?download=1",
        "protein": "https://zenodo.org/records/13987985/files/05.7_combined_set_protein_targets.tsv.xz?download=1",
    },
    "05.6": {
        "bioactivity": "https://zenodo.org/records/7373214/files/05.6_combined_set_without_stereochemistry.tsv.xz?download=1",
        "readme": "https://zenodo.org/records/7373214/files/README.txt?download=1",
        "protein": "https://zenodo.org/records/7373214/files/05.6_combined_set_protein_targets.tsv.xz?download=1",
    },
}


def available_versions() -> tuple[str, ...]:
    return tuple(PAPYRUS_VERSIONS.keys())


def latest() -> str:
    return available_versions()[-1]


def get_sources(version: str) -> dict[str, str]:
    return PAPYRUS_VERSIONS[str(version)]


def ensure_raw(
    version: str, cache_dir: Path | str | Any | None = None
) -> dict[str, Path]:
    """Download if missing, return path to raw file."""
    if cache_dir is None:
        cache_dir = BASE_DIR / "papyrus" / version
    cache_dir = Path(cache_dir)

    return {
        name: ensure_url(url=url, path=cache_dir / f"{file_suffix_from_url(url)}")
        for name, url in get_sources(version).items()
    }


def clean(raw_filepath_dict: dict[str, Path]) -> dict[str, pl.LazyFrame]:
    """Transform raw → clean, return lazy frames. No I/O."""
    schema_overrides = {
        "Year": pl.Int32,
        "pchembl_value_Mean": pl.Float64,
        "pchembl_value_StdDev": pl.Float64,
        "pchembl_value_SEM": pl.Float64,
        "pchembl_value_N": pl.Float64,
        "pchembl_value_Median": pl.Float64,
        "pchembl_value_MAD": pl.Float64,
    }

    return {
        "protein": (
            scan_tsvxz(raw_filepath_dict["protein"], separator="\t")
            .pipe(lowercase_columns)
            .rename({"uniprotid": "uniprot_id"})
        ),
        "bioactivity": (
            scan_tsvxz(
                raw_filepath_dict["bioactivity"],
                separator="\t",
                infer_schema=False,
                schema_overrides=schema_overrides,
                null_values="NA",  # which values to take as None
            )
            .pipe(lowercase_columns)
            .cast({"pchembl_value_n": pl.Int64})
        ),
    }


def compose(lfs: dict[str, pl.LazyFrame]) -> ComposedLFDict:
    """Join/combine lazy frames. Optional, returns single LF."""
    return {
        "bioactivity": lfs["bioactivity"].join(
            lfs["protein"],
            on="target_id",
            how="left",
            maintain_order="left",
            validate="m:1",  # one unique protein only from right, can reoccur within compounds.
        ),
        "compounds": lfs["bioactivity"]
        .drop(
            "activity_id",
        )
        .unique(("connectivity", "inchikey", "inchi")),
        "full": lfs["bioactivity"].join(
            lfs["protein"],
            on="target_id",
            how="left",
            maintain_order="left",
            validate="m:1",  # one unique protein only from right, can reoccur within compounds.
        ),
    }


def help() -> None:
    """prints out example usage"""
    print("""
        Example usage:
        ```
        raw = ensure_raw("05.7")
        lfs = clean(raw)
        final_lf = compose(lfs)
        final_lf.sink_parquet("output.parquet")
        ```
        """)
