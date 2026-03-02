"""Papyrus dataset utilities for downloading, cleaning, and joining drug and protein data.

This module provides functions to ensure the presence of raw and cleaned Papyrus dataset files,
and defines the Papyrus_57 database configuration.
"""

from pathlib import Path

import polars as pl
import pystow  # @TODO: remove pystow dependency

from fairfetched.get._utils import (
    Database,
    file_suffix_from_url,
    lowercase_columns,
    scan_tsvxz,
)

PAPYRUS_57 = Database(
    name="papyrus",
    version="05.7",
    sources={
        "drug": "https://zenodo.org/records/13987985/files/05.7_combined_set_without_stereochemistry.tsv.xz?download=1",
        "readme": "https://zenodo.org/records/13987985/files/README.txt?download=1",
        "protein": "https://zenodo.org/records/13987985/files/05.7_combined_set_protein_targets.tsv.xz?download=1",
    },
)


def _version_picker(version: str = "latest") -> Database:
    match version:
        case "05.7" | "latest":
            return PAPYRUS_57
        case _:
            raise ValueError("version not (yet) implemented")


def ensure_raw(
    sources: dict[str, str] = PAPYRUS_57.sources,
    pystow_module: pystow.Module = pystow.module("papyrus", "raw"),
) -> dict[str, Path]:
    """
    Download and ensure the presence of raw Papyrus dataset files.

    Args:
        sources (dict[str, str]): Mapping of file keys to download URLs.
        pystow_module (pystow.Module): Pystow module for storage.

    Returns:
        dict[str, Path]: Mapping of file keys to local file paths.
    """

    return {
        k: pystow_module.ensure("papyrus", name=f"{k}{file_suffix_from_url(v)}", url=v)
        for k, v in sources.items()
    }


def ensure_clean(
    raw_filepath_dict: dict[str, Path],
    clean_filepath: Path | str = pystow.module("papyrus").join("clean")
    / "papyrus_clean.parquet",
) -> Path:
    """
    Ensure the presence of a cleaned and joined Papyrus dataset as a Parquet file.
    Data is not changed, only formats and datatypes (and None values)
    Downloads raw files if needed, processes and joins drug and protein data,
    and saves the result as a Parquet file.

    Args:
    ---
        raw_filepath_dict (dict[str, Path]): a dictionary of paths to each raw file as returned by `ensure_raw`
        pystow_module (pystow.Module): Pystow module for storage.

    Returns:
    ---
        Path: Path to the cleaned Parquet data file.

    Example:
    ---
        ```
        raw_paths = ensure_raw(
            PAPYRUS_57.sources, pystow.module("papyrus", "raw")
        )
        clean_path = ensure_clean(raw_paths, clean_papyrus.parquet)

        # pl.read_parquet(clean_path).pipe(mol_standardisation_fnx) etc...
        ```
    """

    data_file = Path(clean_filepath)
    if not data_file.exists():
        schema_overrides = {
            "Year": pl.Int32,
            "pchembl_value_Mean": pl.Float64,
            "pchembl_value_StdDev": pl.Float64,
            "pchembl_value_SEM": pl.Float64,
            "pchembl_value_N": pl.Float64,
            "pchembl_value_Median": pl.Float64,
            "pchembl_value_MAD": pl.Float64,
        }

        proteins = (
            scan_tsvxz(raw_filepath_dict["protein"], separator="\t")
            .pipe(lowercase_columns)
            .rename({"uniprotid": "uniprot_id"})
        )
        compounds = (
            scan_tsvxz(
                raw_filepath_dict["drug"],
                separator="\t",
                infer_schema=False,
                schema_overrides=schema_overrides,
                null_values="NA",  # which values to take as None
            )
            .pipe(lowercase_columns)
            .cast({"pchembl_value_n": pl.Int64})
        )

        compounds.join(
            proteins,
            on="target_id",
            how="left",
            maintain_order="left",
            validate="m:1",  # one unique protein only from right, can reoccur within compounds.
        ).sink_parquet(data_file)
    return data_file


def ensure_raw_and_clean(
    version: str = "05.7",
    pystow_module_raw: pystow.Module = pystow.module("papyrus", "raw"),
    clean_filepath: Path | str = pystow.module("papyrus").join("clean"),
) -> tuple[dict[str, Path], Path]:
    papyrus: Database = _version_picker(version)
    raw_filepaths = ensure_raw(sources=papyrus.sources, pystow_module=pystow_module_raw)
    return raw_filepaths, ensure_clean(raw_filepaths, clean_filepath)
