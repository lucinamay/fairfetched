"""CAPRICHO preset for ChEMBL: flags `drop_*` / `note_*` on the bioactivity view,
written once to a cache parquet that every view then scans.

TODO: aggregation (CAPRICHO `aggregate_data`).
TODO: unit conversion (CAPRICHO `unit_conversions.py`).
TODO: pChEMBL calculation from `standard_value` (CAPRICHO `convert_to_log10`).
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import polars as pl

from fairfetched.utils import BASE_DIR

Engine = Literal["polars", "duckdb"]
CACHE_DIR = BASE_DIR / "_cache"


def assay_sizes(activities: pl.LazyFrame) -> pl.LazyFrame:
    """`assay_id`, `assay_size`: distinct `molregno` per assay over all of
    `activities`, as CAPRICHO `get_assay_size_sql`."""
    raise NotImplementedError


def flag(
    lfs: dict[str, pl.LazyFrame],
    bioactivity: pl.LazyFrame,
    *,
    confidence_scores: Sequence[int] | None = (7, 8, 9),
    assay_types: Sequence[str] | None = ("B", "F"),
    standard_relation: Sequence[str] | None = ("=",),
    standard_units: Sequence[str] | None = None,
    min_assay_size: int | None = None,
    max_assay_size: int | None = None,
    min_assay_overlap: int = 0,
    strict_mutant_removal: bool = False,
    chembl_release: int | None = None,
    engine: Engine = "polars",
) -> pl.LazyFrame:
    """`bioactivity` with one pl.Boolean column per CAPRICHO flag.

    Defaults are those of the CAPRICHO CLI (`cli/main.py` DEFAULTS). Selection
    criteria CAPRICHO applies as SQL WHERE (confidence, assay type, relation,
    units, release) become `drop_*` flags, so excluded rows stay inspectable;
    the self-join flags skip rows already flagged for dropping.
    """
    raise NotImplementedError


def cache_path(version: str, params: dict) -> Path:
    """`CACHE_DIR/chembl/<version>/capricho/<hash of params>.parquet`."""
    raise NotImplementedError


def ensure_flagged(
    lfs: dict[str, pl.LazyFrame], bioactivity: pl.LazyFrame, version: str, **params
) -> Path:
    """Sink `flag(...)` to `cache_path` unless it exists; return the path."""
    raise NotImplementedError
