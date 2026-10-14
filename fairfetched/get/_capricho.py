"""CAPRICHO preset for ChEMBL: flags `drop_*` / `note_*` on the bioactivity view,
written once to a cache parquet that every view then scans.

Rows are flagged, not dropped; drop with
`lf.remove(pl.any_horizontal(cs.starts_with("drop_")))`.

TODO: aggregation (CAPRICHO `aggregate_data`).
TODO: unit conversion (CAPRICHO `unit_conversions.py`).
TODO: engine="duckdb" for the self-join flags, pending `dev/bench_capricho.py`.
"""

import hashlib
import json
import logging as lg
from collections.abc import Sequence
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Literal

import polars as pl

from fairfetched.prep import data_expr as de
from fairfetched.utils import BASE_DIR

_lg = lg.getLogger(__name__)

Engine = Literal["polars", "duckdb"]
AssaySizes = Literal["auto", "database", "frame"]
CACHE_DIR = BASE_DIR / "_cache"
_PARENT = "__parent__"


def assay_sizes(activities: pl.LazyFrame) -> pl.LazyFrame:
    """`assay_id`, `assay_size`: distinct `molregno` per assay over all of
    `activities`, as CAPRICHO `get_assay_size_sql`."""
    return activities.group_by("assay_id").agg(
        assay_size=pl.col("molregno").drop_nulls().n_unique()
    )


def _selection(
    confidence_scores: Sequence[int] | None,
    assay_types: Sequence[str] | None,
    standard_relation: Sequence[str] | None,
    standard_units: Sequence[str] | None,
    standard_types: Sequence[str] | None,
    chembl_release: int | None,
) -> dict[str, pl.Expr]:
    """The criteria CAPRICHO applies as SQL WHERE; None means no criterion.
    `drop_missing_standard_value` is always applied, as in CAPRICHO."""
    criteria = {
        "drop_confidence": ("confidence_score", confidence_scores),
        "drop_assay_type": ("assay_type", assay_types),
        "drop_relation": ("standard_relation", standard_relation),
        "drop_units": ("standard_units", standard_units),
        "drop_standard_type": ("standard_type", standard_types),
    }
    out = {
        name: de.is_not_in(col, values)
        for name, (col, values) in criteria.items()
        if values is not None
    }
    out["drop_missing_standard_value"] = de.is_missing("standard_value")
    if chembl_release is not None:  # null release kept, as CAPRICHO
        out["drop_release"] = (pl.col("chembl_release_id") > chembl_release).fill_null(
            False
        )
    return out


def _with_assay_size(
    lf: pl.LazyFrame,
    lfs: dict[str, pl.LazyFrame] | None,
    source: AssaySizes,
    lo: int | None,
    hi: int | None,
) -> pl.LazyFrame:
    """Add `assay_size` and `drop_assay_size` (outside [`lo`, `hi`], None = open).

    "database" counts over all of `lfs["activities"]` (CAPRICHO); "frame" over
    the rows of `lf`; "auto" takes "database" when `lfs` is given, else "frame"
    with a warning.
    """
    if source == "auto":
        source = "database" if lfs is not None else "frame"
        if source == "frame":
            _lg.warning(
                "assay_size counted over the given rows only, not the whole assay; "
                "pass `lfs` for CAPRICHO's database-wide count"
            )
    if source == "database":
        if lfs is None:
            raise ValueError("assay_size_source='database' needs `lfs`")
        lf = lf.join(
            assay_sizes(lfs["activities"]),
            on="assay_id",
            how="left",
            maintain_order="left",
        )
    else:
        lf = lf.with_columns(assay_size=de.assay_size())
    size = pl.col("assay_size")
    too_small = size < lo if lo is not None else pl.lit(False)
    too_large = size > hi if hi is not None else pl.lit(False)
    return lf.with_columns(drop_assay_size=(too_small | too_large).fill_null(False))


def _with_mol_flags(
    lf: pl.LazyFrame, chirality: bool, drop_unassigned_chiral: bool, parallel: bool
) -> pl.LazyFrame:
    """Standardize distinct `canonical_smiles` and add flags, each step as
    CAPRICHO 1.0.4 `get_standardize_and_clean_workflow`:

    - chemFilters `ChemStandardizer(method="chembl")` (`STEPS_CAPRICHO`):
      chembl_structure_pipeline `get_parent_mol`, then `standardize_mol`.
      Stereo is removed unless `chirality` (CAPRICHO `--no-chirality`).
    - `drop_standardization_failed`: no structure comes out (CAPRICHO drops
      the row).
    - `drop_mixture`: not exactly one fragment left after CAPRICHO's
      `clean_mixtures` (`de.fragments_without_salts`), so also salt-only
      structures; ChEMBL's own salt list misses e.g. charge-separated
      perchlorate.
    - `drop_undefined_stereo` (with `drop_unassigned_chiral`): CAPRICHO
      `find_undefined_stereocenters`.
    - `note_stereo_removed` (without `chirality`): stereo specified in
      `canonical_smiles`, CAPRICHO `flag_stereochemistry_removal`.

    `parallel` spreads RDKit over processes.
    """
    from fairfetched.prep.mol_expr import MolExpr
    from fairfetched.prep.mol_functions import remove_stereo
    from fairfetched.prep.pipeline import STEPS_CAPRICHO

    kw = {"parallel": parallel, "dedup": True}
    source = MolExpr.from_smiles("canonical_smiles", **kw)
    steps = STEPS_CAPRICHO if chirality else [*STEPS_CAPRICHO, remove_stereo]
    parent = MolExpr.col(_PARENT)
    flags = {
        "drop_standardization_failed": pl.col(_PARENT).is_null()
        & pl.col("canonical_smiles").is_not_null(),
        "drop_mixture": (
            de.fragments_without_salts(parent.to_smiles(parallel)).list.len() != 1
        ).fill_null(False),
    }
    if drop_unassigned_chiral:
        undefined = parent.num_undefined_stereocenters(**kw) > 0
        flags["drop_undefined_stereo"] = undefined.fill_null(False)
    if not chirality:
        flags["note_stereo_removed"] = source.has_stereo(**kw).fill_null(False)
    return (
        lf.with_columns(source.standardize(*steps, **kw).alias(_PARENT))
        .with_columns(**flags)
        .drop(_PARENT)
    )


def flag(
    bioactivity: pl.LazyFrame,
    lfs: dict[str, pl.LazyFrame] | None = None,
    *,
    confidence_scores: Sequence[int] | None = (7, 8, 9),
    assay_types: Sequence[str] | None = ("B", "F"),
    standard_relation: Sequence[str] | None = ("=",),
    standard_units: Sequence[str] | None = None,
    standard_types: Sequence[str] | None = None,
    chembl_release: int | None = None,
    min_assay_size: int | None = None,
    max_assay_size: int | None = None,
    assay_size_source: AssaySizes = "auto",
    min_assay_overlap: int = 0,
    strict_mutant_removal: bool = False,
    calculate_pchembl: bool = False,
    require_doc_date: bool = False,
    curate_annotation_errors: bool = True,
    mol_flags: bool = False,
    chirality: bool = True,
    drop_unassigned_chiral: bool = False,
    parallel: bool = False,
    engine: Engine = "polars",
) -> pl.LazyFrame:
    """`bioactivity` (columns of `Chembl.view.bioactivity`) with one pl.Boolean
    column per CAPRICHO flag.

    Defaults are those of the CAPRICHO 1.0.4 `capricho get` options. Selection
    criteria CAPRICHO applies as SQL WHERE (confidence, assay type, relation,
    units, standard type, release) become `drop_*` flags, so excluded rows stay
    inspectable; rows they flag take no part in the self-join flags, as
    CAPRICHO never fetches them. `lfs` (the raw tables) gives database-wide
    assay sizes. `calculate_pchembl` fills a null `pchembl_value` from
    `standard_value` (`de.pchembl`); as CAPRICHO, `note_pchembl_calculated` marks
    every attempt (also those giving null) and `drop_incompatible_units` the null
    values in other units. `require_doc_date` makes rows without a year
    (`drop_missing_year`) take no part in the cross-document and overlap flags,
    as CAPRICHO removes them before those. `curate_annotation_errors` adds
    `drop_unit_annotation_error`. `mol_flags` standardizes every distinct
    structure with RDKit (`_with_mol_flags`; ~4 ms each, ~3 h for all of ChEMBL
    on one process; `parallel` uses cpu_count // 2.2 processes); `chirality` and
    `drop_unassigned_chiral` need it. Values stay float64; CAPRICHO's float32
    cast makes it miss unit annotation errors (284 of 6181 ADORA3 rows).
    """
    if engine != "polars":
        raise NotImplementedError(f"engine={engine!r}; only 'polars' so far")
    if not mol_flags and (drop_unassigned_chiral or not chirality):
        raise ValueError("chirality=False and drop_unassigned_chiral need mol_flags")
    selection = _selection(
        confidence_scores,
        assay_types,
        standard_relation,
        standard_units,
        standard_types,
        chembl_release,
    )
    unfetched = pl.any_horizontal(pl.lit(False), *selection)
    if standard_types is not None and calculate_pchembl:
        # CAPRICHO also fetches "Log <type>" and "-Log <type>": they take part in
        # its steps on the whole fetch (pChEMBL, unit errors, overlap), then it
        # drops them before standardizing (`get_standardize_and_clean_workflow`)
        logged = [f"{sign}Log {t}" for t in standard_types for sign in ("", "-")]
        unfetched = pl.any_horizontal(
            pl.lit(False),
            *(name for name in selection if name != "drop_standard_type"),
            de.is_not_in("standard_type", [*standard_types, *logged]),
        )
    if calculate_pchembl:
        missing, convertible = de.is_missing("pchembl_value"), ~de.has_non_molar_units()
        bioactivity = bioactivity.with_columns(
            pchembl_value=pl.coalesce("pchembl_value", de.pchembl()),
            note_pchembl_calculated=missing & convertible,
            drop_incompatible_units=missing & ~convertible,
        )
    undated = de.is_missing("year") if require_doc_date else pl.lit(False)
    removed = (
        de.is_missing("pchembl_value") | de.is_missing("canonical_smiles") | undated
    )
    if standard_types is not None:
        removed = removed | pl.col("drop_standard_type")
    if mol_flags:
        bioactivity = _with_mol_flags(
            bioactivity, chirality, drop_unassigned_chiral, parallel
        )
        removed = removed | pl.col("drop_standardization_failed")
    lf = bioactivity.with_columns(
        **selection
    ).with_columns(
        drop_validity_comment=de.has_validity_comment(),
        drop_potential_duplicate=de.is_potential_duplicate(),
        drop_missing_value=de.is_missing("pchembl_value"),
        drop_missing_year=de.is_missing("year"),
        drop_missing_structure=de.is_missing("canonical_smiles"),
        drop_censored_comment=de.is_censored_by_comment(),
        note_salt_removed=pl.col("canonical_smiles").str.contains(".", literal=True),
        note_cross_document=de.is_cross_document_duplicate(  # over the rows CAPRICHO keeps
            [
                *(k for k in de.DUPLICATE_KEYS if k != "mutation"),
                pl.col("mutation").fill_null("WT"),  # as CAPRICHO's downloader
            ],
            skip=unfetched | removed,
        ),
    )
    if strict_mutant_removal:
        lf = lf.with_columns(drop_mutant=de.mentions_mutant())
    if curate_annotation_errors:
        lf = de.flag_unit_annotation_errors(
            lf, skip=unfetched, name="drop_unit_annotation_error"
        )
    if min_assay_size is not None or max_assay_size is not None:
        lf = _with_assay_size(
            lf, lfs, assay_size_source, min_assay_size, max_assay_size
        )
        unfetched = unfetched | pl.col("drop_assay_size")
    if min_assay_overlap > 0:
        lf = de.flag_low_assay_overlap(
            lf,
            min_assay_overlap,
            skip=unfetched | undated,
            name="drop_low_assay_overlap",
        )
    return lf


def cache_path(version: str, params: dict, cache_dir: Path | str | None = None) -> Path:
    """`<cache_dir>/chembl/<version>/capricho/<hash>.parquet`, hashing `params`
    and the fairfetched version.

    ponytail: a flag-logic change without a version bump reuses a stale cache;
    delete the folder (or bump the version) after editing `flag`.
    """
    key = json.dumps(
        {"params": params, "fairfetched": package_version("fairfetched")},
        sort_keys=True,
    )
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    root = Path(cache_dir) if cache_dir is not None else CACHE_DIR
    return root / "chembl" / version / "capricho" / f"{digest}.parquet"


def ensure_flagged(
    bioactivity: pl.LazyFrame,
    lfs: dict[str, pl.LazyFrame],
    version: str,
    cache_dir: Path | str | None = None,
    **params,
) -> Path:
    """Sink `flag(...)` to `cache_path` unless it exists; return the path.

    Written to a temporary file first, so an interrupted run leaves no cache.
    """
    path = cache_path(version, params, cache_dir)
    if not path.exists():
        tmp = path.with_suffix(".tmp")
        flag(bioactivity, lfs, **params).sink_parquet(tmp, mkdir=True)
        tmp.replace(path)
    return path
