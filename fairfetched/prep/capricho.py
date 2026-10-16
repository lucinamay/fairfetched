"""CAPRICHO preset for ChEMBL: flags `drop_*` / `note_*` on the bioactivity view,
written once to a cache parquet that every view then scans.

Rows are flagged, not dropped; drop with
`lf.remove(pl.any_horizontal(cs.starts_with("drop_")))`, then `aggregate`.

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
import polars.selectors as cs

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
    activities: pl.LazyFrame | None,
    source: AssaySizes,
    lo: int | None,
    hi: int | None,
) -> pl.LazyFrame:
    """Add `assay_size` and `drop_assay_size` (outside [`lo`, `hi`], None = open).

    "database" counts over all of `activities` (CAPRICHO); "frame" over
    the rows of `lf`; "auto" takes "database" when `activities` is given, else "frame"
    with a warning.
    """
    if source == "auto":
        source = "database" if activities is not None else "frame"
        if source == "frame":
            _lg.warning(
                "assay_size counted over the given rows only, not the whole assay; "
                "pass `activities` for CAPRICHO's database-wide count"
            )
    if source == "database":
        if activities is None:
            raise ValueError("assay_size_source='database' needs `activities`")
        lf = lf.join(
            assay_sizes(activities),
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
    """Standardize distinct `canonical_smiles` into `standard_smiles` (salts
    removed by `clean_mixtures`) and add flags, each step as CAPRICHO 1.0.4
    `get_standardize_and_clean_workflow`:

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
    fragments = de.fragments_without_salts(parent.to_smiles(parallel))
    flags = {
        "drop_standardization_failed": pl.col(_PARENT).is_null()
        & pl.col("canonical_smiles").is_not_null(),
    }
    if drop_unassigned_chiral:
        undefined = parent.num_undefined_stereocenters(**kw) > 0
        flags["drop_undefined_stereo"] = undefined.fill_null(False)
    if not chirality:
        flags["note_stereo_removed"] = source.has_stereo(**kw).fill_null(False)
    return (
        lf.with_columns(source.standardize(*steps, **kw).alias(_PARENT))
        .with_columns(**flags, standard_smiles=fragments)
        .with_columns(
            drop_mixture=(pl.col("standard_smiles").list.len() != 1).fill_null(False),
            standard_smiles=pl.col("standard_smiles").list.join("."),
        )
        .drop(_PARENT)
    )


def flag(
    bioactivity: pl.LazyFrame,
    activities: pl.LazyFrame | None = None,
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
    convert_units: bool = False,
    engine: Engine = "polars",
) -> pl.LazyFrame:
    """`bioactivity` (columns of `Chembl.view.bioactivity`) with one pl.Boolean
    column per CAPRICHO flag.

    Defaults are those of the CAPRICHO 1.0.4 `capricho get` options. Selection
    criteria CAPRICHO applies as SQL WHERE (confidence, assay type, relation,
    units, standard type, release) become `drop_*` flags, so excluded rows stay
    inspectable; rows they flag take no part in the self-join flags, as
    CAPRICHO never fetches them. `activities` (unfiltered, e.g.
    `chembl.lfs["activities"]`) gives database-wide assay sizes, as CAPRICHO's
    separate count; the other self-join flags (cross-document duplicates, unit
    annotation errors, assay overlap) compare only the rows of `bioactivity`, as
    CAPRICHO compares only the rows it fetches by target, molecule, document or
    assay ID. A `bioactivity` filtered on other columns can therefore flag
    differently than its whole targets. `calculate_pchembl` fills a null `pchembl_value` from
    `standard_value` (`de.pchembl`); as CAPRICHO, `note_pchembl_calculated` marks
    every attempt (also those giving null) and `drop_incompatible_units` the null
    values in other units. `require_doc_date` makes rows without a year
    (`drop_missing_year`) take no part in the cross-document and overlap flags,
    as CAPRICHO removes them before those. `curate_annotation_errors` adds
    `drop_unit_annotation_error`. `mol_flags` standardizes every distinct
    structure with RDKit into `standard_smiles`, which `aggregate` needs
    (`_with_mol_flags`; ~4 ms each, ~3 h for all of ChEMBL on one process;
    `parallel` uses cpu_count // 2.2 processes); `chirality` and
    `drop_unassigned_chiral` need it. `convert_units` converts `standard_value`
    and `standard_units` (`de.convert_units`, originals in `note_unit_converted`)
    after every other flag, as CAPRICHO, except rows with
    `drop_unit_type_mismatch` (`de.has_unit_type_mismatch`, not in CAPRICHO,
    always added). Values stay float64; CAPRICHO's float32
    cast makes it miss unit annotation errors (284 of 6181 ADORA3 rows).

    CAPRICHO 1.0.4 (and main at 0ec273e) converts `--value-column`, by default
    `pchembl_value`, not `standard_value`: a pChEMBL calculated from 1 uM (6.0)
    becomes 6000.0. Here `pchembl_value` is never converted.
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
        drop_unit_type_mismatch=de.has_unit_type_mismatch(),
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
            lf, activities, assay_size_source, min_assay_size, max_assay_size
        )
        unfetched = unfetched | pl.col("drop_assay_size")
    if min_assay_overlap > 0:
        lf = de.flag_low_assay_overlap(
            lf,
            min_assay_overlap,
            skip=unfetched | undated,
            name="drop_low_assay_overlap",
        )
    if convert_units:
        lf = lf.with_columns(**de.convert_units())
    return lf


CompoundEquality = Literal["inchikey", "inchi", "connectivity", "smiles"]
MULTIVAL = (  # CAPRICHO `multiple_value_cols` and `include_metadata`, view names
    "standard_smiles",
    "canonical_smiles",
    "standard_value",
    "standard_units",
    "assay_chembl_id",
    "description_assay",
    "activity_id",
    "assay_type",
    "standard_type",
    "confidence_score",
    "organism_tgt",
    "molecule_chembl_id",
    "document_chembl_id",
    "assay_tissue",
    "assay_cell_type",
    "relationship_type",
    "doc_type",
    "doi",
    "journal",
    "year",
    "chembl_release_id",
    "activity_comment",
)
_KEY = "__key__"


def _identifier(smiles: str, equality: CompoundEquality, parallel: bool) -> pl.Expr:
    """Compound identifier from a SMILES column, as CAPRICHO `_identifier_map`."""
    from fairfetched.prep.mol_expr import MolExpr

    if equality == "smiles":
        return pl.col(smiles)
    mol = MolExpr.from_smiles(smiles, parallel, dedup=True)
    if equality == "inchi":
        return mol.to_inchi(parallel)
    inchikey = mol.to_inchikey(parallel)
    return inchikey.str.slice(0, 14) if equality == "connectivity" else inchikey


def _stats(value: str) -> dict[str, pl.Expr]:
    """`<value>_mean`, `_std`, `_median`, `_counts` per group, as CAPRICHO
    `assign_stats`: geometric mean and std (scipy `gmean`, `gstd`) of the
    pChEMBL values themselves, arithmetic otherwise; censored and single-row
    groups take the first value."""
    v = pl.col(value)
    if value == "pchembl_value":
        mean, std = v.log().mean().exp(), v.log().std().exp()
    else:
        mean, std = v.mean(), v.std(ddof=0)
    first_only = (pl.col("standard_relation").first() != "=") | (pl.len() == 1)
    first = v.first()
    return {
        f"{value}_mean": pl.when(first_only).then(first).otherwise(mean),
        f"{value}_std": pl.when(first_only).then(None).otherwise(std),
        f"{value}_median": pl.when(first_only).then(first).otherwise(v.median()),
        f"{value}_counts": pl.len(),
    }


def aggregate(
    lf: pl.LazyFrame,
    *,
    chirality: bool = True,
    compound_equality: CompoundEquality = "inchikey",
    value: str = "pchembl_value",
    extra_id_cols: Sequence[str] = (),
    extra_multival_cols: Sequence[str] = (),
    aggregate_mutants: bool = False,
    parallel: bool = False,
) -> pl.LazyFrame:
    """One row per compound, `target_chembl_id`, `standard_relation` (null as
    "="), `mutation` (null as "WT"; unless `aggregate_mutants`) and
    `extra_id_cols`, as
    CAPRICHO 1.0.4 `aggregate_data`. Needs `standard_smiles`
    (`flag(mol_flags=True)`); aggregates every row given, so drop flagged rows
    first.

    Compounds are equal by `compound_equality` of `standard_smiles`, with stereo
    removed first unless `chirality` (always for "connectivity"); censored rows
    (relation not "=") only join rows with the same relation and `value`
    (rounded to 2 decimals for pChEMBL, else 4). `MULTIVAL`, `value`, the
    `drop_*` / `note_*` flags and `extra_multival_cols` become lists in row
    order, `standard_smiles` as `processed_smiles`; `smiles` is its first entry,
    canonicalized again.
    Adds `_stats(value)`, `connectivity` (and `inchikey` / `inchi` when that is
    the equality), `shared_identifier_group` (rows sharing identifier and target
    but split by mutation or `extra_id_cols`; dense from 1, else null) and, when
    stereo is removed, `might_be_racemic` (the compound has more than one row).

    Stats run on the float values; CAPRICHO first rounds them to 2 decimals.
    """
    from fairfetched.prep.mol_expr import MolExpr

    keep_stereo = chirality and compound_equality != "connectivity"
    if not keep_stereo:
        lf = lf.with_columns(
            standard_smiles=MolExpr.from_smiles("standard_smiles", parallel, dedup=True)
            .remove_stereo(parallel, dedup=True)
            .to_smiles(parallel)
        )
    relation = pl.col("standard_relation").fill_null("=")
    decimals = 2 if value == "pchembl_value" else 4
    key = pl.col(_KEY)
    censored_key = pl.format("{}_{}_{}", key, relation, pl.col(value).round(decimals))
    ids = [*extra_id_cols, "target_chembl_id", "standard_relation"]
    multival = [value, *(c for c in MULTIVAL if c != value), *extra_multival_cols]
    if aggregate_mutants:
        multival.append("mutation")
    else:
        ids.append("mutation")
    out = (
        # key as a column first: a parallel (non-elementwise) key UDF used twice in
        # the when/then below panics polars 1.35.2 in the group_by plan
        lf.with_columns(
            **{_KEY: _identifier("standard_smiles", compound_equality, parallel)}
        )
        .with_columns(
            standard_relation=relation,
            mutation=pl.col("mutation").fill_null("WT"),  # as CAPRICHO's downloader
            **{_KEY: pl.when(relation != "=").then(censored_key).otherwise(key)},
        )
        .group_by(_KEY, *ids, maintain_order=True)
        .agg(
            *multival,
            cs.starts_with("drop_", "note_"),
            **_stats(value),
        )
        .with_columns(smiles=pl.col("standard_smiles").list.first())
        .with_columns(  # again, as RDKit canonical SMILES is not idempotent
            smiles=MolExpr.from_smiles("smiles", parallel, dedup=True).to_smiles(
                parallel
            )
        )
        .rename({"standard_smiles": "processed_smiles"})
        .with_columns(
            connectivity=_identifier("smiles", "connectivity", parallel),
            **(
                {compound_equality: _identifier("smiles", compound_equality, parallel)}
                if compound_equality in ("inchikey", "inchi")
                else {}
            ),
        )
    )
    shared = pl.len().over(compound_equality, "target_chembl_id") > 1
    out = out.with_columns(
        shared_identifier_group=pl.when(shared)
        .then(pl.struct(compound_equality, "target_chembl_id"))
        .rank("dense")
        .cast(pl.Int64)
    )
    if not keep_stereo:
        out = out.with_columns(
            might_be_racemic=pl.col(f"{value}_counts").sum().over(_KEY) > 1
        )
    return out.drop(_KEY)


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
    activities: pl.LazyFrame,
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
        flag(bioactivity, activities, **params).sink_parquet(tmp, mkdir=True)
        tmp.replace(path)
    return path
