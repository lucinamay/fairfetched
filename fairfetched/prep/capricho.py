"""CAPRICHO 1.0.4 for ChEMBL: `flag` adds `drop_*` / `note_*` columns, `aggregate`
collapses the rows left after `lf.remove(pl.any_horizontal(cs.starts_with("drop_")))`.
"""

import hashlib
import inspect
import json
import os
from collections.abc import Sequence
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Literal

import polars as pl
import polars.selectors as cs

from fairfetched.prep import data_expr as de
from fairfetched.utils import BASE_DIR

AssaySizes = Literal["database", "frame"]
CACHE_DIR = BASE_DIR / "_cache"
_PARENT = "__parent__"


def assay_sizes(activities: pl.LazyFrame) -> pl.LazyFrame:
    """`assay_id`, `assay_size` (distinct `molregno`), as CAPRICHO `get_assay_size_sql`."""
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
    """CAPRICHO's SQL WHERE criteria as `drop_*` flags; None skips one."""
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
    """Add `assay_size` and `drop_assay_size` (outside [`lo`, `hi`], None = open),
    counted over all of `activities` ("database", CAPRICHO) or the rows of `lf`."""
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
    inside = pl.col("assay_size").is_between(
        lo or 0, float("inf") if hi is None else hi
    )
    return lf.with_columns(drop_assay_size=~inside.fill_null(True))


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
    assay_size_source: AssaySizes = "database",
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
) -> pl.LazyFrame:
    """`bioactivity` (`Chembl.view.bioactivity` columns) with one column per
    CAPRICHO 1.0.4 flag (pl.Boolean; `note_unit_converted` a struct); defaults
    are those of `capricho get`.

    CAPRICHO's SQL WHERE criteria become `drop_*` flags, and rows they flag take
    no part in the self-join flags (cross-document duplicates, unit annotation
    errors, assay overlap). Those compare only the rows of `bioactivity`, as
    CAPRICHO compares the rows it fetches by target, molecule, document or assay,
    so filtering on other columns changes them. Assay sizes count over
    `activities` (unfiltered `chembl.lfs["activities"]`).

    - `calculate_pchembl`: fill a null `pchembl_value` from `standard_value`
      (`de.pchembl`); `note_pchembl_calculated` marks every attempt,
      `drop_incompatible_units` the non-molar ones.
    - `require_doc_date`: rows without `year` take no part in the cross-document
      and overlap flags.
    - `mol_flags`: standardize every distinct structure into `standard_smiles`
      (`_with_mol_flags`, ~3 h for all of ChEMBL on one process); `aggregate`,
      `chirality=False` and `drop_unassigned_chiral` need it.
    - `convert_units`: convert `standard_value` and `standard_units` last
      (`de.convert_units`), except rows with `drop_unit_type_mismatch`.

    Unlike CAPRICHO, values stay float64 (its float32 cast misses 284 of 6181
    ADORA3 unit annotation errors), and `pchembl_value` is never unit-converted
    (CAPRICHO 1.0.4 turns a calculated 6.0 into 6000.0).
    """
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
    fetch = dict(selection)
    if standard_types is not None and calculate_pchembl:
        # CAPRICHO also fetches "Log <type>" and "-Log <type>": they take part in
        # its steps on the whole fetch (pChEMBL, unit errors, overlap), then it
        # drops them before standardizing (`get_standardize_and_clean_workflow`)
        logged = [f"{sign}Log {t}" for t in standard_types for sign in ("", "-")]
        fetch["drop_standard_type"] = de.is_not_in(
            "standard_type", [*standard_types, *logged]
        )
    unfetched = pl.any_horizontal(*fetch.values())
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
        note_multi_fragment=pl.col("canonical_smiles").str.contains(".", literal=True),
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
    "standard_smiles", "canonical_smiles", "standard_value", "standard_units",
    "assay_chembl_id", "description_assay", "activity_id", "assay_type",
    "standard_type", "confidence_score", "organism_tgt", "molecule_chembl_id",
    "document_chembl_id", "assay_tissue", "assay_cell_type", "relationship_type",
    "doc_type", "doi", "journal", "year", "chembl_release_id", "activity_comment",
)  # fmt: skip
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
    """`<value>_mean`, `_std`, `_median`, `_counts`, as CAPRICHO `assign_stats`:
    geometric for pChEMBL, else arithmetic; censored or single-row groups take
    the first value."""
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
    `extra_id_cols`, as CAPRICHO 1.0.4 `aggregate_data`. Needs `standard_smiles`
    (`flag(mol_flags=True)`) and aggregates every row, so drop flagged rows first.

    Compounds are equal by `compound_equality` of `standard_smiles` (stereo
    removed unless `chirality`; always for "connectivity"); censored rows only
    join rows with the same relation and rounded `value`. `MULTIVAL`, the flags
    and `extra_multival_cols` become lists; `standard_smiles` becomes
    `processed_smiles`, and `smiles` is its first entry recanonicalized. Adds
    `_stats(value)`, `connectivity`, `shared_identifier_group` (same identifier
    and target, split by mutation or `extra_id_cols`) and, without stereo,
    `might_be_racemic`. Unlike CAPRICHO, stats use unrounded values.
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
    """`<cache_dir>/chembl/<version>/capricho/<hash>.parquet`; the hash covers the
    fairfetched version and `flag`'s options (defaults filled, sequences as
    sorted sets, `parallel` left out).

    ponytail: editing `flag` without a version bump reuses a stale cache.
    """
    args = inspect.signature(flag).bind(None, None, **params)
    args.apply_defaults()
    options = {
        k: sorted(set(v)) if isinstance(v, Sequence) and not isinstance(v, str) else v
        for k, v in args.arguments.items()
        if k not in ("bioactivity", "activities", "parallel")
    }
    key = json.dumps([options, package_version("fairfetched")], sort_keys=True)
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
    """Sink `flag(...)` to `cache_path` unless it exists, via a renamed
    `<hash>.<pid>.tmp` so no run leaves a partial cache; return the path."""
    path = cache_path(version, params, cache_dir)
    if not path.exists():
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        flag(bioactivity, activities, **params).sink_parquet(tmp, mkdir=True)
        tmp.replace(path)
    return path
