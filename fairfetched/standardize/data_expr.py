"""Data expressions: flags over several columns of a full dataset (pl.Boolean).

Ports the filters of CAPRICHO (github.com/David-Araripe/Capricho,
`chembl/data_flag_functions.py`, `chembl/processing.py`). Default column names
are those of `Chembl.view.bioactivity`; pass other names for custom datasets.
Rows are flagged, not dropped.

TODO: `DataExpr` namespace (`pl.api.register_lazyframe_namespace("data")`), so
the frame-level flags read `lf.data.flag_unit_annotation_errors(...)`.
TODO: move `split_expr.kfold` here once the module shape settles.
"""

from collections.abc import Sequence

import polars as pl

INACTIVE_COMMENTS = (  # CAPRICHO REVIEW_ACTIVITY_COMMENTS
    "not active",
    "inactive",
    "no activity",
    "below threshold",
    "below detection",
    "inconclusive",
    "not tested",
    "not determined",
)
MOLAR_UNITS = ("nM", "uM", "µM", "mM")
MUTANT_WORDS = ("mutant", "mutation", "variant")
DUPLICATE_KEYS = (
    "molregno",
    "pchembl_value",
    "standard_relation",
    "tid",
    "mutation",
    "organism_tgt",
)


# --- row-level ---


def has_validity_comment(col: str = "data_validity_comment") -> pl.Expr:
    """`col` is not null."""
    raise NotImplementedError


def is_potential_duplicate(col: str = "potential_duplicate") -> pl.Expr:
    """`col` == 1."""
    raise NotImplementedError


def is_zero(col: str = "standard_value") -> pl.Expr:
    """`col` == 0."""
    raise NotImplementedError


def has_non_molar_units(
    col: str = "standard_units", molar: Sequence[str] = MOLAR_UNITS
) -> pl.Expr:
    """`col` is not null and not in `molar` (no pChEMBL can be computed)."""
    raise NotImplementedError


def is_missing(col: str = "year") -> pl.Expr:
    """`col` is null (CAPRICHO: missing document date)."""
    raise NotImplementedError


def mentions_mutant(
    col: str = "description_assay", words: Sequence[str] = MUTANT_WORDS
) -> pl.Expr:
    """`col` contains any of `words`, case-insensitive."""
    raise NotImplementedError


def is_censored_by_comment(
    comment: str = "activity_comment",
    relation: str = "standard_relation",
    terms: Sequence[str] = INACTIVE_COMMENTS,
) -> pl.Expr:
    """`relation` == "=" while `comment` contains a whole-word inactivity term."""
    raise NotImplementedError


def is_not_in(col: str, values: Sequence) -> pl.Expr:
    """`col` not in `values`; for confidence score, assay type, relation, units."""
    raise NotImplementedError


# --- window ---


def assay_size(assay: str = "assay_id", mol: str = "molregno") -> pl.Expr:
    """Distinct `mol` per `assay`, counted over the rows of the frame (pl.UInt32).

    Differs from CAPRICHO, which counts over the whole assay in ChEMBL; the
    Chembl preset uses `fairfetched.get._capricho.assay_sizes` for that.
    """
    raise NotImplementedError


def is_cross_document_duplicate(
    keys: Sequence[str] = DUPLICATE_KEYS,
    *,
    doc: str = "doc_id",
    relation: str = "standard_relation",
) -> pl.Expr:
    """`relation` == "=" and rows sharing `keys` come from more than one `doc`.

    A note, not a drop: the same value reported in several papers.
    """
    raise NotImplementedError


# --- frame-level (self-joins; not expressible as one pl.Expr) ---


def flag_unit_annotation_errors(
    lf: pl.LazyFrame,
    *,
    mol: str = "molregno",
    assay: str = "assay_id",
    value: str = "pchembl_value",
    skip: pl.Expr | None = None,
    tol: float = 1e-9,
    name: str = "unit_annotation_error",
) -> pl.LazyFrame:
    """Add `name`: the row pairs with another row of the same `mol` in a different
    `assay` whose `value` differs by a positive multiple of 3 (within `tol`).

    Rows where `skip` is true take no part in the pairing. Differs from CAPRICHO,
    which casts `value` to float32 first and so misses e.g. 6.55 / 3.55.
    Quadratic in the rows per `mol`.
    """
    raise NotImplementedError


def flag_low_assay_overlap(
    lf: pl.LazyFrame,
    min_overlap: int,
    *,
    mol: str = "molregno",
    assay: str = "assay_id",
    target: str = "tid",
    doc: str = "doc_id",
    value: str = "pchembl_value",
    skip: pl.Expr | None = None,
    tol: float = 1e-9,
    name: str = "low_assay_overlap",
) -> pl.LazyFrame:
    """Add `name`: the row's `assay` has no partner assay of the same `target`
    sharing >= `min_overlap` `mol`s, counting only pairs from different `doc`s
    with different `value`s that are not a unit annotation error.

    Assays where `skip` is true (CAPRICHO: size-flagged) are neither partners
    nor flagged. Quadratic in the rows per `target`.
    """
    raise NotImplementedError


__all__ = [
    "assay_size",
    "flag_low_assay_overlap",
    "flag_unit_annotation_errors",
    "has_non_molar_units",
    "has_validity_comment",
    "is_censored_by_comment",
    "is_cross_document_duplicate",
    "is_missing",
    "is_not_in",
    "is_potential_duplicate",
    "is_zero",
    "mentions_mutant",
]
