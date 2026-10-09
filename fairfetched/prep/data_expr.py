"""Data expressions: flags over several columns of a full dataset (pl.Boolean).

Ports the filters of CAPRICHO (github.com/David-Araripe/Capricho,
`chembl/data_flag_functions.py`, `chembl/processing.py`). Default column names
are those of `Chembl.view.bioactivity`; pass other names for custom datasets.
Rows are flagged, not dropped; a null input gives False, except in `is_not_in`.

TODO: `DataExpr` namespace (`pl.api.register_lazyframe_namespace("data")`), so
the frame-level flags read `lf.data.flag_unit_annotation_errors(...)`.
TODO: move `split_expr.kfold` here once the module shape settles.
"""

import re
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
_SKIP = "__skip__"


# --- row-level ---


def has_validity_comment(col: str = "data_validity_comment") -> pl.Expr:
    """`col` is not null."""
    return pl.col(col).is_not_null()


def is_potential_duplicate(col: str = "potential_duplicate") -> pl.Expr:
    """`col` == 1."""
    return (pl.col(col) == 1).fill_null(False)


def is_zero(col: str = "standard_value") -> pl.Expr:
    """`col` == 0."""
    return (pl.col(col) == 0).fill_null(False)


def has_non_molar_units(
    col: str = "standard_units", molar: Sequence[str] = MOLAR_UNITS
) -> pl.Expr:
    """`col` is not null and not in `molar` (no pChEMBL can be computed)."""
    return (pl.col(col).is_not_null() & ~pl.col(col).is_in(molar)).fill_null(False)


def is_missing(col: str = "year") -> pl.Expr:
    """`col` is null (CAPRICHO: missing document date)."""
    return pl.col(col).is_null()


def mentions_mutant(
    col: str = "description_assay", words: Sequence[str] = MUTANT_WORDS
) -> pl.Expr:
    """`col` contains any of `words` as a substring, case-insensitive."""
    pattern = "(?i)" + "|".join(map(re.escape, words))
    return pl.col(col).str.contains(pattern).fill_null(False)


def is_censored_by_comment(
    comment: str = "activity_comment",
    relation: str = "standard_relation",
    terms: Sequence[str] = INACTIVE_COMMENTS,
) -> pl.Expr:
    """`relation` == "=" while `comment` contains a whole-word inactivity term."""
    pattern = r"(?i)\b(?:" + "|".join(map(re.escape, terms)) + r")\b"
    return (
        (pl.col(relation) == "=") & pl.col(comment).str.contains(pattern)
    ).fill_null(False)


def is_not_in(col: str, values: Sequence) -> pl.Expr:
    """`col` not in `values`; for confidence score, assay type, relation, units.

    Null is not in `values` (True), as SQL `IN` in CAPRICHO excludes it.
    """
    return ~pl.col(col).is_in(values).fill_null(False)


# --- window ---


def assay_size(assay: str = "assay_id", mol: str = "molregno") -> pl.Expr:
    """Distinct non-null `mol` per `assay`, counted over the rows of the frame (pl.UInt32).

    Differs from CAPRICHO, which counts over the whole assay in ChEMBL; the
    Chembl preset uses `fairfetched.get._capricho.assay_sizes` for that.
    """
    return pl.col(mol).drop_nulls().n_unique().over(assay)


def is_cross_document_duplicate(
    keys: Sequence[str] = DUPLICATE_KEYS,
    *,
    doc: str = "doc_id",
    relation: str = "standard_relation",
) -> pl.Expr:
    """`relation` == "=" and the "=" rows sharing `keys` come from more than one
    `doc` (null counts as a `doc`).

    A note, not a drop: the same value reported in several papers. Differs from
    CAPRICHO, whose pandas `groupby` drops rows with a null key, so rows with a
    null `mutation` (wild type) are never flagged there.
    """
    exact = pl.col(relation) == "="
    n_docs = pl.col(doc).filter(exact).n_unique().over(keys)
    return (exact & (n_docs > 1)).fill_null(False)


# --- frame-level (self-joins; not expressible as one pl.Expr) ---


def _is_unit_error(diff: pl.Expr, tol: float) -> pl.Expr:
    """|`diff`| is a positive multiple of 3, as `np.isclose(rtol=tol, atol=tol)`."""
    d = diff.abs()
    nearest = (d / 3).round() * 3
    return (nearest > 0) & ((d - nearest).abs() <= tol + tol * nearest)


def _with_skip(lf: pl.LazyFrame, skip: pl.Expr | None) -> pl.LazyFrame:
    mask = pl.lit(False) if skip is None else skip.fill_null(False)
    return lf.with_columns(mask.alias(_SKIP))


def _attach(
    lf: pl.LazyFrame, hits: pl.LazyFrame, on: list[str], name: str
) -> pl.LazyFrame:
    """Left-join `hits` (unique `on`) as `name`; False where absent or skipped."""
    return (
        lf.join(
            hits.with_columns(pl.lit(True).alias(name)),
            on=on,
            how="left",
            maintain_order="left",
        )
        .with_columns(pl.col(name).fill_null(False) & ~pl.col(_SKIP))
        .drop(_SKIP)
    )


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
    Quadratic in the distinct (`assay`, `value`) per `mol`.
    """
    lf = _with_skip(lf, skip)
    keys = [mol, assay, value]
    points = (
        lf.filter(~pl.col(_SKIP), pl.col(value).is_not_null()).select(keys).unique()
    )
    hits = (
        points.join(points, on=mol, suffix="_r")
        .filter(
            pl.col(assay) != pl.col(f"{assay}_r"),
            _is_unit_error(pl.col(value) - pl.col(f"{value}_r"), tol),
        )
        .select(keys)
        .unique()
    )
    return _attach(lf, hits, keys, name)


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
    sharing >= `min_overlap` distinct `mol`s, counting only pairs from different
    non-null `doc`s with different non-null `value`s that are not a unit
    annotation error.

    Rows where `skip` is true (CAPRICHO: size-flagged assays) take no part and
    are not flagged. As in CAPRICHO, a `target` with fewer than 2 assays is not
    checked. Differs from CAPRICHO, which counts row pairs rather than distinct
    `mol`s, and counts pairs with a null `value` or `doc` as differing.
    Quadratic in the distinct (`assay`, `doc`, `value`) per (`target`, `mol`).
    """
    if min_overlap < 1:
        raise ValueError(f"min_overlap must be >= 1, got {min_overlap}")
    lf = _with_skip(lf, skip)
    rows = lf.remove(pl.col(_SKIP)).select(target, assay, mol, doc, value).unique()
    a_r, d_r, v_r = f"{assay}_r", f"{doc}_r", f"{value}_r"
    partners = (
        rows.join(rows, on=[target, mol], suffix="_r")
        .filter(
            pl.col(assay) < pl.col(a_r),
            pl.col(doc) != pl.col(d_r),
            pl.col(value) != pl.col(v_r),
            ~_is_unit_error(pl.col(value) - pl.col(v_r), tol),
        )
        .group_by(assay, a_r)
        .agg(pl.col(mol).n_unique().alias("n"))
        .filter(pl.col("n") >= min_overlap)
    )
    partnered = pl.concat(
        [partners.select(assay), partners.select(pl.col(a_r).alias(assay))]
    ).unique()
    hits = (
        rows.select(target, assay)
        .unique()
        .filter(pl.col(assay).n_unique().over(target) >= 2)
        .join(partnered, on=assay, how="anti")
        .select(assay)
        .unique()
    )
    return _attach(lf, hits, [assay], name)


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
