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
MOLAR_EXPONENT = {"nM": -9, "uM": -6, "µM": -6, "mM": -3}  # CAPRICHO convert_to_log10 units
MUTANT_WORDS = ("mutant", "mutation", "variant")
DUPLICATE_KEYS = (
    "molregno",
    "pchembl_value",
    "standard_relation",
    "tid",
    "mutation",
    "organism_tgt",
)
_SKIP_COL = "__skip__"
# CAPRICHO 1.0.4 `core/smiles_utils.py` MIXTURE_REGEX, verbatim: salts/solvents
# CAPRICHO strips after the ChEMBL pipeline. Not ChEMBL's `salts.smi` (lists
# perchlorate only as OCl(=O)(=O)=O) nor Papyrus's `prep.papyrus.SALTS` (also
# methane, dioxane, trimethylamine, ...).
MIXTURE_REGEX = (
    r"^("  # start of the string
    r"Na\+?|Cl\-?|Br\-?|K\+?|F\-?|I\-?|Ca\+{2}?|Mg\+{2}?|Zn\+{2}?|OH\-?|"  # Salts (bare)
    r"\[Na\+\]?|\[K\+\]?|\[Cl-\]?|\[Br-\]?|\[I-\]?|"  # Salts (bracketed, monovalent)
    r"\[Zn\+2\]?|\[Ca\+2\]?|\[Mg\+2\]?|\[Li\+\]?|"  # Salts (bracketed, divalent + Li)
    r"CCCC\(=O\)\[O-\]|"  # Butyrate
    r"CCCCC\(=O\)\[O-\]|"  # Pentanoate
    r"O=C\(\[O-\]\)c1ccccc1|"  # Benzoate (RDKit-canonical)
    r"\[O-\]\[Cl\+3\]\(\[O-\]\)\(\[O-\]\)\[O-\]|"  # Perchlorate [O-][Cl+3]([O-])([O-])[O-]
    r"c1ccncc1|"  # Pyridine (RDKit-canonical)
    r"CN\(C\)C=O|"  # N,N-Dimethylformamide (RDKit-canonical)
    r"\[N\]=O|"  # Nitric oxide (RDKit-canonical)
    r"c1ccc\(\[B-\]\(c2ccccc2\)\(c2ccccc2\)c2ccccc2\)cc1|"  # Tetraphenylborate
    r"^O$|^N$"  # Match full strings for O and N
    r")$"  # end of the string
)  # fmt: skip


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


def pchembl(value: str = "standard_value", units: str = "standard_units") -> pl.Expr:
    """-log10(`value` in M), unrounded, as CAPRICHO `convert_to_log10` (pl.Float64);
    null for `value` <= 0 and for null or non-molar `units`.

    Float64 as ChEMBL's SQLite stores it; CAPRICHO casts to float32 (~1e-7 off).
    The unit's exponent is subtracted exactly, so 1 nM gives 9.0 as in ChEMBL;
    -log10(1 * 1e-9) gives 8.999999999999998 in float64.
    """
    exponent = pl.col(units).replace_strict(
        MOLAR_EXPONENT, default=None, return_dtype=pl.Float64
    )
    return pl.when(pl.col(value) > 0).then(-pl.col(value).log10() - exponent)


def fragments_without_salts(smiles: str | pl.Expr = "standard_smiles") -> pl.Expr:
    """Distinct "."-separated fragments of `smiles` (pl.List(pl.String), sorted); when
    there are several, those matching `MIXTURE_REGEX` and empty ones are removed,
    so a salt-only `smiles` gives []. As CAPRICHO `clean_mixtures`, which then
    calls every result other than one fragment a mixture ("." for []).

    Expects RDKit-canonical SMILES, as `MIXTURE_REGEX` matches text.
    """
    expr = pl.col(smiles) if isinstance(smiles, str) else smiles
    frags = expr.str.split(".").list.unique().list.sort()  # np.unique sorts
    salt = pl.element().str.contains(MIXTURE_REGEX) | (pl.element() == "")
    return (
        pl.when(frags.list.len() > 1)
        .then(frags.list.eval(pl.element().filter(~salt)))
        .otherwise(frags)
    )


# --- window ---


def assay_size(assay: str = "assay_id", mol: str = "molregno") -> pl.Expr:
    """unique non-null `mol` per `assay`, over the frame's rows (pl.UInt32).

    c.f. CAPRICHO counts over whole assay in ChEMBL.
    the Chembl preset uses `fairfetched.get._capricho.assay_sizes` for that.
    """
    return pl.col(mol).drop_nulls().n_unique().over(assay)


def is_cross_document_duplicate(
    keys: Sequence[str | pl.Expr] = DUPLICATE_KEYS,
    *,
    doc: str = "doc_id",
    relation: str = "standard_relation",
    skip: pl.Expr | None = None,
) -> pl.Expr:
    """`relation` == "=" and the "=" rows sharing `keys` come from more than one
    `doc` (null counts as a `doc`). Rows where `skip` is true or a key is null
    are neither counted nor flagged.

    A note, not a drop: the same value reported in several papers. As CAPRICHO
    `flag_inter_document_duplication`, whose pandas `groupby` leaves out rows
    with a null key; its pipeline fills a null `mutation` with "WT" first, so
    pass `pl.col("mutation").fill_null("WT")` as a key for that.
    """
    cols = [pl.col(k) if isinstance(k, str) else k for k in keys]
    null_key = pl.any_horizontal(c.is_null() for c in cols)
    skipped = skip.fill_null(False) if skip is not None else pl.lit(False)
    exact = (pl.col(relation) == "=") & ~skipped & ~null_key
    n_docs = pl.col(doc).filter(exact).n_unique().over(cols)
    return (exact & (n_docs > 1)).fill_null(False)


# --- frame-level (self-joins; not expressible as one pl.Expr) ---


def _is_unit_error(diff: pl.Expr, tol: float) -> pl.Expr:
    """|`diff`| is a positive multiple of 3, as `np.isclose(rtol=tol, atol=tol)`."""
    d = diff.abs()
    nearest = (d / 3).round() * 3
    return (nearest > 0) & ((d - nearest).abs() <= tol + tol * nearest)


def _with_skip(lf: pl.LazyFrame, skip: pl.Expr | None) -> pl.LazyFrame:
    mask = pl.lit(False) if skip is None else skip.fill_null(False)
    return lf.with_columns(mask.alias(_SKIP_COL))


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
        .with_columns(pl.col(name).fill_null(False) & ~pl.col(_SKIP_COL))
        .drop(_SKIP_COL)
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
        lf.filter(~pl.col(_SKIP_COL), pl.col(value).is_not_null()).select(keys).unique()
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
    with >= `min_overlap` row pairs of the same `mol`, from different `doc`s, at
    different `value`s that are not a unit annotation error.

    As CAPRICHO `flag_insufficient_assay_overlap`: rows are paired, not molecules
    (a `mol` measured twice in each assay is 4 pairs), and a null `doc` or `value`
    counts as different, so a pair without values counts. Rows where `skip` is
    true (CAPRICHO: size-flagged assays) take no part and are not flagged; a
    `target` with fewer than 2 assays is not checked. Quadratic in the rows per
    (`target`, `mol`).
    """
    if min_overlap < 1:
        raise ValueError(f"min_overlap must be >= 1, got {min_overlap}")
    lf = _with_skip(lf, skip)
    rows = lf.remove(pl.col(_SKIP_COL)).select(target, assay, mol, doc, value)
    a_r, d_r, v_r = f"{assay}_r", f"{doc}_r", f"{value}_r"
    partners = (
        rows.join(rows, on=[target, mol], suffix="_r")
        .filter(
            pl.col(assay) < pl.col(a_r),
            (pl.col(doc) != pl.col(d_r)).fill_null(True),
            (pl.col(value) != pl.col(v_r)).fill_null(True),
            ~_is_unit_error(pl.col(value) - pl.col(v_r), tol).fill_null(False),
        )
        .group_by(assay, a_r)
        .agg(n=pl.len())
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
    "fragments_without_salts",
    "has_non_molar_units",
    "has_validity_comment",
    "is_censored_by_comment",
    "is_cross_document_duplicate",
    "is_missing",
    "is_not_in",
    "is_potential_duplicate",
    "is_zero",
    "mentions_mutant",
    "pchembl",
]
