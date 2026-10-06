"""Expected values come from the pure-Python references below (loops, sets,
`re`), which use neither fairfetched nor polars expressions. `TestAgainstCapricho`
runs CAPRICHO itself (`uv run --group capricho pytest`); its strict xfails are the
documented divergences."""

import random
import re
from collections import defaultdict
from functools import partial

import polars as pl
import pytest

from fairfetched.standardize import data_expr as de

# column names CAPRICHO hard-codes, so both sides read the same frame
MOL, ASSAY, TARGET, DOC = (
    "molecule_chembl_id",
    "assay_chembl_id",
    "target_chembl_id",
    "document_chembl_id",
)
VALUE, RELATION = "pchembl_value", "standard_relation"
KEYS = (MOL, VALUE, RELATION, TARGET, "mutation")
unit_errors = partial(de.flag_unit_annotation_errors, mol=MOL, assay=ASSAY, value=VALUE)
low_overlap = partial(
    de.flag_low_assay_overlap, mol=MOL, assay=ASSAY, value=VALUE, target=TARGET, doc=DOC
)
# 6.55 / 3.55 and 9.55 / 6.55 differ by 3 in float64 but not in float32
VALUES = (3.55, 4.1, 5.0, 6.0, 6.55, 7.1, 8.0, 9.0, 9.55)


def _frame(
    seed: int,
    n: int = 150,
    n_mols: int = 25,
    nulls: bool = True,
    repeats: bool = True,
) -> pl.DataFrame:
    """Random activities; an assay has one target (assay % 3) and one document
    (assay % 5, so assays 0 and 5 share D0)."""
    rng = random.Random(seed)
    rows, seen = [], set()
    while len(rows) < n:
        assay, mol = rng.randrange(8), rng.randrange(n_mols)
        if not repeats and (assay, mol) in seen:
            continue
        seen.add((assay, mol))
        rows.append(
            {
                MOL: f"M{mol}",
                ASSAY: f"A{assay}",
                TARGET: f"T{assay % 3}",
                DOC: None if nulls and assay == 7 else f"D{assay % 5}",
                VALUE: rng.choice(VALUES + ((None,) if nulls else ())),
                RELATION: rng.choice(("=", "=", "<") + ((None,) if nulls else ())),
                "mutation": rng.choice((None, "A12B")) if nulls else "WT",
            }
        )
    return pl.DataFrame(rows)


def _flag(df: pl.DataFrame, expr: pl.Expr) -> list[bool]:
    return df.select(expr.alias("f"))["f"].to_list()


# --- references ---


def ref_is_unit_error(a: float, b: float, tol: float = 1e-9) -> bool:
    """|a - b| within np.isclose(rtol=tol, atol=tol) of a positive multiple of 3."""
    d = abs(a - b)
    nearest = round(d / 3) * 3
    return nearest > 0 and abs(d - nearest) <= tol + tol * nearest


def ref_unit_errors(rows: list[dict], skip: list[bool]) -> list[bool]:
    out = []
    for i, r in enumerate(rows):
        out.append(
            not skip[i]
            and r[VALUE] is not None
            and any(
                not skip[j]
                and s[MOL] == r[MOL]
                and s[ASSAY] != r[ASSAY]
                and s[VALUE] is not None
                and ref_is_unit_error(r[VALUE], s[VALUE])
                for j, s in enumerate(rows)
            )
        )
    return out


def ref_low_overlap(rows: list[dict], min_overlap: int, skip: list[bool]) -> list[bool]:
    kept = [r for r, s in zip(rows, skip, strict=True) if not s]
    by_assay = defaultdict(list)
    for r in kept:
        by_assay[r[ASSAY]].append(r)
    flagged = set()
    for target in {r[TARGET] for r in kept}:
        assays = sorted({r[ASSAY] for r in kept if r[TARGET] == target})
        if len(assays) < 2:
            continue
        partnered = set()
        for a in assays:
            for b in assays:
                if a >= b:
                    continue
                shared = {
                    x[MOL]
                    for x in by_assay[a]
                    for y in by_assay[b]
                    if x[MOL] == y[MOL]
                    and None not in (x[DOC], y[DOC], x[VALUE], y[VALUE])
                    and x[DOC] != y[DOC]
                    and x[VALUE] != y[VALUE]
                    and not ref_is_unit_error(x[VALUE], y[VALUE])
                }
                if len(shared) >= min_overlap:
                    partnered |= {a, b}
        flagged |= set(assays) - partnered
    return [r[ASSAY] in flagged and not s for r, s in zip(rows, skip, strict=True)]


def ref_cross_document(rows: list[dict], keys=KEYS) -> list[bool]:
    docs = defaultdict(set)
    for r in rows:
        if r[RELATION] == "=":
            docs[tuple(r[k] for k in keys)].add(r[DOC])
    return [
        r[RELATION] == "=" and len(docs[tuple(r[k] for k in keys)]) > 1 for r in rows
    ]


def ref_assay_size(rows: list[dict]) -> list[int]:
    mols = defaultdict(set)
    for r in rows:
        if r[MOL] is not None:
            mols[r[ASSAY]].add(r[MOL])
    return [len(mols[r[ASSAY]]) for r in rows]


# --- tests ---


class TestRowFlags:
    df = pl.DataFrame(
        {
            "data_validity_comment": ["Outside typical range", None, None, None, "x"],
            "potential_duplicate": [1, 0, None, 1, 0],
            "standard_value": [0.0, 1.5, None, -0.0, 2.0],
            "standard_units": ["nM", "ug.mL-1", None, "µM", "%"],
            "year": [2001, None, 1999, None, 2020],
            "description_assay": [
                "Inhibition of MUTANT EGFR",
                "variants",
                None,
                "x",
                "y",
            ],
            "activity_comment": [
                "Not Active",
                "inactive",
                "inactives",
                None,
                "reinactive",
            ],
            "standard_relation": ["=", "<", "=", "=", "="],
            "confidence_score": [9, 8, None, 5, 7],
        }
    )
    rows = df.rows(named=True)
    inactive = re.compile(
        r"\b(" + "|".join(de.INACTIVE_COMMENTS) + r")\b", re.IGNORECASE
    )

    @pytest.mark.parametrize(
        ("expr", "ref"),
        [
            (
                de.has_validity_comment(),
                lambda r: r["data_validity_comment"] is not None,
            ),
            (de.is_potential_duplicate(), lambda r: r["potential_duplicate"] == 1),
            (de.is_zero(), lambda r: r["standard_value"] == 0),
            (
                de.has_non_molar_units(),
                lambda r: r["standard_units"] not in (None, "nM", "uM", "µM", "mM"),
            ),
            (de.is_missing(), lambda r: r["year"] is None),
            (
                de.mentions_mutant(),
                lambda r: any(
                    w in (r["description_assay"] or "").lower() for w in de.MUTANT_WORDS
                ),
            ),
            (
                de.is_not_in("confidence_score", [7, 8, 9]),
                lambda r: r["confidence_score"] not in (7, 8, 9),
            ),
        ],
        ids=[
            "validity",
            "potential_duplicate",
            "zero",
            "non_molar",
            "missing",
            "mutant",
            "not_in",
        ],
    )
    def test_matches_reference(self, expr, ref):
        assert _flag(self.df, expr) == [ref(r) for r in self.rows]

    def test_censored_by_comment_needs_whole_word_and_exact_relation(self):
        expected = [
            r["standard_relation"] == "="
            and bool(self.inactive.search(r["activity_comment"] or ""))
            for r in self.rows
        ]
        assert expected == [True, False, False, False, False]
        assert _flag(self.df, de.is_censored_by_comment()) == expected


class TestAssaySize:
    @pytest.mark.parametrize("seed", range(3))
    def test_matches_reference(self, seed):
        df = _frame(seed)
        out = df.select(n=de.assay_size(ASSAY, MOL))["n"].to_list()
        assert out == ref_assay_size(df.rows(named=True))

    def test_null_molecule_is_not_counted(self):
        df = pl.DataFrame({ASSAY: ["A1", "A1", "A1"], MOL: ["M1", None, "M1"]})
        out = df.select(n=de.assay_size(ASSAY, MOL))["n"].to_list()
        assert out == ref_assay_size(df.rows(named=True)) == [1, 1, 1]


class TestCrossDocumentDuplicate:
    @pytest.mark.parametrize("seed", range(5))
    def test_matches_reference(self, seed):
        df = _frame(seed, n_mols=6)
        expected = ref_cross_document(df.rows(named=True))
        assert any(expected) and not all(expected)
        expr = de.is_cross_document_duplicate(KEYS, doc=DOC, relation=RELATION)
        assert _flag(df, expr) == expected

    @pytest.mark.parametrize(
        "keys", [KEYS, (MOL, VALUE, TARGET)], ids=["with_relation", "without"]
    )
    def test_censored_rows_neither_flagged_nor_counted(self, keys):
        df = pl.DataFrame(
            {k: ["M1", "M1", "M1"] for k in (MOL, TARGET, "mutation")}
            | {VALUE: [6.0] * 3, RELATION: ["=", "<", "<"], DOC: ["D1", "D2", "D3"]}
        )
        expr = de.is_cross_document_duplicate(keys, doc=DOC, relation=RELATION)
        assert ref_cross_document(df.rows(named=True), keys) == [False] * 3
        assert _flag(df, expr) == [False] * 3


class TestUnitAnnotationErrors:
    @pytest.mark.parametrize("seed", range(5))
    def test_matches_reference(self, seed):
        df = _frame(seed)
        expected = ref_unit_errors(df.rows(named=True), [False] * df.height)
        assert any(expected) and not all(expected)
        out = unit_errors(df.lazy()).collect()
        assert out["unit_annotation_error"].to_list() == expected
        assert out.drop("unit_annotation_error").equals(df)

    @pytest.mark.parametrize("seed", range(3))
    def test_skipped_rows_take_no_part(self, seed):
        df = _frame(seed).with_row_index()
        skip = [i % 3 == 0 for i in range(df.height)]
        out = unit_errors(df.lazy(), skip=pl.col("index") % 3 == 0).collect()
        assert out["unit_annotation_error"].to_list() == ref_unit_errors(
            df.rows(named=True), skip
        )

    def test_float64_difference_of_three_is_flagged(self):
        """6.55 - 3.55 is 3 in float64; CAPRICHO's float32 cast makes it 3.00000024."""
        df = pl.DataFrame({MOL: ["M1", "M1"], ASSAY: ["A1", "A2"], VALUE: [6.55, 3.55]})
        assert ref_is_unit_error(6.55, 3.55)
        out = unit_errors(df.lazy()).collect()
        assert out["unit_annotation_error"].to_list() == [True, True]

    def test_same_assay_is_not_a_pair(self):
        df = pl.DataFrame({MOL: ["M1", "M1"], ASSAY: ["A1", "A1"], VALUE: [9.0, 6.0]})
        out = unit_errors(df.lazy()).collect()
        assert out["unit_annotation_error"].to_list() == [False, False]


class TestLowAssayOverlap:
    @pytest.mark.parametrize(("seed", "min_overlap"), [(0, 2), (1, 3), (2, 4), (3, 3)])
    def test_matches_reference(self, seed, min_overlap):
        df = _frame(seed, n=120, n_mols=40)
        expected = ref_low_overlap(
            df.rows(named=True), min_overlap, [False] * df.height
        )
        assert any(expected) and not all(expected)
        out = low_overlap(df.lazy(), min_overlap).collect()
        assert out["low_assay_overlap"].to_list() == expected

    @pytest.mark.parametrize("seed", range(3))
    def test_skipped_assays_take_no_part(self, seed):
        df = _frame(seed, n=120, n_mols=40)
        skip = [r[ASSAY] in ("A0", "A4") for r in df.rows(named=True)]
        out = low_overlap(
            df.lazy(), 3, skip=pl.col(ASSAY).is_in(["A0", "A4"])
        ).collect()
        assert out["low_assay_overlap"].to_list() == ref_low_overlap(
            df.rows(named=True), 3, skip
        )

    def test_only_conflicting_values_from_other_documents_count(self):
        """A1/A2 share M1 at different values in different documents; A3 shares
        M2 with A1 only within D1, A4 shares M3 with A2 only at a unit-error
        distance (9.0 vs 6.0)."""
        df = pl.DataFrame(
            {
                MOL: ["M1", "M1", "M2", "M2", "M3", "M3"],
                ASSAY: ["A1", "A2", "A3", "A1", "A4", "A2"],
                TARGET: ["T1"] * 6,
                DOC: ["D1", "D2", "D1", "D1", "D3", "D2"],
                VALUE: [6.0, 6.5, 7.0, 7.5, 9.0, 6.0],
            }
        )
        out = low_overlap(df.lazy(), 1).collect()
        expected = [False, False, True, False, True, False]
        assert ref_low_overlap(df.rows(named=True), 1, [False] * 6) == expected
        assert out["low_assay_overlap"].to_list() == expected

    def test_single_assay_target_is_not_checked(self):
        df = pl.DataFrame(
            {MOL: ["M1"], ASSAY: ["A1"], TARGET: ["T1"], DOC: ["D1"], VALUE: [6.0]}
        )
        out = low_overlap(df.lazy(), 1).collect()
        assert out["low_assay_overlap"].to_list() == [False]

    def test_rejects_min_overlap_below_one(self):
        with pytest.raises(ValueError, match="min_overlap"):
            de.flag_low_assay_overlap(pl.LazyFrame(), 0)


class TestAgainstCapricho:
    """Same frame through CAPRICHO 1.0.4 and fairfetched."""

    @pytest.fixture(autouse=True)
    def capricho(self):
        pytest.importorskip("Capricho")

    @staticmethod
    def _capricho_flag(pdf, comment: str, kind: str = "data_dropping_comment"):
        if kind not in pdf.columns:
            return [False] * len(pdf)
        return pdf[kind].fillna("").str.contains(comment, regex=False).tolist()

    @pytest.mark.parametrize("seed", range(3))
    def test_unit_annotation_errors(self, seed):
        from Capricho.chembl.processing import curate_activity_pairs

        df = _frame(seed)
        pdf = curate_activity_pairs(df.to_pandas(), MOL, ASSAY, VALUE)
        out = unit_errors(df.lazy()).collect()
        assert out["unit_annotation_error"].to_list() == self._capricho_flag(
            pdf, "Unit Annotation Error"
        )

    @pytest.mark.xfail(
        strict=True, reason="CAPRICHO casts pchembl_value to float32 before pairing"
    )
    def test_unit_annotation_errors_after_float32_cast(self):
        from Capricho.chembl.processing import curate_activity_pairs

        df = pl.DataFrame({MOL: ["M1", "M1"], ASSAY: ["A1", "A2"], VALUE: [6.55, 3.55]})
        pdf = df.to_pandas().astype({VALUE: "float32"})
        pdf = curate_activity_pairs(pdf, MOL, ASSAY, VALUE)
        out = unit_errors(df.lazy()).collect()
        assert out["unit_annotation_error"].to_list() == self._capricho_flag(
            pdf, "Unit Annotation Error"
        )

    def _cross_document(self, df: pl.DataFrame):
        from Capricho.chembl.data_flag_functions import flag_inter_document_duplication

        pdf = flag_inter_document_duplication(
            df.to_pandas(), key_subset=list(KEYS), diff_subset=[DOC]
        )
        expr = de.is_cross_document_duplicate(KEYS, doc=DOC, relation=RELATION)
        assert _flag(df, expr) == self._capricho_flag(
            pdf, "pChEMBL Duplication Across Documents", "data_processing_comment"
        )

    @pytest.mark.parametrize("seed", range(3))
    def test_cross_document_without_null_keys(self, seed):
        self._cross_document(_frame(seed, n_mols=6, nulls=False))

    @pytest.mark.xfail(
        strict=True, reason="CAPRICHO's pandas groupby drops rows with a null key"
    )
    def test_cross_document_with_null_mutation(self):
        self._cross_document(_frame(0, n_mols=6))

    def _overlap(self, df: pl.DataFrame, min_overlap: int):
        from Capricho.chembl.data_flag_functions import flag_insufficient_assay_overlap

        pdf = flag_insufficient_assay_overlap(
            df.to_pandas(), min_overlap, MOL, ASSAY, TARGET
        )
        out = low_overlap(df.lazy(), min_overlap).collect()
        assert out["low_assay_overlap"].to_list() == self._capricho_flag(
            pdf, "Insufficient assay overlap"
        )

    @pytest.mark.parametrize(("seed", "min_overlap"), [(0, 2), (1, 3), (2, 4)])
    def test_overlap_without_nulls_or_repeats(self, seed, min_overlap):
        self._overlap(
            _frame(seed, n=120, n_mols=40, nulls=False, repeats=False), min_overlap
        )

    @pytest.mark.xfail(strict=True, reason="CAPRICHO counts row pairs, not molecules")
    def test_overlap_with_repeated_measurements(self):
        df = pl.DataFrame(
            {
                MOL: ["M1", "M1", "M1"],
                ASSAY: ["A1", "A1", "A2"],
                TARGET: ["T1"] * 3,
                DOC: ["D1", "D1", "D2"],
                VALUE: [6.0, 6.2, 7.0],
            }
        )
        self._overlap(df, 2)

    @pytest.mark.xfail(
        strict=True, reason="CAPRICHO counts a null document or value as differing"
    )
    def test_overlap_with_null_document(self):
        df = pl.DataFrame(
            {
                MOL: ["M1", "M1"],
                ASSAY: ["A1", "A2"],
                TARGET: ["T1"] * 2,
                DOC: ["D1", None],
                VALUE: [6.0, 7.0],
            }
        )
        self._overlap(df, 1)

    def test_row_flags(self):
        from Capricho.chembl import data_flag_functions as cf

        df = TestRowFlags.df.rename({"description_assay": "assay_description"})
        cases = [
            (
                cf.flag_with_data_validity_comment,
                de.has_validity_comment(),
                "Data Validity",
            ),
            (
                cf.flag_potential_duplicate,
                de.is_potential_duplicate(),
                "Potential Duplicate",
            ),
            (cf.flag_zero_values, de.is_zero(), "Zero Value"),
            (
                cf.flag_incompatible_units,
                de.has_non_molar_units(),
                "Incompatible units",
            ),
            (cf.flag_missing_document_date, de.is_missing(), "Missing document date"),
            (
                lambda d: cf.flag_strict_mutant_assays(d, strict_mutant_removal=True),
                de.mentions_mutant("assay_description"),
                "Mutation keyword",
            ),
            (
                cf.flag_censored_activity_comment,
                de.is_censored_by_comment(),
                "inactivity-like",
            ),
        ]
        for capricho_fn, expr, comment in cases:
            pdf = capricho_fn(df.to_pandas())
            assert _flag(df, expr) == self._capricho_flag(pdf, comment), comment
