"""The ChEMBL preset (`fairfetched.get._capricho`) on `Chembl.demo()` and on small
hand-built frames. Expected values come from plain Python over the rows; the
flag logic itself is tested in `test_data_expr.py`."""

import logging
import math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import polars as pl
import polars.selectors as cs
import pytest

from fairfetched.get import Chembl, _capricho
from fairfetched.get.chembl import CHEMBL_DIR

SELECTION = {
    "drop_confidence": ("confidence_score", (7, 8, 9)),
    "drop_assay_type": ("assay_type", ("B", "F")),
    "drop_relation": ("standard_relation", ("=",)),
}


@pytest.fixture(scope="module")
def chembl() -> Chembl:
    return Chembl.demo()


@pytest.fixture(scope="module")
def bioactivity(chembl) -> pl.LazyFrame:
    return chembl.view.bioactivity


def _bioactivity(rows: list[dict]) -> pl.LazyFrame:
    """Minimal frame with the columns `flag` reads; `rows` override the defaults."""
    default = {
        "activity_id": 0,
        "molregno": 1,
        "assay_id": 1,
        "tid": 1,
        "doc_id": 1,
        "pchembl_value": 6.0,
        "standard_value": 1000.0,
        "standard_relation": "=",
        "standard_type": "Ki",
        "standard_units": "nM",
        "confidence_score": 9,
        "assay_type": "B",
        "chembl_release_id": 1,
        "data_validity_comment": None,
        "potential_duplicate": 0,
        "canonical_smiles": "CCO",
        "year": 2000,
        "activity_comment": None,
        "mutation": None,
        "organism_tgt": "Homo sapiens",
        "description_assay": "x",
    }
    return pl.LazyFrame(
        [default | {"activity_id": i} | r for i, r in enumerate(rows)],
        schema_overrides=dict.fromkeys(
            ("data_validity_comment", "activity_comment", "mutation"), pl.String
        )
        | {"pchembl_value": pl.Float64},
    )


class TestSelectionFlags:
    def test_defaults_match_reference(self, bioactivity):
        out = _capricho.flag(bioactivity).collect()
        for name, (col, allowed) in SELECTION.items():
            expected = [v not in allowed for v in out[col].to_list()]
            assert out[name].to_list() == expected, name

    def test_release_keeps_null_and_older(self, bioactivity):
        out = _capricho.flag(bioactivity, chembl_release=20).collect()
        expected = [
            r is not None and r > 20 for r in out["chembl_release_id"].to_list()
        ]
        assert any(expected) and not all(expected)
        assert out["drop_release"].to_list() == expected

    def test_missing_standard_value_is_never_fetched(self):
        """Assay 2's null-value row would partner assay 1 in the overlap;
        assay 3 shares nothing."""
        rows = [
            {"assay_id": 1, "doc_id": 1, "pchembl_value": 6.0},
            {"assay_id": 2, "doc_id": 2, "pchembl_value": None, "standard_value": None},
            {"assay_id": 3, "doc_id": 3, "molregno": 2},
        ]
        out = _capricho.flag(_bioactivity(rows), min_assay_overlap=1).collect()
        assert out["drop_missing_standard_value"].to_list() == [False, True, False]
        assert out["drop_low_assay_overlap"].to_list() == [True, False, True]

    def test_null_release_is_kept(self):
        rows = [{"chembl_release_id": r} for r in (None, 10, 30)]
        out = _capricho.flag(_bioactivity(rows), chembl_release=20).collect()
        assert out["drop_release"].to_list() == [False, False, True]

    def test_none_disables_a_criterion(self, bioactivity):
        out = _capricho.flag(bioactivity, confidence_scores=None).collect()
        assert "drop_confidence" not in out.columns
        assert "drop_release" not in out.columns

    def test_log_types_take_part_in_unit_errors_only_with_calculate_pchembl(self):
        """A Ki 9.0 and a "Log Ki" 6.0 of the same molecule: CAPRICHO fetches the
        latter with `--calculate-pchembl`, pairs it, then drops it."""
        rows = [
            {"standard_type": "Ki", "assay_id": 1, "pchembl_value": 9.0},
            {"standard_type": "Log Ki", "assay_id": 2, "pchembl_value": 6.0},
        ]
        plain = _capricho.flag(_bioactivity(rows), standard_types=["Ki"]).collect()
        calc = _capricho.flag(
            _bioactivity(rows), standard_types=["Ki"], calculate_pchembl=True
        ).collect()
        for out in (plain, calc):
            assert out["drop_standard_type"].to_list() == [False, True]
        assert plain["drop_unit_annotation_error"].to_list() == [False, False]
        assert calc["drop_unit_annotation_error"].to_list() == [True, True]

    def test_log_types_take_no_part_in_cross_document(self):
        rows = [
            {"standard_type": "Ki", "doc_id": 1},
            {"standard_type": "Log Ki", "doc_id": 2, "assay_id": 2},
        ]
        out = _capricho.flag(
            _bioactivity(rows), standard_types=["Ki"], calculate_pchembl=True
        ).collect()
        assert out["note_cross_document"].to_list() == [False, False]

    def test_unfetched_rows_take_no_part_in_unit_errors(self):
        rows = [
            {"molregno": 1, "assay_id": 1, "pchembl_value": 9.0},
            {"molregno": 1, "assay_id": 2, "pchembl_value": 6.0},
        ]
        fetched = _capricho.flag(_bioactivity(rows)).collect()
        rows[1]["confidence_score"] = 5
        unfetched = _capricho.flag(_bioactivity(rows)).collect()
        assert fetched["drop_unit_annotation_error"].to_list() == [True, True]
        assert unfetched["drop_unit_annotation_error"].to_list() == [False, False]


class TestCalculatePchembl:
    def test_fills_only_missing_values(self):
        """As CAPRICHO, every attempt in molar or null units is noted, also the
        one that gives null; other units are incompatible."""
        rows = [
            {"pchembl_value": 6.64, "standard_value": 230.0},  # ChEMBL's value kept
            {"pchembl_value": None, "standard_value": 100.0},  # 1e-7 M
            {"pchembl_value": None, "standard_value": 10.0, "standard_units": None},
            {"pchembl_value": None, "standard_value": 10.0, "standard_units": "%"},
        ]
        out = _capricho.flag(_bioactivity(rows), calculate_pchembl=True).collect()
        assert out["pchembl_value"].to_list() == pytest.approx([6.64, 7.0, None, None])
        assert out["note_pchembl_calculated"].to_list() == [False, True, True, False]
        assert out["drop_incompatible_units"].to_list() == [False, False, False, True]
        assert out["drop_missing_value"].to_list() == [False, False, True, True]

    def test_off_by_default(self):
        out = _capricho.flag(_bioactivity([{"pchembl_value": None}])).collect()
        assert "note_pchembl_calculated" not in out.columns
        assert "drop_incompatible_units" not in out.columns
        assert out["pchembl_value"].to_list() == [None]


class TestConvertUnits:
    def test_after_pchembl(self):
        """pChEMBL is calculated from the original 1 uM; CAPRICHO would then
        multiply it by 1000 (`flag` docstring)."""
        rows = [{"pchembl_value": None, "standard_value": 1.0, "standard_units": "uM"}]
        out = _capricho.flag(
            _bioactivity(rows), calculate_pchembl=True, convert_units=True
        ).collect()
        assert out["pchembl_value"].to_list() == [6.0]
        assert out["standard_value"].to_list() == [1000.0]
        assert out["standard_units"].to_list() == ["nM"]
        assert out["note_unit_converted"].to_list() == [
            {"standard_value": 1.0, "standard_units": "uM"}
        ]

    def test_off_by_default(self):
        out = _capricho.flag(_bioactivity([{}])).collect_schema().names()
        assert "note_unit_converted" not in out


class TestCrossDocument:
    def test_null_mutation_is_wild_type_and_null_organism_left_out(self):
        """Pairs of the same value in two documents: null mutation (CAPRICHO's
        downloader fills "WT") and null organism (left out by pandas groupby)."""
        rows = [
            {"doc_id": 1, "molregno": 1},
            {"doc_id": 2, "molregno": 1, "assay_id": 2},
            {"doc_id": 1, "molregno": 2, "organism_tgt": None},
            {"doc_id": 2, "molregno": 2, "assay_id": 2, "organism_tgt": None},
        ]
        out = _capricho.flag(_bioactivity(rows)).collect()
        assert out["note_cross_document"].to_list() == [True, True, False, False]


class TestRequireDocDate:
    def test_missing_year_flagged_either_way(self):
        rows = [{"year": None}, {"year": 2000}]
        for require in (False, True):
            out = _capricho.flag(_bioactivity(rows), require_doc_date=require)
            assert out.collect()["drop_missing_year"].to_list() == [True, False]

    def test_missing_year_counts_for_cross_document_only_by_default(self):
        """The same value in two documents, one without a year."""
        rows = [{"doc_id": 1, "year": None}, {"doc_id": 2, "assay_id": 2}]
        default = _capricho.flag(_bioactivity(rows)).collect()
        strict = _capricho.flag(_bioactivity(rows), require_doc_date=True).collect()
        assert default["note_cross_document"].to_list() == [True, True]
        assert strict["note_cross_document"].to_list() == [False, False]

    def test_missing_year_counts_for_overlap_only_by_default(self):
        """Assays 1 and 2 share molecule 1, assay 2 has no year; assay 3 shares
        nothing."""
        rows = [
            {"assay_id": 1, "doc_id": 1, "pchembl_value": 6.0},
            {"assay_id": 2, "doc_id": 2, "pchembl_value": 7.0, "year": None},
            {"assay_id": 3, "doc_id": 3, "molregno": 2},
        ]
        default = _capricho.flag(_bioactivity(rows), min_assay_overlap=1).collect()
        strict = _capricho.flag(
            _bioactivity(rows), min_assay_overlap=1, require_doc_date=True
        )
        assert default["drop_low_assay_overlap"].to_list() == [False, False, True]
        assert strict.collect()["drop_low_assay_overlap"].to_list() == [
            True,
            False,
            True,
        ]


class TestCurateAnnotationErrors:
    def test_off_leaves_out_the_flag(self):
        lf = _capricho.flag(_bioactivity([{}]), curate_annotation_errors=False)
        assert "drop_unit_annotation_error" not in lf.collect_schema()


class TestAssaySize:
    def test_database_counts_whole_assay(self, chembl, bioactivity):
        subset = bioactivity.head(3)
        mols = defaultdict(set)
        for r in chembl.lfs["activities"].collect().rows(named=True):
            mols[r["assay_id"]].add(r["molregno"])
        out = _capricho.flag(subset, chembl.lfs, min_assay_size=2).collect()
        assert out["assay_size"].to_list() == [
            len(mols[a]) for a in out["assay_id"].to_list()
        ]
        assert out["drop_assay_size"].to_list() == [
            len(mols[a]) < 2 for a in out["assay_id"].to_list()
        ]

    def test_database_differs_from_frame(self):
        """The frame holds 1 of the 3 molecules assay 1 measured."""
        activities = pl.LazyFrame({"assay_id": [1, 1, 1], "molregno": [1, 2, 3]})
        frame = _bioactivity([{"assay_id": 1, "molregno": 1}])
        out = _capricho.flag(
            frame, {"activities": activities}, min_assay_size=2
        ).collect()
        assert out["assay_size"].to_list() == [3]
        assert out["drop_assay_size"].to_list() == [False]

    def test_frame_counts_given_rows_and_warns_on_auto(self, caplog):
        rows = [{"assay_id": 1, "molregno": m} for m in (1, 2, 2)] + [
            {"assay_id": 2, "molregno": 3}
        ]
        with caplog.at_level(logging.WARNING):
            out = _capricho.flag(_bioactivity(rows), max_assay_size=1).collect()
        assert "given rows only" in caplog.text
        assert out["assay_size"].to_list() == [2, 2, 2, 1]
        assert out["drop_assay_size"].to_list() == [True, True, True, False]

    def test_database_without_tables_raises(self):
        with pytest.raises(ValueError, match="needs `lfs`"):
            _capricho.flag(
                _bioactivity([{}]), min_assay_size=1, assay_size_source="database"
            )


class TestOverlapSkipsSizeFlaggedAssays:
    def test_size_flagged_assay_is_neither_partner_nor_flagged(self):
        """A1's only partner is A2 (M1, 6.0 vs 6.5, other document); A3 shares
        nothing. A2 is too small, so A1 and A3 are left without partners. Were
        A2 not skipped, A1 and A2 would pair and only A3 be flagged."""
        rows = [
            {"assay_id": 1, "molregno": 1, "doc_id": 1, "pchembl_value": 6.0},
            {"assay_id": 1, "molregno": 2, "doc_id": 1, "pchembl_value": 7.0},
            {"assay_id": 2, "molregno": 1, "doc_id": 2, "pchembl_value": 6.5},
            {"assay_id": 3, "molregno": 3, "doc_id": 3, "pchembl_value": 5.0},
            {"assay_id": 3, "molregno": 4, "doc_id": 3, "pchembl_value": 5.5},
        ]
        out = _capricho.flag(
            _bioactivity(rows),
            min_assay_size=2,
            assay_size_source="frame",
            min_assay_overlap=1,
        ).collect()
        assert out["drop_assay_size"].to_list() == [False, False, True, False, False]
        assert out["drop_low_assay_overlap"].to_list() == [
            True,
            True,
            False,
            True,
            True,
        ]


class TestMolFlags:
    """Oracle: CAPRICHO 1.0.4's own chemFilters `ChemStandardizer`,
    `clean_mixtures` and `find_undefined_stereocenters`, without fairfetched."""

    @staticmethod
    def _oracle(smiles: list[str]) -> tuple[list[bool], list[bool]]:
        pytest.importorskip("Capricho")
        from Capricho.core.smiles_utils import (  # pyright: ignore[reportMissingImports]
            clean_mixtures,
        )
        from Capricho.core.stereo import (  # pyright: ignore[reportMissingImports]
            find_undefined_stereocenters,
        )
        from chemFilters.chem.standardizers import (  # pyright: ignore[reportMissingImports]
            ChemStandardizer,
        )

        std = ChemStandardizer(from_smi=True, n_jobs=1, isomeric=True, progress=False)
        cleaned = [clean_mixtures(s) for s in std(smiles)]
        mixture = ["." in s for s in cleaned]
        undefined = [len(find_undefined_stereocenters(s)) > 0 for s in cleaned]
        return mixture, undefined

    def _check(self, out: pl.DataFrame) -> tuple[list[bool], list[bool]]:
        mixture, undefined = self._oracle(out["canonical_smiles"].to_list())
        assert out["drop_mixture"].to_list() == mixture
        assert out["drop_undefined_stereo"].to_list() == undefined
        return mixture, undefined

    def test_demo(self, bioactivity):
        out = _capricho.flag(bioactivity, mol_flags=True, drop_unassigned_chiral=True)
        self._check(out.collect())

    def test_salt_mixture_and_racemate(self):
        """Ethylamine loses its HCl; aspirin + caffeine stays a mixture;
        1-aminoethanol has one undefined center. ChEMBL keeps every fragment of
        sodium acetate and of potassium bromide (all on its salt list); CAPRICHO
        then strips [Na+], leaving acetate, and both [K+] and [Br-], leaving
        nothing, which it counts as a mixture. Copper(II) acetate keeps its
        acetates: ChEMBL's exclusion check applies to the copper parent, and
        chemFilters runs `get_parent_mol` with it before `standardize_mol`."""
        smiles = [
            "CCN.Cl",
            "CC(=O)Oc1ccccc1C(=O)O.Cn1cnc2c1c(=O)n(C)c(=O)n2C",
            "CC(N)O",
            "CC(=O)[O-].[Na+]",
            "[K+].[Br-]",
            "[Cu+2].CC(=O)[O-].CC(=O)[O-]",
        ]
        rows = [{"canonical_smiles": s} for s in smiles]
        serial = _capricho.flag(
            _bioactivity(rows), mol_flags=True, drop_unassigned_chiral=True
        ).collect()
        assert self._check(serial) == (
            [False, True, False, False, True, True],
            [False, False, True, False, False, False],
        )
        parallel = _capricho.flag(
            _bioactivity(rows),
            mol_flags=True,
            drop_unassigned_chiral=True,
            parallel=True,
        )
        assert parallel.collect().equals(serial)

    def test_undefined_stereo_needs_drop_unassigned_chiral(self):
        out = _capricho.flag(_bioactivity([{}]), mol_flags=True).collect()
        assert "drop_undefined_stereo" not in out.columns

    def test_standardization_failure(self):
        """An unclosed ring does not parse; a missing structure is not a failure."""
        rows = [{"canonical_smiles": s} for s in ("C1CC", "CCO", None)]
        out = _capricho.flag(_bioactivity(rows), mol_flags=True).collect()
        assert out["drop_standardization_failed"].to_list() == [True, False, False]
        assert out["drop_mixture"].to_list() == [False, False, False]

    def test_standardization_failure_takes_no_part_in_cross_document(self):
        """The same value in two documents, for a SMILES that does not parse."""
        rows = [
            {"doc_id": 1, "canonical_smiles": "C1CC"},
            {"doc_id": 2, "assay_id": 2, "canonical_smiles": "C1CC"},
        ]
        plain = _capricho.flag(_bioactivity(rows)).collect()
        mol = _capricho.flag(_bioactivity(rows), mol_flags=True).collect()
        assert plain["note_cross_document"].to_list() == [True, True]
        assert mol["note_cross_document"].to_list() == [False, False]

    def test_without_chirality(self):
        """Removing stereo leaves the center of (R)-1-aminoethanol undefined."""
        rows = [{"canonical_smiles": s} for s in ("C[C@@H](N)O", "CC(N)O", "CCO")]
        out = _capricho.flag(
            _bioactivity(rows),
            mol_flags=True,
            chirality=False,
            drop_unassigned_chiral=True,
        ).collect()
        assert out["note_stereo_removed"].to_list() == [True, False, False]
        assert out["drop_undefined_stereo"].to_list() == [True, True, False]

    @pytest.mark.parametrize(
        "kw", [{"chirality": False}, {"drop_unassigned_chiral": True}]
    )
    def test_stereo_options_need_mol_flags(self, kw):
        with pytest.raises(ValueError, match="mol_flags"):
            _capricho.flag(_bioactivity([{}]), **kw)


def _aggregated(rows: list[dict], **kw) -> pl.DataFrame:
    """`aggregate` on rows of `standard_smiles` etc., other columns null."""
    default = {
        "standard_smiles": "CCO",
        "pchembl_value": 6.0,
        "standard_relation": "=",
        "mutation": None,
        "target_chembl_id": "T1",
        "drop_x": False,
    }
    columns = dict.fromkeys(_capricho.MULTIVAL)
    lf = pl.LazyFrame(
        [columns | default | {"activity_id": i} | r for i, r in enumerate(rows)],
        infer_schema_length=None,
    )
    return _capricho.aggregate(lf, **kw).collect()


class TestAggregate:
    def test_geometric_stats(self):
        """scipy `gmean([6, 8]) = sqrt(48)`; `gstd` = exp of the ddof=1 std of
        the logs, for two values |ln 8 - ln 6| / sqrt(2)."""
        out = _aggregated([{"pchembl_value": 6.0}, {"pchembl_value": 8.0}])
        row = out.row(0, named=True)
        assert row["pchembl_value_mean"] == pytest.approx(math.sqrt(48))
        assert row["pchembl_value_std"] == pytest.approx(
            math.exp(math.log(8 / 6) / math.sqrt(2))
        )
        assert row["pchembl_value_median"] == 7.0
        assert row["pchembl_value_counts"] == 2
        assert row["activity_id"] == [0, 1]
        assert row["drop_x"] == [False, False]

    def test_singleton_std_is_null(self):
        assert _aggregated([{}])["pchembl_value_std"].to_list() == [None]

    def test_censored_join_only_equal_values(self):
        out = _aggregated(
            [
                {"standard_relation": "<", "pchembl_value": 5.0},
                {"standard_relation": "<", "pchembl_value": 5.0},
                {"standard_relation": "<", "pchembl_value": 4.0},
            ]
        )
        assert out["pchembl_value_counts"].to_list() == [2, 1]
        assert out["pchembl_value_mean"].to_list() == [5.0, 4.0]
        assert out["pchembl_value_std"].to_list() == [None, None]

    def test_null_relation_is_equal(self):
        out = _aggregated([{"standard_relation": None}, {}])
        assert out["standard_relation"].to_list() == ["="]

    def test_mutants_kept_apart_and_share_identifier_group(self):
        out = _aggregated([{}, {"mutation": "A1B"}])
        assert out["mutation"].to_list() == ["WT", "A1B"]
        assert out["shared_identifier_group"].to_list() == [1, 1]

    def test_aggregate_mutants(self):
        out = _aggregated([{}, {"mutation": "A1B"}], aggregate_mutants=True)
        assert out["mutation"].to_list() == [["WT", "A1B"]]
        assert out["shared_identifier_group"].to_list() == [None]

    ENANTIOMERS = (
        {"standard_smiles": "C[C@H](N)O"},
        {"standard_smiles": "C[C@@H](N)O"},
    )

    def test_enantiomers_apart_with_chirality(self):
        out = _aggregated(list(self.ENANTIOMERS))
        assert out["smiles"].to_list() == ["C[C@H](N)O", "C[C@@H](N)O"]
        assert "might_be_racemic" not in out.columns

    def test_enantiomers_joined_without_chirality(self):
        out = _aggregated(list(self.ENANTIOMERS), chirality=False)
        assert out["smiles"].to_list() == ["CC(N)O"]
        assert out["might_be_racemic"].to_list() == [True]


class TestCache:
    def test_written_once_per_params(self, chembl, tmp_path):
        first = chembl.view.capricho(cache_dir=tmp_path)
        (path,) = tmp_path.rglob("*.parquet")
        mtime = path.stat().st_mtime_ns
        again = chembl.view.capricho(cache_dir=tmp_path)
        assert path.stat().st_mtime_ns == mtime
        assert first.collect().equals(again.collect())
        chembl.view.capricho(cache_dir=tmp_path, min_assay_overlap=1)
        assert len(list(tmp_path.rglob("*.parquet"))) == 2
        assert not list(tmp_path.rglob("*.tmp"))

    def test_cache_holds_flagged_bioactivity(self, chembl, tmp_path):
        cached = chembl.view.capricho(cache_dir=tmp_path).collect()
        direct = _capricho.flag(chembl.view.bioactivity, chembl.lfs).collect()
        assert cached.equals(direct)

    def test_duckdb_engine_not_implemented(self, bioactivity):
        with pytest.raises(NotImplementedError, match="duckdb"):
            _capricho.flag(bioactivity, engine="duckdb")


REFERENCE = Path(__file__).parent / "reference/capricho/1.0.4"
FETCHED = (  # CAPRICHO's SQL WHERE
    "drop_confidence",
    "drop_assay_type",
    "drop_relation",
    "drop_units",
    "drop_standard_type",
    "drop_release",
    "drop_missing_standard_value",
)
CAPRICHO_COMMENTS = {
    "drop_validity_comment": "Data Validity Comment Present",
    "drop_potential_duplicate": "Potential Duplicate",
    "drop_unit_annotation_error": "Unit Annotation Error",
    "drop_mutant": "Mutation keyword in assay description",
    "drop_assay_size": "Assay size",
    "drop_censored_comment": "inactivity-like",
    "drop_low_assay_overlap": "Insufficient assay overlap",
    "drop_mixture": "Mixture in SMILES",
    "drop_undefined_stereo": "Undefined Stereochemistry",
    "note_cross_document": "pChEMBL Duplication Across Documents",
    "note_pchembl_calculated": "Calculated pChEMBL",
    "drop_incompatible_units": "Incompatible units for pChEMBL",
    "note_salt_removed": "Salt/solvent removed",
    "note_stereo_removed": "Stereochemistry removed",
}
FLOAT32_UNIT_ERRORS = (
    "CAPRICHO casts pchembl_value and standard_value to float32, so ~1e-7 noise "
    "hides exact multiples of 3 at tol=1e-9"
)


def _ours(target: str, recipe: dict) -> pl.DataFrame:
    """`flag` on `target` in ChEMBL 36, without the rows CAPRICHO never fetches."""
    if not (CHEMBL_DIR / "36" / "parquet").exists():
        pytest.skip("ChEMBL 36 not downloaded")
    chembl = Chembl.from_version("36")
    bioactivity = chembl.view.bioactivity.filter(target_chembl_id=target)
    lf = _capricho.flag(bioactivity, chembl.lfs, **recipe)
    fetched = cs.by_name(*FETCHED, require_all=False)
    return lf.remove(pl.any_horizontal(fetched)).collect()


def _theirs(name: str) -> pl.DataFrame:
    """CAPRICHO's output and removed rows, the latter marked `removed`."""
    return pl.concat(
        [
            pl.read_parquet(REFERENCE / f"{name}_data_{part}.parquet").with_columns(
                removed=pl.lit(part == "removed_subset")
            )
            for part in ("not_aggregated", "removed_subset")
        ],
        how="diagonal_relaxed",
    )


def _joined(ours: pl.DataFrame, theirs: pl.DataFrame) -> pl.DataFrame:
    comments = pl.concat_str(
        "data_dropping_comment", "data_processing_comment", ignore_nulls=True
    )
    return ours.join(
        theirs.select(
            "activity_id",
            "removed",
            comments=comments.fill_null(""),
            pchembl_theirs=pl.col("pchembl_value").fill_nan(None),
        ),
        on="activity_id",
    )


def _flag_params(flags: Sequence[str], divergences: dict[str, str]) -> list:
    return [
        pytest.param(
            f,
            marks=pytest.mark.xfail(strict=True, reason=divergences[f])
            if f in divergences
            else (),
        )
        for f in flags
    ]


def _disagreeing(joined: pl.DataFrame, flag: str) -> list[int]:
    """`activity_id`s where `flag` and CAPRICHO's comment differ."""
    theirs = joined["comments"].str.contains(CAPRICHO_COMMENTS[flag], literal=True)
    return joined.filter(joined[flag] != theirs)["activity_id"].to_list()


ADORA3_RECIPE = {  # ADORA3_data_recipe.json
    "chembl_release": 31,
    "min_assay_size": 20,
    "max_assay_size": 100,
    "strict_mutant_removal": True,
    "calculate_pchembl": True,
    "require_doc_date": True,
}
ADORA3_DISCARDED = ("drop_missing_year", "drop_missing_structure")  # in neither file
ADORA3_FLAGS = (
    "drop_validity_comment",
    "drop_potential_duplicate",
    "drop_unit_annotation_error",
    "drop_mutant",
    "drop_assay_size",
    "drop_censored_comment",
    "note_cross_document",
    "note_pchembl_calculated",
    "drop_incompatible_units",
    "note_salt_removed",
)
ADORA3_DIVERGENCES = {"drop_unit_annotation_error": f"{FLOAT32_UNIT_ERRORS} (284 rows)"}


@pytest.fixture(scope="module")
def adora3_ours() -> pl.DataFrame:
    return _ours("CHEMBL256", ADORA3_RECIPE)


@pytest.fixture(scope="module")
def adora3_theirs() -> pl.DataFrame:
    return _theirs("ADORA3")


@pytest.fixture(scope="module")
def adora3(adora3_ours, adora3_theirs) -> pl.DataFrame:
    return _joined(adora3_ours, adora3_theirs).join(
        adora3_theirs.select("activity_id", assay_size_theirs="assay_size"),
        on="activity_id",
    )


class TestAdora3AgainstCapricho:
    """ADORA3 (CHEMBL256) on ChEMBL 36, against the output of CAPRICHO 1.0.4 run
    with `ADORA3_RECIPE` (`tests/reference/generate_capricho.py`).

    The reference holds the rows CAPRICHO fetches, split into its output and its
    removed subset; rows without a year or structure are in neither
    (`ADORA3_DISCARDED`). Skipped unless ChEMBL 36 is downloaded.
    """

    def test_same_activities(self, adora3_ours, adora3_theirs):
        kept = adora3_ours.remove(pl.any_horizontal(ADORA3_DISCARDED))
        assert sorted(kept["activity_id"]) == sorted(adora3_theirs["activity_id"])

    def test_removed_are_missing_value(self, adora3):
        assert adora3["drop_missing_value"].equals(adora3["removed"])

    @pytest.mark.parametrize("flag", _flag_params(ADORA3_FLAGS, ADORA3_DIVERGENCES))
    def test_flag(self, adora3, flag):
        assert _disagreeing(adora3, flag) == []

    def test_assay_size(self, adora3):
        assert adora3["assay_size"].equals(adora3["assay_size_theirs"])

    def test_pchembl(self, adora3):
        """CAPRICHO stores float32, so up to ~5e-7 off."""
        ours, theirs = adora3["pchembl_value"], adora3["pchembl_theirs"]
        assert (ours.is_null() == theirs.is_null()).all()
        assert ((ours - theirs).abs().max() or 0) < 1e-6


CHEMBL281_RECIPE = {  # CHEMBL281_data_recipe.json
    "calculate_pchembl": True,
    "min_assay_overlap": 5,
    "mol_flags": True,
    "drop_unassigned_chiral": True,
}
CHEMBL281_DISCARDED = ("drop_missing_structure", "drop_standardization_failed")
CHEMBL281_FLAGS = ("drop_low_assay_overlap", "drop_mixture", "drop_undefined_stereo")
CHEMBL281_DIVERGENCES: dict[str, str] = {}


@pytest.fixture(scope="module")
def chembl281_ours() -> pl.DataFrame:
    return _ours("CHEMBL281", CHEMBL281_RECIPE)


@pytest.fixture(scope="module")
def chembl281_theirs() -> pl.DataFrame:
    return _theirs("CHEMBL281")


@pytest.fixture(scope="module")
def chembl281(chembl281_ours, chembl281_theirs) -> pl.DataFrame:
    return _joined(chembl281_ours, chembl281_theirs)


class TestChembl281AgainstCapricho:
    """CHEMBL281 on ChEMBL 36, against CAPRICHO 1.0.4 with the options ADORA3
    leaves off: mixtures, undefined stereo (`--drop-unassigned-chiral`) and assay
    overlap. Skipped unless ChEMBL 36 is downloaded.
    """

    def test_same_activities(self, chembl281_ours, chembl281_theirs):
        kept = chembl281_ours.remove(pl.any_horizontal(CHEMBL281_DISCARDED))
        assert sorted(kept["activity_id"]) == sorted(chembl281_theirs["activity_id"])

    @pytest.mark.parametrize(
        "flag", _flag_params(CHEMBL281_FLAGS, CHEMBL281_DIVERGENCES)
    )
    def test_flag(self, chembl281, flag):
        assert _disagreeing(chembl281, flag) == []


CHEMBL281_NONDEFAULT_RECIPE = {  # CHEMBL281_nondefault_data_recipe.json
    "calculate_pchembl": True,
    "chembl_release": 30,
    "standard_types": ("Ki", "IC50"),
    "standard_units": ("nM", "uM"),
    "standard_relation": ("=", "<", ">"),
    "confidence_scores": (8, 9),
    "require_doc_date": True,
    "min_assay_overlap": 3,
    "mol_flags": True,
    "chirality": False,
    "drop_unassigned_chiral": True,
    "curate_annotation_errors": False,
}
CHEMBL281_NONDEFAULT_DISCARDED = (
    "drop_missing_year",
    "drop_missing_structure",
    "drop_standardization_failed",
)
CHEMBL281_NONDEFAULT_FLAGS = (
    "drop_validity_comment",
    "drop_potential_duplicate",
    "drop_censored_comment",
    "drop_low_assay_overlap",
    "drop_mixture",
    "drop_undefined_stereo",
    "note_cross_document",
    "note_pchembl_calculated",
    "drop_incompatible_units",
    "note_salt_removed",
    "note_stereo_removed",
)


@pytest.fixture(scope="module")
def chembl281_nondefault_ours() -> pl.DataFrame:
    return _ours("CHEMBL281", CHEMBL281_NONDEFAULT_RECIPE)


@pytest.fixture(scope="module")
def chembl281_nondefault_theirs() -> pl.DataFrame:
    return _theirs("CHEMBL281_nondefault")


@pytest.fixture(scope="module")
def chembl281_nondefault(
    chembl281_nondefault_ours, chembl281_nondefault_theirs
) -> pl.DataFrame:
    return _joined(chembl281_nondefault_ours, chembl281_nondefault_theirs)


class TestChembl281NondefaultAgainstCapricho:
    """CHEMBL281 on ChEMBL 36, against CAPRICHO 1.0.4 with the switches
    CHEMBL281 leaves at their default (`--no-chirality`,
    `--dont-curate-annotation-errors`, `--require-doc-date` with overlap) and
    non-default selections, including `--bioactivity-type` with
    `--calculate-pchembl`. Skipped unless ChEMBL 36 is downloaded.
    """

    def test_same_activities(
        self, chembl281_nondefault_ours, chembl281_nondefault_theirs
    ):
        ours = chembl281_nondefault_ours
        kept = ours.remove(pl.any_horizontal(CHEMBL281_NONDEFAULT_DISCARDED))
        theirs = chembl281_nondefault_theirs
        assert sorted(kept["activity_id"]) == sorted(theirs["activity_id"])

    def test_removed_are_missing_value_or_mixture(self, chembl281_nondefault):
        df = chembl281_nondefault
        ours = df["drop_missing_value"] | df["drop_mixture"]
        assert ours.equals(df["removed"])

    def test_no_unit_annotation_errors(self, chembl281_nondefault):
        assert "drop_unit_annotation_error" not in chembl281_nondefault.columns
        comments = chembl281_nondefault["comments"]
        assert not comments.str.contains("Unit Annotation Error", literal=True).any()

    @pytest.mark.parametrize("flag", CHEMBL281_NONDEFAULT_FLAGS)
    def test_flag(self, chembl281_nondefault, flag):
        assert _disagreeing(chembl281_nondefault, flag) == []


CHEMBL232_RECIPE = {  # CHEMBL232_data_recipe.json
    "calculate_pchembl": True,
    "standard_types": ("Ki", "Inhibition"),
    "mol_flags": True,
}
CHEMBL232_DISCARDED = ("drop_missing_structure", "drop_standardization_failed")
CHEMBL232_FLAGS = (
    "drop_validity_comment",
    "drop_potential_duplicate",
    "drop_unit_annotation_error",
    "drop_censored_comment",
    "drop_mixture",
    "note_cross_document",
    "note_pchembl_calculated",
    "drop_incompatible_units",
    "note_salt_removed",
)
CHEMBL232_DIVERGENCES: dict[str, str] = {}


@pytest.fixture(scope="module")
def chembl232_ours() -> pl.DataFrame:
    return _ours("CHEMBL232", CHEMBL232_RECIPE)


@pytest.fixture(scope="module")
def chembl232_theirs() -> pl.DataFrame:
    return _theirs("CHEMBL232")


@pytest.fixture(scope="module")
def chembl232(chembl232_ours, chembl232_theirs) -> pl.DataFrame:
    return _joined(chembl232_ours, chembl232_theirs)


class TestChembl232AgainstCapricho:
    """CHEMBL232 on ChEMBL 36, against CAPRICHO 1.0.4 with `--bioactivity-type
    Ki,Inhibition --calculate-pchembl`: real "Log Ki" rows (admitted through
    CAPRICHO's type expansion), inactivity comments and percent units.
    Skipped unless ChEMBL 36 is downloaded.
    """

    def test_same_activities(self, chembl232_ours, chembl232_theirs):
        kept = chembl232_ours.remove(pl.any_horizontal(CHEMBL232_DISCARDED))
        assert sorted(kept["activity_id"]) == sorted(chembl232_theirs["activity_id"])

    def test_recipe_has_log_types(self, chembl232_theirs):
        """ChEMBL has "Log Ki" rows here; CAPRICHO fetches and then drops them,
        which `test_same_activities` checks."""
        log_ki = Chembl.from_version("36").view.bioactivity.filter(
            target_chembl_id="CHEMBL232", standard_type="Log Ki"
        )
        assert log_ki.select(pl.len()).collect().item() > 0
        assert not (chembl232_theirs["standard_type"] == "Log Ki").any()

    def test_removed_are_missing_value_or_mixture(self, chembl232):
        ours = chembl232["drop_missing_value"] | chembl232["drop_mixture"]
        assert ours.equals(chembl232["removed"])

    @pytest.mark.parametrize(
        "flag", _flag_params(CHEMBL232_FLAGS, CHEMBL232_DIVERGENCES)
    )
    def test_flag(self, chembl232, flag):
        assert _disagreeing(chembl232, flag) == []

    def test_pchembl(self, chembl232):
        """CAPRICHO stores float32, so up to ~5e-7 off."""
        ours, theirs = chembl232["pchembl_value"], chembl232["pchembl_theirs"]
        assert (ours.is_null() == theirs.is_null()).all()
        assert ((ours - theirs).abs().max() or 0) < 1e-6


def _activity_ids(ids: pl.Expr) -> pl.Expr:
    """Sorted, "|"-joined `activity_id`s: a group's join key."""
    return ids.list.sort().list.eval(pl.element().cast(pl.String)).list.join("|")


def _aggregated_pair(ours: pl.DataFrame, name: str, chirality: bool) -> pl.DataFrame:
    """`aggregate` on the rows CAPRICHO aggregates (`_not_aggregated`), in its
    order (a group's `smiles` is its first row), joined to CAPRICHO's
    `aggregate_data` on the same rows (`_theirs` suffix)."""
    kept = pl.read_parquet(REFERENCE / f"{name}_data_not_aggregated.parquet")
    ours = kept.select("activity_id").join(
        ours, on="activity_id", how="left", maintain_order="left"
    )
    agg = _capricho.aggregate(ours.lazy(), chirality=chirality).collect()
    theirs = pl.read_parquet(REFERENCE / f"{name}_data_aggregated.parquet")
    ids = pl.col("activity_id").str.split("|").cast(pl.List(pl.Int64))
    return agg.with_columns(ids=_activity_ids(pl.col("activity_id"))).join(
        theirs.with_columns(ids=_activity_ids(ids)),
        on="ids",
        how="full",
        suffix="_theirs",
    )


@pytest.fixture(
    scope="module",
    params=[
        ("CHEMBL281", CHEMBL281_RECIPE),
        ("CHEMBL281_nondefault", CHEMBL281_NONDEFAULT_RECIPE),
    ],
    ids=lambda p: p[0],
)
def aggregated(request) -> pl.DataFrame:
    name, recipe = request.param
    chirality = recipe.get("chirality", True)
    return _aggregated_pair(_ours("CHEMBL281", recipe), name, chirality)


class TestAggregateAgainstCapricho:
    """`aggregate` against CAPRICHO 1.0.4 `aggregate_data` on CAPRICHO's own
    pre-aggregation rows (`generate_capricho.aggregate_reference`). Groups are
    matched by their `activity_id`s. Skipped unless ChEMBL 36 is downloaded.
    """

    def test_same_groups(self, aggregated):
        assert aggregated["ids"].null_count() == 0
        assert aggregated["ids_theirs"].null_count() == 0

    @pytest.mark.parametrize(
        "column",
        [
            "smiles",
            "connectivity",
            "inchikey",
            "shared_identifier_group",
            "standard_relation",
            "mutation",
            "pchembl_value_counts",
        ],
    )
    def test_column(self, aggregated, column):
        assert aggregated[column].equals(aggregated[f"{column}_theirs"])

    def test_might_be_racemic(self, aggregated):
        if "might_be_racemic_theirs" not in aggregated.columns:
            assert "might_be_racemic" not in aggregated.columns
        else:
            assert aggregated["might_be_racemic"].equals(
                aggregated["might_be_racemic_theirs"]
            )

    @pytest.mark.parametrize("stat", ["mean", "median", "std"])
    def test_stat(self, aggregated, stat):
        """CAPRICHO rounds values to 2 decimals first (here at most 0.005 off); its NaN
        (log of a negative calculated pChEMBL) is stored as null."""
        ours = aggregated[f"pchembl_value_{stat}"].fill_nan(None)
        theirs = aggregated[f"pchembl_value_{stat}_theirs"]
        assert (ours.is_null() == theirs.is_null()).all()
        assert ((ours - theirs).abs().max() or 0) <= 0.005
