"""Failure records (`fairfetched.prep.failures`): every step failure, warning
and filter drop becomes one record naming step, molecule, reason and the user
line that built the expression; `FailureCollector` receives them while open,
`logging` receives them subject to the user's configuration, and
`read_failure_log` recovers them from any log file.

Expected values come from outside the code under test: RDKit's own parse
message and tautomer status, `inspect` for the line number, and literal
strings for everything else.
"""

import inspect
import logging
from pathlib import Path

import polars as pl
import pytest
from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize.rdMolStandardize import TautomerEnumerator
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt

from fairfetched.prep import failures as fx
from fairfetched.prep.failures import (
    Failure,
    FailureCollector,
    mol_id,
    read_failure_log,
)
from fairfetched.prep.mol_expr import MolExpr
from fairfetched.prep.mol_functions import canonical_tautomer, safe_step
from fairfetched.prep.pipeline import STEPS_CHEMBL, STEPS_PAPYRUS

GOOD = ["CCO", "c1ccccc1"]
BAD = "C1CC"
PENTAVALENT = "CN(C)(C)(C)C"
CANONICAL = {s: Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in GOOD}
RDKIT_PARSE_ERROR = f"SMILES Parse Error: unclosed ring for input: '{BAD}'"
HERE = str(Path(__file__).resolve())


@pytest.fixture(autouse=True)
def rdkit_error_log_enabled():
    """Other test modules call `RDLogger.DisableLog("rdApp.*")` at import."""
    rdBase.EnableLog("rdApp.error")
    yield
    rdBase.DisableLog("rdApp.error")


@safe_step
def boom(mol):
    raise ValueError("kaboom")


def rows(collected: FailureCollector, *columns: str) -> list[tuple]:
    return collected.frame().select(columns).rows()


def parse(smiles: list[str | None]) -> pl.DataFrame:
    return pl.DataFrame({"smiles": smiles}).with_columns(
        MolExpr.from_smiles("smiles").to_smiles()
    )


class TestRecords:
    def test_step_failure_names_step_error_and_canonical_smiles(self) -> None:
        with FailureCollector() as failures:
            pl.DataFrame({"smiles": GOOD}).with_columns(
                MolExpr.from_smiles("smiles").standardize(boom).to_smiles()
            )
        assert rows(failures, "level", "step", "mol", "error") == [
            ("ERROR", "boom", CANONICAL[s], "ValueError: kaboom") for s in GOOD
        ]

    def test_parse_failure_carries_rdkit_message_and_input_text(self) -> None:
        with FailureCollector() as failures:
            parse([BAD])
        assert rows(failures, "step", "mol", "error") == [
            ("_smiles_to_binary", BAD, RDKIT_PARSE_ERROR)
        ]

    def test_valence_failure_is_reported_without_the_rdkit_log(self) -> None:
        """Sanitization problems come from `DetectChemistryProblems`, so they
        read the same whether or not RDKit's log is on."""
        (problem,) = Chem.DetectChemistryProblems(
            Chem.MolFromSmiles(PENTAVALENT, sanitize=False)
        )
        rdBase.DisableLog("rdApp.error")
        with FailureCollector() as failures:
            parse([PENTAVALENT])
        assert rows(failures, "mol", "error") == [(PENTAVALENT, problem.Message())]

    def test_syntax_failure_with_rdkit_log_disabled_says_so(self) -> None:
        rdBase.DisableLog("rdApp.error")
        with FailureCollector() as failures:
            parse([BAD])
        assert rows(failures, "error") == [
            ("MolFromSmiles returned None (RDKit log disabled)",)
        ]

    def test_success_and_null_input_leave_no_record(self) -> None:
        with FailureCollector() as failures:
            pl.DataFrame({"smiles": [*GOOD, None]}).with_columns(
                MolExpr.from_smiles("smiles").to_inchikey()
            )
        assert failures.failures == []

    def test_callsite_is_the_user_line(self) -> None:
        with FailureCollector() as failures:
            line = inspect.currentframe().f_lineno + 1  # type: ignore[union-attr]
            expr = MolExpr.from_smiles("smiles").to_smiles()
            pl.DataFrame({"smiles": [BAD]}).with_columns(expr)
        assert rows(failures, "file", "lineno") == [(HERE, line)]
        assert "MolExpr.from_smiles" in failures.failures[0].code  # type: ignore[operator]

    def test_callsite_through_the_mol_namespace(self) -> None:
        """`pl.col(...).mol` builds the MolExpr from inside polars; the record
        must still cite this file, not polars or the dataclass `__init__`."""
        with FailureCollector() as failures:
            line = inspect.currentframe().f_lineno + 1  # type: ignore[union-attr]
            expr = pl.col("mol").mol.to_smiles()  # pyright: ignore[reportAttributeAccessIssue]
            pl.DataFrame({"mol": [b"junk"]}).with_columns(expr)
        assert rows(failures, "step", "file", "lineno") == [
            ("_binary_to_smiles", HERE, line)
        ]

    def test_chained_expression_shares_one_callsite(self) -> None:
        """Parse failure and step failure from one chain cite the same line."""
        with FailureCollector() as failures:
            pl.DataFrame({"smiles": [BAD, *GOOD]}).with_columns(
                MolExpr.from_smiles("smiles").standardize(boom).to_smiles()
            )
        steps = failures.frame()["step"].to_list()
        assert sorted(steps) == ["_smiles_to_binary", "boom", "boom"]
        assert failures.frame()["lineno"].n_unique() == 1

    @pytest.mark.parametrize("dedup,n", [(True, 1), (False, 3)], ids=["dedup", "all"])
    def test_dedup_records_once_per_unique_input(self, dedup: bool, n: int) -> None:
        with FailureCollector() as failures:
            pl.DataFrame({"smiles": [BAD] * 3}).with_columns(
                MolExpr.from_smiles("smiles", dedup=dedup).to_smiles()
            )
        assert len(failures.failures) == n

    def test_parallel_workers_records_reach_the_parent(self) -> None:
        with FailureCollector() as failures:
            pl.DataFrame({"smiles": [BAD, *GOOD]}).with_columns(
                MolExpr.from_smiles("smiles")
                .standardize(boom, parallel=True)
                .to_smiles()
            )
        assert sorted(rows(failures, "step", "mol", "error")) == sorted(
            [
                (
                    "_smiles_to_binary",
                    BAD,
                    RDKIT_PARSE_ERROR,
                ),  # serial, before the pool
                *[("boom", CANONICAL[s], "ValueError: kaboom") for s in GOOD],
            ]
        )

    def test_parallel_workers_write_nothing_to_stderr(self, capfd) -> None:
        """Only the parent logs; a worker logging too would reach the inherited
        stderr through `logging.lastResort` and print every record twice."""
        pl.DataFrame({"smiles": GOOD}).with_columns(
            MolExpr.from_smiles("smiles").standardize(boom, parallel=True)
        )
        assert fx.MARKER not in capfd.readouterr().err

    def test_records_reach_standard_logging_without_a_collector(self, caplog) -> None:
        with caplog.at_level(logging.ERROR, logger="fairfetched.prep"):
            parse([BAD])
        # RDKit's own message is on the `rdkit` logger and also propagates to root
        (record,) = [r for r in caplog.records if r.name == "fairfetched.prep"]
        assert record.failure.step == "_smiles_to_binary"  # type: ignore[attr-defined]
        assert "fairfetched.failure\t" in record.getMessage()


class TestCollector:
    """A collector is a plain registry entry: every open collector gets every
    record, a closed one gets nothing, and the logger is never reconfigured."""

    def test_nested_collectors_both_receive(self) -> None:
        with FailureCollector() as outer:
            parse([BAD])
            with FailureCollector() as inner:
                parse([BAD])
        assert len(outer.failures) == 2
        assert len(inner.failures) == 1

    def test_closed_collector_receives_nothing_and_leaves_the_registry(self) -> None:
        with FailureCollector() as failures:
            pass
        parse([BAD])
        assert failures.failures == []
        assert fx._COLLECTORS == []

    def test_logger_level_and_handlers_are_untouched(self) -> None:
        logger = logging.getLogger("fairfetched.prep")
        before = (logger.level, list(logger.handlers), logger.propagate)
        with FailureCollector():
            assert (logger.level, list(logger.handlers), logger.propagate) == before
            parse([BAD])
        assert (logger.level, list(logger.handlers), logger.propagate) == before


class TestMolId:
    def test_string_is_itself(self) -> None:
        assert mol_id(BAD) == BAD

    def test_binary_is_canonical_smiles(self) -> None:
        b = Chem.MolFromSmiles("OCC").ToBinary()
        assert mol_id(b) == "CCO"

    def test_unreadable_binary_falls_back_to_base64(self) -> None:
        assert mol_id(b"not a mol").startswith("b64:")


class TestWarnings:
    ENOL = "OC(=CC(=O)c1ccccc1)c1ccccc1"

    def test_truncated_enumeration_warns_and_still_returns(self) -> None:
        enumerator = TautomerEnumerator()
        enumerator.SetMaxTautomers(1)
        status = enumerator.Enumerate(Chem.MolFromSmiles(self.ENOL)).status.name
        with FailureCollector() as failures:
            out = pl.DataFrame({"smiles": [self.ENOL]}).with_columns(
                MolExpr.from_smiles("smiles")
                .standardize(canonical_tautomer(max_tautomers=1))
                .to_smiles()
                .alias("out")
            )
        assert out["out"][0] is not None
        assert rows(failures, "level", "step", "error") == [
            ("WARNING", "canonical_tautomer", f"tautomer enumeration {status}")
        ]

    def test_complete_enumeration_is_silent(self) -> None:
        with FailureCollector() as failures:
            pl.DataFrame({"smiles": [self.ENOL]}).with_columns(
                MolExpr.from_smiles("smiles")
                .standardize(canonical_tautomer())
                .to_smiles()
            )
        assert failures.failures == []

    def test_chembl_standardize_warns_where_the_library_gate_skips(self) -> None:
        """`standardize_mol()` returns a mol with a METAL_LIST atom untouched,
        so the ammonium stays charged; the record says so."""
        pytest.importorskip("chembl_structure_pipeline")
        with FailureCollector() as failures:
            out = pl.DataFrame({"smiles": ["CC[NH3+].[Fe]", "CC[NH3+]"]}).with_columns(
                MolExpr.from_smiles("smiles").standardize(*STEPS_CHEMBL).to_smiles()
            )
        assert out["smiles"].to_list() == ["CC[NH3+].[Fe]", "CCN"]
        assert rows(failures, "level", "step", "mol") == [
            ("WARNING", "chembl_standardize", "CC[NH3+].[Fe]")
        ]


class TestFilterNotes:
    """Filter steps drop by design; each drop is one INFO record naming the step,
    so a filtered row is told from a failed one. Inputs are the Papyrus
    `standardize()` rejections (inorganic, mixture, under 200 Da)."""

    SILANE = "C[Si](C)(C)c1ccc(cc1)C(=O)Nc1ccc(cc1)C(C)(C)C"
    MIXTURE = "CCCCCCCCCCc1ccccc1.CCCCCCCCCCCc1ccccc1"
    SMALL = "CC[NH3+]"

    def test_papyrus_rejections_each_leave_one_note(self) -> None:
        pytest.importorskip("chembl_structure_pipeline")
        with FailureCollector() as failures:
            out = pl.DataFrame(
                {"smiles": [self.SILANE, self.MIXTURE, self.SMALL]}
            ).with_columns(MolExpr.from_smiles("smiles").standardize(*STEPS_PAPYRUS))
        assert out["smiles"].to_list() == [None, None, None]
        assert rows(failures, "level", "step") == [
            ("INFO", "papyrus_only_organic"),
            ("INFO", "papyrus_no_mixtures"),
            ("INFO", "mw_between"),
        ]

    def test_mw_note_states_the_mass_and_window(self) -> None:
        mass = CalcExactMolWt(Chem.MolFromSmiles("CCO"))
        with FailureCollector() as failures:
            pl.DataFrame({"smiles": ["CCO"]}).with_columns(
                MolExpr.from_smiles("smiles").mw_between(200, 800)
            )
        assert rows(failures, "error") == [
            (f"exact mass {mass:.2f} outside [200, 800]",)
        ]

    def test_notes_are_not_lost_in_parallel_workers(self) -> None:
        with FailureCollector() as failures:
            pl.DataFrame({"smiles": ["CCO", "CCCO"]}).with_columns(
                MolExpr.from_smiles("smiles").mw_between(200, 800, parallel=True)
            )
        assert rows(failures, "level", "step") == [("INFO", "mw_between")] * 2

    def test_notes_reach_logging_only_at_info_level(self, caplog) -> None:
        expr = MolExpr.from_smiles("smiles").mw_between(200, 800)
        with caplog.at_level(logging.WARNING, logger="fairfetched.prep"):
            pl.DataFrame({"smiles": ["CCO"]}).with_columns(expr)
        assert [r for r in caplog.records if r.name == "fairfetched.prep"] == []
        with caplog.at_level(logging.INFO, logger="fairfetched.prep"):
            pl.DataFrame({"smiles": ["CCO"]}).with_columns(expr)
        (record,) = [r for r in caplog.records if r.name == "fairfetched.prep"]
        assert record.failure.step == "mw_between"  # type: ignore[attr-defined]


class TestLogFile:
    FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

    def test_parser_recovers_records_from_a_mixed_log(self, tmp_path: Path) -> None:
        path = tmp_path / "run.log"
        handler = logging.FileHandler(path)
        handler.setFormatter(logging.Formatter(self.FORMAT))
        root = logging.getLogger()
        root.addHandler(handler)
        try:
            logging.getLogger("other").error("noise before")
            pl.DataFrame({"smiles": [BAD, *GOOD]}).with_columns(
                MolExpr.from_smiles("smiles").standardize(boom).to_smiles()
            )
            logging.getLogger("other").error("noise after")
        finally:
            root.removeHandler(handler)
            handler.close()
        out = read_failure_log(path)
        assert out.schema == pl.Schema(Failure.schema())
        assert out.select("step", "mol", "error").rows() == [
            ("_smiles_to_binary", BAD, RDKIT_PARSE_ERROR),
            *[("boom", CANONICAL[s], "ValueError: kaboom") for s in GOOD],
        ]
        assert out["file"].to_list() == [HERE] * 3
        assert out["time"].null_count() == 0

    def test_parser_matches_collector(self, tmp_path: Path) -> None:
        path = tmp_path / "run.log"
        handler = logging.FileHandler(path)
        logging.getLogger("fairfetched.prep").addHandler(handler)
        try:
            with FailureCollector() as failures:
                parse([BAD])
        finally:
            logging.getLogger("fairfetched.prep").removeHandler(handler)
            handler.close()
        assert read_failure_log(path).equals(failures.frame())

    def test_file_without_records_gives_empty_frame(self, tmp_path: Path) -> None:
        path = tmp_path / "quiet.log"
        path.write_text("INFO something else\n")
        out = read_failure_log(path)
        assert out.is_empty() and out.schema == pl.Schema(Failure.schema())
