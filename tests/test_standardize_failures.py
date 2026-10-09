"""Failure records (`fairfetched.prep.failures`): every step failure and
warning becomes one logging record naming step, molecule, error and the user
line that built the expression; `FailureLedger` collects them and
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

from fairfetched.prep.failures import (
    SCHEMA,
    FailureLedger,
    mol_id,
    read_failure_log,
)
from fairfetched.prep.mol_expr import MolExpr
from fairfetched.prep.mol_functions import canonical_tautomer, safe_step

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


def rows(ledger: FailureLedger, *columns: str) -> list[tuple]:
    return ledger.frame().select(columns).rows()


class TestRecords:
    def test_step_failure_names_step_error_and_canonical_smiles(self) -> None:
        with FailureLedger() as ledger:
            pl.DataFrame({"smiles": GOOD}).with_columns(
                MolExpr.from_smiles("smiles").standardize(boom).to_smiles()
            )
        assert rows(ledger, "level", "step", "mol", "error") == [
            ("ERROR", "boom", CANONICAL[s], "ValueError: kaboom") for s in GOOD
        ]

    def test_parse_failure_carries_rdkit_message_and_input_text(self) -> None:
        with FailureLedger() as ledger:
            pl.DataFrame({"smiles": [BAD]}).with_columns(
                MolExpr.from_smiles("smiles").to_smiles()
            )
        assert rows(ledger, "step", "mol", "error") == [
            ("_smiles_to_binary", BAD, RDKIT_PARSE_ERROR)
        ]

    def test_valence_failure_is_reported_without_the_rdkit_log(self) -> None:
        """Sanitization problems come from `DetectChemistryProblems`, so they
        read the same whether or not RDKit's log is on."""
        (problem,) = Chem.DetectChemistryProblems(
            Chem.MolFromSmiles(PENTAVALENT, sanitize=False)
        )
        rdBase.DisableLog("rdApp.error")
        with FailureLedger() as ledger:
            pl.DataFrame({"smiles": [PENTAVALENT]}).with_columns(
                MolExpr.from_smiles("smiles").to_smiles()
            )
        assert rows(ledger, "mol", "error") == [(PENTAVALENT, problem.Message())]

    def test_syntax_failure_with_rdkit_log_disabled_says_so(self) -> None:
        rdBase.DisableLog("rdApp.error")
        with FailureLedger() as ledger:
            pl.DataFrame({"smiles": [BAD]}).with_columns(
                MolExpr.from_smiles("smiles").to_smiles()
            )
        assert rows(ledger, "error") == [
            ("MolFromSmiles returned None (RDKit log disabled)",)
        ]

    def test_success_and_null_input_leave_no_record(self) -> None:
        with FailureLedger() as ledger:
            pl.DataFrame({"smiles": [*GOOD, None]}).with_columns(
                MolExpr.from_smiles("smiles").to_inchikey()
            )
        assert ledger.failures == []

    def test_callsite_is_the_user_line(self) -> None:
        with FailureLedger() as ledger:
            line = inspect.currentframe().f_lineno + 1  # type: ignore[union-attr]
            expr = MolExpr.from_smiles("smiles").to_smiles()
            pl.DataFrame({"smiles": [BAD]}).with_columns(expr)
        assert rows(ledger, "file", "lineno") == [(HERE, line)]
        assert "MolExpr.from_smiles" in ledger.failures[0].code  # type: ignore[operator]

    def test_callsite_through_the_mol_namespace(self) -> None:
        """`pl.col(...).mol` builds the MolExpr from inside polars; the record
        must still cite this file, not polars or the dataclass `__init__`."""
        with FailureLedger() as ledger:
            line = inspect.currentframe().f_lineno + 1  # type: ignore[union-attr]
            expr = pl.col("mol").mol.to_smiles()  # pyright: ignore[reportAttributeAccessIssue]
            pl.DataFrame({"mol": [b"junk"]}).with_columns(expr)
        assert rows(ledger, "step", "file", "lineno") == [
            ("_binary_to_smiles", HERE, line)
        ]

    def test_chained_expression_shares_one_callsite(self) -> None:
        """Parse failure and step failure from one chain cite the same line."""
        with FailureLedger() as ledger:
            pl.DataFrame({"smiles": [BAD, *GOOD]}).with_columns(
                MolExpr.from_smiles("smiles").standardize(boom).to_smiles()
            )
        steps = ledger.frame()["step"].to_list()
        assert sorted(steps) == ["_smiles_to_binary", "boom", "boom"]
        assert ledger.frame()["lineno"].n_unique() == 1

    @pytest.mark.parametrize("dedup,n", [(True, 1), (False, 3)], ids=["dedup", "all"])
    def test_dedup_records_once_per_unique_input(self, dedup: bool, n: int) -> None:
        with FailureLedger() as ledger:
            pl.DataFrame({"smiles": [BAD] * 3}).with_columns(
                MolExpr.from_smiles("smiles", dedup=dedup).to_smiles()
            )
        assert len(ledger.failures) == n

    def test_parallel_workers_records_reach_the_parent(self) -> None:
        with FailureLedger() as ledger:
            pl.DataFrame({"smiles": [BAD, *GOOD]}).with_columns(
                MolExpr.from_smiles("smiles")
                .standardize(boom, parallel=True)
                .to_smiles()
            )
        assert sorted(rows(ledger, "step", "mol", "error")) == sorted(
            [
                (
                    "_smiles_to_binary",
                    BAD,
                    RDKIT_PARSE_ERROR,
                ),  # serial, before the pool
                *[("boom", CANONICAL[s], "ValueError: kaboom") for s in GOOD],
            ]
        )

    def test_records_reach_standard_logging_without_a_ledger(self, caplog) -> None:
        with caplog.at_level(logging.ERROR, logger="fairfetched.prep"):
            pl.DataFrame({"smiles": [BAD]}).with_columns(
                MolExpr.from_smiles("smiles").to_smiles()
            )
        # RDKit's own message is on the `rdkit` logger and also propagates to root
        (record,) = [r for r in caplog.records if r.name == "fairfetched.prep"]
        assert record.failure.step == "_smiles_to_binary"  # type: ignore[attr-defined]
        assert "fairfetched.failure\t" in record.getMessage()


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
        with FailureLedger() as ledger:
            out = pl.DataFrame({"smiles": [self.ENOL]}).with_columns(
                MolExpr.from_smiles("smiles")
                .standardize(canonical_tautomer(max_tautomers=1))
                .to_smiles()
                .alias("out")
            )
        assert out["out"][0] is not None
        assert rows(ledger, "level", "step", "error") == [
            ("WARNING", "canonical_tautomer", f"tautomer enumeration {status}")
        ]

    def test_complete_enumeration_is_silent(self) -> None:
        with FailureLedger() as ledger:
            pl.DataFrame({"smiles": [self.ENOL]}).with_columns(
                MolExpr.from_smiles("smiles")
                .standardize(canonical_tautomer())
                .to_smiles()
            )
        assert ledger.failures == []


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
        assert out.schema == pl.Schema(SCHEMA)
        assert out.select("step", "mol", "error").rows() == [
            ("_smiles_to_binary", BAD, RDKIT_PARSE_ERROR),
            *[("boom", CANONICAL[s], "ValueError: kaboom") for s in GOOD],
        ]
        assert out["file"].to_list() == [HERE] * 3
        assert out["time"].null_count() == 0

    def test_parser_matches_ledger(self, tmp_path: Path) -> None:
        path = tmp_path / "run.log"
        handler = logging.FileHandler(path)
        logging.getLogger("fairfetched.prep").addHandler(handler)
        try:
            with FailureLedger() as ledger:
                pl.DataFrame({"smiles": [BAD]}).with_columns(
                    MolExpr.from_smiles("smiles").to_smiles()
                )
        finally:
            logging.getLogger("fairfetched.prep").removeHandler(handler)
            handler.close()
        assert read_failure_log(path).equals(ledger.frame())

    def test_file_without_records_gives_empty_frame(self, tmp_path: Path) -> None:
        path = tmp_path / "quiet.log"
        path.write_text("INFO something else\n")
        out = read_failure_log(path)
        assert out.is_empty() and out.schema == pl.Schema(SCHEMA)
