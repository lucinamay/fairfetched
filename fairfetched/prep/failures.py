"""Failure records for `fairfetched.prep`: every step that returns None
because it raised, and every warning a step chooses to emit, becomes one
`logging` record on the `fairfetched.prep` logger that names the step,
the molecule, the error, and the line of user code that built the expression.

Nothing has to be switched on. The records flow through whatever logging
configuration exists; `FailureLedger` collects them into a DataFrame and
`read_failure_log` recovers them from a log file written by any formatter.

    >>> import logging; logging.basicConfig(filename="run.log")
    >>> with FailureLedger() as ledger:
    ...     out = df.with_columns(
    ...         MolExpr.from_smiles("smiles").standardize(*STEPS_CHEMBL).to_inchikey()
    ...     )
    >>> ledger.frame()
    shape: (2, 8)
    ┌─────────────────────┬───────┬───────────┬────────┬──────────┬───────────────────┬───────┬─────────────────────────────┐
    │ time                ┆ level ┆ file      ┆ lineno ┆ code     ┆ step              ┆ mol   ┆ error                       │
    ├─────────────────────┼───────┼───────────┼────────┼──────────┼───────────────────┼───────┼─────────────────────────────┤
    │ 2026-10-08 13:46:20 ┆ ERROR ┆ notebook… ┆ 3      ┆ out = d… ┆ _smiles_to_binary ┆ C1CC  ┆ SMILES Parse Error: unclos… │
    │ 2026-10-08 13:46:20 ┆ ERROR ┆ notebook… ┆ 3      ┆ out = d… ┆ chembl_normalize  ┆ CCO[… ┆ ValueError: …               │
    └─────────────────────┴───────┴───────────┴────────┴──────────┴───────────────────┴───────┴─────────────────────────────┘
    >>> read_failure_log("run.log").equals(ledger.frame())
    True

`mol` is the input text for parse failures and RDKit's canonical SMILES of the
structure entering the failing expression otherwise, so a back-join uses
`MolExpr.from_smiles("smiles").to_smiles()` on the source frame, or the raw
column for parse failures:

    >>> df.join(ledger.frame(), left_on="smiles", right_on="mol", how="inner")

`parallel=True` workers collect their records and the parent re-emits them,
so the ledger sees the same rows either way.
"""

from __future__ import annotations

import base64
import logging
import traceback
from contextvars import ContextVar
from dataclasses import astuple, dataclass, fields
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Self

import polars as pl
from rdkit.Chem import Mol, MolToSmiles

logger = logging.getLogger("fairfetched.prep")
MARKER = "fairfetched.failure\t"
_SKIP = (str(Path(__file__).resolve().parent.parent), str(Path(pl.__file__).parent))


@dataclass(frozen=True)
class CallSite:
    file: str
    lineno: int | None
    code: str | None


@dataclass(frozen=True)
class Failure:
    time: str
    level: str
    file: str
    lineno: int | None
    code: str | None
    step: str
    mol: str
    error: str

    def message(self) -> str:
        """One line, tab-separated after `MARKER`, so any formatter keeps it parseable."""
        cells = (
            "" if v is None else str(v).replace("\t", " ").replace("\n", " ")
            for v in astuple(self)
        )
        return MARKER + "\t".join(cells)


SCHEMA = {
    "time": pl.Datetime("ms"),
    "level": pl.String,
    "file": pl.String,
    "lineno": pl.Int64,
    "code": pl.String,
    "step": pl.String,
    "mol": pl.String,
    "error": pl.String,
}
assert tuple(SCHEMA) == tuple(f.name for f in fields(Failure))
_TEXT_SCHEMA = {**SCHEMA, "time": pl.String}
_PARSE_TIME = pl.col("time").str.to_datetime("%Y-%m-%d %H:%M:%S", time_unit="ms")

# (call site, element being processed) for the duration of one step call;
# `_map` sets it per element, so `safure_step` never has to be told who failed
_CONTEXT: ContextVar[tuple[CallSite, object]] = ContextVar("fairfetched_mol_context")
_UNKNOWN = CallSite("?", None, None)


def callsite() -> CallSite:
    """First stack frame outside the fairfetched package: the user line that
    built the expression. In Jupyter the file is `.../ipykernel_<pid>/<n>.py`
    with the cell source resolved by linecache; marimo and `python -i` give
    `<cell-…>` or `<stdin>` and no source line."""
    for frame in reversed(traceback.extract_stack()):
        if frame.filename.startswith(_SKIP) or frame.filename == "<string>":
            continue  # fairfetched, polars (`.mol` namespace), dataclass __init__
        return CallSite(frame.filename, frame.lineno, frame.line or None)
    return _UNKNOWN


def mol_id(element: object) -> str:
    """Text identifying an input: the string itself, RDKit canonical SMILES for
    mol binary, base64 of the binary where even SMILES writing fails (lossless:
    `Chem.Mol(base64.b64decode(...))`)."""
    if isinstance(element, str):
        return element
    if isinstance(element, bytes):
        try:
            return MolToSmiles(Mol(element))  # ty: ignore[no-matching-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
        except Exception:  # noqa: BLE001  # any RDKit failure falls through to the lossless form
            return "b64:" + base64.b64encode(element).decode()
    return repr(element)


def record(step: str, error: str, level: int = logging.ERROR) -> None:
    """Emit one failure record for the element currently in `_CONTEXT`."""
    site, element = _CONTEXT.get((_UNKNOWN, None))
    failure = Failure(
        time=datetime.now().isoformat(sep=" ", timespec="seconds"),  # noqa: DTZ005  # local, as logging's asctime
        level=logging.getLevelName(level),
        file=site.file,
        lineno=site.lineno,
        code=site.code,
        step=step,
        mol=mol_id(element),
        error=error,
    )
    logger.log(level, failure.message(), extra={"failure": failure})


def warn(step: str, message: str) -> None:
    record(step, message, logging.WARNING)


def call_in_context(fn, site: CallSite, element):
    """`fn(element)` with `_CONTEXT` naming `element` for any record it emits."""
    token = _CONTEXT.set((site, element))
    try:
        return fn(element)
    finally:
        _CONTEXT.reset(token)


class FailureLedger(logging.Handler):
    """Collects `Failure` records while attached; `with` attaches and detaches.

    `frame()` has one row per record with the `SCHEMA` columns.
    """

    def __init__(self) -> None:
        super().__init__()
        self.failures: list[Failure] = []

    def emit(self, record: logging.LogRecord) -> None:
        failure = getattr(record, "failure", None)
        if failure is not None:
            self.failures.append(failure)

    def frame(self) -> pl.DataFrame:
        rows = [astuple(f) for f in self.failures]
        return pl.DataFrame(rows, schema=_TEXT_SCHEMA, orient="row").with_columns(
            _PARSE_TIME
        )

    def __enter__(self) -> Self:
        logger.addHandler(self)
        return self

    def __exit__(self, *exc) -> None:
        logger.removeHandler(self)


# --- parallel workers: records are collected per call and re-emitted by the parent ---

_worker_ledger: FailureLedger | None = None


def _worker_ledger_installed() -> FailureLedger:
    global _worker_ledger
    if _worker_ledger is None:
        _worker_ledger = FailureLedger()
        logger.addHandler(_worker_ledger)
        logger.propagate = False  # the parent prints them, not the worker's stderr
    return _worker_ledger


def call_collecting(fn, site: CallSite, element) -> tuple[object, list[Failure]]:
    """Worker-side `call_in_context` that returns the records it produced."""
    ledger = _worker_ledger_installed()
    ledger.failures.clear()
    result = call_in_context(fn, site, element)
    return result, list(ledger.failures)


def reemit(failures: list[Failure]) -> None:
    for failure in failures:
        logger.log(
            logging.getLevelName(failure.level),
            failure.message(),
            extra={"failure": failure},
        )


def read_failure_log(path: str | Path) -> pl.DataFrame:
    """Failure records from a log file, whatever formatter wrote it: every line
    containing `MARKER` is parsed from the marker on; other lines are ignored."""
    lines = [
        line[i + len(MARKER) :]
        for line in Path(path).read_text().splitlines()
        if (i := line.find(MARKER)) >= 0
    ]
    if not lines:
        return pl.DataFrame(schema=SCHEMA)
    return pl.read_csv(
        StringIO("\n".join(lines)),
        separator="\t",
        has_header=False,
        new_columns=list(SCHEMA),
        schema_overrides=_TEXT_SCHEMA,
        quote_char=None,
    ).with_columns(_PARSE_TIME)
