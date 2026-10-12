"""Failure records for `fairfetched.prep`: a step that raised (ERROR), a
warning a step emits (WARNING), or a molecule a filter step drops (INFO), each
naming step, molecule, reason and the user line that built the expression.

Every record goes to two independent sinks: every open `FailureCollector`
(always, whatever the logging configuration) and the `fairfetched.prep` logger
(subject to the user's level and handlers). `read_failure_log` parses the
records back out of a log file written by any formatter.

    >>> with FailureCollector() as failures:
    ...     out = df.with_columns(MolExpr.from_smiles("smiles").standardize(*STEPS_CHEMBL))
    >>> failures.frame()  # columns: SCHEMA

`mol` is the input text for parse failures and RDKit's canonical SMILES of the
structure entering the failing expression otherwise.
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

# (call site, element being processed, records sink) for one step call; `_map`
# sets it per element, so `safe_step` never has to be told who failed
_CONTEXT: ContextVar[tuple[CallSite, object, list[Failure]]] = ContextVar(
    "fairfetched_mol_context"
)
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
    """One failure record for the element currently in `_CONTEXT`; dispatched
    at once when the step was called outside `call_collecting`."""
    site, element, sink = _CONTEXT.get((_UNKNOWN, None, None))
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
    if sink is None:
        dispatch([failure])
    else:
        sink.append(failure)


def warn(step: str, message: str) -> None:
    record(step, message, logging.WARNING)


def note(step: str, reason: str) -> None:
    """INFO record"""
    record(step, reason, logging.INFO)


def call_collecting(fn, site: CallSite, element) -> tuple[object, list[Failure]]:
    """`fn(element)` and the records it emitted, nothing logged: the parent
    process dispatches them, so worker processes never touch the logger."""
    sink: list[Failure] = []
    token = _CONTEXT.set((site, element, sink))
    try:
        return fn(element), sink
    finally:
        _CONTEXT.reset(token)


def dispatch(failures: list[Failure]) -> None:
    """Every record to every open `FailureCollector` and to the logger."""
    levels = logging.getLevelNamesMapping()
    for failure in failures:
        for collector in _COLLECTORS:
            collector.failures.append(failure)
        logger.log(levels[failure.level], failure.message(), extra={"failure": failure})


class FailureCollector:
    """Receives every `Failure` recorded while open (`with`), independent of
    the logging configuration; `frame()` has one row per record (`SCHEMA`)."""

    def __init__(self) -> None:
        self.failures: list[Failure] = []

    def frame(self) -> pl.DataFrame:
        rows = [astuple(f) for f in self.failures]
        return pl.DataFrame(rows, schema=_TEXT_SCHEMA, orient="row").with_columns(
            _PARSE_TIME
        )

    def __enter__(self) -> Self:
        _COLLECTORS.append(self)
        return self

    def __exit__(self, *exc) -> None:
        _COLLECTORS.remove(self)


_COLLECTORS: list[FailureCollector] = []


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
