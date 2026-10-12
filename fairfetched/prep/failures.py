"""Failure records for `fairfetched.prep`: a step that raised (ERROR), a
warning a step emits (WARNING), or a molecule a filter step drops (INFO), each
naming step, molecule, reason and the user line that built the expression.

Every record goes to two independent sinks: every open `FailureCollector`
(always, whatever the logging configuration) and the `fairfetched.prep` logger
(subject to the user's level and handlers). `read_failure_log` parses the
records back out of a log file written by any formatter.

    >>> with FailureCollector() as failures:
    ...     out = df.with_columns(MolExpr.from_smiles("smiles").standardize(*STEPS_CHEMBL))
    >>> failures.frame()  # one row per record, columns as `Failure`

`mol` is the input text for parse failures and RDKit's canonical SMILES of the
structure entering the failing expression otherwise.
"""

from __future__ import annotations

import base64
import logging
import traceback
from contextvars import ContextVar
from dataclasses import astuple, dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Self, get_args, get_type_hints

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
class Failure(CallSite):
    time: datetime
    level: str
    step: str
    mol: str
    error: str

    @classmethod
    def schema(cls) -> dict[str, pl.DataType]:
        """polars dtypes from the field annotations above."""
        dtypes = {datetime: pl.Datetime("us"), str: pl.String, int: pl.Int64}
        return {
            name: dtypes[(get_args(hint) or (hint,))[0]]  # `X | None` -> X
            for name, hint in get_type_hints(cls).items()
        }

    def message(self) -> str:
        """One line, tab-separated after `MARKER`, so any formatter keeps it parseable."""
        cells = (
            "" if v is None else str(v).replace("\t", " ").replace("\n", " ")
            for v in astuple(self)
        )
        return MARKER + "\t".join(cells)


# (call site, element being processed, records sink) for one step call; `_map`
# sets it per element, so `safe_step` never has to be told who failed
_CONTEXT: ContextVar[tuple[CallSite, object, list[Failure]]] = ContextVar(
    "fairfetched_mol_context"
)
_UNKNOWN = CallSite("?", None, None)


def callsite() -> CallSite:
    """The user's own line that built the expression: the innermost stack frame
    outside fairfetched and polars. `code` is None where Python keeps no cell
    source (marimo, `python -i`); Jupyter cells resolve."""
    for frame in reversed(traceback.extract_stack()):
        if frame.filename.startswith(_SKIP) or frame.filename == "<string>":
            continue  # fairfetched, polars (`.mol` namespace), dataclass __init__
        return CallSite(frame.filename, frame.lineno, frame.line or None)
    return _UNKNOWN


def mol_id(element: object) -> str:
    """string->RDKit canonical SMILES->lossless binary"""
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
        *astuple(site),
        time=datetime.now(),  # noqa: DTZ005  # local, as logging's asctime
        level=logging.getLevelName(level),
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
    the logging configuration; `frame()` has one row per record."""

    def __init__(self) -> None:
        self.failures: list[Failure] = []

    def frame(self) -> pl.DataFrame:
        return pl.DataFrame(self.failures, schema=Failure.schema())

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
    schema = Failure.schema()
    if not lines:
        return pl.DataFrame(schema=schema)
    return pl.read_csv(
        StringIO("\n".join(lines)),
        separator="\t",
        has_header=False,
        new_columns=list(schema),
        schema=schema,
        quote_char=None,
    )
