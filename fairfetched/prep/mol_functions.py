from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, fields
from functools import lru_cache, partial, wraps
from typing import Any, ParamSpec, TypeVar

from numpy import uint8
from numpy.typing import NDArray

# from fairfetched.standardization.pipeline import CHEMBL_PIPELINE, MolFn, mol_pipeline
from rdkit import Chem, rdBase
from rdkit.Chem import (
    InchiToInchiKey,
    Mol,
    MolFromInchi,
    MolToInchi,
    MolToInchiAndAuxInfo,
    RemoveStereochemistry,
)
from rdkit.Chem.MolStandardize.rdMolStandardize import (
    TautomerEnumerator,
    TautomerEnumeratorStatus,
)
from rdkit.Chem.rdFingerprintGenerator import FingerprintGenerator64, GetMorganGenerator
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt
from rdkit.Chem.rdmolfiles import MolFromSmiles, MolToSmiles
from rdkit.Chem.Scaffolds import MurckoScaffold

from fairfetched.prep.failures import note, record, warn

P = ParamSpec("P")
T = TypeVar("T")

logger = logging.getLogger(__name__)


# RDKit's most recent log line in this thread, routed in via `LogToPythonLogger`;
# a ContextVar because polars runs UDFs for sibling expressions on separate threads
_RDKIT_MESSAGE: ContextVar[str] = ContextVar("fairfetched_rdkit_message", default="")


class _LastMessage(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        _RDKIT_MESSAGE.set(record.getMessage().strip().split("] ", 1)[-1])


# TODO(globals): import-time reconfiguration of RDKit logging for the whole
# process; move into a `functools.cache`d installer called at the top of the
# `safe_step` wrapper (before `func` runs, or the first parse failure reads "")
rdBase.LogToPythonLogger()
logging.getLogger("rdkit").addHandler(_LastMessage())


class MolParseError(ValueError):
    """RDKit returned None for a SMILES/InChI; the message is the reason."""


def _error_text(e: Exception) -> str:
    return str(e) if isinstance(e, MolParseError) else f"{type(e).__name__}: {e}"


def _parse_reason(smiles: str) -> str:
    """Why `MolFromSmiles` returned None. Sanitization problems are reported
    by `DetectChemistryProblems` without any log; syntax errors only reach us
    through RDKit's log, which `RDLogger.DisableLog` silences."""
    unsanitized = MolFromSmiles(smiles, sanitize=False)
    if unsanitized is not None:
        problems = Chem.DetectChemistryProblems(unsanitized)
        if problems:
            return "; ".join(p.Message() for p in problems)
    return _RDKIT_MESSAGE.get() or "MolFromSmiles returned None (RDKit log disabled)"


def safe_step_function(
    name: str | None = None,
) -> Callable[[Callable[P, T | None]], Callable[P, T | None]]:
    """
    Decorator:
      - returns None if first argument is None
      - catches all exceptions and returns None
      - emits one `failures.record` naming the step, the error and the molecule
    """

    def deco(func: Callable[P, T | None]) -> Callable[P, T | None]:
        step = name or getattr(func, "__name__", repr(func))

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T | None:
            if (args[0] if args else None) is None:
                return None
            _RDKIT_MESSAGE.set("")  # a parse failure then reads this step's message
            try:
                return func(*args, **kwargs)
            except Exception as e:  # noqa: BLE001  # the step contract: None plus a record
                record(step, _error_text(e))
                return None

        return wrapper

    return deco


safe_step = safe_step_function()


MolFn = Callable[[Mol], Mol | None]
BinaryMolFn = Callable[[bytes], bytes | None]
BinaryToAnyFn = Callable[[bytes | None], Any | None]


@lru_cache(maxsize=1)
def _get_morgan_generator(
    radius: int, fp_size: int, **kwargs
) -> FingerprintGenerator64:
    return GetMorganGenerator(radius=radius, fpSize=fp_size, **kwargs)


@safe_step
def remove_stereo(mol: Mol) -> Mol | None:
    """`rdkit.Chem.RemoveStereochemistry`, applied and mol returned"""
    RemoveStereochemistry(mol)
    return mol


@safe_step
def via_inchi(mol: Mol) -> Mol | None:
    """Round-trip through InChI to canonicalise."""
    inchi = MolToInchi(mol)
    return MolFromInchi(inchi) if inchi else None


@safe_step_function("mw_between")
def _mw_between(mol: Mol, lo: float | None, hi: float | None) -> Mol | None:
    mw = CalcExactMolWt(mol)
    if (lo is not None and mw < lo) or (hi is not None and mw > hi):
        note("mw_between", f"exact mass {mw:.2f} outside [{lo}, {hi}]")
        return None
    return mol


def mw_between(lo: float | None = None, hi: float | None = None) -> MolFn:
    """Step returning None for mols whose exact (monoisotopic) mass is outside
    [lo, hi], both inclusive; a None bound is open. `mw_between(200, 800)` is
    the cut Papyrus applies by default.

    Place it after `get_parent`: before it, counter-ions and solvents count
    towards the mass, so a salt can pass or fail where its parent would not.
    """
    return partial(_mw_between, lo=lo, hi=hi)


@safe_step
def valid_inchi(mol: Mol) -> Mol | None:
    """Returns mol only if it produces a valid InChI."""
    inchi = MolToInchi(mol)
    if inchi and MolFromInchi(inchi):
        return mol
    note("valid_inchi", "no valid InChI")
    return None


@safe_step
def sanitize(mol: Mol) -> Mol | None:
    """`Chem.SanitizeMol`; None where it raises."""
    Chem.SanitizeMol(mol)
    return mol


@safe_step_function("canonical_tautomer")
def _canonical_tautomer(
    mol: Mol, max_tautomers: int, allow_stereo_removal: bool
) -> Mol | None:
    enumerator = TautomerEnumerator()
    enumerator.SetMaxTautomers(max_tautomers)
    enumerator.SetRemoveSp3Stereo(allow_stereo_removal)
    enumerator.SetRemoveBondStereo(allow_stereo_removal)
    result = enumerator.Enumerate(mol)
    if result.status != TautomerEnumeratorStatus.Completed:
        warn("canonical_tautomer", f"tautomer enumeration {result.status.name}")
    return enumerator.PickCanonical(result.tautomers)


def canonical_tautomer(
    max_tautomers: int = 1000, allow_stereo_removal: bool = True
) -> MolFn:
    """Step picking RDKit's canonical tautomer, with a WARNING failure record
    when enumeration stopped at `max_tautomers` or RDKit's transform limit.
    `pipeline.papyrus_canonical_tautomer` does the same with Papyrus's
    defaults (no tautomer cap) and its chiral-centre filter."""
    return partial(
        _canonical_tautomer,
        max_tautomers=max_tautomers,
        allow_stereo_removal=allow_stereo_removal,
    )


# top-level functions required for pickling (ProcessPoolExecutor requirement)


@safe_step
def _smiles_to_binary(s: str) -> bytes | None:
    mol = MolFromSmiles(s)
    if mol is None:
        raise MolParseError(_parse_reason(s))
    return mol.ToBinary()


@safe_step
def _inchi_to_binary(s: str) -> bytes | None:
    mol = MolFromInchi(s)
    if mol is None:
        raise MolParseError(
            _RDKIT_MESSAGE.get() or "MolFromInchi returned None (RDKit log disabled)"
        )
    return mol.ToBinary()


@safe_step
def _binary_to_smiles(b: bytes | None) -> str | None:
    return MolToSmiles(Mol(b))  # ty: ignore[no-matching-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]


@safe_step
def _binary_to_kekulized_smiles(b: bytes | None) -> str | None:
    return MolToSmiles(Mol(b), kekuleSmiles=True, isomericSmiles=False)  # ty: ignore[no-matching-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]


@safe_step
def _binary_to_inchi(b: bytes | None) -> str | None:
    return MolToInchi(Mol(b))  # ty: ignore[invalid-return-type, no-matching-overload]  # pyright: ignore[reportCallIssue, reportArgumentType, reportReturnType]


@safe_step
def _binary_to_inchi_and_auxinfo(b: bytes | None) -> str | None:
    return Chem.inchi.MolToInchiAndAuxInfo(Mol(b))  # ty: ignore[no-matching-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]


@safe_step
def _binary_to_inchikey(b: bytes | None) -> str | None:
    return Chem.inchi.MolToInchiKey(Mol(b))  # ty: ignore[no-matching-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]


@dataclass(frozen=True)
class Descriptors:
    inchi: str
    inchi_auxinfo: str
    inchikey: str
    smiles: str

    @classmethod
    def dataclass_schema(cls) -> dict[str, type | str | Any]:
        # hints = get_type_hints(cls)
        return {f.name: str for f in fields(cls)}


@safe_step
def _binary_to_descriptors(b: bytes | None) -> Descriptors | None:
    """returns inchi, inchi_auxinfo, inchikey, kekulised smiles (as 'smiles')"""
    mol = Chem.Mol(b)  # ty: ignore[no-matching-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    inchi, auxinfo = MolToInchiAndAuxInfo(mol)
    return Descriptors(
        inchi=inchi,
        inchi_auxinfo=auxinfo,
        inchikey=InchiToInchiKey(inchi),
        smiles=MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=False),
    )


@safe_step
def _binary_to_morgan_array(
    b: bytes | None, radius: int, fp_size: int, **kwargs
) -> NDArray[uint8] | None:
    mol = Chem.Mol(b)  # ty: ignore[no-matching-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    gen = _get_morgan_generator(radius=radius, fp_size=fp_size, **kwargs)
    return gen.GetFingerprintAsNumPy(mol)


@safe_step
def _binary_to_mol(b: bytes | None) -> Mol | None:
    return Mol(b)  # ty: ignore[no-matching-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]


@safe_step
def _num_atoms(b: bytes | None) -> int | None:
    return Mol(b).GetNumAtoms()  # ty: ignore[no-matching-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]


@safe_step
def _num_heavy_atoms(b: bytes | None) -> int | None:
    return Mol(b).GetNumHeavyAtoms()  # ty: ignore[no-matching-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]


@safe_step
def _has_stereo(b: bytes | None) -> bool | None:
    """SMILES changes when stereo is removed (CAPRICHO `flag_stereochemistry_removal`)."""
    mol = Mol(b)  # ty: ignore[no-matching-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    flat = Mol(mol)
    RemoveStereochemistry(flat)
    return MolToSmiles(mol) != MolToSmiles(flat)


@safe_step
def _num_undefined_stereocenters(b: bytes | None) -> int | None:
    """As CAPRICHO `find_undefined_stereocenters`: centers with CHI_UNSPECIFIED tag."""
    mol = Mol(b)  # ty: ignore[no-matching-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    centers = Chem.FindMolChiralCenters(
        mol, includeUnassigned=True, useLegacyImplementation=False
    )
    unspecified = Chem.ChiralType.CHI_UNSPECIFIED
    return sum(mol.GetAtomWithIdx(i).GetChiralTag() == unspecified for i, _ in centers)


@safe_step
def _num_fragments(b: bytes | None) -> int | None:
    return len(Chem.GetMolFrags(Mol(b)))  # ty: ignore[no-matching-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]


@safe_step
def _binary_to_scaffold_smiles(b: bytes | None, generic: bool = False) -> str | None:
    """Bemis–Murcko scaffold SMILES; `generic=True` also makes atoms/bonds generic."""
    scaffold = MurckoScaffold.GetScaffoldForMol(Mol(b))  # ty: ignore[no-matching-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    if generic:
        scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
    return MolToSmiles(scaffold)


def _stable_hash64(s: str | None) -> int | None:
    """Process-, version- and platform-independent 64-bit hash (hashlib.blake2b, not `hash()`/`pl.Expr.hash`)."""
    if s is None:
        return None
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")
