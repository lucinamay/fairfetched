from __future__ import annotations

import hashlib
import logging
import threading
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

from ._optional import (
    _chembl_standardize,
    chembl_exclude_flag,
    chembl_get_parent_mol,
    chembl_standardizer,
    papyrus_standardizer,
)

# from rdkit.Chem.rdinchi import MolToInchi #returns something different (int64?)
from ._optional import _papyrus_standardize as _papyrus_standardize_impl
from .failures import record, warn

P = ParamSpec("P")
T = TypeVar("T")

logger = logging.getLogger(__name__)


# RDKit's most recent log line in this thread, routed in via `LogToPythonLogger`;
# a ContextVar because polars runs UDFs for sibling expressions on separate threads
_RDKIT_MESSAGE: ContextVar[str] = ContextVar("fairfetched_rdkit_message", default="")


class _LastMessage(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        _RDKIT_MESSAGE.set(record.getMessage().strip().split("] ", 1)[-1])


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


@safe_step
def chembl_standardize(mol: Mol) -> Mol | None:
    """`chembl_structure_pipeline.standardize_mol` with its defaults."""
    return _chembl_standardize(mol)


@safe_step
def get_parent(mol: Mol, exclude_by_chembl_standards: bool = False) -> Mol | None:
    """`chembl_structure_pipeline.get_parent_mol`: strips salts/solvents and
    neutralises.

    ChEMBL flags structures it would not register (any metal atom, or more than
    7 borons, checked on the input); `exclude_by_chembl_standards=True` returns
    None for those, e.g. cisplatin. Default False keeps them.
    """
    parent, exclude = chembl_get_parent_mol(
        mol, check_exclusion=exclude_by_chembl_standards
    )
    return None if exclude and exclude_by_chembl_standards else parent


@safe_step
def _mw_between(mol: Mol, lo: float | None, hi: float | None) -> Mol | None:
    mw = CalcExactMolWt(mol)
    if (lo is not None and mw < lo) or (hi is not None and mw > hi):
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
def papyrus_standardize(mol, *args, **kwargs):
    return _papyrus_standardize_impl(mol, *args, **kwargs)


@safe_step
def valid_inchi(mol: Mol) -> Mol | None:
    """Returns mol only if it produces a valid InChI."""
    inchi = MolToInchi(mol)
    return mol if (inchi and MolFromInchi(inchi)) else None


@safe_step
def no_mixtures(mol: Mol) -> Mol | None:
    """returns none if mol is mixture. untested naive implementation (checks for period in smiles)"""
    logger.warning(
        "fairfetched.standardization.no_mixtures() is still an untested naive implementation"
    )
    return None if "." in MolToSmiles(mol) else mol


@safe_step  # @TODO: check implementation for this
def only_organic(mol: Mol) -> Mol | None:
    """returns none if mol is organic. untested naive implementation, checks if
    all atoms are ∈ {C,N,O,F,P,S,Cl,Br,I}"""
    logger.warning(
        "fairfetched.standardization.only_organic() is still an untested naive implementation"
    )
    organic = {6, 7, 8, 9, 15, 16, 17, 35, 53}
    return mol if all(a.GetAtomicNum() in organic for a in mol.GetAtoms()) else None


# --- chembl_structure_pipeline.standardize_mol, one step per library call ---
# `standardize_mol(check_exclusion=True)` returns an excluded mol untouched; a
# step list cannot branch, so `STEPS_CHEMBL_STANDARDIZE_MOL` equals
# `check_exclusion=False` and `chembl_exclude` drops excluded mols instead.


@safe_step
def chembl_exclude(mol: Mol) -> Mol | None:
    """None where ChEMBL's `exclude_flag` is set: any atom of its metal list,
    or more than 7 borons."""
    return None if chembl_exclude_flag(mol, includeRDKitSanitization=False) else mol


@safe_step
def chembl_update_valences(mol: Mol) -> Mol | None:
    return chembl_standardizer.update_mol_valences(mol)


@safe_step
def chembl_remove_sgroups(mol: Mol) -> Mol | None:
    return chembl_standardizer.remove_sgroups_from_mol(mol)


@safe_step
def chembl_kekulize(mol: Mol) -> Mol | None:
    return chembl_standardizer.kekulize_mol(mol)


@safe_step
def chembl_remove_hs(mol: Mol) -> Mol | None:
    """`Chem.RemoveHs` keeping wedged, stereo-bearing and odd-valence Hs."""
    return chembl_standardizer.remove_hs_from_mol(mol)


@safe_step
def chembl_normalize(mol: Mol) -> Mol | None:
    """ChEMBL's SMIRKS normalizations (nitro, sulfoxide, azide, ...) and
    alkali-alkoxide bond splitting."""
    return chembl_standardizer.normalize_mol(mol)


@safe_step
def chembl_uncharge(mol: Mol) -> Mol | None:
    """`rdMolStandardize.Uncharger(canonicalOrder=True)`."""
    return chembl_standardizer.uncharge_mol(mol)


@safe_step
def chembl_flatten_tartrate(mol: Mol) -> Mol | None:
    """Unsets the two stereocentres of free tartrate/tartaric acid fragments."""
    return chembl_standardizer.flatten_tartrate_mol(mol)


@safe_step
def chembl_cleanup_drawing(mol: Mol) -> Mol | None:
    """Straightens triple bonds and allenes in a 2D conformer; no-op without one."""
    return chembl_standardizer.cleanup_drawing_mol(mol)


@safe_step
def sanitize(mol: Mol) -> Mol | None:
    """`Chem.SanitizeMol`; None where it raises."""
    Chem.SanitizeMol(mol)
    return mol


# --- chembl_structure_pipeline.get_parent_mol, one step per library call ---


@safe_step
def chembl_isotope_parent(mol: Mol) -> Mol | None:
    """Isotope labels cleared, then `chembl_remove_hs`."""
    return chembl_standardizer.get_isotope_parent_mol(mol)


@safe_step
def _chembl_fragment_parent(
    mol: Mol, neutralize: bool, exclude_by_chembl_standards: bool
) -> Mol | None:
    parent, excluded = chembl_standardizer.get_fragment_parent_mol(
        mol, check_exclusion=exclude_by_chembl_standards, neutralize=neutralize
    )
    return None if excluded else parent


def chembl_fragment_parent(
    neutralize: bool = True, exclude_by_chembl_standards: bool = False
) -> MolFn:
    """Step stripping ChEMBL's solvents, then salts (keeping everything when
    nothing would remain), then uncharging; `[chembl_isotope_parent,
    chembl_fragment_parent()]` is `get_parent`."""
    return partial(
        _chembl_fragment_parent,
        neutralize=neutralize,
        exclude_by_chembl_standards=exclude_by_chembl_standards,
    )


@safe_step
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
    `papyrus_canonical_tautomer` cannot report that: the library blocks
    RDKit's log and discards the enumeration status."""
    return partial(
        _canonical_tautomer,
        max_tautomers=max_tautomers,
        allow_stereo_removal=allow_stereo_removal,
    )


# --- papyrus_structure_pipeline.standardize, one step per library call ---
# the weight window is `mw_between(200, 800)`: `is_small_molecule` is the same
# inclusive CalcExactMolWt check


@safe_step
def papyrus_chembl_roundtrip(mol: Mol) -> Mol | None:
    """`standardize_mol`, `get_parent_mol`, then a SMILES write/read; Papyrus
    runs it first and last."""
    return papyrus_standardizer._apply_chembl_standardization(mol)


_SALTS_LOCK = threading.Lock()


@safe_step
def _papyrus_remove_salts(mol: Mol, include_metals: bool) -> Mol | None:
    if include_metals:
        return papyrus_standardizer._remove_supplementary_salts(
            mol, include_metals=True
        )
    # include_metals=False makes the library overwrite its module-level SALTS
    # list in place with Mol objects, so every later call would raise: restore it
    with _SALTS_LOCK:
        saved = list(papyrus_standardizer.SALTS)
        try:
            return papyrus_standardizer._remove_supplementary_salts(
                mol, include_metals=False
            )
        finally:
            papyrus_standardizer.SALTS[:] = saved


def papyrus_remove_salts(include_metals: bool = True) -> MolFn:
    """Step stripping Papyrus's extra salt list and, with `include_metals`
    (the Papyrus default), every metal atom."""
    return partial(_papyrus_remove_salts, include_metals=include_metals)


@safe_step
def papyrus_no_mixtures(mol: Mol) -> Mol | None:
    """None where `Chem.GetMolFrags` finds more than one fragment."""
    return None if papyrus_standardizer.is_mixture(mol) else mol


@safe_step
def papyrus_only_organic(mol: Mol) -> Mol | None:
    """None where ChEMBL's `exclude_flag` is set, no C~C bond exists, or an
    element is outside {C,H,O,N,P,S,F,Cl,Br,I}."""
    return mol if papyrus_standardizer.is_organic(mol) else None


@safe_step
def papyrus_uncharge(mol: Mol) -> Mol | None:
    """`rdMolStandardize.Uncharger`, then B3DB's SMARTS neutralizations."""
    return papyrus_standardizer._uncharge(mol)


@safe_step
def _papyrus_canonical_tautomer(
    mol: Mol, allow_stereo_removal: bool, max_tautomers: int
) -> Mol | None:
    return papyrus_standardizer._canonicalize_tautomer(
        mol, allow_stereo_removal=allow_stereo_removal, max_tautomers=max_tautomers
    )


def papyrus_canonical_tautomer(
    allow_stereo_removal: bool = True, max_tautomers: int = 2**32 - 1
) -> MolFn:
    """Step picking RDKit's canonical tautomer. The Papyrus default
    `allow_stereo_removal=True` lets tautomerization erase stereocentres."""
    return partial(
        _papyrus_canonical_tautomer,
        allow_stereo_removal=allow_stereo_removal,
        max_tautomers=max_tautomers,
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
