from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, fields
from functools import lru_cache, partial, wraps
from typing import Any, ParamSpec, TypeVar

from numpy import uint8
from numpy.typing import NDArray

# from fairfetched.standardization.pipeline import CHEMBL_PIPELINE, MolFn, mol_pipeline
from rdkit import Chem
from rdkit.Chem import (
    InchiToInchiKey,
    Mol,
    MolFromInchi,
    MolToInchi,
    MolToInchiAndAuxInfo,
    RemoveStereochemistry,
)
from rdkit.Chem.rdFingerprintGenerator import FingerprintGenerator64, GetMorganGenerator
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt
from rdkit.Chem.rdmolfiles import MolFromSmiles, MolToSmiles
from rdkit.Chem.Scaffolds import MurckoScaffold

from ._optional import _chembl_standardize, chembl_get_parent_mol

# from rdkit.Chem.rdinchi import MolToInchi #returns something different (int64?)
from ._optional import _papyrus_standardize as _papyrus_standardize_impl

P = ParamSpec("P")
T = TypeVar("T")

logger = logging.getLogger(__name__)


def safe_step_function(
    name: str | None = None,
) -> Callable[[Callable[P, T | None]], Callable[P, T | None]]:
    """
    Decorator:
      - returns None if first argument is None
      - catches all exceptions and returns None
      - logs the failing step with module-level logger
    """

    def deco(func: Callable[P, T | None]) -> Callable[P, T | None]:
        step = name or getattr(func, "__name__", repr(func))

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T | None:
            if (args[0] if args else None) is None:
                return None
            try:
                return func(*args, **kwargs)
            except Exception:
                logger.exception("Failure at step '%s'", step)
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
def chembl_standardize(mol, *args, **kwargs):
    return _chembl_standardize(mol, *args, **kwargs)


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


# top-level functions required for pickling (ProcessPoolExecutor requirement)


@safe_step
def _smiles_to_binary(s: str | None) -> bytes | None:
    return MolFromSmiles(s).ToBinary()


@safe_step
def _inchi_to_binary(s: str) -> bytes | None:
    return MolFromInchi(s).ToBinary()


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
