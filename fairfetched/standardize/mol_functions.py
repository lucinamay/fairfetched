from __future__ import annotations

import logging
from functools import wraps
from typing import Callable, ParamSpec, TypeVar

# from fairfetched.standardization.pipeline import CHEMBL_PIPELINE, MolFn, mol_pipeline
from rdkit import Chem
from rdkit.Chem import Mol, MolFromInchi, MolToInchi, RemoveStereochemistry
from rdkit.Chem.rdmolfiles import MolFromSmiles, MolToSmiles

# from rdkit.Chem.rdinchi import MolToInchi #returns something different (int64?)
from ._optional import _papyrus_standardize, chembl_standardize

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
        step = name or func.__name__

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T | None:
            if not args or args[0] is None:
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
def chembl_standardise(mol, *args, **kwargs):
    return chembl_standardize(mol, *args, **kwargs)


@safe_step
def papyrus_standardise(mol, *args, **kwargs):
    return _papyrus_standardize(mol, *args, **kwargs)


@safe_step
def valid_inchi(mol: Mol) -> Mol | None:
    """Returns mol only if it produces a valid InChI."""
    inchi = MolToInchi(mol)
    return mol if (inchi and MolFromInchi(inchi)) else None


@safe_step
def no_mixtures(mol: Mol) -> Mol | None:
    """returns none if mol is mixture. untested naive implementation (checks for period in smiles)"""
    logging.warning(
        "fairfetched.standardization.no_mixtures() is still an untested naive implementation"
    )
    return None if "." in MolToSmiles(mol) else mol


@safe_step  # @TODO: check implementation for this
def only_organic(mol: Mol) -> Mol | None:
    """returns none if mol is organic. untested naive implementation, checks if
    all atoms are ∈ {C,N,O,F,P,S,Cl,Br,I}"""
    logging.warning(
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
    return MolToSmiles(Mol(b))


@safe_step
def _binary_to_kekulized_smiles(b: bytes | None) -> str | None:
    return MolToSmiles(Mol(b), kekuleSmiles=True, isomericSmiles=False)  # ty: ignore[no-matching-overload]


@safe_step
def _binary_to_inchi(b: bytes | None) -> str | None:
    return MolToInchi(Mol(b))  # ty: ignore[invalid-return-type, no-matching-overload]


@safe_step
def _binary_to_inchi_and_auxinfo(b: bytes | None) -> str | None:
    return Chem.inchi.MolToInchiAndAuxInfo(Mol(b))


@safe_step
def _binary_to_inchikey(b: bytes | None) -> str | None:
    return Chem.inchi.MolToInchiKey(Mol(b))


@safe_step
def _binary_to_mol(b: bytes | None) -> Mol | None:
    return Mol(b)


@safe_step
def _num_atoms(b: bytes | None) -> int | None:
    return Mol(b).GetNumAtoms()


@safe_step
def _num_heavy_atoms(b: bytes | None) -> int | None:
    return Mol(b).GetNumHeavyAtoms()
