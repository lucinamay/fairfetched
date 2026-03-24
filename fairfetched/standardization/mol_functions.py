import logging
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterable

# from fairfetched.standardization.pipeline import CHEMBL_PIPELINE, MolFn, mol_pipeline
import polars as pl
from chembl_structure_pipeline import standardize_mol
from rdkit import Chem
from rdkit.Chem import Mol, MolFromInchi, MolToInchi, RemoveStereochemistry
from rdkit.Chem.rdmolfiles import MolToSmiles

# from rdkit.Chem.rdinchi import MolToInchi #returns something different (int64?)
from ._optional import _papyrus_standardize, chembl_standardize

MolFn = Callable[[Chem.Mol], Chem.Mol | None]


def remove_stereo(mol: Chem.Mol) -> Chem.Mol | None:
    try:
        RemoveStereochemistry(mol)
        return mol
    except Exception:
        return None


def via_inchi(mol: Chem.Mol) -> Chem.Mol | None:
    """Round-trip through InChI to canonicalise."""
    try:
        inchi = Chem.MolToInchi(mol)
        return Chem.MolFromInchi(inchi) if inchi else None
    except Exception:
        return None


def chembl_standardise(mol: Chem.Mol) -> Chem.Mol | None:
    try:
        return chembl_standardize(mol)
    except Exception:
        return None


def valid_inchi(mol: Chem.Mol) -> Chem.Mol | None:
    """Returns mol only if it produces a valid InChI."""
    try:
        inchi = Chem.MolToInchi(mol)
        return mol if (inchi and Chem.MolFromInchi(inchi)) else None
    except Exception:
        return None


def no_mixtures(mol: Chem.Mol) -> Chem.Mol | None:
    return None if "." in Chem.MolToSmiles(mol) else mol


def only_organic(mol: Chem.Mol) -> Chem.Mol | None:
    organic = {6, 7, 8, 9, 15, 16, 17, 35, 53}
    return mol if all(a.GetAtomicNum() in organic for a in mol.GetAtoms()) else None


# top-level functions required for pickling (ProcessPoolExecutor requirement)
def _smiles_to_binary(s: str | None) -> bytes | None:
    if s is None:
        return None
    mol = Chem.MolFromSmiles(s)
    return mol.ToBinary() if mol else None


def _inchi_to_binary(s: str | None) -> bytes | None:
    if s is None:
        return None
    mol = MolFromInchi(s)
    return mol.ToBinary() if mol else None


def _binary_to_smiles(b: bytes | None) -> str | None:
    return MolToSmiles(Mol(b)) if b else None


def _binary_to_kekulized_smiles(b: bytes | None) -> str | None:
    return MolToSmiles(Mol(b), kekuleSmiles=True, isomericSmiles=False) if b else None  # ty: ignore[no-matching-overload]


def _binary_to_inchi(b: bytes | None) -> str | None:
    return MolToInchi(Mol(b)) if b else None  # ty: ignore[invalid-return-type, no-matching-overload]


def _binary_to_inchi_and_auxinfo(b: bytes | None) -> str | None:
    return Chem.inchi.MolToInchiAndAuxInfo(Mol(b)) if b else None


def _binary_to_inchikey(b: bytes | None) -> str | None:
    return Chem.inchi.MolToInchiKey(Mol(b)) if b else None


def _binary_to_mol(b: bytes | None) -> Mol | None:
    return Mol(b) if b else None


def _num_atoms(b: bytes | None) -> int | None:
    return Mol(b).GetNumAtoms() if b else None


def _num_heavy_atoms(b: bytes | None) -> int | None:
    return Mol(b).GetNumHeavyAtoms() if b else None
