import logging as lg
from pathlib import Path
from typing import Iterable

from rdkit.Chem import (
    Kekulize,
    Mol,
    MolFromInchi,
    MolFromSmiles,
    MolToInchi,
    MolToInchiAndAuxInfo,
    MolToSmiles,
    RemoveStereochemistry,
)
from rdkit.Chem.inchi import InchiToInchiKey

from fairfetched.utils._track import track

from .optional import papyrus_standardise


def _smiles_to_clean_mol(smiles: str) -> Mol | None:
    """Returns cleaned molecules from SMILES string or None upon failure or invalid molecules"""
    assert isinstance(smiles, str)
    if not smiles:
        lg.warning("empty smiles")
        return None
    mol = MolFromSmiles(smiles)
    if not isinstance(mol, Mol):
        lg.debug("Failed on initial `MolFromSmiles` conversion")
        return None
    try:
        RemoveStereochemistry(mol)
    except Exception as e:
        lg.error(f"Failed in RemoveStereochemistry(mol) with error {e}")
        return None
    # @TODO: check if stereochemistry removal should be before or after standardisation
    if not mol:
        lg.error(f"{smiles} could not be converted to a mol")
        return None
    try:
        mol, message = papyrus_standardise(
            mol,
            filter_mixtures=True,  # @TODO: check if this is desired
            filter_inorganic=False,
            filter_non_small_molecule=False,
            return_type=True,
            raise_error=False,
        )
        if not mol:
            lg.warning(
                f"{smiles} could not be cleaned during Papyrus-type standardisation: {message}"
            )
            return None
        mol = MolFromInchi(MolToInchi(mol))
        if not mol:
            lg.warning(f"{smiles} did not give valid inchi: {message}")
        return mol
    except Exception as e:
        lg.error(
            f"{smiles} could not be cleaned in papyrus-style standardsation with exception: {e}"
        )
        return None
    return None


def _safe_inchi_to_mol(inchi: str) -> Mol | None:
    try:
        return MolFromInchi(inchi)
    except Exception as e:
        lg.error(e)
        return None


def inchis_to_mols(
    inchis: Iterable[str], cache: Path | None = None
) -> list[Mol | None]:
    """Convert InChIs to RDKit Mol objects"""
    return [
        _safe_inchi_to_mol(i) for i in track(inchis, desc="Converting InChI to mol")
    ]


def smiles_to_clean_mols(smiles: Iterable[str]) -> list[Mol | None]:
    return [_smiles_to_clean_mol(s) for s in track(smiles, desc="cleaning smiles")]


def _mol_to_kekulised_smiles(mol: Mol) -> str:
    """Returns a SMILES string suitable for re-loading the molecule later.
    Removes stereochemistry.
    """
    try:
        Kekulize(mol, clearAromaticFlags=True)
    except Exception as e:
        lg.debug(f"Kekulization failed with error {e}")
        return MolToSmiles(mol, isomericSmiles=False)
    return MolToSmiles(mol, isomericSmiles=False)


def mols_to_kekulised_smiles(mols: Iterable[Mol]) -> list[str]:
    """Returns kekulised SMILES strings suitable for re-loading the molecule later.
    Removes stereochemistry.
    """
    return [
        _mol_to_kekulised_smiles(m)
        for m in track(mols, desc="converting mols to kekulised smiles")
    ]


def _mol_to_inchi_and_auxinfo(mol: Mol) -> tuple[str, str]:
    return MolToInchiAndAuxInfo(mol)


def _mol_to_inchi(mol: Mol) -> str:
    return MolToInchiAndAuxInfo(mol)[0]  # to maintain consistency


def mols_to_inchis(mols: Iterable[Mol]) -> list[str]:
    return [_mol_to_inchi(m) for m in track(mols, desc="converting mols to inchis")]


def mols_to_inchis_and_auxinfo(mols: Iterable[Mol]) -> list[tuple[str, str]]:
    return [
        _mol_to_inchi_and_auxinfo(m)
        for m in track(mols, desc="converting mols to inchis with auxinfo")
    ]


def inchis_to_inchikeys(inchis: Iterable[str]) -> list[str]:
    return [
        InchiToInchiKey(i) for i in track(inchis, desc="converting inchis to inchikeys")
    ]


def mols_to_inchikeys(mols: Iterable[Mol]) -> list[str]:
    return [
        InchiToInchiKey(MolToInchiAndAuxInfo(m)[0])
        for m in track(mols, desc="converting mols to inchikeys")
    ]


mol_to_kekulized_smiles = _mol_to_kekulised_smiles  # American spelling alias
inchi_to_inchikey = InchiToInchiKey  # convenience alias
