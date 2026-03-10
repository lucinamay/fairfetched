"""
Contains Compounds and Protein classes for consolidation
of unique Compounds/Proteins within datasets.
The goal of this module is to provide an easy API to explore /
entire data used throughout the model training.

This module does not contain curated datasets with clean input-output labelsm
which is part of the datasets.py module
"""

import logging as lg
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from rdkit.Chem import (
    Kekulize,
    Mol,
    MolFromInchi,
    MolFromSmiles,
    MolToInchi,
    MolToInchiAndAuxInfo,
    MolToInchiKey,
    MolToSmiles,
    RemoveStereochemistry,
)
from rdkit.Chem.inchi import InchiToInchiKey

from fairfetched.utils._track import track

from ._optional import _papyrus_standardize

# Dedicated logger for _smiles_to_clean_mol
mlg = lg.getLogger("fairfetched.standardization.compound_fns")
if not mlg.hasHandlers():
    handler = lg.FileHandler("fairfetched_standardisation.log")
    formatter = lg.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    mlg.addHandler(handler)
mlg.setLevel(lg.DEBUG)


def _aeffect_standardise_mol(smiles: str) -> Mol | None:
    """Returns cleaned molecules from SMILES string or None upon failure or invalid molecules"""
    assert isinstance(smiles, str)
    if not smiles:
        mlg.warning("empty smiles")
        return None

    mol = MolFromSmiles(smiles)
    if not isinstance(mol, Mol):
        mlg.debug("Failed on initial `MolFromSmiles` conversion")
        return None

    try:
        # @TODO: check if stereochemistry removal should be before or after standardisation
        RemoveStereochemistry(mol)
    except Exception as e:
        mlg.error(f"Failed in RemoveStereochemistry(mol) with error {e}")
        return None
    finally:
        if not mol:
            mlg.error(f"{smiles} could not be converted to a mol")
            return None
    try:
        # molclean_lg.debug("Starting Papyrus-type standardisation")
        mol, message = _papyrus_standardize(
            mol,
            filter_mixtures=True,  # @TODO: check if this is desired
            filter_inorganic=False,
            filter_non_small_molecule=False,
            return_type=True,
            raise_error=False,
        )
        if not mol:
            mlg.warning(
                f"{smiles} could not be cleaned during Papyrus-type standardisation: {message}"
            )
            return None
        mol = MolFromInchi(MolToInchi(mol))
        if not mol:
            mlg.warning(f"{smiles} did not give valid inchi: {message}")
        return mol
    except Exception as e:
        mlg.error(
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


def _safe_smiles_to_mol(inchi: str) -> Mol | None:
    try:
        return MolFromSmiles(inchi)
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
    return [_aeffect_standardise_mol(s) for s in track(smiles, desc="cleaning smiles")]


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
        # _mol_to_kekulised_smiles(m)
        MolToSmiles(m, kekuleSmiles=True, isomericSmiles=False)
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


async def fetch_atc(session, cid: str) -> str:
    """
    Fetch human ATC codes from PubChem, excluding veterinary codes.
    Returns comma-separated ATC codes (only the most specific per hierarchy).
    """
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON?heading=ATC+Code"

    data = await _safe_fetch_json(session, url)
    if not data:
        return ""

    sections = data.get("Record", {}).get("Section", [])

    atc_codes = _extract_atc_codes(sections)

    if not atc_codes:
        lg.debug(f"No human ATC codes found for cid {cid}")
        return ""

    return ",".join(_filter_most_specific(atc_codes))


async def _safe_fetch_json(session, url: str) -> dict[str, Any]:
    try:
        async with session.get(url) as resp:
            if resp.status != 200:
                return {}
            data = await resp.json()
    except Exception as e:
        lg.warning(f"Failed to fetch ATC from url {url}: {e}")
        return {}

    # Check for no data
    if data.get("Fault", {}).get("Message") == "No data found":
        return {}
    return data


def _extract_act_code_section(json: dict) -> list[str]:
    """Recurses through json to find ATC Code heading"""
    texts = []
    sections = json.get("Record", {}).get("Section", [])
    for section in sections:
        if section.get("TOCHeading") != "Pharmacology and Biochemistry":
            continue
        for subsection in section.get("Section", []):
            if subsection.get("TOCHeading") != "ATC Code":
                continue
            for info in subsection.get("Information", []):
                # Skip veterinary codes
                if "vet" in info.get("Name", "").lower():
                    continue

                # Extract ATC codes from StringWithMarkup
                for markup in info.get("Value", {}).get("StringWithMarkup", []):
                    text = markup.get("String", "").strip()
                    if not text:
                        continue
                    texts.append(text)
    return texts


def _extract_atc_codes(texts: Iterable[str]) -> set[str]:
    """From a given iterable of texts, returns a set of ATC codes"""
    atc_codes = set()
    for text in texts:
        # Extract only the code part (before any dash or description)
        # ATC codes are alphanumeric, typically like A01AB12
        match_ = re.match(r"^([A-Z]\d{2}[A-Z]{0,2}\d{0,2})", text)
        if not match_:
            continue
        atc_codes.add(match_.group(1))
    return atc_codes


def _filter_most_specific(atc_codes: Iterable[str]) -> list:
    """Keep only the most specific code per hierarchy
    Group by prefix and keep longest"""
    filtered_codes = set()
    sorted_codes = sorted(set(atc_codes), key=len, reverse=True)

    for code in sorted_codes:
        # Check if this code is a prefix of any already-added code
        if not any(longer_code.startswith(code) for longer_code in filtered_codes):
            filtered_codes.add(code)
    return sorted(map(str, sorted_codes))

    return sorted(sorted_codes)


def inchi_to_key(x: str) -> str:
    "As the regular function maps less well in polars"
    return InchiToInchiKey(x)
