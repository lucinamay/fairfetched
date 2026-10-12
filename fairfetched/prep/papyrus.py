"""The Papyrus standardization functions, copied from papyrus_structure_pipeline
0.0.5 (`standardizer.py`, MIT, copyright 2023 Olivier J. M. Béquignon; the
neutralization SMARTS in `uncharge` are Hans de Winter's contribution to B3DB).

Copied rather than imported so that the package is typed, mutates no module
state (`_remove_supplementary_salts` overwrites the library's SALTS in place),
and lets RDKit's messages reach the failure records (the library wraps every
call in `BlockLogs`). `tests/test_standardize_steps.py` checks each function
against the library on every test input.
"""

from __future__ import annotations

from functools import lru_cache

from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem.MolStandardize.rdMolStandardize import (
    TautomerEnumerator,
    TautomerEnumeratorStatus,
    Uncharger,
)
from rdkit.Chem.rdmolops import ReplaceSubstructs

from fairfetched.prep._optional import chembl_exclude_flag, chembl_standardizer

# version copied and sha256 of its standardizer.py; the dev tests fail when the
# installed library differs, so the copy gets re-diffed instead of trusted
UPSTREAM = (
    "0.0.5",
    "7497ea51314324395f1014d6763e5c4a9f609977a69cb4e5f14fef9b68fac230",
)

# salts not flagged by the ChEMBL structure pipeline
SALTS = [
    "[Na+]", "[K+]", "Cl", "C1=NC=CC=C1", "[O-][Cl+3]([O-])([O-])[O-]", "O=CN(C)C",
    "[No]", "C", "C1COCCO1", "CCCCCCN(C)C", "CCCCCC", "CN(C)C(=N)N", "CN(C)C",
    "c1ccc([B-](c2ccccc2)(c2ccccc2)c2ccccc2)cc1",
]  # fmt: skip

# every element but H, B, C, N, O, P, F, S, Cl, Br, I
METALS = [
    "He", "Li", "Be", "Ne", "Na", "Mg", "Al", "Si", "Ar", "K", "Ca", "Sc", "Ti", "V",
    "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Kr", "Rb", "Sr",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te",
    "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho",
    "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl",
    "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am",
    "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt",
    "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]  # fmt: skip

ORGANIC_ATOMS = ["C", "H", "O", "N", "P", "S", "F", "Cl", "I", "Br"]

_CC_BOND = Chem.MolFromSmarts("[#6]~[#6]")

_NEUTRALIZATIONS = [
    ("[n+;H]", "n"),  # imidazoles
    ("[N+;!H0]", "N"),  # amines
    ("[$([O-]);!$([O-][#7])]", "O"),  # carboxylic acids and alcohols
    ("[S-;X1]", "S"),  # thiols
    ("[$([N-;X2]S(=O)=O)]", "N"),  # sulfonamides
    ("[$([N-;X2][C,N]=C)]", "N"),  # enamines
    ("[n-]", "[nH]"),  # tetrazoles
    ("[$([S-]=O)]", "S"),  # sulfoxides
    ("[$([N-]C=O)]", "N"),  # amides
]
_NEUTRALIZING_REACTIONS = [
    (Chem.MolFromSmarts(reactant), Chem.MolFromSmiles(product, False))
    for reactant, product in _NEUTRALIZATIONS
]


def chembl_roundtrip(mol: Mol) -> Mol:
    """`standardize_mol`, `get_parent_mol`, then a SMILES write and read."""
    parent, _ = chembl_standardizer.get_parent_mol(
        chembl_standardizer.standardize_mol(mol)
    )
    smiles = Chem.MolToSmiles(parent)
    out = Chem.MolFromSmiles(smiles)
    if out is None:
        raise ValueError(f"Could not parse standardized SMILES: {smiles}")
    return out


@lru_cache(maxsize=2)
def _salt_mols(include_metals: bool) -> tuple[Mol, ...]:
    smiles = SALTS + [f"[{m}]" for m in METALS] if include_metals else SALTS
    return tuple(Chem.MolFromSmiles(s) for s in smiles)


def remove_supplementary_salts(mol: Mol, include_metals: bool = True) -> Mol:
    """Drops every fragment that equals a `SALTS` entry (or a lone `METALS`
    atom); returns the input unchanged if nothing would remain."""
    salts = _salt_mols(include_metals)
    frags = []
    for frag in Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False):
        frag = Chem.RemoveHs(frag, sanitize=False)
        frag.UpdatePropertyCache(strict=False)
        Chem.SetAromaticity(frag)
        frags.append(frag)
    keep = [True] * len(frags)
    for i, frag in enumerate(frags):
        for salt in salts:
            if (
                keep[i]
                and frag.GetNumAtoms() == salt.GetNumAtoms()
                and frag.GetNumBonds() == salt.GetNumBonds()
                and frag.HasSubstructMatch(salt)
                and salt.HasSubstructMatch(frag)
            ):
                keep[i] = False
            if not any(keep):
                return Chem.Mol(mol)
    kept = [frag for frag, k in zip(frags, keep, strict=True) if k]
    out = kept[0]
    for frag in kept[1:]:
        out = Chem.CombineMols(out, frag)
    return out


def is_mixture(mol: Mol) -> bool:
    return len(Chem.GetMolFrags(mol)) > 1


def is_organic(mol: Mol) -> bool:
    """No ChEMBL exclusion flag, at least one C~C bond, elements within
    `ORGANIC_ATOMS`."""
    if chembl_exclude_flag(Chem.MolToMolBlock(mol)):
        return False
    if not mol.GetSubstructMatches(_CC_BOND):
        return False
    return not {atom.GetSymbol() for atom in mol.GetAtoms()}.difference(ORGANIC_ATOMS)


def uncharge(mol: Mol) -> Mol:
    """`rdMolStandardize.Uncharger`, then a second pass over zwitterions with
    B3DB's neutralization SMARTS."""
    mol = Uncharger().uncharge(mol)
    romol = Chem.RWMol(mol)
    for reactant, product in _NEUTRALIZING_REACTIONS:
        while romol.HasSubstructMatch(reactant):
            romol = ReplaceSubstructs(romol, reactant, product)[0]
    mol = Chem.Mol(romol)
    Chem.SanitizeMol(mol)
    return mol


def _num_chiral_centers(mol: Mol) -> int:
    return sum(
        atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        for atom in mol.GetAtoms()
    )


def canonicalize_tautomer(
    mol: Mol, allow_stereo_removal: bool = True, max_tautomers: int = 2**32 - 1
) -> tuple[Mol, TautomerEnumeratorStatus]:
    """RDKit's canonical tautomer and the enumeration status. With
    `allow_stereo_removal=False`, tautomers that change the number of chiral
    centres are discarded before picking."""
    enumerator = TautomerEnumerator()
    enumerator.SetMaxTautomers(max_tautomers)
    enumerator.SetRemoveSp3Stereo(allow_stereo_removal)
    enumerator.SetRemoveBondStereo(allow_stereo_removal)
    result = enumerator.Enumerate(mol)
    tautomers = result.tautomers
    if not allow_stereo_removal:
        n = _num_chiral_centers(mol)
        if n > 0:
            tautomers = [t for t in tautomers if _num_chiral_centers(t) == n]
    canonical = enumerator.PickCanonical(tautomers)
    if canonical is None:
        raise ValueError(
            f"Could not obtain canonical tautomer: {Chem.MolToSmiles(mol)}"
        )
    return canonical, result.status
