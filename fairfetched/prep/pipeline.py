"""Steps bound to the ChEMBL and Papyrus structure pipelines, and the step
lists that reproduce each library call sequence."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial

from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem.MolStandardize.rdMolStandardize import TautomerEnumeratorStatus

from fairfetched.prep import papyrus
from fairfetched.prep._optional import (
    _chembl_standardize,
    chembl_exclude_flag,
    chembl_get_parent_mol,
    chembl_standardizer,
)
from fairfetched.prep.failures import note, warn
from fairfetched.prep.mol_functions import (
    MolFn,
    mw_between,
    remove_stereo,
    safe_step,
    safe_step_function,
    sanitize,
)


@dataclass(frozen=True)
class MolPipeline:
    steps: Iterable[MolFn]

    @property
    def __name__(self) -> str:
        return (
            "MolPipeline("
            + ",".join(getattr(s, "__name__", "?") for s in self.steps)
            + ")"
        )

    def __call__(self, b: bytes | None) -> bytes | None:
        if b is None:
            return None
        mol = Mol(b)  # ty: ignore[no-matching-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
        for step in self.steps:
            mol = step(mol)
            if mol is None:
                return None
        return mol.ToBinary()


def make_mol_pipeline(*steps: MolFn) -> MolPipeline:
    return MolPipeline(steps=tuple(steps))


# --- chembl_structure_pipeline ---


@safe_step
def chembl_exclude(mol: Mol) -> Mol | None:
    """None where ChEMBL's `exclude_flag` is set: any atom of its metal list,
    or more than 7 borons. ChEMBL does not register such structures."""
    if chembl_exclude_flag(mol, includeRDKitSanitization=False):
        note("chembl_exclude", "ChEMBL exclude flag set")
        return None
    return mol


@safe_step
def chembl_standardize(mol: Mol) -> Mol | None:
    """`chembl_structure_pipeline.standardize_mol` with its defaults, which
    return a mol `chembl_exclude` would drop untouched: no normalization, no
    uncharging. Such mols get a WARNING record; `STEPS_CHEMBL_STANDARDIZE_MOL`
    standardizes them instead."""
    if chembl_exclude_flag(mol, includeRDKitSanitization=False):
        warn("chembl_standardize", "ChEMBL exclude flag set; returned unstandardized")
    return _chembl_standardize(mol)


@safe_step
def get_parent(mol: Mol, exclude_by_chembl_standards: bool = False) -> Mol | None:
    """`chembl_structure_pipeline.get_parent_mol`: strips salts/solvents and
    neutralises. `exclude_by_chembl_standards=True` returns None where
    `chembl_exclude` would drop the salt-stripped parent, e.g. cisplatin;
    the default keeps them."""
    parent, exclude = chembl_get_parent_mol(
        mol, check_exclusion=exclude_by_chembl_standards
    )
    if exclude:
        note("get_parent", "ChEMBL exclude flag set")
        return None
    return parent


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
def chembl_isotope_parent(mol: Mol) -> Mol | None:
    """Isotope labels cleared, then `chembl_remove_hs`."""
    return chembl_standardizer.get_isotope_parent_mol(mol)


@safe_step_function("chembl_fragment_parent")
def _chembl_fragment_parent(
    mol: Mol, neutralize: bool, exclude_by_chembl_standards: bool
) -> Mol | None:
    parent, excluded = chembl_standardizer.get_fragment_parent_mol(
        mol, check_exclusion=exclude_by_chembl_standards, neutralize=neutralize
    )
    if excluded:
        note("chembl_fragment_parent", "ChEMBL exclude flag set")
        return None
    return parent


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


# --- papyrus_structure_pipeline (`papyrus.py` holds the copied functions) ---


@safe_step
def papyrus_chembl_roundtrip(mol: Mol) -> Mol | None:
    """`standardize_mol`, `get_parent_mol`, then a SMILES write/read; Papyrus
    runs it first and last."""
    return papyrus.chembl_roundtrip(mol)


@safe_step_function("papyrus_remove_salts")
def _papyrus_remove_salts(mol: Mol, include_metals: bool) -> Mol | None:
    return papyrus.remove_supplementary_salts(mol, include_metals=include_metals)


def papyrus_remove_salts(include_metals: bool = True) -> MolFn:
    """Step stripping Papyrus's extra salt list and, with `include_metals`
    (the Papyrus default), every metal atom."""
    return partial(_papyrus_remove_salts, include_metals=include_metals)


@safe_step
def papyrus_no_mixtures(mol: Mol) -> Mol | None:
    """None where `Chem.GetMolFrags` finds more than one fragment."""
    if papyrus.is_mixture(mol):
        note(
            "papyrus_no_mixtures", f"mixture of {len(Chem.GetMolFrags(mol))} fragments"
        )
        return None
    return mol


@safe_step
def papyrus_only_organic(mol: Mol) -> Mol | None:
    """None where `chembl_exclude` would drop the mol, no C~C bond exists, or
    an element is outside {C,H,O,N,P,S,F,Cl,Br,I}."""
    if papyrus.is_organic(mol):
        return mol
    note("papyrus_only_organic", "inorganic")
    return None


@safe_step
def papyrus_uncharge(mol: Mol) -> Mol | None:
    """`rdMolStandardize.Uncharger`, then B3DB's SMARTS neutralizations."""
    return papyrus.uncharge(mol)


@safe_step_function("papyrus_canonical_tautomer")
def _papyrus_canonical_tautomer(
    mol: Mol, allow_stereo_removal: bool, max_tautomers: int
) -> Mol | None:
    canonical, status = papyrus.canonicalize_tautomer(
        mol, allow_stereo_removal=allow_stereo_removal, max_tautomers=max_tautomers
    )
    if status != TautomerEnumeratorStatus.Completed:
        warn("papyrus_canonical_tautomer", f"tautomer enumeration {status.name}")
    return canonical


def papyrus_canonical_tautomer(
    allow_stereo_removal: bool = True, max_tautomers: int = 2**32 - 1
) -> MolFn:
    """Step picking RDKit's canonical tautomer as Papyrus does: the default
    `allow_stereo_removal=True` lets tautomerization erase stereocentres;
    False keeps only tautomers with the input's number of chiral centres.
    A WARNING record when enumeration stopped at `max_tautomers` or RDKit's
    transform limit."""
    return partial(
        _papyrus_canonical_tautomer,
        allow_stereo_removal=allow_stereo_removal,
        max_tautomers=max_tautomers,
    )


# --- step lists ---

STEPS_CHEMBL: list[MolFn] = [chembl_standardize]
STEPS_CHEMBL_PARENT: list[MolFn] = [chembl_standardize, get_parent]
# `standardize_mol` as one step per library call, in library order.
# `standardize_mol(check_exclusion=True)` returns an excluded mol untouched; a
# step list cannot branch, so this list equals `check_exclusion=False`, and
# `chembl_exclude` drops excluded mols instead.
STEPS_CHEMBL_STANDARDIZE_MOL: list[MolFn] = [
    chembl_update_valences,
    chembl_remove_sgroups,
    chembl_kekulize,
    chembl_remove_hs,
    chembl_normalize,
    chembl_uncharge,
    chembl_flatten_tartrate,
    chembl_cleanup_drawing,
    sanitize,
]
STEPS_CHEMBL_GET_PARENT_MOL: list[MolFn] = [
    chembl_isotope_parent,
    chembl_fragment_parent(),
]
# Papyrus `standardize()` defaults (`raise_error=False`), one step per library
# call. Tautomer canonicalisation may erase stereocentres
# (`allow_stereo_removal=True`) without saying so.
STEPS_PAPYRUS: list[MolFn] = [
    papyrus_chembl_roundtrip,
    papyrus_remove_salts(),
    papyrus_no_mixtures,
    papyrus_only_organic,
    papyrus_uncharge,
    mw_between(200, 800),
    papyrus_canonical_tautomer(),
    papyrus_chembl_roundtrip,
]
STEPS_PAPYRUS_ANY_SIZE: list[MolFn] = [  # `filter_non_small_molecule=False`
    papyrus_chembl_roundtrip,
    papyrus_remove_salts(),
    papyrus_no_mixtures,
    papyrus_only_organic,
    papyrus_uncharge,
    papyrus_canonical_tautomer(),
    papyrus_chembl_roundtrip,
]
STEPS_PAPYRUS_NOSTEREO: list[MolFn] = [remove_stereo, *STEPS_PAPYRUS]
