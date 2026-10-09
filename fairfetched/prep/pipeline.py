from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial

from rdkit.Chem import Mol

from ._optional import _papyrus_standardize
from .mol_functions import (
    MolFn,
    chembl_cleanup_drawing,
    chembl_flatten_tartrate,
    chembl_fragment_parent,
    chembl_isotope_parent,
    chembl_kekulize,
    chembl_normalize,
    chembl_remove_hs,
    chembl_remove_sgroups,
    chembl_standardize,
    chembl_uncharge,
    chembl_update_valences,
    get_parent,
    mw_between,
    papyrus_canonical_tautomer,
    papyrus_chembl_roundtrip,
    papyrus_no_mixtures,
    papyrus_only_organic,
    papyrus_remove_salts,
    papyrus_uncharge,
    remove_stereo,
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


STEPS_CHEMBL: list[MolFn] = [
    chembl_standardize
]  # uses the mol_functions wrapper, not the rdkit impl
STEPS_CHEMBL_PARENT: list[MolFn] = [
    chembl_standardize,
    get_parent,
]  # salts/solvents stripped
# TODO: Papyrus standardization may remove stereocentres during tautomer
# canonicalisation (tautomer_allow_stereo_removal defaults to True) without
# saying so; state this wherever STEPS_PAPYRUS is documented.
STEPS_PAPYRUS = [_papyrus_standardize]
STEPS_PAPYRUS_ANY_SIZE = [
    partial(_papyrus_standardize, filter_non_small_molecule=False)
]
STEPS_PAPYRUS_NOSTEREO = [remove_stereo, _papyrus_standardize]

# the library functions above, one step per library call, in library order
STEPS_CHEMBL_STANDARDIZE_MOL: list[MolFn] = [  # standardize_mol(check_exclusion=False)
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
# papyrus standardize() defaults, raise_error=False
STEPS_PAPYRUS_STANDARDIZE: list[MolFn] = [
    papyrus_chembl_roundtrip,
    papyrus_remove_salts(),
    papyrus_no_mixtures,
    papyrus_only_organic,
    papyrus_uncharge,
    mw_between(200, 800),
    papyrus_canonical_tautomer(),
    papyrus_chembl_roundtrip,
]
