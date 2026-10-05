from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial

from rdkit.Chem import Mol

from ._optional import _papyrus_standardize
from .mol_functions import MolFn, chembl_standardize, get_parent, remove_stereo


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
        mol = Mol(b)  # ty: ignore[no-matching-overload]
        for step in self.steps:
            mol = step(mol)
            if mol is None:
                return None
        return mol.ToBinary()


def make_mol_pipeline(*steps: MolFn) -> MolPipeline:
    return MolPipeline(steps=tuple(steps))


STEPS_CHEMBL = [
    chembl_standardize
]  # uses the mol_functions wrapper, not the rdkit impl
STEPS_CHEMBL_PARENT = [chembl_standardize, get_parent]  # salts/solvents stripped
# TODO: Papyrus standardization may remove stereocentres during tautomer
# canonicalisation (tautomer_allow_stereo_removal defaults to True) without
# saying so; state this wherever STEPS_PAPYRUS is documented.
STEPS_PAPYRUS = [_papyrus_standardize]
STEPS_PAPYRUS_ANY_SIZE = [
    partial(_papyrus_standardize, filter_non_small_molecule=False)
]
STEPS_PAPYRUS_NOSTEREO = [remove_stereo, _papyrus_standardize]
