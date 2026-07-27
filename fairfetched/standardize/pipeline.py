from dataclasses import dataclass
from typing import Iterable

from rdkit.Chem import Mol

from ._optional import _papyrus_standardize
from .mol_functions import MolFn, chembl_standardize, remove_stereo


@dataclass(frozen=True)
class MolPipeline:
    steps: Iterable[MolFn]

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
STEPS_PAPYRUS = [_papyrus_standardize]
STEPS_PAPYRUS_NOSTEREO = [remove_stereo, _papyrus_standardize]
