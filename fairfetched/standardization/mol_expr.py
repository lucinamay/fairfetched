import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

# from fairfetched.standardization.pipeline import CHEMBL_PIPELINE, MolFn, mol_pipeline
import polars as pl
from chembl_structure_pipeline import standardize_mol
from rdkit import Chem
from rdkit.Chem import RemoveStereochemistry

mp.set_start_method("spawn", force=True)  # @TODO: check if other options are better

MolFn = Callable[[Chem.Mol], Chem.Mol | None]

# //2 for roughly the physical cores - slightly less
_N_WORKERS: int = round((os.cpu_count() or 2.2) // 2.2)


# --- primitive steps ---


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
        return standardize_mol(mol)
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


# --- pipeline builder ---


@dataclass(frozen=True)
class MolPipeline:
    steps: tuple[MolFn, ...]

    def __call__(self, b: bytes | None) -> bytes | None:
        if b is None:
            return None
        mol = Chem.Mol(b)  # ty: ignore[no-matching-overload]
        for step in self.steps:
            mol = step(mol)
            if mol is None:
                return None
        return mol.ToBinary()


def mol_pipeline(*steps: MolFn) -> MolPipeline:
    return MolPipeline(steps=tuple(steps))


# top-level functions required for pickling (ProcessPoolExecutor requirement)
def _smiles_to_binary(s: str | None) -> bytes | None:
    if s is None:
        return None
    mol = Chem.MolFromSmiles(s)
    return mol.ToBinary() if mol else None


def _inchi_to_binary(s: str | None) -> bytes | None:
    if s is None:
        return None
    mol = Chem.inchi.MolFromInchi(s)
    return mol.ToBinary() if mol else None


def _binary_to_smiles(b: bytes | None) -> str | None:
    return Chem.MolToSmiles(Chem.Mol(b)) if b else None


def _binary_to_kekulized_smiles(b: bytes | None) -> str | None:
    return (
        Chem.MolToSmiles(Chem.Mol(b), kekuleSmiles=True, isomericSmiles=False)
        if b
        else None
    )


def _binary_to_inchi(b: bytes | None) -> str | None:
    return Chem.inchi.MolToInchi(Chem.Mol(b)) if b else None


def _binary_to_inchi_and_auxinfo(b: bytes | None) -> str | None:
    return Chem.inchi.MolToInchiAndAuxInfo(Chem.Mol(b)) if b else None


def _binary_to_inchikey(b: bytes | None) -> str | None:
    return Chem.inchi.MolToInchiKey(Chem.Mol(b)) if b else None


def _num_atoms(b: bytes | None) -> int | None:
    return Chem.Mol(b).GetNumAtoms() if b else None


def _num_heavy_atoms(b: bytes | None) -> int | None:
    return Chem.Mol(b).GetNumHeavyAtoms() if b else None


def _parallel_map(
    fn,
    series: pl.Series,
    return_dtype: pl.DataTypeExpr | pl.DataType | Any,
) -> pl.Series:
    with ProcessPoolExecutor(_N_WORKERS) as pool:
        results = list(pool.map(fn, series.to_list(), chunksize=256))
    return pl.Series(series.name, results, dtype=return_dtype)


def _map(
    fn,
    series: pl.Series,
    return_dtype: pl.DataTypeExpr | pl.DataType | Any,
    parallel: bool = False,
) -> pl.Series:
    if parallel:
        with ProcessPoolExecutor(_N_WORKERS) as pool:
            results = list(pool.map(fn, series.to_list(), chunksize=256))
        return pl.Series(series.name, results, dtype=return_dtype)
    return pl.Series(series.name, tuple(map(fn, series)), dtype=return_dtype)


# this setup was suggested by Claude, should still be tested
@pl.api.register_expr_namespace("mol")
class MolExpr:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def from_smiles(self, parallel=False) -> pl.Expr:
        return self._expr.map_batches(
            lambda s: _parallel_map(_smiles_to_binary, s, pl.Binary),
            return_dtype=pl.Binary,
        )

    def from_inchi(self) -> pl.Expr:
        return self._expr.map_batches(
            lambda s: _parallel_map(_inchi_to_binary, s, pl.Binary),
            return_dtype=pl.Binary,
        )

    def to_smiles(self) -> pl.Expr:
        return self._expr.map_batches(
            lambda s: _parallel_map(_binary_to_smiles, s, pl.String),
            return_dtype=pl.String,
        )

    def to_inchi(self) -> pl.Expr:
        return self._expr.map_batches(
            lambda s: _parallel_map(_binary_to_inchi, s, pl.String),
            return_dtype=pl.String,
        )

    def to_inchi_and_auxinfo(self) -> pl.Expr:
        dtype = pl.Struct({"inchi": pl.String, "inchi_auxinfo": pl.String})
        return self._expr.map_batches(
            lambda s: _parallel_map(_binary_to_inchi_and_auxinfo, s, dtype),
            return_dtype=dtype,
        )

    def to_inchikey(self) -> pl.Expr:
        return self._expr.map_batches(
            lambda s: _parallel_map(_binary_to_inchikey, s, pl.String),
            return_dtype=pl.String,
        )

    def to_kekulised_smiles(self) -> pl.Expr:
        return self._expr.map_batches(
            lambda s: _parallel_map(_binary_to_kekulized_smiles, s, pl.String),
            return_dtype=pl.String,
        )

    def num_atoms(self) -> pl.Expr:
        return self._expr.map_batches(
            lambda s: _parallel_map(_num_atoms, s, pl.Int32),
            return_dtype=pl.Int32,
        )

    def num_heavy_atoms(self) -> pl.Expr:
        return self._expr.map_batches(
            lambda s: _parallel_map(_num_heavy_atoms, s, pl.Int32),
            return_dtype=pl.Int32,
        )

    def standardise(self, *steps: MolFn) -> pl.Expr:
        fn = mol_pipeline(*steps)
        return self._expr.map_batches(
            lambda s: _parallel_map(fn, s, pl.Binary),
            return_dtype=pl.Binary,
        )

    # ... rest of namespace


CHEMBL_PIPELINE = (remove_stereo, chembl_standardise, valid_inchi)
