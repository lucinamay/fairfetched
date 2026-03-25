import logging
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

# from fairfetched.standardization.pipeline import CHEMBL_PIPELINE, MolFn, mol_pipeline
import polars as pl
from rdkit import Chem

from .mol_functions import (
    MolFn,
    _binary_to_inchi,
    _binary_to_inchi_and_auxinfo,
    _binary_to_inchikey,
    _binary_to_kekulized_smiles,
    _binary_to_mol,
    _binary_to_smiles,
    _inchi_to_binary,
    _num_atoms,
    _smiles_to_binary,
)
from .pipeline import (
    PIPELINE_CHEMBL,
    PIPELINE_PAPYRUS,
    PIPELINE_PAPYRUS_NOSTEREO,
    MolPipeline,
    mol_pipeline,
)

# //2 for roughly the physical cores - slightly less
_N_WORKERS: int = round((os.cpu_count() or 2.2) // 2.2)
_CTX = mp.get_context("spawn")


# --- primitive steps ---

# --- pipeline builder ---


def _map(
    fn,
    series: pl.Series,
    return_dtype: pl.DataTypeExpr | pl.DataType | Any,
    parallel: bool = False,
) -> pl.Series:
    if parallel:
        try:
            mp.set_start_method(
                "spawn", force=True
            )  # @TODO: check where else to put this that is not the main module
            with ProcessPoolExecutor(_N_WORKERS, mp_context=_CTX) as pool:
                results = list(pool.map(fn, series.to_list(), chunksize=256))
            return pl.Series(series.name, results, dtype=return_dtype)
        except Exception:
            logging.warning(
                "'parallel' execution did not work, resorting to native polars map_batches"
                "consider passing parallel=False, as this at least allows subdivision into batches"
            )
    return pl.Series(series.name, tuple(map(fn, series)), dtype=return_dtype)


@pl.api.register_expr_namespace("mol")
class PlMolExpr:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def from_smiles(self, parallel: bool = False) -> pl.Expr:
        return self._expr.map_batches(
            # lambda s: _parallel_map(_smiles_to_binary, s, pl.Binary),
            lambda s: _map(_smiles_to_binary, s, pl.Binary, parallel),
            return_dtype=pl.Binary,
            is_elementwise=not parallel,
        )

    def from_inchi(self, parallel: bool = False) -> pl.Expr:
        return self._expr.map_batches(
            lambda s: _map(_inchi_to_binary, s, pl.Binary, parallel),
            return_dtype=pl.Binary,
            is_elementwise=not parallel,
        )

    def to_smiles(self, parallel: bool = False) -> pl.Expr:
        return self._expr.map_batches(
            lambda s: _map(_binary_to_smiles, s, pl.String, parallel),
            return_dtype=pl.String,
            is_elementwise=not parallel,
        )

    def to_inchi(self, parallel: bool = False) -> pl.Expr:
        return self._expr.map_batches(
            lambda s: _map(_binary_to_inchi, s, pl.String, parallel),
            return_dtype=pl.String,
            is_elementwise=not parallel,
        )

    def to_inchi_and_auxinfo(self, parallel: bool = False) -> pl.Expr:
        dtype = pl.Struct({"inchi": pl.String, "inchi_auxinfo": pl.String})
        return self._expr.map_batches(
            lambda s: _map(_binary_to_inchi_and_auxinfo, s, dtype, parallel),
            return_dtype=dtype,
            is_elementwise=not parallel,
        )

    def to_inchikey(self, parallel: bool = False) -> pl.Expr:
        return self._expr.map_batches(
            lambda s: _map(_binary_to_inchikey, s, pl.String, parallel),
            return_dtype=pl.String,
            is_elementwise=not parallel,
        )

    def to_kekulised_smiles(self, parallel: bool = False) -> pl.Expr:
        return self._expr.map_batches(
            lambda s: _map(_binary_to_kekulized_smiles, s, pl.String, parallel),
            return_dtype=pl.String,
            is_elementwise=not parallel,
        )

    def num_atoms(self, parallel: bool = False) -> pl.Expr:
        return self._expr.map_batches(
            lambda s: _map(_num_atoms, s, pl.Int32, parallel),
            return_dtype=pl.Int32,
            is_elementwise=not parallel,
        )

    def num_heavy_atoms(self, parallel: bool = False) -> pl.Expr:
        return self._expr.map_batches(
            lambda s: _map(_num_heavy_atoms, s, pl.Int32, parallel),
            return_dtype=pl.Int32,
            is_elementwise=not parallel,
        )

    def standardise(self, *steps: MolFn, parallel: bool = False) -> pl.Expr:
        """to a mol, apply standardisation steps (mol->mol | None)"""
        fn = mol_pipeline(*steps)
        return self._expr.map_batches(
            lambda s: _map(fn, s, pl.Binary, parallel),
            return_dtype=pl.Binary,
            is_elementwise=not parallel,
        )

    # ... rest of namespace


@dataclass(frozen=True)
class MolExpr(pl.Expr):
    """Lightweight pl.Expr copy for mol-specific functionalities"""

    _expr: pl.Expr
    _parallel: bool = False
    # --- entry points ---

    @property
    def _pyexpr(self):
        return self._expr._pyexpr

    @classmethod
    def from_smiles(cls, col: str = "smiles", parallel: bool = False) -> "MolExpr":
        return cls(
            pl.col(col).map_batches(
                lambda s: _map(_smiles_to_binary, s, pl.Binary, cls._parallel),
                return_dtype=pl.Binary,
                is_elementwise=not cls._parallel,
            )
        )

    @classmethod
    def from_inchi(cls, col: str = "inchi", parallel: bool = False) -> "MolExpr":
        return cls(
            pl.col(col).map_batches(
                lambda s: _map(_inchi_to_binary, s, pl.Binary, parallel),
                return_dtype=pl.Binary,
                is_elementwise=not parallel,
            )
        )

    @classmethod
    def from_col(cls, col: str, parallel: bool = False) -> "MolExpr":
        """Wrap an existing binary mol column."""
        return cls(pl.col(col))

    # --- transforms ---

    def standardise(self, *steps: MolFn, parallel: bool = False) -> "MolExpr":
        pipeline = MolPipeline(steps=tuple(steps))
        return MolExpr(
            self._expr.map_batches(
                lambda s: _map(pipeline, s, pl.Binary, parallel),
                return_dtype=pl.Binary,
                is_elementwise=not parallel,
            )
        )

    def alias(self, name: str) -> "MolExpr":
        return MolExpr(self._expr.alias(name))

    # --- 'sinks' ---

    def to_binary(self, parallel: bool = False) -> pl.Expr:
        return self._expr

    def to_smiles(self, parallel: bool = False) -> pl.Expr:
        return self._expr.map_batches(
            lambda s: _map(_binary_to_smiles, s, pl.String, parallel),
            return_dtype=pl.String,
            is_elementwise=not parallel,
        )

    def to_inchi(self, parallel: bool = False) -> pl.Expr:
        return self._expr.map_batches(
            lambda s: _map(_binary_to_inchi, s, pl.String, parallel),
            return_dtype=pl.String,
            is_elementwise=not parallel,
        )

    def to_inchikey(self, parallel: bool = False) -> pl.Expr:
        return self._expr.map_batches(
            lambda s: _map(_binary_to_inchikey, s, pl.String, parallel),
            return_dtype=pl.String,
            is_elementwise=not parallel,
        )

    def to_kekulised_smiles(self, parallel: bool = False) -> pl.Expr:
        return self._expr.map_batches(
            lambda s: _map(_binary_to_kekulized_smiles, s, pl.String, parallel),
            return_dtype=pl.String,
            is_elementwise=not parallel,
        )

    def to_inchi_and_auxinfo(self, parallel: bool = False) -> pl.Expr:
        """from mol to a pl.struct of inchi, inchi_auxinfo, both pl.String types"""
        dtype = pl.Struct({"inchi": pl.String, "inchi_auxinfo": pl.String})
        return self._expr.map_batches(
            lambda s: _map(_binary_to_inchi_and_auxinfo, s, dtype, parallel),
            return_dtype=dtype,
            is_elementwise=not parallel,
        )

    def num_atoms(self, parallel: bool = False) -> pl.Expr:
        return self._expr.map_batches(
            lambda s: _map(_num_atoms, s, pl.Int32, parallel),
            return_dtype=pl.Int32,
            is_elementwise=not parallel,
        )

    def num_heavy_atoms(self, parallel: bool = False) -> pl.Expr:
        return self._expr.map_batches(
            lambda s: _map(_num_heavy_atoms, s, pl.Int32, parallel),
            return_dtype=pl.Int32,
            is_elementwise=not parallel,
        )

    def to_mol_objects(self, parallel: bool = False) -> pl.Expr:
        """convert to actual Chem.Mol objects. Cannot be written to parquet"""
        return self._expr.map_batches(
            lambda s: _map(_binary_to_mol, s, pl.Object, parallel),
            return_dtype=pl.Object,
            is_elementwise=not parallel,
        )


# CHEMBL_PIPELINE = (remove_stereo, chembl_standardise, valid_inchi)
__all__ = [
    PIPELINE_CHEMBL,
    PIPELINE_PAPYRUS,
    PIPELINE_PAPYRUS_NOSTEREO,
    PlMolExpr,
    MolFn,
]


# ================ experimental ========================


def _make_batch_mapper(fn, return_dtype, parallel: bool):
    if parallel:

        def mapper(s: pl.Series) -> pl.Series:
            with ProcessPoolExecutor(_N_WORKERS, mp_context=_CTX) as pool:
                out = list(pool.map(fn, s.to_list(), chunksize=256))
            return pl.Series(s.name, out, dtype=return_dtype)

        is_elementwise = False
    else:

        def mapper(s: pl.Series) -> pl.Series:
            return pl.Series(s.name, tuple(map(fn, s)), dtype=return_dtype)

        is_elementwise = True

    return mapper, is_elementwise


def _map_directly(expr: pl.Expr, fn, return_dtype, parallel: bool) -> pl.Expr:
    mapper, is_elementwise = _make_batch_mapper(fn, return_dtype, parallel)

    return expr.map_batches(
        mapper,
        return_dtype=return_dtype,
        is_elementwise=is_elementwise,
    )
