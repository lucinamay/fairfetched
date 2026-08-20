import logging
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import polars as pl

from fairfetched.utils._track import track

from .mol_functions import (
    Descriptors,
    MolFn,
    _binary_to_descriptors,
    _binary_to_inchi,
    _binary_to_inchi_and_auxinfo,
    _binary_to_inchikey,
    _binary_to_kekulized_smiles,
    _binary_to_mol,
    _binary_to_morgan_array,
    _binary_to_smiles,
    _inchi_to_binary,
    _num_atoms,
    _num_heavy_atoms,
    _smiles_to_binary,
)
from .pipeline import (
    STEPS_CHEMBL,
    STEPS_PAPYRUS,
    STEPS_PAPYRUS_NOSTEREO,
    MolPipeline,
)

# //2 for roughly the physical cores - slightly less
_N_WORKERS: int = round((os.cpu_count() or 2.2) // 2.2)
_CTX = mp.get_context("spawn")


# --- primitive steps ---

# --- pipeline builder ---


def _map_nodedup(
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
                results = list(
                    track(
                        pool.map(fn, series.to_list(), chunksize=256),
                        desc=getattr(fn, "__name__", ""),
                        total=len(series),
                    )
                )
            return pl.Series(series.name, results, dtype=return_dtype)
        except Exception as e:
            logging.exception(
                f"'parallel' execution did not work with exception: {e}, resorting to native polars map_batches. "
                "consider passing parallel=False, as this at least allows subdivision into batches"
            )
    return pl.Series(
        series.name,
        tuple(map(fn, series)),
        dtype=return_dtype,
    )


def _map(
    fn,
    series: pl.Series,
    return_dtype: pl.DataTypeExpr | pl.DataType | Any,
    parallel: bool,
    dedup: bool = False,
) -> pl.Series:
    if not dedup:
        return _map_nodedup(fn, series, return_dtype, parallel=parallel)
    unique = series.unique()
    results = _map_nodedup(fn, unique, return_dtype, parallel=parallel)
    mapping = pl.DataFrame({"k": unique, "v": results})
    return (
        series.to_frame("k").join(mapping, on="k", how="left")["v"].rename(series.name)
    )


@pl.api.register_expr_namespace("mol")
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
    def from_smiles(
        cls, col: str = "smiles", parallel: bool = False, dedup: bool = False
    ) -> "MolExpr":
        return cls(
            pl.col(col).map_batches(
                lambda s: _map(_smiles_to_binary, s, pl.Binary, parallel, dedup),
                return_dtype=pl.Binary,
                is_elementwise=not parallel,
            )
        )

    @classmethod
    def from_inchi(
        cls, col: str = "inchi", parallel: bool = False, dedup: bool = False
    ) -> "MolExpr":
        return cls(
            pl.col(col).map_batches(
                lambda s, return_dtype: _map(
                    _inchi_to_binary, s, pl.Binary, parallel, dedup
                ),
                return_dtype=pl.Binary,
                is_elementwise=not parallel,
            )
        )

    @classmethod
    def col(
        cls, col: str = "mol", parallel: bool = False, dedup: bool = False
    ) -> "MolExpr":
        """Wrap an existing binary mol column."""
        return cls(pl.col(col))

    @classmethod
    def from_col_infer(
        cls, col: str, parallel: bool = False, dedup: bool = False
    ) -> "MolExpr":
        """Infer mol source from column dtype or first non-null value."""

        def _infer(s: pl.Series) -> pl.Series:
            if s.dtype == pl.Binary:
                return s
            first = next((v for v in s if v is not None), None)
            if first is None:
                return s
            if isinstance(first, str) and first.startswith("InChI="):
                return _map(_inchi_to_binary, s, pl.Binary, parallel, dedup)
            return _map(_smiles_to_binary, s, pl.Binary, parallel, dedup)

        return cls(
            pl.col(col).map_batches(
                _infer,
                return_dtype=pl.Binary,
                is_elementwise=not parallel,
            )
        )

    # --- transforms ---

    def standardize(
        self, *steps: MolFn, parallel: bool = False, dedup: bool = False
    ) -> "MolExpr":
        pipeline = MolPipeline(steps=tuple(steps))
        return MolExpr(
            self._expr.map_batches(
                lambda s: _map(pipeline, s, pl.Binary, parallel, dedup),
                return_dtype=pl.Binary,
                is_elementwise=not parallel,
            )
        )

    def alias(self, name: str) -> "MolExpr":
        return MolExpr(self._expr.alias(name))

    # --- 'sinks' ---
    def _apply(self, fn, dtype, parallel: bool = False, dedup: bool = False) -> pl.Expr:
        """convert objects within expression, with specifyable dtype"""
        return self._expr.map_batches(
            lambda s: _map(fn, s, dtype, parallel, dedup),
            return_dtype=dtype,
            is_elementwise=not parallel,
        )

    def to_binary(self, parallel: bool = False) -> pl.Expr:
        return self._expr

    def to_smiles(self, parallel: bool = False) -> pl.Expr:
        return self._apply(_binary_to_smiles, pl.String, parallel)

    def to_inchi(self, parallel: bool = False) -> pl.Expr:
        return self._apply(_binary_to_inchi, pl.String, parallel)

    def to_inchikey(self, parallel: bool = False) -> pl.Expr:
        return self._apply(_binary_to_inchikey, pl.String, parallel)

    def to_kekulized_smiles(self, parallel: bool = False) -> pl.Expr:
        return self._apply(_binary_to_kekulized_smiles, pl.String, parallel)

    def to_inchi_and_auxinfo(self, parallel: bool = False) -> pl.Expr:
        """from mol to a pl.struct of inchi, inchi_auxinfo, both pl.String types"""
        dtype = pl.Struct({"inchi": pl.String, "inchi_auxinfo": pl.String})
        return self._apply(
            _binary_to_inchi_and_auxinfo, dtype, parallel
        ).struct.unnest()

    def to_descriptors(
        self,
        parallel: bool = False,
        dedup: bool = False,
    ) -> pl.Expr:
        """returns inchi, inchi_auxinfo, inchikey, kekulised smiles (as ‘smiles’)"""
        dtype = pl.Struct(Descriptors.dataclass_schema())  # ty:ignore[invalid-argument-type]
        fn = partial(_binary_to_descriptors)
        return self._apply(fn, dtype, parallel, dedup).struct.unnest()

    def to_morgan_fp(
        self,
        radius: int = 3,
        fp_size: int = 2048,
        parallel: bool = False,
        dedup: bool = False,
        **kwargs,
    ) -> pl.Expr:
        """convenience function for morgan fingerprint generation. expression generates an array of dtype pl.Array(pl.UInt8, fp_size)"""
        fn = partial(_binary_to_morgan_array, radius=radius, fp_size=fp_size, **kwargs)
        dtype = pl.Array(pl.UInt8, fp_size)
        return self._apply(fn, dtype, parallel, dedup)

    def num_atoms(self, parallel: bool = False, dedup: bool = False) -> pl.Expr:
        return self._apply(_num_atoms, pl.Int32, parallel, dedup)

    def num_heavy_atoms(self, parallel: bool = False, dedup: bool = False) -> pl.Expr:
        return self._apply(_num_heavy_atoms, pl.Int32, parallel, dedup)

    def to_mol_objects(self, parallel: bool = False, dedup: bool = False) -> pl.Expr:
        """convert to actual Chem.Mol objects. Cannot be written to parquet"""
        return self._apply(_binary_to_mol, pl.Object, parallel, dedup)

    def to_custom(
        self,
        function: Callable,
        return_dtype: pl.DataTypeExpr,
        parallel: bool = False,
        dedup: bool = False,
    ) -> pl.Expr:
        return self._apply(function, return_dtype, parallel, dedup)


__all__ = [
    "STEPS_CHEMBL",
    "STEPS_PAPYRUS",
    "STEPS_PAPYRUS_NOSTEREO",
    "MolExpr",
]
