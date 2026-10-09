import logging
import multiprocessing as mp
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from polars._typing import PolarsDataType

from fairfetched.utils._track import track

from .failures import CallSite, call_collecting, call_in_context, callsite, reemit
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
    _binary_to_scaffold_smiles,
    _binary_to_smiles,
    _has_stereo,
    _inchi_to_binary,
    _num_atoms,
    _num_fragments,
    _num_heavy_atoms,
    _num_undefined_stereocenters,
    _smiles_to_binary,
    _stable_hash64,
    get_parent,
    mw_between,
    remove_stereo,
)
from .pipeline import (
    STEPS_CHEMBL,
    STEPS_CHEMBL_GET_PARENT_MOL,
    STEPS_CHEMBL_PARENT,
    STEPS_CHEMBL_STANDARDIZE_MOL,
    STEPS_PAPYRUS,
    STEPS_PAPYRUS_ANY_SIZE,
    STEPS_PAPYRUS_NOSTEREO,
    STEPS_PAPYRUS_STANDARDIZE,
    MolPipeline,
)

logger = logging.getLogger(__name__)

# //2 for roughly the physical cores - slightly less
_N_WORKERS: int = round((os.cpu_count() or 2.2) // 2.2)
_CTX = mp.get_context("spawn")


# --- primitive steps ---

# --- pipeline builder ---


def _map_nodedup(
    fn,
    series: pl.Series,
    return_dtype: "PolarsDataType",
    parallel: bool = False,
    site: CallSite | None = None,
) -> pl.Series:
    """`fn` per element, each call in a failure-record context naming the
    element and `site` (the user line that built the expression)."""
    site = site or callsite()
    if parallel:
        try:
            mp.set_start_method(
                "spawn", force=True
            )  # @TODO: check where else to put this that is not the main module
            with ProcessPoolExecutor(_N_WORKERS, mp_context=_CTX) as pool:
                pairs = list(
                    track(
                        pool.map(
                            partial(call_collecting, fn, site),
                            series.to_list(),
                            chunksize=256,
                        ),
                        desc=getattr(fn, "__name__", ""),
                        total=len(series),
                    )
                )
            for _, failures in pairs:
                reemit(failures)
            return pl.Series(series.name, [r for r, _ in pairs], dtype=return_dtype)
        except Exception:
            logger.exception(
                "'parallel' execution failed, resorting to native polars map_batches. "
                "consider passing parallel=False, as this at least allows subdivision into batches"
            )
    return pl.Series(
        series.name,
        tuple(call_in_context(fn, site, x) for x in series),
        dtype=return_dtype,
    )


def _map(
    fn,
    series: pl.Series,
    return_dtype: "PolarsDataType",
    parallel: bool,
    dedup: bool = False,
    site: CallSite | None = None,
) -> pl.Series:
    if not dedup:
        return _map_nodedup(fn, series, return_dtype, parallel=parallel, site=site)
    unique = series.unique()
    results = _map_nodedup(fn, unique, return_dtype, parallel=parallel, site=site)
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
    # user line that built the expression; every failure record cites it
    _site: CallSite | None = None

    def __post_init__(self) -> None:
        if self._site is None:
            object.__setattr__(self, "_site", callsite())

    # --- entry points ---

    @property
    def _pyexpr(self):  # pyright: ignore[reportIncompatibleVariableOverride]
        return self._expr._pyexpr

    @classmethod
    def from_smiles(
        cls, col: str = "smiles", parallel: bool = False, dedup: bool = False
    ) -> "MolExpr":
        site = callsite()
        return cls(
            pl.col(col).map_batches(
                lambda s, **_: _map(
                    _smiles_to_binary, s, pl.Binary, parallel, dedup, site
                ),
                return_dtype=pl.Binary,
                is_elementwise=not parallel,
            ),
            _site=site,
        )

    @classmethod
    def from_inchi(
        cls, col: str = "inchi", parallel: bool = False, dedup: bool = False
    ) -> "MolExpr":
        site = callsite()
        return cls(
            pl.col(col).map_batches(
                lambda s, **_: _map(
                    _inchi_to_binary, s, pl.Binary, parallel, dedup, site
                ),
                return_dtype=pl.Binary,
                is_elementwise=not parallel,
            ),
            _site=site,
        )

    @classmethod
    def col(
        cls, col: str = "mol", parallel: bool = False, dedup: bool = False
    ) -> "MolExpr":
        """Wrap an existing binary mol column."""
        return cls(pl.col(col), _site=callsite())

    @classmethod
    def from_col_infer(
        cls, col: str, parallel: bool = False, dedup: bool = False
    ) -> "MolExpr":
        """Infer mol source from column dtype or first non-null value."""
        site = callsite()

        def _infer(s: pl.Series, **_) -> pl.Series:
            if s.dtype == pl.Binary:
                return s
            first = next((v for v in s if v is not None), None)
            if first is None:
                return s
            if isinstance(first, str) and first.startswith("InChI="):
                return _map(_inchi_to_binary, s, pl.Binary, parallel, dedup, site)
            return _map(_smiles_to_binary, s, pl.Binary, parallel, dedup, site)

        return cls(
            pl.col(col).map_batches(
                _infer,
                return_dtype=pl.Binary,
                is_elementwise=not parallel,
            ),
            _site=site,
        )

    # --- transforms ---

    def standardize(
        self, *steps: MolFn, parallel: bool = False, dedup: bool = False
    ) -> "MolExpr":
        pipeline = MolPipeline(steps=tuple(steps))
        site = self._site
        return MolExpr(
            self._expr.map_batches(
                lambda s, **_: _map(pipeline, s, pl.Binary, parallel, dedup, site),
                return_dtype=pl.Binary,
                is_elementwise=not parallel,
            ),
            _site=site,
        )

    def get_parent(
        self,
        exclude_by_chembl_standards: bool = False,
        parallel: bool = False,
        dedup: bool = False,
    ) -> "MolExpr":
        """Strips salts/solvents and neutralises (see `mol_functions.get_parent`)."""
        step = partial(
            get_parent, exclude_by_chembl_standards=exclude_by_chembl_standards
        )
        return self.standardize(step, parallel=parallel, dedup=dedup)

    def remove_stereo(self, parallel: bool = False, dedup: bool = False) -> "MolExpr":
        """Removes tetrahedral and double-bond stereochemistry."""
        return self.standardize(remove_stereo, parallel=parallel, dedup=dedup)

    def mw_between(
        self,
        lo: float | None = None,
        hi: float | None = None,
        parallel: bool = False,
        dedup: bool = False,
    ) -> "MolExpr":
        """Nulls mols whose exact mass is outside [lo, hi] (inclusive; None =
        open); drop them with `.drop_nulls()` on the frame. Use after
        `get_parent`, otherwise counter-ions count towards the mass."""
        return self.standardize(mw_between(lo, hi), parallel=parallel, dedup=dedup)

    def alias(self, name: str) -> "MolExpr":
        return MolExpr(self._expr.alias(name), _site=self._site)

    # --- 'sinks' ---
    def _apply(self, fn, dtype, parallel: bool = False, dedup: bool = False) -> pl.Expr:
        """convert objects within expression, with specifyable dtype"""
        site = self._site
        return self._expr.map_batches(
            lambda s, **_: _map(fn, s, dtype, parallel, dedup, site),
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
        dtype = pl.Struct(Descriptors.dataclass_schema())  # ty:ignore[invalid-argument-type]  # pyright: ignore[reportArgumentType]
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

    def to_scaffold(
        self, generic: bool = False, parallel: bool = False, dedup: bool = False
    ) -> pl.Expr:
        """Bemis–Murcko scaffold SMILES per molecule (pl.String)."""
        fn = partial(_binary_to_scaffold_smiles, generic=generic)
        return self._apply(fn, pl.String, parallel, dedup)

    def scaffold_cluster(
        self, generic: bool = False, parallel: bool = False, dedup: bool = False
    ) -> pl.Expr:
        """Dataset-independent cluster id per scaffold: stable 64-bit hash of the scaffold SMILES (pl.UInt64)."""
        return self.to_scaffold(generic, parallel, dedup).map_batches(
            lambda s, **_: _map(_stable_hash64, s, pl.UInt64, False, dedup, self._site),
            return_dtype=pl.UInt64,
            is_elementwise=True,
        )

    def has_stereo(self, parallel: bool = False, dedup: bool = False) -> pl.Expr:
        """Atom or double-bond stereo is specified (pl.Boolean); CAPRICHO notes it before stereo removal."""
        return self._apply(_has_stereo, pl.Boolean, parallel, dedup)

    def num_undefined_stereocenters(
        self, parallel: bool = False, dedup: bool = False
    ) -> pl.Expr:
        """Stereocenters with unspecified chirality (pl.Int32), as CAPRICHO `find_undefined_stereocenters`."""
        return self._apply(_num_undefined_stereocenters, pl.Int32, parallel, dedup)

    def num_fragments(self, parallel: bool = False, dedup: bool = False) -> pl.Expr:
        """Disconnected fragments (pl.Int32); > 1 after `get_parent` is a mixture."""
        return self._apply(_num_fragments, pl.Int32, parallel, dedup)

    def to_mol_objects(self, parallel: bool = False, dedup: bool = False) -> pl.Expr:
        """convert to actual Chem.Mol objects. Cannot be written to parquet"""
        return self._apply(_binary_to_mol, pl.Object, parallel, dedup)

    def to_custom(
        self,
        function: Callable,
        return_dtype: "PolarsDataType",
        parallel: bool = False,
        dedup: bool = False,
    ) -> pl.Expr:
        return self._apply(function, return_dtype, parallel, dedup)


__all__ = [
    "STEPS_CHEMBL",
    "STEPS_CHEMBL_GET_PARENT_MOL",
    "STEPS_CHEMBL_PARENT",
    "STEPS_CHEMBL_STANDARDIZE_MOL",
    "STEPS_PAPYRUS",
    "STEPS_PAPYRUS_ANY_SIZE",
    "STEPS_PAPYRUS_NOSTEREO",
    "STEPS_PAPYRUS_STANDARDIZE",
    "MolExpr",
]
