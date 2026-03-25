import logging as lg
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import Pool
from typing import Callable, Iterable, Literal

import polars as pl
from rdkit.Chem import Mol

from fairfetched.utils._track import track
from fairfetched.utils.polars import apply_to_unique, map_batches_pooled

from .compound_fns import (
    _remove_stereo_papyrus_standardise_check_inchi,
    _safe_inchi_to_mol,
    _safe_smiles_to_mol,
    inchi_to_inchikey,
    mols_to_inchikeys,
    mols_to_inchis,
    mols_to_inchis_and_auxinfo,
    mols_to_kekulised_smiles,
    standardised_nostereo_to_smiles_inchi_aux_inchikey,
)

lg.warning(
    "warning: pipe might change soon, preferrably use the mol_expr and its syntax instead"
)


def apply_to_unique_strings(
    lf: pl.LazyFrame,
    function: Callable[[str], Mol | None] = _safe_inchi_to_mol,
    from_col: str = "inchi",
    to_col: str = "mol",
    parallel=True,
    chunks_per_worker: int = 8,
) -> pl.LazyFrame:
    """
    for ~1M mols: 1h07m for fully linear one, 15m for this impl. (now actually <1m)
    chunksize is calculated as max(1, len(unique_molstrings) // (workers * chunks_per_worker))
    so for smaller chunks (good for larger operations), you would need more chunks per worker (800 or so)
    """
    if parallel:
        unique_molstrings = (
            lf.select(pl.col(from_col)).unique().collect().get_column(from_col)  # ty: ignore[unresolved-attribute]
        )
        workers = round(os.cpu_count() or 1 * 0.8)
        chunksize = max(1, len(unique_molstrings) // (workers * chunks_per_worker))
        with ProcessPoolExecutor(max_workers=workers) as p:
            res = track(
                p.map(function, unique_molstrings, chunksize=chunksize),
                total=len(unique_molstrings),
            )

            mols_df = pl.DataFrame(
                [
                    pl.Series(unique_molstrings, dtype=pl.Utf8).alias(from_col),
                    pl.Series(list(res), dtype=pl.Object).alias(to_col),
                ]
            )
        return lf.join(
            mols_df.lazy(),
            on=from_col,
            how="left",
            maintain_order="left",
        )

    return lf.join(
        lf.select(pl.col(from_col))
        .unique()
        .with_columns(
            pl.col(from_col)
            .map_elements(function, return_dtype=pl.Object)
            .alias(to_col)
        ),
        on=from_col,
        how="left",
        maintain_order="left",
    )


def with_cleaned_mol_descriptors_struct(
    lf: pl.LazyFrame,
    descriptor_to_descriptors_func: Callable[[str], tuple[str | None, ...]],
    from_col: str,
    to_col: str,
    return_dtype: pl.DataTypeExpr,
    parallel=True,
    **kwargs,
) -> pl.LazyFrame:
    return lf.pipe(
        apply_to_unique,  # creates unique list, iterates over pooled,
        descriptor_to_descriptors_func,
        from_col=from_col,
        to_col=to_col,
        return_dtype=return_dtype,
        parallel=True,
    )


def with_cleaned_mol_descriptors(
    lf: pl.LazyFrame,
    smiles_col: str = "smiles",
    **kwargs,
) -> pl.LazyFrame:
    """cleans and standardises mols by:
    1) removing stereochemistry
    2) running papyrus standardisation
    3) validating inchis
    returns None if invalid, but does not drop
    """
    fields = ["smiles", "inchi", "inchi_auxinfo", "inchikey"]
    to_col = "descriptors_struct"
    lf = lf.pipe(
        with_cleaned_mol_descriptors_struct,
        descriptor_to_descriptors_func=standardised_nostereo_to_smiles_inchi_aux_inchikey,
        from_col=smiles_col,
        to_col=to_col,
        return_dtype=pl.Struct(
            {k: pl.Utf8 for k in ["smiles", "inchi", "inchi_auxinfo", "inchikey"]}
        ),
        **kwargs,
    )
    return (
        lf.cast({to_col: pl.Struct({k: pl.Utf8 for k in fields})})
        .rename({k: f"{k}_original" for k in fields if k in lf.collect_schema()})
        .with_columns(pl.col(to_col).struct.unnest())
        .drop(to_col)
    )


def with_mols_from_inchi_batched(lf, alias="mol", parallel=False):
    function = partial(
        map_batches_pooled,
        fn=_safe_inchi_to_mol,
        return_dtype=pl.Object,
        parallel=parallel,
        desc="loading mols",  # used only if parallel => tracks progress
    )
    return lf.with_columns(
        pl.col("inchi")
        .map_batches(function, return_dtype=pl.Object, is_elementwise=not parallel)
        .alias(alias)
    )


def with_mols_from_smiles_batched(lf, alias="mol", parallel=False):
    function = partial(
        map_batches_pooled,
        fn=_safe_smiles_to_mol,
        return_dtype=pl.Object,
        parallel=parallel,
        desc="loading mols",  # used only if parallel => tracks progress
    )
    return lf.with_columns(
        pl.col("smiles")
        .map_batches(function, return_dtype=pl.Object, is_elementwise=not parallel)
        .alias(alias)
    )


def with_mols_from_inchi_pool(
    lf: pl.LazyFrame, alias="mol", parallel=True
) -> pl.LazyFrame:
    """
    for ~1M mols: 1h07m for fully linear one, 15m for this impl.
    """
    if parallel:
        unique_inchis = (
            lf.select(pl.col("inchi")).unique().collect().get_column("inchi")  # ty: ignore[unresolved-attribute]
        )
        with Pool() as p:
            res = p.map(_safe_inchi_to_mol, unique_inchis)
        mols_df = pl.DataFrame(
            [
                pl.Series(unique_inchis, dtype=pl.Utf8).alias("inchi"),
                pl.Series(res, dtype=pl.Object).alias(alias),
            ]
        )
        return lf.join(
            mols_df.lazy(),
            on="inchi",
            how="left",
            maintain_order="left",
        )
    return lf.join(
        lf.select(pl.col("inchi"))
        .unique()
        .with_columns(
            pl.col("inchi")
            .map_elements(_safe_inchi_to_mol, return_dtype=pl.Object)
            .alias(alias)
        ),
        on="inchi",
        how="left",
        maintain_order="left",
    )


def with_mols_from_inchi_thread(
    lf: pl.LazyFrame,
    alias="mol",
    parallel=True,
    join=True,
) -> pl.LazyFrame:
    """
    Returns a new DataFrame with a column of RDKit Mol objects from InChI strings.
    """
    if parallel:
        if join:
            inchis = (
                lf.select(pl.col("inchi")).unique().collect().get_column("inchi")  # ty: ignore[unresolved-attribute]
            )
        else:
            lf = lf.sort("inchi")
            inchis = lf.select("inchi").collect().get_column("inchi")  # ty: ignore[unresolved-attribute]
            inchis = lf.select("inchi").collect().get_column("inchi")  # ty: ignore[unresolved-attribute]
        workers = round((os.cpu_count() or 1) * 0.9)
        chunksize = max(1, len(inchis) // (workers * 4))
        print("n_workers=", workers)
        print("chunksize=", chunksize)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            res = list(
                executor.map(
                    _safe_inchi_to_mol,
                    track(inchis, total=len(inchis), desc="loading mols"),
                    chunksize=chunksize,
                )
            )
            mol_series = pl.Series(res, dtype=pl.Object).alias(alias)
        if join:
            mols_df = pl.DataFrame(
                [pl.Series(inchis, dtype=pl.Utf8).alias("inchi"), mol_series]
            )
            return lf.join(
                mols_df.lazy(),
                on="inchi",
                how="left",
                maintain_order="left",
            )
        else:
            lf.with_columns(mol_series)
    return lf.join(
        lf.select(pl.col("inchi"))
        .unique()
        .with_columns(
            pl.col("inchi")
            .map_elements(_safe_inchi_to_mol, return_dtype=pl.Object)
            .alias(alias)
        ),
        on="inchi",
        how="left",
        maintain_order="left",
    )


def with_clean_mols_from_smiles(df: pl.LazyFrame, alias="mol") -> pl.LazyFrame:
    return df.with_columns(
        pl.col("smiles")
        .map_elements(
            _remove_stereo_papyrus_standardise_check_inchi,
            return_dtype=pl.Object,
            strategy="threading",
        )
        .alias("mol")
    )


def _smiles_and_cleaned_mols_df(smiles: Iterable[str]) -> pl.DataFrame:
    with Pool() as p:
        res = p.map(_remove_stereo_papyrus_standardise_check_inchi, smiles)
    return pl.DataFrame(
        [
            pl.Series(smiles, dtype=pl.Utf8).alias("smiles"),
            pl.Series(res, dtype=pl.Object).alias("mol"),
        ]
    )


def _series_mols_to_string_representation(
    mols: pl.Series,
    type: Literal["smiles", "inchi", "inchi_and_auxinfo", "inchikey"] = "smiles",
) -> pl.Series:
    match type:
        case "smiles":
            func = mols_to_kekulised_smiles
            dtype = pl.Utf8
        case "inchi":
            func = mols_to_inchis
            dtype = pl.Utf8
        case "inchi_and_auxinfo":
            func = mols_to_inchis_and_auxinfo
            dtype = pl.Struct({"inchi": pl.Utf8, "inchi_auxinfo": pl.Utf8})
        case "inchikey":
            func = mols_to_inchikeys
            dtype = pl.Utf8
        case _:
            raise NotImplementedError
    return pl.Series(name=type, values=func(mols), dtype=dtype)


# def _unique_with_clean_string_representations(df: pl.LazyFrame) -> pl.LazyFrame:
#     """cleans and standardises molecules, replacing original string
#     representations (which are tagged _papyrus) and adding new ones
#     without the suffix. This is to ensure that all string representations
#     are consistent and can be used for downstream tasks without confusion."""
#     df = df.pipe(with_clean_mols_from_smiles).rename(
#         {k: f"{k}_papyrus" for k in ["smiles", "inchi", "inchi_auxinfo", "inchikey"]},
#         strict=False,
#     )
#     mols = df.lazy().select("mol").collect().get_column("mol")
#     return df.with_columns(
#         _series_mols_to_string_representation(mols, type="smiles"),
#         _series_mols_to_string_representation(
#             mols, type="inchi_and_auxinfo"
#         ).struct.unnest(),
#     ).with_columns(
#         pl.col("inchi")
#         .map_elements(inchi_to_inchikey, return_dtype=pl.Utf8)
#         .alias("inchikey")
#     )


def with_clean_string_representations(df: pl.LazyFrame) -> pl.LazyFrame:
    """Variant of with_clean_string_representations that uses a join on unique mols
    to avoid rerunning similar mols. Throws away all invalid mols"""
    unique_smiles_df: pl.DataFrame = _smiles_and_cleaned_mols_df(
        df.select("smiles").unique().collect()["smiles"]  # ty: ignore
    ).drop_nulls("mol")

    original_col_order = list(df.collect_schema().keys())
    mols: pl.Series = unique_smiles_df["mol"]
    return (
        (
            unique_smiles_df.rename({"smiles": "smiles_"})
            .with_columns(
                # sm iles
                _series_mols_to_string_representation(mols, type="smiles"),
                # inchis + auxinfo
                *(
                    _series_mols_to_string_representation(
                        mols, type="inchi_and_auxinfo"
                    )
                    .struct.unnest()
                    .iter_columns()
                ),
            )
            # inchikeys directly from the clean inchis
            .with_columns(
                pl.col("inchi")
                .map_elements(inchi_to_inchikey, return_dtype=pl.Utf8)
                .alias("inchikey")
            )
        )
        .lazy()
        .join(
            df.lazy(),
            left_on="smiles_",
            right_on="smiles",
            how="right",
            maintain_order="right",
            suffix="_old",
        )
        .select(*original_col_order, pl.exclude(original_col_order).exclude("mol"))
    )


def __conditional_rename(
    df: pl.DataFrame | pl.LazyFrame, dict_: dict[str, str] | None
) -> pl.DataFrame | pl.LazyFrame:
    if not dict_:
        return df
    dict_ = {k: v for k, v in dict_.items() if k in df.collect_schema()}
    return df.rename(dict_)


# ====== utils =======


def task_readout_per_activity(
    flat_df: pl.DataFrame | pl.LazyFrame,
    readout_fmt: str = "^type_.*$",
    group_by="activity_id",
) -> pl.DataFrame | pl.LazyFrame:
    """calculates the sum of readouts per activity. useful for stratification for splitting.
    Args:
        - flat_df (pl.DataFrame | pl.LazyFrame): dataframe with single datapoints
            that have one-hot-encodings for each activity column
        - readout_fmt (str): regex expression (or single string) for selecting the readout columns
        - group_by (str): column name for grouping by, e.g. `activity_id`,`target_id`,`smiles`

    Returns:
        - pl.LazyFrame with columns [group_by, readout_fmt] where readout_fmt
        is the sum of the readouts for each group_by unit.

    Example usage:
        `pl.scan_parquet('papyrus.parquet').pipe(semicolon_to_flat).pipe(task_readout_per_activity)`
    """
    return (
        flat_df.select(pl.col(readout_fmt), group_by)
        .group_by(group_by)
        .agg(pl.col(readout_fmt).cast(pl.Int8).sum())
    )


def task_protein_per_mol(
    flat_df, target_col="target_id", mol_col="inchi"
) -> pl.LazyFrame:
    return flat_df.select(target_col, mol_col).pivot(
        on=target_col,
        on_columns=flat_df.select(target_col).unique().collect().to_series(),
        index=mol_col,
        values=target_col,
        aggregate_function="len",
    )


def filter_is_in(lf, lf_series_filter):
    """join-based filter is_in for a lazyframe.
    Args:
        lf (pl.LazyFrame): lazyframe to filter
        lf_series_filter (pl.LazyFrame): lf for which values to keep. if not single-column, will use all columns. Usually only single-columned LF.

    Note: could be used with pl.DataFrames too but I generally avoid joins with eager df's due to memory concerns
    """
    return lf.lazy().join(
        lf_series_filter.unique().lazy(),
        on=lf_series_filter.collect_schema().names(),  # all cols
        how="semi",  # keep left if in right
        maintain_order="left",
    )
