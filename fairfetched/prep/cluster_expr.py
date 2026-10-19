from functools import partial
from typing import Literal

import polars as pl

from fairfetched.prep.split_expr import (
    _X,
    Records,
    _as_records,
    _pack,
    _records,
    _unique_apply,
)

ClusterMethod = Literal[
    "kmeans",
    "dbscan",
    "hdbscan",
    "sphere_exclusion",
    "bitbirch",
    "maxmin",
    "leader_picker",
    "hash_dummy",
    "random",
]


def cluster(
    descriptors: str | pl.Expr,
    *,
    method: ClusterMethod,
    **kwargs,
) -> pl.Expr:
    """Cluster id per row (pl.Int32), from a descriptor column.

    Wraps `nanoom.cluster` (extra: ``fairfetched[split]``).

    descriptors: pl.Array column, e.g. a fingerprint or embedding.
    method: bit-vector methods ("sphere_exclusion", "bitbirch", "maxmin",
        "leader_picker") need 0/1 descriptors; others take any numeric array.
    kwargs: passed to `nanoom.cluster`, e.g. `n_clusters`, `threshold`.

    Output feeds `kfold(groups=...)`. "dbscan" and "hdbscan" label noise
    points -1 unchanged, so `kfold` treats all noise rows as one group.

    Raises on null arrays or null elements.

    >>> df = df.with_columns(cluster=cluster("fp", method="leader_picker"))  # doctest: +SKIP
    """
    expr = pl.col(descriptors) if isinstance(descriptors, str) else descriptors
    return expr.map_batches(
        partial(_array_to_clusters, method=method, **kwargs),
        return_dtype=pl.Int32,
    )


def _array_to_clusters(s: pl.Series, method: ClusterMethod, **kwargs) -> pl.Series:
    """`map_batches` worker: pl.Array column -> cluster id per row."""
    from nanoom import cluster as nanoom_cluster

    if s.explode().null_count():
        raise ValueError("`descriptors` contains nulls")
    labels = nanoom_cluster(s.to_numpy(), method, **kwargs)
    return pl.Series(s.name, labels, dtype=pl.Int32)


def groups(
    X: Records,
    *,
    splitter: str,
    seed: int = 0,
    n_jobs: int = 1,
    **params,
) -> pl.Expr:
    """Group id per row (pl.Int32, dense 0..g-1), from a group-forming
    `chemsplit` splitter (extra: ``fairfetched[split]``): "murcko_scaffold",
    "generic_scaffold", "scaffold_tree", "butina", "sphere_exclusion", ...;
    `split_expr.splitters()` lists them (`group_forming`).

    X: a SMILES column, a pl.Array descriptor column, or a (compound, target)
        pair of columns; structures must already be standardized. Groups are
        formed on the distinct `X` (one RDKit parse per compound) and mapped
        back, so equal structures share a group.
    params: splitter parameters, e.g. `cutoff` for "butina".

    Output feeds `kfold(groups=...)` or `holdout(groups=...)`, like `cluster()`.
    Dataset-independent scaffold ids are `MolExpr.scaffold_cluster` instead.

    Raises on null `X`. Similarity splitters build pairwise distances and raise
    chemsplit's `ScalabilityError` above their limits.

    >>> df = df.with_columns(scaffold=groups("canonical_smiles", splitter="murcko_scaffold"))  # doctest: +SKIP
    """
    return _pack(X=_as_records(X)).map_batches(
        partial(
            _struct_to_groups,
            splitter=splitter,
            seed=seed,
            n_jobs=n_jobs,
            params=params,
        ),
        return_dtype=pl.Int32,
    )


def _struct_to_groups(
    s: pl.Series, splitter: str, seed: int, n_jobs: int, params: dict
) -> pl.Series:
    """`map_batches` worker: struct{X} -> group id per row."""
    from chemsplit import get_splitter

    sp = get_splitter(splitter, random_state=seed, n_jobs=n_jobs, **params)

    def fn(distinct: pl.DataFrame) -> pl.Series:
        labels = sp.compute_groups(_records(distinct.get_column(_X)), standardize=False)
        return pl.Series(labels, dtype=pl.Int32)

    return _unique_apply(s, fn)


__all__ = ["cluster", "groups"]
