from functools import partial
from typing import Literal

import polars as pl

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


__all__ = ["cluster"]
