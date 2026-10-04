from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Literal

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

Metric = Literal["tanimoto", "cosine", "euclidean", "cityblock"]
Multitask = Literal["mean_std", "rms_std"]
Replicates = Literal["pooled", "mean"]
_Y = "__y__"
_N = "__n__"
_VAR = "__var__"


def roughness(
    y: str | pl.Expr,
    *,
    X: dict[str, Metric],
    weights: dict[str, float] | None = None,
    rogi_xd: bool = True,
    multitask: Multitask = "mean_std",
    max_rows: int = 10_000,
    min_dt: float = 0.01,
) -> pl.Expr:
    """Roughness index of `y` over the descriptor columns in `X` (pl.Float64 scalar).

    Reimplements `rogi` (Aldeghi 2022) and `rogi-xd` (Graff 2023) on scipy
    (extra: ``fairfetched[rough]``), extended to several descriptor columns
    and to vector `y`.

    y: Float column (one row per sample, or per drug-target pair in long
        format), or pl.Array(Float, n_tasks) (one row per drug, wide format).
        Each task is min-max scaled to [0, 1]. Null array elements are skipped
        per task.
    X: column name -> metric; columns are pl.Array, e.g.
        {"fp": "tanimoto", "prot_emb": "cosine"}. Each distance is scaled to
        [0, 1] before the weighted mean over columns.
    weights: column name -> weight in that mean. None weighs columns equally.
    rogi_xd: True integrates over 1 - log(n_clusters) / log(n) (ROGI-XD),
        comparable across descriptors of different dimensionality; False
        integrates over the distance threshold (original ROGI).
    multitask: how tasks of a vector `y` combine into one dispersion.
        "mean_std" is the mean of the per-task standard deviations; without
        nulls the index equals the mean of the per-task indices. "rms_std" is
        the root of the mean per-task variance, which weighs high-variance
        tasks more.
    max_rows: pairwise distances take n * (n - 1) / 2 float64 (0.4 GB at
        10,000 rows), plus a transient n x n float64 (0.8 GB) while gathering
        a column that has repeated rows.
    min_dt: smallest step between clustering thresholds, on the [0, 1] distance.

    Raises on nulls in `X`, on a null scalar `y`, on a `y` task without any
    value, on fewer than 2 rows, and above `max_rows` (subsample first with
    `df.sample`).

    Reduces to one value, so it is valid per group:

    >>> df.select(
    ...     rogi=roughness("pchembl", X={"fp": "tanimoto", "prot_emb": "cosine"})
    ... )  # doctest: +SKIP
    >>> df.group_by("target_id").agg(
    ...     rogi=roughness("pchembl", X={"fp": "tanimoto"})
    ... )  # doctest: +SKIP
    """
    y = pl.col(y) if isinstance(y, str) else y
    return pl.struct(y.alias(_Y), *X).map_batches(
        partial(
            _struct_to_rogi,
            metrics=X,
            weights=weights,
            rogi_xd=rogi_xd,
            multitask=multitask,
            max_rows=max_rows,
            min_dt=min_dt,
        ),
        return_dtype=pl.Float64,
        returns_scalar=True,
    )


def _struct_to_rogi(
    s: pl.Series,
    metrics: dict[str, Metric],
    weights: dict[str, float] | None,
    rogi_xd: bool,
    multitask: Multitask,
    max_rows: int,
    min_dt: float,
) -> float:
    """`map_batches` worker: struct{y, *X} -> validate -> `_combined_distance` -> `_rogi`."""
    import numpy as np

    df = s.struct.unnest()
    if df.height < 2:
        raise ValueError(f"roughness needs at least 2 rows, got {df.height}")
    if df.height > max_rows:
        raise ValueError(
            f"{df.height} rows exceed max_rows={max_rows}; subsample first, "
            f"e.g. df.sample({max_rows}, seed=0), or raise max_rows"
        )
    if n := df.select(pl.any_horizontal(pl.col(list(metrics)).is_null()).sum()).item():
        raise ValueError(f"{n} rows have a null in `X`")
    y = df.get_column(_Y)
    if isinstance(y.dtype, pl.Array):
        y = y.to_numpy().astype(float)
        if (empty := np.flatnonzero(np.isnan(y).all(0))).size:
            raise ValueError(f"`y` tasks {empty.tolist()} have no non-null value")
    elif y.null_count():
        raise ValueError(f"`y` contains {y.null_count()} nulls")
    else:
        y = y.to_numpy().astype(float).reshape(-1, 1)

    low = np.nanmin(y, axis=0)
    span = np.nanmax(y, axis=0) - low
    y = (y - low) / np.where(span > 0, span, 1)

    arrays = {c: df.get_column(c).to_numpy() for c in metrics}
    d = _combined_distance(arrays, metrics, weights)
    return _rogi(d, y, rogi_xd, multitask, min_dt)


def replicate_variance(
    y: str | pl.Expr,
    *,
    X: Sequence[str],
    aggregate: Replicates = "pooled",
    normalize: bool = False,
) -> pl.Expr:
    """Variance of `y` among rows with identical `X` (pl.Float64 scalar).

    y: Float column, one row per measurement.
    X: columns that identify a point, e.g. ["fp", "prot_emb"] or id columns.
        Rows are replicates when all of `X` is exactly equal (float arrays
        included). Points with a single row carry no information and are
        left out.
    aggregate: how the per-point variances (ddof=1) combine.
        "pooled" weighs each point by its replicate count - 1 (the residual
        mean square of a one-way ANOVA on the points): the noise of one
        measurement, for training on the unaggregated rows.
        "mean" weighs each point equally, so 2 replicates count as much as
        50. Recommended when replicates are aggregated to one value per point
        before training (e.g. Papyrus), where each point is one sample.
    normalize: divide by the variance of all of `y`, giving the fraction of
        the `y` variance that is replicate noise, comparable across datasets.

    Raises on nulls in `X` or `y`, and when no point has 2 or more rows.

    Reduces to one value, so it is valid per group:

    >>> df.group_by("target_id").agg(
    ...     noise=replicate_variance("pchembl", X=["fp"], aggregate="mean")
    ... )  # doctest: +SKIP
    """
    y = pl.col(y) if isinstance(y, str) else y
    return pl.struct(y.alias(_Y), *X).map_batches(
        partial(
            _struct_to_replicate_variance,
            X=list(X),
            aggregate=aggregate,
            normalize=normalize,
        ),
        return_dtype=pl.Float64,
        returns_scalar=True,
    )


def _struct_to_replicate_variance(
    s: pl.Series, X: list[str], aggregate: Replicates, normalize: bool
) -> float:
    """`map_batches` worker: struct{y, *X} -> validate -> variance per point -> combine."""
    df = s.struct.unnest()
    if n := df.select(pl.any_horizontal(pl.col(X).is_null()).sum()).item():
        raise ValueError(f"{n} rows have a null in `X`")
    if n := df.get_column(_Y).null_count():
        raise ValueError(f"`y` contains {n} nulls")

    points = (
        df.group_by(X)
        .agg(pl.len().alias(_N), pl.col(_Y).var().alias(_VAR))
        .filter(pl.col(_N) > 1)
    )
    if points.is_empty():
        raise ValueError("no two rows share the same `X`")

    if aggregate == "pooled":
        dof = pl.col(_N) - 1
        out = points.select((dof * pl.col(_VAR)).sum() / dof.sum()).item()
    else:
        out = points.get_column(_VAR).mean()
    return out / df.get_column(_Y).var() if normalize else out


def _unit_distance(x: np.ndarray, metric: Metric) -> np.ndarray:
    """Condensed pairwise distance of the rows of `x`, scaled to [0, 1].

    Scaling follows `rogi`: tanimoto is jaccard on bool; cosine / 2; euclidean
    and cityblock / the distance between the per-feature min and max corners.
    """
    import numpy as np
    from scipy.spatial.distance import pdist

    if metric == "tanimoto":
        return pdist(x.astype(bool), "jaccard")
    if metric == "cosine":
        return pdist(x, "cosine") / 2
    corners = np.stack([x.min(0), x.max(0)])
    return pdist(x, metric) / pdist(corners, metric)[0]


def _combined_distance(
    arrays: dict[str, np.ndarray],
    metrics: dict[str, Metric],
    weights: dict[str, float] | None,
) -> np.ndarray:
    """Weighted mean of `_unit_distance` over columns (condensed, in [0, 1]).

    Long format repeats each drug and target over many rows, so each column's
    distance is computed on its unique rows and gathered to row pairs.
    """
    import numpy as np
    from scipy.spatial.distance import squareform

    weights = weights or dict.fromkeys(metrics, 1.0)
    total = 0.0
    for col, metric in metrics.items():
        x = arrays[col]
        unique, inverse = np.unique(x, axis=0, return_inverse=True)
        if len(unique) == len(x):
            d = _unit_distance(x, metric)
        else:
            # ponytail: gathers through an n x n square; index the condensed form directly if memory matters
            square = squareform(_unit_distance(unique, metric))
            inverse = inverse.ravel()
            d = squareform(square[np.ix_(inverse, inverse)], checks=False)
        total = total + weights[col] * d
    return total / sum(weights[col] for col in metrics)


def _dispersion(y: np.ndarray, clusters: np.ndarray, multitask: Multitask) -> float:
    """2 * std of the cluster means (weighted by cluster size), combined over tasks.

    y is (n, n_tasks) in [0, 1] with NaN for missing; clusters holds labels
    1..k. n_tasks == 1 gives `rogi`'s normalised standard deviation.
    """
    import numpy as np

    valid = ~np.isnan(y)
    shape = (clusters.max(), y.shape[1])
    counts, sums = np.zeros(shape), np.zeros(shape)
    np.add.at(counts, clusters - 1, valid)
    np.add.at(sums, clusters - 1, np.where(valid, y, 0))

    means = sums / np.maximum(counts, 1)
    n = counts.sum(0)
    var = (counts * (means - sums.sum(0) / n) ** 2).sum(0) / n
    if multitask == "mean_std":
        return float(2 * np.sqrt(var).mean())
    return float(2 * np.sqrt(var.mean()))


def _rogi(
    d: np.ndarray, y: np.ndarray, rogi_xd: bool, multitask: Multitask, min_dt: float
) -> float:
    """Complete linkage on `d` -> `_dispersion` per threshold -> dispersion[0] - AUC."""
    import numpy as np
    from scipy.cluster.hierarchy import complete, fcluster
    from scipy.integrate import trapezoid

    n = len(y)
    z = complete(d)
    thresholds, previous = [], -1.0
    for t in z[:, 2]:
        if t >= previous + min_dt:
            thresholds.append(t)
            previous = t

    # endpoints: every row its own cluster, then all rows in one
    clusterings = [
        np.arange(1, n + 1),
        *(fcluster(z, t, "distance") for t in thresholds),
        np.ones(n, dtype=int),
    ]
    dispersion = np.array([_dispersion(y, c, multitask) for c in clusterings])
    if rogi_xd:
        x = 1 - np.log([c.max() for c in clusterings]) / np.log(n)
    else:
        x = np.array([0.0, *thresholds, 1.0])
    return float(dispersion[0] - trapezoid(dispersion, x))


__all__ = ["replicate_variance", "roughness"]
