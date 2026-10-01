from collections.abc import Sequence
from functools import partial
from typing import Literal

import polars as pl

SplitMethod = Literal["sklearn", "tricarico"]
_GROUPS = "groups"


def kfold(
    n_splits: int = 5,
    *,
    stratify: str | pl.Expr | Sequence[str | pl.Expr],
    groups: str | pl.Expr | None = None,
    n_bins: int = 5,
    method: SplitMethod = "sklearn",
    seed: int = 0,
    **kwargs,
) -> pl.Expr:
    """Fold number 0..n_splits-1 per row (pl.UInt8), from several columns.

    Wraps `nanoom.split` (extra: ``fairfetched[split]``).

    stratify: columns balanced across folds (target id, y, ...). Float columns
        are quantile-binned into `n_bins`; "tricarico" treats a float column
        holding only whole numbers as classes instead.
    groups: cluster / scaffold / compound id; a group never spans two folds.
        None splits rows independently.
    method: "sklearn" is StratifiedGroupKFold (one stratify column, uses
        `seed`); "tricarico" is the LP balancer (several stratify columns).
    kwargs: passed to `nanoom.split`, e.g. `relative_gap`, `time_limit_seconds`
        for "tricarico", which returns the solver's best solution at the limit.

    Raises on null `groups`, and on rows nanoom cannot stratify: any null
    `stratify` value for "sklearn" (it would be binned as a number), a group
    with no non-null `stratify` value for "tricarico" (it would get a null fold).

    Use for leakage-free cross-validation: `groups` keeps near-identical
    compounds together, `stratify` keeps the label distribution equal per fold.
    Call at frame level (`with_columns`, eager or lazy), not inside `.over()` or
    `group_by().agg()`, which would split each group separately.

    >>> df = df.with_columns(
    ...     split=kfold(5, stratify="pchembl_value_mean", groups="cluster")
    ... )  # doctest: +SKIP
    >>> test = df.filter(split=0)  # doctest: +SKIP
    >>> train = df.remove(split=0)  # doctest: +SKIP
    """
    stratify = [stratify] if isinstance(stratify, (str, pl.Expr)) else list(stratify)
    fields = [_as_expr(e).alias(f"y{i}") for i, e in enumerate(stratify)]
    if groups is not None:
        fields.append(_as_expr(groups).alias(_GROUPS))
    if method == "sklearn":
        kwargs["random_state"] = seed
    return pl.struct(fields).map_batches(
        partial(
            _struct_to_folds,
            n_splits=n_splits,
            method=method,
            n_bins_for_regression=n_bins,
            **kwargs,
        ),
        return_dtype=pl.UInt8,
    )


def _as_expr(e: str | pl.Expr) -> pl.Expr:
    return pl.col(e) if isinstance(e, str) else e


def _struct_to_folds(s: pl.Series, method: SplitMethod, **kwargs) -> pl.Series:
    """`map_batches` worker: struct{y0.., groups} -> fold number per row."""
    from nanoom import split

    df = s.struct.unnest()
    grouped = _GROUPS in df.columns
    y_cols = [c for c in df.columns if c != _GROUPS]

    if grouped and df.get_column(_GROUPS).null_count():
        raise ValueError("`groups` contains nulls")
    if method == "sklearn":
        unstratifiable = pl.any_horizontal(pl.col(y_cols).is_null())
    else:
        unstratifiable = pl.all_horizontal(pl.col(y_cols).is_null())
        if grouped:
            unstratifiable = unstratifiable.all().over(_GROUPS)
    if n := df.select(unstratifiable.sum()).item():
        raise ValueError(f"{n} rows have no usable `stratify` value for {method!r}")

    out = split(
        df,
        y_cols=y_cols,
        cluster_col=_GROUPS if grouped else None,
        method=method,
        **kwargs,
    )
    return out.get_column("split").cast(pl.UInt8)


__all__ = ["kfold"]
