from collections.abc import Callable, Sequence
from functools import partial
from typing import Literal

import polars as pl

SplitMethod = Literal["sklearn", "tricarico"]
Partition = pl.Enum(["train", "valid", "test"])
_GROUPS = "groups"
_X = "X"
_OUT = "__out__"
Records = str | pl.Expr | tuple[str | pl.Expr, str | pl.Expr]


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


def holdout(
    X: Records,
    *,
    splitter: str,
    y: str | pl.Expr | None = None,
    dates: str | pl.Expr | None = None,
    groups: str | pl.Expr | None = None,
    seed: int = 0,
    n_jobs: int = 1,
    **params,
) -> pl.Expr:
    """Partition per row, "train" / "valid" / "test" (`Partition` enum); null for
    records the splitter discards.

    Wraps a single-split `chemsplit` splitter (extra: ``fairfetched[split]``);
    `splitters()` lists them, the chemsplit README maps deployment scenarios to ids.

    X: the records: a SMILES column, a pl.Array descriptor column, or a
        (compound, target) pair of columns for the cold-start splitters.
        Structures must already be standardized. Splitting runs on the distinct
        `X`, so `train_size` etc. count compounds (pairs), not rows, and every
        row of one compound lands in the same partition.
    y: labels, for splitters with `requires_labels`.
    dates: pl.Date column, for "temporal" and other `requires_dates` splitters.
    groups: precomputed group id, e.g. from `cluster()` or `groups()`, for
        splitters that consume groups ("predefined", "group_k_fold" with
        `n_splits=1` is not a holdout).
    params: splitter parameters, e.g. `train_size`, `test_size`, `valid_size`,
        `cutoff`.

    Raises on null `X`, when `y` / `dates` / `groups` vary within one `X`
    (aggregate per compound first), and when the splitter yields more than one
    split ("monte_carlo", "repeated", "k_fold": use `kfold` for folds).

    Call at frame level, as `kfold`.

    >>> df = df.with_columns(
    ...     split=holdout("canonical_smiles", splitter="temporal", dates="date",
    ...                   train_size=0.8, test_size=0.2)
    ... )  # doctest: +SKIP
    """
    return _pack(X=_as_records(X), y=y, dates=dates, groups=groups).map_batches(
        partial(
            _struct_to_partition,
            splitter=splitter,
            seed=seed,
            n_jobs=n_jobs,
            params=params,
        ),
        return_dtype=Partition,
    )


def splitters() -> pl.DataFrame:
    """The chemsplit registry: `id`, `family`, `strictness`, `group_forming`,
    `requires_labels`, `requires_dates`, ... one row per splitter.

    `group_forming` ids go to `groups()`; the rest to `holdout()`. The chemsplit
    README's "24 deployment scenarios" says which id answers which question.
    """
    from chemsplit import list_splitters

    return pl.from_pandas(list_splitters())


def _as_records(X: Records) -> pl.Expr:
    if isinstance(X, tuple):
        compound, target = X
        return pl.struct(
            _as_expr(compound).alias("compound"), _as_expr(target).alias("target")
        )
    return _as_expr(X)


def _pack(**fields: str | pl.Expr | None) -> pl.Expr:
    """`pl.struct` of the non-None fields, each aliased to its keyword."""
    return pl.struct(
        *(_as_expr(e).alias(name) for name, e in fields.items() if e is not None)
    )


def _records(s: pl.Series) -> list | object:
    """chemsplit `X` from a distinct `X` series: SMILES list, 2-D array, or
    (compound, target) tuples."""
    if s.dtype == pl.String:
        return s.to_list()
    if isinstance(s.dtype, pl.Array):
        return s.to_numpy()
    if isinstance(s.dtype, pl.Struct):
        return s.struct.unnest().rows()
    raise TypeError(
        f"X must be pl.String (SMILES), pl.Array (features) or a (compound, target) "
        f"pair, got {s.dtype}"
    )


def _unique_apply(s: pl.Series, fn: Callable[[pl.DataFrame], pl.Series]) -> pl.Series:
    """`map_batches` helper: struct{X, ...} -> `fn` on the distinct rows, mapped
    back onto every row.

    Raises on null `X` and when another field varies within one `X`.
    """
    df = s.struct.unnest()
    if df.get_column(_X).null_count():
        raise ValueError("`X` contains nulls")
    distinct = df.unique(maintain_order=True)
    if distinct.select(_X).unique().height != distinct.height:
        raise ValueError(
            "`y` / `dates` / `groups` vary within one X; aggregate per X first"
        )
    out = fn(distinct)
    return df.join(
        distinct.select(_X).with_columns(out.alias(_OUT)),
        on=_X,
        how="left",
        maintain_order="left",
    ).get_column(_OUT)


def _struct_to_partition(
    s: pl.Series, splitter: str, seed: int, n_jobs: int, params: dict
) -> pl.Series:
    """`map_batches` worker: struct{X, y?, dates?, groups?} -> partition per row."""
    from chemsplit import get_splitter

    sp = get_splitter(splitter, random_state=seed, n_jobs=n_jobs, **params)

    def fn(distinct: pl.DataFrame) -> pl.Series:
        kw = {c: distinct.get_column(c).to_numpy() for c in distinct.columns if c != _X}
        results = sp.split_result(
            _records(distinct.get_column(_X)),
            kw.pop("y", None),
            kw.pop(_GROUPS, None),
            standardize=False,
            **kw,
        )
        if len(results) != 1:
            raise ValueError(
                f"{splitter!r} yields {len(results)} splits; `holdout` takes one"
            )
        (r,) = results
        out: list[str | None] = [None] * r.n_records
        for name, idx in (("train", r.train), ("valid", r.valid), ("test", r.test)):
            for i in idx:
                out[i] = name
        return pl.Series(out, dtype=Partition)

    return _unique_apply(s, fn)


__all__ = ["Partition", "holdout", "kfold", "splitters"]
