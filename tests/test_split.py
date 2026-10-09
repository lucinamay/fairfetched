import numpy as np
import polars as pl
import pytest

pytest.importorskip("nanoom")

from sklearn.model_selection import StratifiedGroupKFold

from fairfetched.prep.split_expr import kfold


@pytest.fixture
def df() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    n = 300
    return pl.DataFrame(
        {
            "cluster": rng.integers(0, 40, n),
            "active": rng.integers(0, 2, n),
            "target": rng.choice(["P1", "P2", "P3"], n),
        }
    )


class TestKfold:
    def test_matches_sklearn_stratified_group_kfold(self, df):
        """Oracle: sklearn called directly, on an integer label so no binning is involved."""
        expected = np.full(df.height, -1)
        splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=3)
        folds = splitter.split(
            np.zeros((df.height, 1)), df["active"].to_numpy(), df["cluster"].to_numpy()
        )
        for k, (_, test_idx) in enumerate(folds):
            expected[test_idx] = k

        out = df.select(s=kfold(4, stratify="active", groups="cluster", seed=3))["s"]

        assert out.dtype == pl.UInt8
        assert out.to_list() == expected.tolist()

    def test_tricarico_keeps_groups_whole_over_two_stratify_columns(self, df):
        out = df.with_columns(
            split=kfold(
                3,
                stratify=["active", pl.col("target")],
                groups="cluster",
                method="tricarico",
                relative_gap=0.05,
                time_limit_seconds=5,
            )
        )
        assert sorted(out["split"].unique().to_list()) == [0, 1, 2]
        folds_per_cluster = out.group_by("cluster").agg(pl.col("split").n_unique())
        assert folds_per_cluster["split"].max() == 1

    def test_groups_none_stratifies_rows(self, df):
        """StratifiedKFold property: per class, fold sizes differ by at most one row."""
        out = df.with_columns(split=kfold(5, stratify="active"))
        counts = out.group_by("active", "split").len()
        spread = counts.group_by("active").agg(
            d=pl.col("len").max() - pl.col("len").min()
        )
        assert counts.height == 10
        assert spread["d"].max() <= 1


class TestKfoldNulls:
    """Inputs for which nanoom returns folds without an error: a null fold, or
    a null y binned as a number."""

    @pytest.mark.parametrize("method", ["sklearn", "tricarico"])
    def test_null_group_raises(self, df, method):
        df = df.with_columns(cluster=pl.when(pl.col("cluster") > 0).then("cluster"))
        with pytest.raises(ValueError, match="`groups` contains nulls"):
            df.select(kfold(3, stratify="active", groups="cluster", method=method))

    def test_sklearn_null_stratify_raises(self, df):
        df = df.with_columns(active=pl.when(pl.col("cluster") > 0).then("active"))
        with pytest.raises(ValueError, match="no usable `stratify`"):
            df.select(kfold(3, stratify="active", groups="cluster"))

    def test_tricarico_group_without_stratify_value_raises(self, df):
        df = df.with_columns(
            pl.when(pl.col("cluster") > 0).then(pl.col("active", "target"))
        )
        with pytest.raises(ValueError, match="no usable `stratify`"):
            df.select(
                kfold(
                    3,
                    stratify=["active", "target"],
                    groups="cluster",
                    method="tricarico",
                )
            )

    def test_tricarico_accepts_partly_null_stratify(self, df):
        """Sparse multi-task y: cluster 0 has no `active`, and row 0 has no
        stratify value at all, but each cluster keeps one non-null value."""
        row = pl.int_range(pl.len())
        df = df.with_columns(
            active=pl.when((pl.col("cluster") > 0) & (row > 0)).then("active"),
            target=pl.when(row > 0).then("target"),
        )
        assert df.filter(cluster=df["cluster"][0])["target"].count() > 0
        out = df.select(
            s=kfold(
                3,
                stratify=["active", "target"],
                groups="cluster",
                method="tricarico",
                relative_gap=0.05,
                time_limit_seconds=5,
            )
        )["s"]
        assert out.null_count() == 0
