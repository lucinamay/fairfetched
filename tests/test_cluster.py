import numpy as np
import polars as pl
import pytest

pytest.importorskip("nanoom")

from fairfetched.prep.cluster_expr import cluster
from fairfetched.prep.split_expr import kfold


@pytest.fixture
def df() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    fp = rng.integers(0, 2, (60, 16)).astype(np.uint8)
    return pl.DataFrame(
        {"fp": fp, "active": rng.integers(0, 2, 60)},
        schema_overrides={"fp": pl.Array(pl.UInt8, 16)},
    )


class TestCluster:
    def test_kwargs_reach_nanoom_and_row_order_is_kept(self, df):
        """Oracle: sklearn KMeans on the same matrix; `random_state` only
        reaches it through `**kwargs`."""
        from sklearn.cluster import KMeans

        expected = KMeans(n_clusters=4, random_state=1).fit_predict(df["fp"].to_numpy())
        out = df.select(c=cluster("fp", method="kmeans", n_clusters=4, random_state=1))[
            "c"
        ]

        assert out.dtype == pl.Int32
        assert out.to_list() == expected.tolist()

    def test_accepts_expression_and_lazy_frame(self, df):
        out = (
            df.lazy()
            .with_columns(c=cluster(pl.col("fp"), method="random", n_clusters=3))
            .collect()
        )
        assert out["c"].n_unique() == 3

    def test_bit_methods_reject_non_binary_descriptors(self, df):
        scaled = df.with_columns(fp=pl.col("fp").cast(pl.Array(pl.Float32, 16)) * 0.5)
        with pytest.raises(Exception, match="binary"):
            scaled.select(cluster("fp", method="maxmin"))

    @pytest.mark.parametrize(
        "bad",
        [
            [[0] * 4, None],
            [[0, None, 1, 0], [1, 0, 0, 1]],
        ],
        ids=["null_array", "null_element"],
    )
    def test_raises_on_nulls(self, bad):
        d = pl.DataFrame({"fp": bad}, schema={"fp": pl.Array(pl.Int8, 4)})
        with pytest.raises(Exception, match="nulls"):
            d.select(cluster("fp", method="random", n_clusters=2))

    def test_output_is_valid_kfold_groups(self, df):
        out = df.with_columns(
            c=cluster("fp", method="random", n_clusters=12)
        ).with_columns(split=kfold(3, stratify="active", groups="c"))
        assert out.group_by("c").agg(pl.col("split").n_unique())["split"].max() == 1
