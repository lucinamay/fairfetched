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


SMILES = [f"c1ccccc1{'C' * i}" for i in range(6)] + [
    f"C1CCNCC1{'O' * i}" for i in range(6)
]


class TestGroups:
    def test_matches_chemsplit_on_distinct_smiles_and_maps_back_to_rows(self):
        """Oracle: chemsplit `compute_groups` on the distinct SMILES list; rows
        repeat each compound, so the column repeats the label."""
        pytest.importorskip("chemsplit")
        from chemsplit import get_splitter

        from fairfetched.prep.cluster_expr import groups

        expected = get_splitter("murcko_scaffold").compute_groups(SMILES).tolist()
        df = pl.DataFrame({"smiles": SMILES * 3})
        out = df.select(g=groups("smiles", splitter="murcko_scaffold"))["g"]
        assert out.dtype == pl.Int32
        assert out.to_list() == expected * 3

    def test_features_feed_kfold(self, df):
        """pl.Array in; the ids are a valid `groups` for nanoom's `kfold`."""
        pytest.importorskip("chemsplit")
        from fairfetched.prep.cluster_expr import groups

        out = df.with_columns(
            g=groups("fp", splitter="butina", cutoff=0.6)
        ).with_columns(split=kfold(3, stratify="active", groups="g"))
        assert out.group_by("g").agg(pl.col("split").n_unique())["split"].max() == 1

    def test_null_smiles_raise(self):
        pytest.importorskip("chemsplit")
        from fairfetched.prep.cluster_expr import groups

        df = pl.DataFrame({"smiles": SMILES[:3] + [None]})
        with pytest.raises(ValueError, match="null"):
            df.select(groups("smiles", splitter="murcko_scaffold"))
