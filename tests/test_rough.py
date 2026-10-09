import numpy as np
import polars as pl
import pytest

pytest.importorskip("scipy")

from scipy.spatial.distance import pdist

from fairfetched.prep.rough_expr import (
    Metric,
    _combined_distance,
    _dispersion,
    _rogi,
    _unit_distance,
    replicate_variance,
    roughness,
)

# Graff et al. 2023 (arXiv 2305.08238), Figure 2C/D legends: d -> (ROGI, ROGI-XD)
FIGURE_2 = {
    4: (0.42, 0.42),
    16: (0.26, 0.41),
    64: (0.15, 0.41),
    256: (0.09, 0.42),
    1024: (0.05, 0.43),
}
X_COLS: dict[str, Metric] = {"fp": "tanimoto", "emb": "cosine"}


@pytest.fixture(scope="module")
def figure_2_inputs() -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """d -> (distances, y), sampled as in coleygroup/rogi-xd notebooks/toy_examples.ipynb."""
    rng = np.random.default_rng(24)
    inputs = {}
    for d in FIGURE_2:
        x = rng.uniform(size=(1000, d))
        y = rng.uniform(size=1000)
        dist = pdist(x)
        inputs[d] = (dist / dist.max(), y[:, None])
    return inputs


@pytest.fixture
def df() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    n = 60
    return pl.DataFrame(
        {
            "target": rng.choice(["P1", "P2"], n),
            "y": rng.normal(size=n),
            "ys": pl.Series(rng.normal(size=(n, 3)), dtype=pl.Array(pl.Float64, 3)),
            "fp": pl.Series(rng.integers(0, 2, (n, 32)), dtype=pl.Array(pl.UInt8, 32)),
            "emb": pl.Series(rng.normal(size=(n, 8)), dtype=pl.Array(pl.Float64, 8)),
        }
    )


class TestRogi:
    @pytest.mark.parametrize("d", FIGURE_2)
    def test_original_rogi_matches_figure_2c(self, figure_2_inputs, d):
        dist, y = figure_2_inputs[d]
        assert _rogi(dist, y, False, "mean_std", 0.01) == pytest.approx(
            FIGURE_2[d][0], abs=0.005
        )

    @pytest.mark.parametrize("d", FIGURE_2)
    def test_rogi_xd_matches_figure_2d(self, figure_2_inputs, d):
        dist, y = figure_2_inputs[d]
        assert _rogi(dist, y, True, "mean_std", 0.01) == pytest.approx(
            FIGURE_2[d][1], abs=0.005
        )


class TestDispersion:
    """Expected values computed by hand."""

    def test_nan_is_left_out_of_its_task_only(self):
        # per task: cluster means 0.5 (n=2) and 1 (n=1) around 2/3 -> variance 1/18
        y = np.array([[0, np.nan], [1, 1], [1, 0], [np.nan, 1]])
        clusters = np.array([1, 1, 2, 2])
        assert _dispersion(y, clusters, "mean_std") == pytest.approx(
            2 * np.sqrt(1 / 18)
        )

    @pytest.mark.parametrize(
        ("multitask", "expected"),
        [("mean_std", 0.5 + np.sqrt(3) / 4), ("rms_std", 2 * np.sqrt(7 / 32))],
    )
    def test_tasks_with_variance_one_quarter_and_three_sixteenths(
        self, multitask, expected
    ):
        y = np.array([[0, 0], [1, 0], [0, 0], [1, 1]], dtype=float)
        assert _dispersion(y, np.arange(1, 5), multitask) == pytest.approx(expected)


class TestUnitDistance:
    """Expected values computed by hand."""

    def test_tanimoto(self):
        x = np.array([[1, 1, 0], [1, 0, 1]], dtype=np.uint8)
        assert _unit_distance(x, "tanimoto") == pytest.approx([2 / 3])

    def test_cosine_spans_zero_to_one(self):
        x = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        assert _unit_distance(x, "cosine") == pytest.approx([0.5, 1.0, 0.5])

    @pytest.mark.parametrize(
        ("metric", "expected"),
        [("euclidean", [1.0, 0.6, 0.8]), ("cityblock", [1.0, 3 / 7, 4 / 7])],
    )
    def test_unbounded_metric_is_divided_by_corner_distance(self, metric, expected):
        x = np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 0.0]])
        assert _unit_distance(x, metric) == pytest.approx(expected)


class TestCombinedDistance:
    def test_repeated_rows_match_plain_pdist(self):
        """Oracle: scipy pdist on the full, undeduplicated rows."""
        rng = np.random.default_rng(1)
        fp = rng.integers(0, 2, (5, 16))[rng.integers(0, 5, 30)]
        emb = rng.normal(size=(4, 6))[rng.integers(0, 4, 30)]
        expected = (
            3 * pdist(fp.astype(bool), "jaccard") + pdist(emb, "cosine") / 2
        ) / 4

        out = _combined_distance(
            {"fp": fp, "emb": emb},
            {"fp": "tanimoto", "emb": "cosine"},
            {"fp": 3.0, "emb": 1.0},
        )

        assert out == pytest.approx(expected)


class TestRoughness:
    def test_reduces_to_one_float(self, df):
        out = df.select(r=roughness("y", X=X_COLS))
        assert out.schema == {"r": pl.Float64}
        assert out.height == 1
        assert 0 <= out.item() <= 1

    def test_per_group_matches_filtered_frames(self, df):
        out = df.group_by("target").agg(r=roughness("y", X=X_COLS))
        for target, r in out.iter_rows():
            alone = df.filter(target=target).select(roughness("y", X=X_COLS)).item()
            assert r == pytest.approx(alone)

    @pytest.mark.parametrize("rogi_xd", [True, False])
    def test_dense_mean_std_is_mean_of_per_task_indices(self, df, rogi_xd):
        per_task = [
            df.select(
                roughness(pl.col("ys").arr.get(i), X=X_COLS, rogi_xd=rogi_xd)
            ).item()
            for i in range(3)
        ]
        out = df.select(roughness("ys", X=X_COLS, rogi_xd=rogi_xd)).item()
        assert out == pytest.approx(np.mean(per_task))

    def test_raises_on_null_x(self, df):
        nulled = df.with_columns(
            pl.when(pl.int_range(pl.len()) > 0).then("fp").alias("fp")
        )
        with pytest.raises(ValueError, match="1 rows have a null in `X`"):
            nulled.select(roughness("y", X=X_COLS))

    def test_raises_on_null_scalar_y(self, df):
        nulled = df.with_columns(
            pl.when(pl.int_range(pl.len()) > 1).then("y").alias("y")
        )
        with pytest.raises(ValueError, match="`y` contains 2 nulls"):
            nulled.select(roughness("y", X=X_COLS))

    def test_raises_on_task_without_values(self, df):
        nulled = df.with_columns(
            ys=pl.concat_arr(pl.col("y"), pl.lit(None, dtype=pl.Float64))
        )
        with pytest.raises(ValueError, match=r"tasks \[1\] have no non-null value"):
            nulled.select(roughness("ys", X=X_COLS))

    def test_raises_on_single_row(self, df):
        with pytest.raises(ValueError, match="at least 2 rows, got 1"):
            df.head(1).select(roughness("y", X=X_COLS))

    def test_raises_above_max_rows_and_suggests_sample(self, df):
        with pytest.raises(ValueError, match=r"df\.sample\(50"):
            df.select(roughness("y", X=X_COLS, max_rows=50))


class TestReplicateVariance:
    """Expected values computed by hand.

    Points A (y = 1, 2, 4; variance 7/3), B (y = 5, 7; variance 2) and a
    single-row C (y = 9); the variance of all six y is 136/15.
    """

    X = ("fp", "emb")

    @pytest.fixture
    def replicates(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "target": ["P1"] * 6,
                "fp": pl.Series(
                    [[1, 0]] * 3 + [[0, 1]] * 2 + [[1, 1]], dtype=pl.Array(pl.UInt8, 2)
                ),
                "emb": pl.Series(
                    [[0.1, 0.2]] * 3 + [[0.3, 0.4]] * 2 + [[0.5, 0.6]],
                    dtype=pl.Array(pl.Float64, 2),
                ),
                "y": [1.0, 2.0, 4.0, 5.0, 7.0, 9.0],
            }
        )

    @pytest.mark.parametrize(
        ("aggregate", "expected"),
        [("pooled", (2 * 7 / 3 + 1 * 2) / 3), ("mean", (7 / 3 + 2) / 2)],
    )
    def test_aggregate(self, replicates, aggregate, expected):
        out = replicates.select(
            r=replicate_variance("y", X=self.X, aggregate=aggregate)
        )
        assert out.schema == {"r": pl.Float64}
        assert out.item() == pytest.approx(expected)

    def test_normalize_divides_by_total_variance(self, replicates):
        out = replicates.select(replicate_variance("y", X=self.X, normalize=True))
        assert out.item() == pytest.approx(25 / 102)

    def test_points_differing_in_one_column_are_not_replicates(self, replicates):
        # with fp constant, fp alone makes all six rows one point
        same_fp = replicates.with_columns(fp=pl.col("fp").first())
        out = same_fp.select(
            both=replicate_variance("y", X=self.X),
            fp=replicate_variance("y", X=["fp"]),
        )
        assert out.row(0) == pytest.approx((20 / 9, 136 / 15))

    def test_valid_per_group(self, replicates):
        # P2 is the same frame with y doubled -> variance x 4
        both = pl.concat(
            [
                replicates,
                replicates.with_columns(
                    pl.lit("P2").alias("target"), y=pl.col("y") * 2
                ),
            ]
        )
        out = both.group_by("target").agg(r=replicate_variance("y", X=self.X))
        assert dict(out.iter_rows()) == pytest.approx({"P1": 20 / 9, "P2": 80 / 9})

    def test_raises_without_replicates(self, replicates):
        with pytest.raises(ValueError, match="no two rows share the same `X`"):
            replicates.unique(self.X).select(replicate_variance("y", X=self.X))

    def test_raises_on_null_y(self, replicates):
        nulled = replicates.with_columns(
            pl.when(pl.int_range(pl.len()) > 0).then("y").alias("y")
        )
        with pytest.raises(ValueError, match="`y` contains 1 nulls"):
            nulled.select(replicate_variance("y", X=self.X))

    def test_raises_on_null_x(self, replicates):
        nulled = replicates.with_columns(
            pl.when(pl.int_range(pl.len()) > 0).then("fp").alias("fp")
        )
        with pytest.raises(ValueError, match="1 rows have a null in `X`"):
            nulled.select(replicate_variance("y", X=self.X))
