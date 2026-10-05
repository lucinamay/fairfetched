"""ToxCast source: offline checks on member matching, the mc5-6 extraction,
the joins and the file ids. No download."""

import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

import polars as pl
import pytest

from fairfetched.get import toxcast
from fairfetched.utils import raw
from fairfetched.utils.ensure import ensure_url

# Clowder file ids/sizes as listed by the Clowder API for the v4.3 dataset
_IDS = {
    "assay_annotations": "68af6bd3e4b02565fc7c3aa8",
    "assay_target_mappings": "68af6bd3e4b02565fc7c3aa0",
    "cytotox": "68af6bd3e4b02565fc7c3aa4",
    "analytical_qc": "68af6bd3e4b02565fc7c3ab8",
    "summary_zip": "68af6b70e4b02565fc7c3a98",
}


def _zip(path: Path, members: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, body in members.items():
            zf.writestr(name, body)
    return path


class TestSourceUrls:
    def test_ids(self):
        assert toxcast.source_urls("4.3") == {
            k: f"https://clowder.edap-cluster.com/files/{v}/blob"
            for k, v in _IDS.items()
        }

    def test_versions(self):
        assert toxcast.available_versions() == ("4.3",)
        assert toxcast.latest() == "4.3"


class TestZipMember:
    def test_matches_by_substring_regardless_of_stamp(self, tmp_path):
        zp = _zip(
            tmp_path / "a.zip",
            {
                "mc5-6_winning_model_fits-c2021_invitrodbv4_3_AUG2024.csv": "x\n1\n",
                "mc5-6_winning_model_fits-c2021_invitrodbv4_3_AUG2024.Rdata": "",
                "mc4_all_model_fits_invitrodbv4_3_AUG2024.csv": "",
            },
        )
        out = raw._extract_zip_member(
            zp, ("mc5-6", "winning_model_fits", ".csv"), tmp_path
        )
        assert out.name.endswith("AUG2024.csv") and out.read_text() == "x\n1\n"

    def test_zero_matches_raises(self, tmp_path):
        zp = _zip(tmp_path / "a.zip", {"other.csv": ""})
        with pytest.raises(ValueError):
            raw._extract_zip_member(zp, ("mc5-6",), tmp_path)

    def test_ambiguous_raises(self, tmp_path):
        zp = _zip(tmp_path / "a.zip", {"mc5-6_a.csv": "", "mc5-6_b.csv": ""})
        with pytest.raises(ValueError):
            raw._extract_zip_member(zp, ("mc5-6",), tmp_path)


def _mc56(tmp_path: Path, body: str) -> pl.DataFrame:
    zp = _zip(tmp_path / "s.zip", {"mc5-6_winning_model_fits_x.csv": body})
    out = toxcast.ensure_parquet_tables({"summary_zip": zp}, tmp_path / "pq")
    return pl.read_parquet(out["mc5_mc6"])


class TestMc56Extraction:
    def test_na_literal_is_null_and_columns_stay_numeric(self, tmp_path):
        df = _mc56(tmp_path, "aeid,spid,ac50\n1,a,0.5\n1,b,NA\n")
        assert df.schema["ac50"] == pl.Float64
        assert df["ac50"].to_list() == [0.5, None]

    def test_late_scientific_notation_does_not_break_inference(self, tmp_path):
        # R writes round ids as 1.8e+07; only the last row, past any short prefix
        body = "aeid,m4id\n" + "1,14190200\n" * 100_001 + "1,1.8e+07\n"
        assert _mc56(tmp_path, body)["m4id"].max() == 18_000_000


class TestClean:
    @pytest.fixture
    def cleaned(self):
        lf = pl.LazyFrame(
            {
                "aeid": [1, 2],
                "endpoint.x": ["e1", "e2"],
                "endpoint.y": ["e1", "e2"],
                "flag.length": [1, None],
                "tissue": ["NA", "liver"],
                "average_mass": ["193.07", "360.99"],
                "t0": ["A", "D"],
                "m4id": [1.8e7, 14190200.0],
            }
        )
        return toxcast._clean(lf).collect()

    def test_dot_suffix_duplicate_collapses_to_one_column(self, cleaned):
        assert [c for c in cleaned.columns if c.startswith("endpoint")] == ["endpoint"]

    def test_other_dots_become_underscores(self, cleaned):
        assert "flag_length" in cleaned.columns
        assert not [c for c in cleaned.columns if "." in c]

    def test_na_string_is_null(self, cleaned):
        assert cleaned["tissue"].to_list() == [None, "liver"]

    def test_qc_numeric_text_becomes_float(self, cleaned):
        assert cleaned.schema["average_mass"] == pl.Float64
        assert cleaned.schema["t0"] == pl.String  # letter grades stay text

    def test_scientific_notation_model_id_becomes_int(self, cleaned):
        assert cleaned.schema["m4id"] == pl.Int64
        assert cleaned["m4id"].to_list() == [18_000_000, 14_190_200]


class TestBuildViews:
    @pytest.fixture
    def views(self, tmp_path):
        tables = {
            # aeid 3 has no annotation row; a left join must keep it
            # the last row is a blank well (no chemical)
            "mc5_mc6": pl.DataFrame(
                {
                    "aeid": [1, 1, 2, 3],
                    "spid": list("abcd"),
                    "chid": [10, 10, 11, None],
                    "casn": ["c10", "c10", "c11", None],
                }
            ),
            "assay_annotations": pl.DataFrame(
                {c: [c, c] if c != "aeid" else [1, 2] for c in toxcast._ANNOTATION_COLS}
            ),
            "cytotox": pl.DataFrame(
                {
                    "chid": [10, 11],
                    "casn": ["c10", "c11"],
                    "chnm": ["n10", "n11"],
                    "dsstox_substance_id": ["d10", "d11"],
                    "cytotox_lower_bound_um": [1.0, 2.0],
                }
            ),
            "assay_target_mappings": pl.DataFrame(
                {"aeid": [1, 1], "target_type": ["aop", "key_event"]}
            ),
        }
        paths = {}
        for name, df in tables.items():
            paths[name] = tmp_path / f"{name}.parquet"
            df.write_parquet(paths[name])
        return toxcast.build_views(paths)

    def test_bioactivity_drops_rows_without_chemical(self, views):
        bio = views["bioactivity"].collect()
        assert bio["spid"].to_list() == ["a", "b", "c"]

    def test_full_adds_only_cytotox_burst_columns_one_row_per_measurement(self, views):
        full = views["full"].collect()
        assert full.height == views["bioactivity"].collect().height
        added = set(full.columns) - set(views["bioactivity"].collect_schema().names())
        assert added == {"cytotox_lower_bound_um"}
        assert full["cytotox_lower_bound_um"].to_list() == [1.0, 1.0, 2.0]

    def test_long_format_targets_not_joined_into_bioactivity(self, views):
        assert "target_type" not in views["bioactivity"].collect_schema().names()
        assert views["targets"].collect().height == 2

    def test_compounds_is_cytotox(self, views):
        assert views["compounds"].collect()["chid"].to_list() == [10, 11]


class TestEnsureUrlAtomic:
    def test_interrupted_download_leaves_no_final_file(self, tmp_path):
        target = tmp_path / "big.zip"
        resp = Mock()
        resp.read = Mock(side_effect=[b"abc", OSError("connection reset")])
        resp.getheader = Mock(return_value="100")
        resp.__enter__ = Mock(return_value=resp)
        resp.__exit__ = Mock(return_value=None)
        with (
            patch("urllib.request.urlopen", return_value=resp),
            pytest.raises(OSError),
        ):
            ensure_url("http://example.com/big.zip", target)
        assert not target.exists()


_PARQUET_DIR = toxcast.TOXCAST_DIR / "4.3" / "parquet"
_FULL_TABLES = (
    "assay_annotations",
    "assay_target_mappings",
    "cytotox",
    "analytical_qc",
    "mc5_mc6",
)


@pytest.mark.skipif(
    not all((_PARQUET_DIR / f"{t}.parquet").exists() for t in _FULL_TABLES),
    reason="needs the real v4.3 parquet tables (7.5 GB download; run "
    "`python -m fairfetched.get.toxcast`, then ensure_parquet_tables)",
)
class TestFullRelease:
    """Invariants of the real invitrodb v4.3 release. The counts were taken by
    profiling the raw mc5-6 CSV with ``infer_schema=False``, not from this
    module's output; the key and join properties hold for any release."""

    @pytest.fixture
    def lfs(self):
        paths = {t: _PARQUET_DIR / f"{t}.parquet" for t in _FULL_TABLES}
        return toxcast.cleanly_scan_parquet_tables(paths), toxcast.build_views(paths)

    def test_mc5_mc6_shape(self, lfs):
        mc = lfs[0]["mc5_mc6"]
        assert mc.select(pl.len()).collect().item() == 3_527_285
        assert len(mc.collect_schema()) == 65

    def test_ids_are_int_and_unique(self, lfs):
        mc = lfs[0]["mc5_mc6"]
        schema = mc.collect_schema()
        assert schema["m4id"] == schema["m5id"] == pl.Int64
        n = mc.select(pl.col("m4id", "m5id").n_unique()).collect().row(0)
        assert n == (3_527_285, 3_527_285)

    def test_sample_endpoint_pair_is_unique(self, lfs):
        mc = lfs[0]["mc5_mc6"]
        dup = mc.select(pl.len() - pl.struct("spid", "aeid").n_unique()).collect()
        assert dup.item() == 0

    def test_no_dot_in_any_column_name(self, lfs):
        for name, lf in {**lfs[0], **lfs[1]}.items():
            assert not [c for c in lf.collect_schema() if "." in c], name

    def test_bioactivity_join_keeps_every_row_and_annotates_all(self, lfs):
        bio = lfs[1]["bioactivity"]
        assert bio.select(pl.len()).collect().item() == 3_527_285 - 9_383
        assert bio.select(pl.col("chid").is_null().sum()).collect().item() == 0
        assert bio.select(pl.col("organism").is_null().sum()).collect().item() == 0

    def test_every_endpoint_has_an_annotation(self, lfs):
        mc, ann = lfs[0]["mc5_mc6"], lfs[0]["assay_annotations"]
        missing = (
            mc.select("aeid").unique().join(ann.select("aeid"), on="aeid", how="anti")
        )
        assert missing.collect().height == 0
        assert mc.select(pl.col("aeid").n_unique()).collect().item() == 1_583

    def test_every_tested_chemical_is_in_compounds(self, lfs):
        mc, cyto = lfs[0]["mc5_mc6"], lfs[0]["cytotox"]
        tested = mc.select(pl.col("chid").drop_nulls().unique())
        assert tested.collect().height == 9_800
        assert (
            tested.join(cyto.select("chid"), on="chid", how="anti").collect().height
            == 0
        )
        untested = cyto.select("chid").join(tested, on="chid", how="anti")
        assert untested.collect().height == 687

    def test_blank_wells_have_no_chemical(self, lfs):
        mc = lfs[0]["mc5_mc6"]
        blank = mc.filter(pl.col("chid").is_null())
        assert blank.select(pl.len()).collect().item() == 9_383
        assert (
            blank.select(pl.col("spid").str.starts_with("MediaBlank").any())
            .collect()
            .item()
        )

    def test_hitc_equals_hitcall(self, lfs):
        mc = lfs[0]["mc5_mc6"]
        assert mc.select((pl.col("hitc") == pl.col("hitcall")).all()).collect().item()

    def test_winning_models(self, lfs):
        mc = lfs[0]["mc5_mc6"]
        models = mc.select(pl.col("modl").unique()).collect()["modl"].to_list()
        assert sorted(models) == sorted(
            ["poly1", "poly2", "exp2", "exp3", "exp4", "exp5"]
            + ["hill", "gnls", "pow", "loec", "none"]
        )
        loec = mc.filter(modl="loec").select(pl.len()).collect().item()
        assert loec == 208_684

    def test_ac50_null_for_unfitted_rows_not_for_hits(self, lfs):
        mc = lfs[0]["mc5_mc6"]
        assert mc.select(pl.col("ac50").is_null().sum()).collect().item() == 12_304
        hits = mc.filter(pl.col("hitc") >= 0.9)
        assert hits.select(pl.col("ac50").is_null().mean()).collect().item() < 1e-5

    def test_full_keeps_every_bioactivity_row_and_has_cytotox_for_all(self, lfs):
        full = lfs[1]["full"]
        assert full.select(pl.len()).collect().item() == 3_527_285 - 9_383
        nulls = full.select(pl.col("cytotox_lower_bound_um").is_null().sum())
        assert nulls.collect().item() == 0
