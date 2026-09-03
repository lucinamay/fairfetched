"""SIDER source: offline checks on the consolidation, the joins, and the
content-hash pin. No download."""

from pathlib import Path

import polars as pl
import pytest

from fairfetched.get import sider
from fairfetched.utils import manifest

_FREQ = pl.DataFrame(
    {
        "stitch_flat": ["CID100000085", "CID100000085"],
        "stitch_stereo": ["CID000010917", "CID000010917"],
        "umls_label": ["C0000737", "C0000737"],
        "placebo": ["", "placebo"],
        "freq_desc": ["21%", "rare"],
        "lower": [0.21, 0.0001],
        "upper": [0.21, 0.01],
        "term_type": ["PT", "PT"],
        "umls_meddra": ["C0000737", "C0000737"],
        "term_name": ["Abdominal pain", "Abdominal pain"],
    }
)
_ALL_SE = pl.DataFrame(
    {
        "stitch_flat": ["CID100000085"],
        "stitch_stereo": ["CID000010917"],
        "umls_label": ["C0000729"],
        "term_type": ["PT"],
        "umls_meddra": ["C0000737"],
        "term_name": ["Abdominal pain"],
    }
)
_DRUG_NAMES = pl.DataFrame(
    {"stitch_flat": ["CID100000085"], "drug_name": ["carnitine"]}
)
_MEDDRA = pl.DataFrame(
    {
        "umls_id": ["C0000727"],
        "term_type": ["PT"],
        "meddra_id": ["10000647"],
        "term_name": ["Acute abdomen"],
    }
)


@pytest.fixture
def parquet_paths(tmp_path: Path) -> dict[str, Path]:
    frames = {
        "meddra_freq": _FREQ,
        "meddra_all_se": _ALL_SE,
        "drug_names": _DRUG_NAMES,
        "meddra": _MEDDRA,
    }
    paths = {}
    for name, df in frames.items():
        paths[name] = tmp_path / f"{name}.parquet"
        df.write_parquet(paths[name])
    return paths


class TestVersions:
    def test_latest_is_last_available(self):
        assert sider.latest() == sider.available_versions()[-1]

    def test_source_urls_are_https(self):
        urls = sider.source_urls("4.1")
        assert set(urls) == set(sider._FILES)
        assert all(u.startswith("https://") for u in urls.values())


class TestConsolidation:
    def test_readme_is_not_a_parquet_table(self, tmp_path):
        (tmp_path / "README").write_text("format description")
        (tmp_path / "drug_names.tsv").write_text("CID100000085\tcarnitine\n")
        out = sider.ensure_parquet_tables(
            {"readme": tmp_path / "README", "drug_names": tmp_path / "drug_names.tsv"},
            tmp_path / "parquet",
        )
        assert "readme" not in out
        assert out["drug_names"].exists()

    def test_unescaped_quotes_in_term_names_survive(self, tmp_path):
        raw = tmp_path / "meddra.tsv"
        raw.write_text('C0000727\tPT\t10000647\t"Ventilation" pneumonitis\n')
        out = sider.ensure_parquet_tables({"meddra": raw}, tmp_path / "parquet")
        row = pl.read_parquet(out["meddra"]).row(0, named=True)
        assert row["term_name"] == '"Ventilation" pneumonitis'


class TestViews:
    def test_stitch_id_becomes_pubchem_cid(self, parquet_paths):
        drugs = sider.build_views(parquet_paths)["drugs"].collect()
        assert drugs["pubchem_cid"].to_list() == [85]  # CID1 00000085

    def test_placebo_empty_string_is_nulled(self, parquet_paths):
        freq = sider.build_views(parquet_paths)["frequencies"].collect()
        assert freq.sort("freq_desc")["placebo"].to_list() == [None, "placebo"]

    def test_side_effects_carries_drug_name(self, parquet_paths):
        se = sider.build_views(parquet_paths)["side_effects"].collect()
        assert se["drug_name"].to_list() == ["carnitine"]

        class TestPin:
            def test_committed_manifest_covers_every_source_file(self):
                import json

                recorded = json.loads(sider._MANIFEST_PATH.read_text())["files"]
                assert set(recorded) == set(sider._FILES)

            def test_committed_pin_flags_a_drifted_file(self, tmp_path):
                bogus = tmp_path / "drug_names.tsv"
                bogus.write_text("not the real drug_names file")
                with pytest.raises(ValueError, match="differ from the release pinned"):
                    manifest.verify({"drug_names": bogus}, sider._MANIFEST_PATH)
