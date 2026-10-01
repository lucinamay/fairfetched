"""Shared raw -> Parquet consolidation: suffix dispatch, staging, table naming."""

import gzip
import lzma

import polars as pl
import pytest

from fairfetched.utils import raw as raw_mod  # the `raw` fixture below shadows it

_TSV = "a\tb\n1\tx\n2\ty\n"


class TestScanTable:
    @pytest.mark.parametrize(
        "name, body, sep",
        [
            ("t.tsv", _TSV, None),
            ("t.txt", _TSV, None),
            ("t.csv", "a,b\n1,x\n2,y\n", None),
        ],
    )
    def test_separator_from_suffix(self, tmp_path, name, body, sep):
        p = tmp_path / name
        p.write_text(body)
        assert raw_mod.scan_raw(p).collect().to_dict(as_series=False) == {
            "a": [1, 2],
            "b": ["x", "y"],
        }

    def test_gz(self, tmp_path):
        p = tmp_path / "t.tsv.gz"
        p.write_bytes(gzip.compress(_TSV.encode()))
        assert raw_mod.scan_raw(p).collect().shape == (2, 2)

    def test_xz_is_decompressed_beside_archive(self, tmp_path):
        p = tmp_path / "t.tsv.xz"
        p.write_bytes(lzma.compress(_TSV.encode()))
        assert raw_mod.scan_raw(p).collect().shape == (2, 2)

    def test_kwargs_override_inferred_separator(self, tmp_path):
        p = tmp_path / "t.txt"
        p.write_text("1#x\n2#y\n")
        lf = raw_mod.scan_raw(
            p, separator="#", has_header=False, new_columns=["a", "b"]
        )
        assert lf.collect()["b"].to_list() == ["x", "y"]

    @pytest.mark.parametrize("name", ["t.bin", "t.gz", "t"])
    def test_unknown_suffix_raises(self, tmp_path, name):
        (tmp_path / name).write_bytes(b"x")
        with pytest.raises(ValueError):
            raw_mod.scan_raw(tmp_path / name)


class TestEnsureParquetTables:
    @pytest.fixture
    def raw(self, tmp_path):
        d = tmp_path / "raw"
        d.mkdir()
        (d / "one.tsv").write_text(_TSV)
        (d / "two.tsv").write_text(_TSV)
        return {"one": d / "one.tsv", "two": d / "two.tsv"}

    def test_xz_decompressed_file_removed(self, tmp_path):
        p = tmp_path / "raw" / "t.tsv.xz"
        p.parent.mkdir()
        p.write_bytes(lzma.compress(_TSV.encode()))
        out = raw_mod.ensure_parquet_tables({"t": p})
        assert pl.read_parquet(out["t"]).shape == (2, 2)
        assert not p.with_suffix("").exists()

    def test_xz_decompressed_file_removed_on_failure(self, tmp_path):
        p = tmp_path / "raw" / "t.tsv.xz"
        p.parent.mkdir()
        p.write_bytes(lzma.compress(_TSV.encode()))
        with pytest.raises(pl.exceptions.PolarsError):
            raw_mod.ensure_parquet_tables(
                {"t": p}, scan_kwargs={"t": {"schema": {"zz": pl.Int64}}}
            )
        assert not p.with_suffix("").exists()
        assert not (tmp_path / "parquet" / "t.parquet").exists()

    def test_default_dir_names_and_values(self, raw, tmp_path):
        out = raw_mod.ensure_parquet_tables(raw, table_names={"two": "renamed"})
        assert out == {
            "one": tmp_path / "parquet" / "one.parquet",
            "renamed": tmp_path / "parquet" / "renamed.parquet",
        }
        assert pl.read_parquet(out["one"])["a"].to_list() == [1, 2]

    def test_existing_table_is_not_rewritten(self, raw, tmp_path):
        out = raw_mod.ensure_parquet_tables(raw)
        out["one"].write_bytes(b"sentinel")
        raw_mod.ensure_parquet_tables(raw)
        assert out["one"].read_bytes() == b"sentinel"

    def test_failed_scan_leaves_no_table(self, raw, tmp_path):
        def boom(name, path):
            raise RuntimeError("scan failed")

        with pytest.raises(RuntimeError):
            raw_mod.ensure_parquet_tables(raw, scanner=boom)
        assert not list((tmp_path / "parquet").glob("*.parquet"))

    def test_failed_sink_leaves_no_final_file(self, raw, tmp_path):
        bad = pl.scan_csv(raw["one"]).with_columns(pl.col("b").cast(pl.Int64))
        with pytest.raises(pl.exceptions.PolarsError):
            raw_mod.ensure_parquet_tables({"one": raw["one"]}, scanner=lambda n, p: bad)
        assert not (tmp_path / "parquet" / "one.parquet").exists()
