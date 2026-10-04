"""Shared raw -> Parquet consolidation: suffix dispatch, staging, table naming."""

import gzip
import json
import lzma

import polars as pl
import pytest

from fairfetched.utils import raw as raw_mod  # the `raw` fixture below shadows it

_TSV = "a\tb\n1\tx\n2\ty\n"


class TestScanTable:
    @pytest.mark.parametrize(
        "name,body,kwargs",
        [
            ("t.tsv", _TSV, {}),
            ("t.txt", _TSV, {}),
            ("t.csv", "a,b\n1,x\n2,y\n", {}),
            ("t.tsv.gz", _TSV, {}),
            ("t.tsv.xz", _TSV, {}),
            ("t.bin", _TSV, {"separator": "\t"}),
            ("t.gz", _TSV, {"separator": "\t"}),
            ("t", _TSV, {"separator": "\t"}),
        ],
    )
    def test_scan_formats(self, tmp_path, name, body, kwargs):
        p = tmp_path / name
        compress = {".gz": gzip.compress, ".xz": lzma.compress}.get(p.suffix)
        p.write_bytes(compress(body.encode()) if compress else body.encode())
        assert raw_mod.scan_raw(p, **kwargs).collect().to_dict(as_series=False) == {
            "a": [1, 2],
            "b": ["x", "y"],
        }
        if p.suffix == ".xz":
            assert p.with_suffix("").read_text() == body

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

    @pytest.mark.parametrize(
        "suffix,compress", [("gz", gzip.compress), ("xz", lzma.compress)]
    )
    @pytest.mark.parametrize("failure", [False, True])
    def test_decompression_cleanup_and_cache(
        self, tmp_path, monkeypatch, suffix, compress, failure
    ):
        temp = tmp_path / "system_temp"
        temp.mkdir()
        monkeypatch.setattr("tempfile.tempdir", str(temp))
        source = tmp_path / "raw" / f"t.tsv.{suffix}"
        source.parent.mkdir()
        source.write_bytes(compress(_TSV.encode()))
        kwargs = {"t": {"schema": {"a": pl.Int64, "b": pl.Int64}}} if failure else None
        convert = lambda: raw_mod.ensure_parquet_tables(
            {"t": source}, scan_kwargs=kwargs, decompress_first={"t": 2.0}
        )
        if failure:
            with pytest.raises(pl.exceptions.PolarsError):
                convert()
            assert not list((tmp_path / "parquet").iterdir())
        else:
            out = convert()
            assert pl.read_parquet(out["t"]).shape == (2, 2)
            source.write_bytes(b"invalid compressed data")
            assert convert() == out
        assert list(temp.iterdir()) == []
        assert not source.with_suffix("").exists()

    def test_default_dir_names_and_values(self, raw, tmp_path):
        out = raw_mod.ensure_parquet_tables(raw, table_names={"two": "renamed"})
        assert out == {
            "one": tmp_path / "parquet" / "one.parquet",
            "renamed": tmp_path / "parquet" / "renamed.parquet",
        }
        assert pl.read_parquet(out["one"])["a"].to_list() == [1, 2]

    def test_existing_unpinned_table_is_not_rewritten(self, raw, tmp_path):
        out = raw_mod.ensure_parquet_tables(raw)
        (tmp_path / "parquet" / "_tables.json").unlink()
        out["one"].write_bytes(b"sentinel")
        raw_mod.ensure_parquet_tables(raw)
        assert out["one"].read_bytes() == b"sentinel"

    def test_overwritten_pinned_table_raises(self, raw):
        out = raw_mod.ensure_parquet_tables(raw)
        out["one"].write_bytes(b"sentinel")
        with pytest.raises(ValueError, match="different size"):
            raw_mod.ensure_parquet_tables(raw)

    def test_same_size_corruption_raises(self, raw):
        out = raw_mod.ensure_parquet_tables(raw)
        data = bytearray(out["one"].read_bytes())
        data[len(data) // 2] ^= 0xFF
        out["one"].write_bytes(bytes(data))
        with pytest.raises(ValueError, match="differ from the release pinned"):
            raw_mod.ensure_parquet_tables(raw)

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


class TestTablesManifest:
    def test_round_trip_with_directory_table(self, tmp_path):
        (tmp_path / "dd").mkdir()
        (tmp_path / "a.parquet").touch()
        tables = {"a": tmp_path / "a.parquet", "dd": tmp_path / "dd"}
        raw_mod.write_manifest(tmp_path, tables)
        assert json.loads((tmp_path / "_tables.json").read_text())["tables"] == {
            "a": "a.parquet",
            "dd": "dd",
        }
        assert raw_mod.read_manifest(tmp_path) == tables

    def test_directory_table_pins_each_part_file(self, tmp_path):
        d = tmp_path / "dd"
        d.mkdir()
        (d / "part-0.parquet").write_bytes(b"x")
        raw_mod.write_manifest(tmp_path, {"dd": d})
        assert list(json.loads((tmp_path / "_tables.json").read_text())["files"]) == [
            "dd/part-0.parquet"
        ]
        (d / "part-1.parquet").write_bytes(b"y")  # extra part => rebuild
        assert raw_mod.read_manifest(tmp_path) is None

    def test_none_without_manifest(self, tmp_path):
        assert raw_mod.read_manifest(tmp_path) is None

    def test_none_when_a_listed_table_is_missing(self, tmp_path):
        (tmp_path / "a.parquet").touch()
        raw_mod.write_manifest(tmp_path, {"a": tmp_path / "a.parquet"})
        (tmp_path / "a.parquet").unlink()
        assert raw_mod.read_manifest(tmp_path) is None
