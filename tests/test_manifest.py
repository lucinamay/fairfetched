"""Content-hash pin: fairfetched.utils.manifest, reusable across sources."""

import json

import pytest

from fairfetched.utils import manifest


@pytest.fixture
def raw_paths(tmp_path):
    paths = {}
    for name, body in {"a": "alpha", "b": "beta"}.items():
        paths[name] = tmp_path / f"{name}.txt"
        paths[name].write_text(body)
    return paths


class TestWrite:
    def test_records_hash_size_and_meta(self, tmp_path, raw_paths):
        out = manifest.write(raw_paths, tmp_path / "m.json", version="9")
        assert out["version"] == "9"
        assert out["files"]["a"]["bytes"] == len("alpha")
        assert out["files"]["a"]["sha256"] == manifest.sha256(raw_paths["a"])
        assert json.loads((tmp_path / "m.json").read_text()) == out


class TestVerify:
    def test_passes_when_unchanged(self, tmp_path, raw_paths):
        manifest.write(raw_paths, tmp_path / "m.json")
        manifest.verify(raw_paths, tmp_path / "m.json")  # no raise

    def test_raises_when_a_file_changed(self, tmp_path, raw_paths):
        manifest.write(raw_paths, tmp_path / "m.json")
        raw_paths["b"].write_text("BETA")
        with pytest.raises(ValueError, match=r"1 file\(s\) differ"):
            manifest.verify(raw_paths, tmp_path / "m.json")

    def test_ignores_paths_not_in_the_manifest(self, tmp_path, raw_paths):
        manifest.write({"a": raw_paths["a"]}, tmp_path / "m.json")
        manifest.verify(raw_paths, tmp_path / "m.json")  # 'b' unrecorded, no raise

    def test_missing_manifest_warns_and_skips(self, tmp_path, raw_paths, caplog):
        manifest.verify(raw_paths, tmp_path / "absent.json")
        assert "unpinned" in caplog.text
