"""Shared manifest verification for committed and in-memory pins."""

import json

import pytest

from fairfetched.utils import manifest


@pytest.fixture
def raw_paths(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("alpha")
    return {"a": path}


@pytest.mark.parametrize(
    ("algorithm", "checksum"),
    [
        ("sha256", "8ed3f6ad685b959ead7022518e1af76cd816f8e8ec7ccdda1ed4018e8f2223f8"),
        ("md5", "2c1743a391305fbf367df8e4f069f9f9"),
    ],
)
def test_in_memory_hash_and_drift(raw_paths, algorithm, checksum):
    pin = {"files": {"a": {algorithm: checksum}}, "version": "1"}
    manifest.verify(raw_paths, pin, strict=True, algorithm=algorithm, version="1")
    raw_paths["a"].write_text("changed")
    with pytest.raises(ValueError, match="1 file"):
        manifest.verify(raw_paths, pin, strict=True, algorithm=algorithm)


def test_write_records_literal_sha256_and_metadata(tmp_path, raw_paths):
    path = tmp_path / "m.json"
    out = manifest.write(raw_paths, path, version="9")
    assert out["files"]["a"]["sha256"] == (
        "8ed3f6ad685b959ead7022518e1af76cd816f8e8ec7ccdda1ed4018e8f2223f8"
    )
    assert out["version"] == "9"
    assert out["files"]["a"]["bytes"] == 5
    assert json.loads(path.read_text()) == out


def test_permissive_vs_strict(tmp_path, raw_paths, caplog):
    manifest.verify(raw_paths, tmp_path / "absent.json")
    assert "unpinned" in caplog.text
    with pytest.raises(ValueError, match="absent"):
        manifest.verify(raw_paths, tmp_path / "absent.json", strict=True)
    pin = {"files": {}, "version": "1"}
    manifest.verify(raw_paths, pin)
    with pytest.raises(ValueError, match="1 file"):
        manifest.verify(raw_paths, pin, strict=True)
    with pytest.raises(ValueError, match="version"):
        manifest.verify(raw_paths, pin, version="2")
