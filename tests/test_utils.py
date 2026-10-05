import json
import os
import sqlite3
import tarfile
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from fairfetched.get import chembl
from fairfetched.utils import _track as track_module
from fairfetched.utils.files import ensure_untarred_sqlite as untar_sqlite
from fairfetched.utils.storage import _get_fairfetched_home_dir


@pytest.fixture
def temp_dir():
    """Create a temporary directory that gets cleaned up after the test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_sqlite_db(temp_dir):
    """Create a sample SQLite database for testing."""
    db_path = temp_dir / "test.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create a simple test table
    cursor.execute("""
        CREATE TABLE test_table (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            value REAL
        )
    """)
    cursor.execute("INSERT INTO test_table (name, value) VALUES ('test1', 1.5)")
    cursor.execute("INSERT INTO test_table (name, value) VALUES ('test2', 2.5)")

    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def tar_gz_archive(sample_sqlite_db, temp_dir):
    """Create a tar.gz archive containing the SQLite database."""
    archive_path = temp_dir / "archive.tar.gz"

    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(sample_sqlite_db, arcname="test.db")

    return archive_path


def test_untar_sqlite_from_tar_gz(tar_gz_archive, temp_dir):
    """Test extracting SQLite database from a tar.gz file."""
    extracted_path = untar_sqlite(tar_gz_archive)

    # Verify the file exists
    assert extracted_path.exists()
    assert extracted_path.suffix == ".db"

    # Verify it's a valid SQLite database
    conn = sqlite3.connect(extracted_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    conn.close()

    assert len(tables) > 0
    assert tables[0][0] == "test_table"


def test_untar_sqlite_invalid_tar_gz(temp_dir):
    """Test error handling when tar.gz file doesn't contain a SQLite database."""
    invalid_tar = temp_dir / "invalid.tar.gz"

    # Create a tar.gz with a non-database file
    with tarfile.open(invalid_tar, "w:gz") as tar:
        text_file = temp_dir / "test.txt"
        text_file.write_text("This is not a database")
        tar.add(text_file, arcname="test.txt")

    with pytest.raises(ValueError, match="No .db file found in archive"):
        untar_sqlite(invalid_tar)


def test_untar_sqlite_database_content(tar_gz_archive):
    """Test that extracted database contains the expected data."""
    extracted_path = untar_sqlite(tar_gz_archive)

    conn = sqlite3.connect(extracted_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM test_table")
    row_count = cursor.fetchone()[0]

    cursor.execute("SELECT name, value FROM test_table ORDER BY id")
    rows = cursor.fetchall()
    conn.close()

    assert row_count == 2
    assert rows[0] == ("test1", 1.5)
    assert rows[1] == ("test2", 2.5)


class TestHomeDir:
    """Test HOME_DIR handling with tilde expansion."""

    def test_fairfetched_home_with_tilde(self):
        """Test FAIRFETCHED_HOME expands ~ correctly."""
        with mock.patch.dict(
            os.environ, {"FAIRFETCHED_HOME": "~/fairfetched_data"}, clear=False
        ):
            result = _get_fairfetched_home_dir()
            expected = Path.home() / "fairfetched_data"
            assert result == expected
            assert "~" not in str(result)

    def test_pystow_home_with_tilde(self):
        """Test PYSTOW_HOME expands ~ correctly."""
        with mock.patch.dict(os.environ, {"PYSTOW_HOME": "~/pystow_data"}, clear=False):
            # Clear FAIRFETCHED_HOME if it exists
            env = os.environ.copy()
            env.pop("FAIRFETCHED_HOME", None)
            with mock.patch.dict(os.environ, env, clear=True):
                result = _get_fairfetched_home_dir()
                expected = Path.home() / "pystow_data"
                assert result == expected
                assert "~" not in str(result)

    def test_fairfetched_home_precedence(self):
        """Test FAIRFETCHED_HOME takes precedence over PYSTOW_HOME."""
        with mock.patch.dict(
            os.environ,
            {"FAIRFETCHED_HOME": "~/fairfetched", "PYSTOW_HOME": "~/pystow"},
            clear=False,
        ):
            result = _get_fairfetched_home_dir()
            expected = Path.home() / "fairfetched"
            assert result == expected

    def test_default_home_dir(self):
        """Test default HOME_DIR when neither env var is set."""
        env = os.environ.copy()
        env.pop("FAIRFETCHED_HOME", None)
        env.pop("PYSTOW_HOME", None)
        with mock.patch.dict(os.environ, env, clear=True):
            result = _get_fairfetched_home_dir()
            expected = Path.home() / ".data"
            assert result == expected

    def test_absolute_path_fairfetched_home(self):
        """Test FAIRFETCHED_HOME with absolute path (no tilde)."""
        with mock.patch.dict(
            os.environ, {"FAIRFETCHED_HOME": "/tmp/fairfetched"}, clear=False
        ):
            result = _get_fairfetched_home_dir()
            assert result == Path("/tmp/fairfetched")

    def test_nested_tilde_path(self):
        """Test nested paths with tilde."""
        with mock.patch.dict(
            os.environ, {"FAIRFETCHED_HOME": "~/data/fairfetched/v1"}, clear=False
        ):
            result = _get_fairfetched_home_dir()
            expected = Path.home() / "data/fairfetched/v1"
            assert result == expected
            assert "~" not in str(result)


class TestTrack:
    def test_simple_fallback_writes_to_stderr(self, monkeypatch, capsys):
        monkeypatch.setattr(track_module, "HAS_TQDM", False)
        monkeypatch.setattr(track_module, "HAS_RICH", False)
        monkeypatch.setattr(track_module, "in_marimo", lambda: False)

        out = list(track_module.track(range(3), desc="work", total=3))

        captured = capsys.readouterr()
        assert out == [0, 1, 2]
        assert captured.out == ""
        assert "work" in captured.err
        assert "100%" in captured.err
        assert "3/3" in captured.err

    def test_track_can_be_disabled(self, monkeypatch, capsys):
        monkeypatch.setattr(track_module, "HAS_TQDM", False)
        monkeypatch.setattr(track_module, "HAS_RICH", False)
        monkeypatch.setattr(track_module, "in_marimo", lambda: False)

        out = list(track_module.track(range(3), desc="work", total=3, disable=True))

        captured = capsys.readouterr()
        assert out == [0, 1, 2]
        assert captured.out == ""
        assert captured.err == ""


class TestChemblTablesManifest:
    @pytest.fixture
    def raw_paths(self, sample_sqlite_db, temp_dir):
        archive = temp_dir / "sql_db.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(sample_sqlite_db, arcname="chembl_test.db")
        sample_sqlite_db.unlink()
        return {"sql_db": archive}

    def test_uncompressed_db_deleted_and_tables_listed(self, raw_paths, temp_dir):
        out = chembl.ensure_parquet_tables(raw_paths, temp_dir / "pq")
        assert list(out) == ["test_table"] and out["test_table"].is_file()
        assert not (temp_dir / "chembl_test.db").exists()
        pinned = json.loads((temp_dir / "pq" / "_tables.json").read_text())
        assert pinned["tables"] == {"test_table": "test_table.parquet"}
        assert list(pinned["files"]) == ["test_table.parquet"]

    def test_reload_needs_neither_archive_nor_db(self, raw_paths, temp_dir):
        first = chembl.ensure_parquet_tables(raw_paths, temp_dir / "pq")
        raw_paths["sql_db"].write_bytes(b"not a tarball")
        assert chembl.ensure_parquet_tables(raw_paths, temp_dir / "pq") == first

    def test_missing_table_is_rebuilt_from_archive(self, raw_paths, temp_dir):
        out = chembl.ensure_parquet_tables(raw_paths, temp_dir / "pq")
        out["test_table"].unlink()
        assert chembl.ensure_parquet_tables(raw_paths, temp_dir / "pq")[
            "test_table"
        ].is_file()

    def test_interrupted_run_writes_no_manifest(self, raw_paths, temp_dir):
        with (
            mock.patch.object(
                chembl, "ensure_sqlite_db_to_parquets", side_effect=RuntimeError
            ),
            pytest.raises(RuntimeError),
        ):
            chembl.ensure_parquet_tables(raw_paths, temp_dir / "pq")
        assert not (temp_dir / "pq" / "_tables.json").exists()


class TestEnsureUntarredSqlite:
    def test_truncated_extraction_is_redone(self, tmp_path):
        src = tmp_path / "src" / "x.db"
        src.parent.mkdir()
        src.write_bytes(b"0123456789" * 100)
        tar = tmp_path / "x.tar.gz"
        with tarfile.open(tar, "w:gz") as f:
            f.add(src, arcname="chembl/x.db")
        out = untar_sqlite(tar)
        out.write_bytes(out.read_bytes()[:10])
        assert untar_sqlite(tar).stat().st_size == 1000
