import sqlite3
import tarfile
import tempfile
from pathlib import Path

import pytest

from fairfetched.utils.files import ensure_untarred_sqlite as untar_sqlite


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

    with pytest.raises(ValueError, match="No .sqlite or .db file found"):
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
