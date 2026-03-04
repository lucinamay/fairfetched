"""Integration tests for the chembl module.

These tests verify the end-to-end workflow of downloading, extracting,
and processing ChEMBL data, using temporary directories that are cleaned
up after each test.
"""

import shutil
import sqlite3
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from fairfetched.get._ensure import ensure_url
from fairfetched.get._utils import sqlite_db_to_parquets, untar_sqlite


@pytest.fixture
def temp_base_dir():
    """Create a temporary base directory for all test data."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup after test
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def sample_chembl_database():
    """Create a mock ChEMBL-like SQLite database with sample tables."""
    temp_db = Path(tempfile.mktemp(suffix=".db"))
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    # Create sample tables mimicking ChEMBL structure
    cursor.execute("""
        CREATE TABLE molecule_dictionary (
            molregno INTEGER PRIMARY KEY,
            chembl_id TEXT UNIQUE NOT NULL,
            pref_name TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE compound_properties (
            molregno INTEGER PRIMARY KEY,
            mw_freebase REAL,
            alogp REAL
        )
    """)

    cursor.execute("""
        CREATE TABLE activities (
            activity_id INTEGER PRIMARY KEY,
            assay_id INTEGER,
            molregno INTEGER,
            standard_value REAL,
            standard_units TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE assays (
            assay_id INTEGER PRIMARY KEY,
            assay_type TEXT,
            description TEXT
        )
    """)

    # Insert sample data
    cursor.execute(
        "INSERT INTO molecule_dictionary VALUES (?, ?, ?)",
        (1, "CHEMBL1", "Aspirin"),
    )
    cursor.execute(
        "INSERT INTO molecule_dictionary VALUES (?, ?, ?)",
        (2, "CHEMBL2", "Ibuprofen"),
    )

    cursor.execute(
        "INSERT INTO compound_properties VALUES (?, ?, ?)",
        (1, 180.16, 1.19),
    )
    cursor.execute(
        "INSERT INTO compound_properties VALUES (?, ?, ?)",
        (2, 206.28, 3.97),
    )

    cursor.execute(
        "INSERT INTO activities VALUES (?, ?, ?, ?, ?)",
        (1, 1, 1, 5.2, "nM"),
    )
    cursor.execute(
        "INSERT INTO activities VALUES (?, ?, ?, ?, ?)",
        (2, 2, 2, 3.1, "nM"),
    )

    cursor.execute(
        "INSERT INTO assays VALUES (?, ?, ?)",
        (1, "B", "Binding assay"),
    )
    cursor.execute(
        "INSERT INTO assays VALUES (?, ?, ?)",
        (2, "F", "Functional assay"),
    )

    conn.commit()
    conn.close()

    yield temp_db

    # Cleanup
    if temp_db.exists():
        temp_db.unlink()


@pytest.fixture
def chembl_tar_gz(sample_chembl_database, temp_base_dir):
    """Create a tar.gz archive containing the mock ChEMBL database."""
    tar_gz_path = temp_base_dir / "chembl_36_sqlite.tar.gz"

    with tarfile.open(tar_gz_path, mode="w:gz") as tar:
        tar.add(sample_chembl_database, arcname="chembl_36.db")

    return tar_gz_path


def test_ensure_url_downloads_tar_gz(chembl_tar_gz, temp_base_dir):
    """Test downloading a tar.gz file (mocked)."""
    download_dir = temp_base_dir / "raw"
    download_dir.mkdir(parents=True, exist_ok=True)

    # Read the actual tar.gz content
    tar_gz_content = chembl_tar_gz.read_bytes()

    with patch("fairfetched.get._ensure.requests.get") as mock_get:
        mock_response = Mock()
        mock_response.iter_content = Mock(return_value=[tar_gz_content])
        mock_get.return_value = mock_response

        result = ensure_url(
            "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_36/chembl_36_sqlite.tar.gz",
            download_dir / "chembl_36_sqlite.tar.gz",
        )

    # Verify the file was downloaded
    assert result.exists()
    assert result.suffix == ".gz"
    assert result.stat().st_size > 0


def test_extract_and_convert_to_parquets(chembl_tar_gz, temp_base_dir):
    """Test extracting tar.gz and converting to parquet files."""
    # Extract the database
    extracted_db = untar_sqlite(chembl_tar_gz)
    assert extracted_db.exists()

    # Convert to parquets
    parquet_dir = temp_base_dir / "clean"
    parquet_paths = sqlite_db_to_parquets(extracted_db, cache_dir=parquet_dir)

    # Verify parquet files were created
    assert len(parquet_paths) > 0
    assert parquet_dir.exists()

    # Verify expected tables
    expected_tables = [
        "molecule_dictionary",
        "compound_properties",
        "activities",
        "assays",
    ]
    for table_name in expected_tables:
        assert table_name in parquet_paths
        parquet_path = parquet_paths[table_name]
        assert parquet_path.exists()
        assert parquet_path.suffix == ".parquet"

    # Cleanup
    extracted_db.unlink()


def test_full_workflow_from_tar_gz_to_parquets(chembl_tar_gz, temp_base_dir):
    """Test the complete workflow: extract tar.gz -> convert to parquets."""
    # Step 1: Extract the database
    extracted_db = untar_sqlite(chembl_tar_gz)
    assert extracted_db.exists()

    # Step 2: Convert to parquets
    parquet_dir = temp_base_dir / "clean"
    parquet_paths = sqlite_db_to_parquets(extracted_db, cache_dir=parquet_dir)

    # Step 3: Verify all parquets exist
    assert len(parquet_paths) == 4  # We created 4 tables

    # Step 4: Verify parquet content with polars
    import polars as pl

    molecules_lf = pl.scan_parquet(parquet_paths["molecule_dictionary"])
    molecules = molecules_lf.collect()

    assert molecules.height == 2
    assert "molregno" in molecules.columns
    assert "chembl_id" in molecules.columns
    assert "pref_name" in molecules.columns

    activities_lf = pl.scan_parquet(parquet_paths["activities"])
    activities = activities_lf.collect()

    assert activities.height == 2
    assert "activity_id" in activities.columns
    assert "standard_value" in activities.columns

    # Cleanup
    extracted_db.unlink()


def test_extracted_database_from_directory_to_parquets(
    sample_chembl_database, temp_base_dir
):
    """Test extraction when database is in a directory (not tar.gz)."""
    # Create a directory structure
    db_dir = temp_base_dir / "extracted"
    db_dir.mkdir()
    db_path = db_dir / sample_chembl_database.name
    shutil.copy(sample_chembl_database, db_path)

    # Use untar_sqlite on directory
    result = untar_sqlite(db_dir)
    assert result == db_path
    assert result.exists()

    # Convert to parquets
    parquet_dir = temp_base_dir / "clean"
    parquet_paths = sqlite_db_to_parquets(result, cache_dir=parquet_dir)

    # Verify all tables were converted
    assert len(parquet_paths) == 4
    for path in parquet_paths.values():
        assert path.exists()
        assert path.suffix == ".parquet"


def test_parquet_files_use_separate_temp_dir(chembl_tar_gz, temp_base_dir):
    """Test that parquet conversion doesn't pollute the original temp_base_dir."""
    extracted_db = untar_sqlite(chembl_tar_gz)

    parquet_dir_1 = temp_base_dir / "clean_1"
    parquet_paths_1 = sqlite_db_to_parquets(extracted_db, cache_dir=parquet_dir_1)

    parquet_dir_2 = temp_base_dir / "clean_2"
    parquet_paths_2 = sqlite_db_to_parquets(extracted_db, cache_dir=parquet_dir_2)

    # Verify both directories are separate
    assert parquet_dir_1 != parquet_dir_2
    assert parquet_dir_1.exists()
    assert parquet_dir_2.exists()

    # Verify both have the same tables but in different locations
    assert set(parquet_paths_1.keys()) == set(parquet_paths_2.keys())

    # Verify no cross-contamination
    for path in parquet_paths_1.values():
        assert "clean_1" in str(path)

    for path in parquet_paths_2.values():
        assert "clean_2" in str(path)

    # Cleanup
    extracted_db.unlink()


def test_multiple_databases_in_separate_temp_dirs(temp_base_dir):
    """Test handling multiple databases in separate temporary directories."""
    temp_dirs = []

    for i in range(3):
        # Create a temporary directory for each database
        temp_dir = Path(tempfile.mkdtemp())
        temp_dirs.append(temp_dir)

        # Create a simple database
        db_path = temp_dir / f"test_{i}.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(f"CREATE TABLE table_{i} (id INTEGER PRIMARY KEY, value TEXT)")
        cursor.execute(f"INSERT INTO table_{i} VALUES (1, 'value_{i}')")
        conn.commit()
        conn.close()

        # Convert to parquets
        parquet_dir = temp_dir / "parquets"
        parquet_paths = sqlite_db_to_parquets(db_path, cache_dir=parquet_dir)

        # Verify conversion
        assert len(parquet_paths) == 1
        assert parquet_paths[f"table_{i}"].exists()

    # Cleanup all temp directories
    for temp_dir in temp_dirs:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
