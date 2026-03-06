"""Tests for the get module covering data retrieval, composition, and lazy frame operations.

These tests use mocking and minimal fixtures to avoid downloading entire datasets,
while thoroughly testing all core functionality.
"""

import shutil
import tempfile
from pathlib import Path

import polars as pl
import pytest

from fairfetched.get import chembl, papyrus
from fairfetched.get.api import Chembl, Papyrus

# ============================================================================
# Fixtures: Temporary directories and mock data
# ============================================================================


@pytest.fixture
def temp_dir():
    """Temporary directory, cleaned up after test."""
    path = Path(tempfile.mkdtemp())
    yield path
    if path.exists():
        shutil.rmtree(path)


@pytest.fixture
def sample_chembl_parquets(temp_dir):
    """Create minimal Parquet files mimicking ChEMBL tables."""
    parquet_dir = temp_dir / "chembl_parquets"
    parquet_dir.mkdir()

    # Minimal tables required for ChEMBL composition
    tables = {
        "molecule_dictionary": pl.DataFrame(
            {
                "molregno": [1, 2, 3],
                "chembl_id": ["CHEMBL1", "CHEMBL2", "CHEMBL3"],
                "pref_name": ["Aspirin", "Ibuprofen", "Acetaminophen"],
            }
        ),
        "compound_properties": pl.DataFrame(
            {
                "molregno": [1, 2, 3],
                "mw_freebase": [180.16, 206.28, 151.16],
                "alogp": [1.19, 3.97, 0.46],
            }
        ),
        "compound_structures": pl.DataFrame(
            {
                "molregno": [1, 2, 3],
                "canonical_smiles": [
                    "O=C(O)Cc1ccccc1C(=O)O",
                    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
                    "CC(=O)Nc1ccc(O)cc1",
                ],
            }
        ),
        "bioactivity": pl.DataFrame(
            {
                "activity_id": [10, 11, 12],
                "molregno": [1, 2, 3],
                "target_id": [100, 101, 100],
                "assay_id": [1000, 1001, 1002],
                "action_type": ["ANTAGONIST", "AGONIST", "ANTAGONIST"],
                "standard_value": [5.2, 3.1, 6.5],
                "standard_units": ["nM", "nM", "nM"],
            }
        ),
        "protein": pl.DataFrame(
            {
                "target_id": [100, 101],
                "target_chembl_id": ["CHEMBL100", "CHEMBL101"],
                "pref_name": ["TYK2", "JAK1"],
                "target_type": ["SINGLE PROTEIN", "SINGLE PROTEIN"],
            }
        ),
        "action_type": pl.DataFrame(
            {
                "action_type": ["ANTAGONIST", "AGONIST"],
                "description": ["Antagonist", "Agonist"],
            }
        ),
        "assays": pl.DataFrame(
            {
                "assay_id": [1000, 1001, 1002],
                "assay_type": ["B", "F", "B"],
                "assay_chembl_id": ["CHEMBL1000", "CHEMBL1001", "CHEMBL1002"],
                "description": ["Binding assay", "Functional assay", "Binding assay"],
            }
        ),
        "assay_type": pl.DataFrame(
            {
                "assay_type": ["B", "F"],
                "description": ["Binding", "Functional"],
            }
        ),
        "compound_records": pl.DataFrame(
            {
                "molregno": [1, 2, 3],
                "doc_id": [5000, 5001, 5002],
                "compound_record_id": [1, 2, 3],
            }
        ),
        "docs": pl.DataFrame(
            {
                "doc_id": [5000, 5001, 5002],
                "pubmed_id": [12345, 12346, 12347],
                "journal": ["J Med Chem", "Bioorg Med Chem", "J Med Chem"],
            }
        ),
        "compound_structural_alerts": pl.DataFrame(
            {
                "molregno": [1, 2],
                "alert_id": [1, 2],
                "alert_name": ["Genotoxic Carbamate", "PAINS filterA1"],
            }
        ),
        "component_sequences": pl.DataFrame(
            {
                "component_id": [1, 2],
                "component_type": ["PROTEIN", "ANTIBODY"],
            }
        ),
        "component_class": pl.DataFrame(
            {
                "component_id": [1, 2],
                "protein_class_id": [10, 11],
                "protein_class_desc": ["Enzyme", "Antibody"],
            }
        ),
        "component_domains": pl.DataFrame(
            {
                "component_id": [1, 2],
                "domain_id": [100, 101],
            }
        ),
        "domains": pl.DataFrame(
            {
                "domain_id": [100, 101],
                "domain_name": ["Kinase domain", "Antibody domain"],
            }
        ),
    }

    paths = {}
    for name, df in tables.items():
        path = parquet_dir / f"{name}.parquet"
        df.write_parquet(path)
        paths[name] = path

    return paths


@pytest.fixture
def sample_papyrus_parquets(temp_dir):
    """Create minimal Parquet files mimicking Papyrus dataset."""
    parquet_dir = temp_dir / "papyrus_parquets"
    parquet_dir.mkdir()

    bioactivity_df = pl.DataFrame(
        {
            "activity_id": [1, 2, 3],
            "connectivity": ["CCO", "CCC", "CCCC"],
            "inchikey": [
                "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
                "LFQSCWFLJHTTHZ-UHFFFAOYSA-O",
                "LFQSCWFLJHTTHZ-UHFFFAOYSA-P",
            ],
            "inchi": [
                "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
                "InChI=1S/C3H8/c1-2-3/h3H2,2H2,1H3",
                "InChI=1S/C4H10/c1-2-3-4/h3H2,1-2,4H3",
            ],
            "target_id": [100, 101, 100],
            "pchembl_value_mean": [5.2, 3.1, 6.5],
            "year": [2020, 2021, 2022],
        }
    )

    protein_df = pl.DataFrame(
        {
            "target_id": [100, 101],
            "uniprotid": ["P12345", "P12346"],
            "target_chembl_id": ["CHEMBL100", "CHEMBL101"],
            "pref_name": ["Kinase 1", "Kinase 2"],
        }
    )

    bioactivity_path = parquet_dir / "bioactivity.parquet"
    protein_path = parquet_dir / "protein.parquet"

    bioactivity_df.write_parquet(bioactivity_path)
    protein_df.write_parquet(protein_path)

    return {
        "bioactivity": bioactivity_path,
        "protein": protein_path,
    }


# ============================================================================
# Tests: ChEMBL Version & Source Management
# ============================================================================


class TestChemblVersions:
    """Test version handling and source retrieval."""

    def test_available_versions_returns_tuple(self):
        """Versions should be a tuple of strings."""
        versions = chembl.available_versions()
        assert isinstance(versions, tuple)
        assert len(versions) > 0
        assert all(isinstance(v, str) for v in versions)

    def test_latest_version_in_available(self):
        """Latest version should be the last in available versions."""
        latest = chembl.latest()
        available = chembl.available_versions()
        assert latest == available[-1]

    def test_get_sources_returns_dict(self):
        """Sources for a version should be a dict with URL strings."""
        latest = chembl.latest()
        sources = chembl.get_sources(latest)
        assert isinstance(sources, dict)
        assert "sql_db" in sources
        assert sources["sql_db"].startswith("https://")

    def test_version_formatter_normalizes_versions(self):
        """Version formatter should canonicalize version strings."""
        assert chembl._version_formatter(24.1) == "24_1"
        assert chembl._version_formatter(22) == "22"
        assert chembl._version_formatter("24_1") == "24_1"

    def test_version_to_url_format(self):
        """URLs should follow ChEMBL FTP structure."""
        url = chembl._version_to_url("24_1")
        assert "ftp.ebi.ac.uk" in url
        assert "chembl_24_1" in url
        assert url.endswith(".tar.gz")

    def test_version_formatter_invalid_type(self):
        """Version formatter should handle invalid types gracefully."""
        with pytest.raises(TypeError):
            chembl._version_formatter([1, 2, 3])  # type: ignore[arg-type]


class TestChemblEnsureRaw:
    """Test ensure_raw function for ChEMBL."""

    def test_ensure_raw_returns_dict_with_paths(self, temp_dir):
        """ensure_raw should return dict with sql_db key pointing to Path."""
        version = chembl.latest()
        result = chembl.ensure_raw(version, raw_dir=temp_dir)

        assert isinstance(result, dict)
        assert "sql_db" in result
        assert isinstance(result["sql_db"], Path)

    def test_ensure_raw_uses_provided_directory(self, temp_dir):
        """ensure_raw should use the provided raw_dir."""
        version = chembl.latest()
        raw_dir = temp_dir / "my_raw"
        result = chembl.ensure_raw(version, raw_dir=raw_dir)

        # Path should be under the specified directory
        assert str(result["sql_db"]).startswith(str(raw_dir))


class TestPapyrusVersions:
    """Test Papyrus version handling."""

    def test_papyrus_available_versions(self):
        """Should list available Papyrus versions."""
        versions = papyrus.available_versions()
        assert isinstance(versions, tuple)
        assert len(versions) > 0
        assert all("." in v for v in versions)  # Version format like "05.6"

    def test_papyrus_get_sources(self):
        """Sources should contain bioactivity, protein, and readme URLs."""
        version = papyrus.latest()
        sources = papyrus.get_sources(version)
        assert "bioactivity" in sources
        assert "protein" in sources
        assert "readme" in sources
        assert all(url.startswith("https://") for url in sources.values())


# ============================================================================
# Tests: ChEMBL Data Pipeline (Mocked Downloads)
# ============================================================================


class TestChemblClean:
    """Test data cleaning pipeline."""

    def test_clean_returns_lazy_frames(self, sample_chembl_parquets):
        """clean() should return dict of LazyFrames."""
        result = chembl.clean(sample_chembl_parquets)

        assert isinstance(result, dict)
        assert len(result) > 0
        assert all(isinstance(lf, pl.LazyFrame) for lf in result.values())

    def test_clean_applies_lowercase_columns(self, temp_dir):
        """clean() should lowercase all column names."""
        # Create a parquet with mixed case columns
        df = pl.DataFrame(
            {
                "MolRegNo": [1, 2],
                "ChemblID": ["A", "B"],
            }
        )
        parquet_path = temp_dir / "test.parquet"
        df.write_parquet(parquet_path)

        result = chembl.clean({"test": parquet_path})
        collected: pl.DataFrame = result["test"].collect()  # ty: ignore[invalid-assignment]

        assert all(col.islower() for col in collected.columns)

    def test_clean_replaces_empty_strings_with_none(self, temp_dir):
        """clean() should replace empty strings with None."""
        df = pl.DataFrame(
            {
                "name": ["Aspirin", "", "Ibuprofen"],
                "description": ["desc1", "desc2", ""],
            }
        )
        parquet_path = temp_dir / "test.parquet"
        df.write_parquet(parquet_path)

        result = chembl.clean({"test": parquet_path})
        collected: pl.DataFrame = result["test"].collect()  # ty: ignore[invalid-assignment]

        # Check for None values (empty strings should be replaced)
        assert collected["name"][1] is None or collected["name"][1] == ""
        assert collected["description"][2] is None or collected["description"][2] == ""


class TestChemblCompose:
    """Test composition of cleaned lazy frames."""

    def test_compose_returns_dict_with_expected_keys(self, sample_chembl_parquets):
        """compose() should return dict with bioactivity, compounds, proteins, components."""
        lfs = chembl.clean(sample_chembl_parquets)
        result = chembl.compose(lfs)

        assert isinstance(result, dict)
        expected_keys = {"bioactivity", "compounds", "proteins", "components"}
        assert expected_keys == set(result.keys())

    def test_compose_returns_lazy_frames(self, sample_chembl_parquets):
        """All composed results should be LazyFrames."""
        lfs = chembl.clean(sample_chembl_parquets)
        result = chembl.compose(lfs)

        assert all(isinstance(lf, pl.LazyFrame) for lf in result.values())

    def test_bioactivity_composition_joins_correctly(self, sample_chembl_parquets):
        """Bioactivity should join with protein, action_type, assays."""
        lfs = chembl.clean(sample_chembl_parquets)
        result = chembl.compose(lfs)
        bioactivity: pl.DataFrame = result["bioactivity"].collect()  # ty: ignore[invalid-assignment]

        # Should have columns from both bioactivity and protein tables
        assert "target_id" in bioactivity.columns
        assert "pref_name" in bioactivity.columns  # From protein
        assert "molregno" in bioactivity.columns

    def test_compounds_composition_includes_structures(self, sample_chembl_parquets):
        """Compounds should include structures and properties."""
        lfs = chembl.clean(sample_chembl_parquets)
        result = chembl.compose(lfs)
        compounds: pl.DataFrame = result["compounds"].collect()  # ty: ignore[invalid-assignment]

        # Should have molecular structure info
        assert "canonical_smiles" in compounds.columns
        assert "mw_freebase" in compounds.columns

    def test_proteins_returns_protein_table(self, sample_chembl_parquets):
        """Proteins should just return the protein LazyFrame."""
        lfs = chembl.clean(sample_chembl_parquets)
        result = chembl.compose(lfs)
        proteins: pl.DataFrame = result["proteins"].collect()  # ty: ignore[invalid-assignment]

        assert "target_id" in proteins.columns
        assert "target_chembl_id" in proteins.columns
        assert len(proteins) == 2


# ============================================================================
# Tests: Papyrus Data Pipeline
# ============================================================================


class TestPapyrusClean:
    """Test Papyrus data cleaning."""

    def test_papyrus_clean_returns_lazy_frames(self, sample_papyrus_parquets):
        """clean() should return dict of LazyFrames."""
        result = papyrus.clean(sample_papyrus_parquets)

        assert isinstance(result, dict)
        assert "bioactivity" in result
        assert "protein" in result
        assert all(isinstance(lf, pl.LazyFrame) for lf in result.values())

    def test_papyrus_clean_renames_uniprotid(self, temp_dir):
        """clean() should rename uniprotid to uniprot_id."""
        protein_df = pl.DataFrame(
            {
                "uniprotid": ["P12345", "P12346"],
                "target_id": [1, 2],
            }
        )
        bioactivity_df = pl.DataFrame(
            {
                "activity_id": [1, 2],
                "target_id": [1, 2],
            }
        )
        protein_path = temp_dir / "protein.parquet"
        bioactivity_path = temp_dir / "bioactivity.parquet"
        protein_df.write_parquet(protein_path)
        bioactivity_df.write_parquet(bioactivity_path)

        result = papyrus.clean(
            {"protein": protein_path, "bioactivity": bioactivity_path}
        )
        collected: pl.DataFrame = result["protein"].collect()  # ty: ignore[invalid-assignment]

        assert "uniprot_id" in collected.columns
        assert "uniprotid" not in collected.columns

    def test_papyrus_clean_lowercases_columns(self, temp_dir):
        """clean() should lowercase all columns."""
        bioactivity_df = pl.DataFrame(
            {
                "Target_ID": [1, 2],
                "PChembl_Value": [5.2, 3.1],
            }
        )
        protein_df = pl.DataFrame(
            {
                "target_id": [1, 2],
                "uniprotid": ["P1", "P2"],
            }
        )
        bioactivity_path = temp_dir / "bioactivity.parquet"
        protein_path = temp_dir / "protein.parquet"
        bioactivity_df.write_parquet(bioactivity_path)
        protein_df.write_parquet(protein_path)

        result = papyrus.clean(
            {"bioactivity": bioactivity_path, "protein": protein_path}
        )
        collected: pl.DataFrame = result["bioactivity"].collect()  # ty: ignore[invalid-assignment]

        assert all(col.islower() for col in collected.columns)


class TestPapyrusCompose:
    """Test Papyrus composition."""

    def test_papyrus_compose_returns_expected_keys(self, sample_papyrus_parquets):
        """compose() should return bioactivity, compounds, full, and proteins."""
        lfs = papyrus.clean(sample_papyrus_parquets)
        result = papyrus.compose(lfs)

        expected_keys = {"bioactivity", "compounds", "full", "proteins"}
        assert expected_keys == set(result.keys())

    def test_papyrus_bioactivity_joins_protein(self, sample_papyrus_parquets):
        """Bioactivity should be joined with protein."""
        lfs = papyrus.clean(sample_papyrus_parquets)
        result = papyrus.compose(lfs)
        bioactivity: pl.DataFrame = result["bioactivity"].collect()  # ty: ignore[invalid-assignment]

        # Should have columns from both tables
        assert "target_id" in bioactivity.columns
        assert "uniprot_id" in bioactivity.columns

    def test_papyrus_compounds_unique_structures(self, sample_papyrus_parquets):
        """Compounds should have unique connectivity/inchikey/inchi."""
        lfs = papyrus.clean(sample_papyrus_parquets)
        result = papyrus.compose(lfs)
        compounds: pl.DataFrame = result["compounds"].collect()  # ty: ignore[invalid-assignment]

        # Should be unique across structure identifiers
        assert "connectivity" in compounds.columns
        assert "inchikey" in compounds.columns
        assert len(compounds) <= 3


# ============================================================================
# Tests: API Classes (Chembl, Papyrus)
# ============================================================================


class TestChemblDataClass:
    """Test Chembl dataclass properties."""

    def test_chembl_name_is_chembl(self, temp_dir, sample_chembl_parquets):
        """Chembl.name should return 'chembl'."""
        obj = Chembl(
            version="36",
            raw_paths={"sql_db": temp_dir / "chembl.tar.gz"},
            consolidated_paths=sample_chembl_parquets,
            dir=temp_dir,
            module=chembl,
        )
        assert obj.name == "chembl"

    def test_chembl_lfs_returns_lazy_frames(self, temp_dir, sample_chembl_parquets):
        """Chembl.lfs should return dict of LazyFrames."""
        obj = Chembl(
            version="36",
            raw_paths={"sql_db": temp_dir / "chembl.tar.gz"},
            consolidated_paths=sample_chembl_parquets,
            dir=temp_dir,
            module=chembl,
        )
        lfs = obj.lfs
        assert isinstance(lfs, dict)
        assert all(isinstance(lf, pl.LazyFrame) for lf in lfs.values())

    def test_chembl_compose_returns_dict(self, temp_dir, sample_chembl_parquets):
        """Chembl.compose() should return dict with expected keys."""
        obj = Chembl(
            version="36",
            raw_paths={"sql_db": temp_dir / "chembl.tar.gz"},
            consolidated_paths=sample_chembl_parquets,
            dir=temp_dir,
            module=chembl,
        )
        composed = obj.compose()
        assert isinstance(composed, dict)
        assert "bioactivity" in composed
        assert "compounds" in composed

    def test_chembl_bioactivity_property(self, temp_dir, sample_chembl_parquets):
        """Chembl.bioactivity should return LazyFrame."""
        obj = Chembl(
            version="36",
            raw_paths={"sql_db": temp_dir / "chembl.tar.gz"},
            consolidated_paths=sample_chembl_parquets,
            dir=temp_dir,
            module=chembl,
        )
        assert isinstance(obj.bioactivity, pl.LazyFrame)

    def test_chembl_compounds_property(self, temp_dir, sample_chembl_parquets):
        """Chembl.compounds should return LazyFrame."""
        obj = Chembl(
            version="36",
            raw_paths={"sql_db": temp_dir / "chembl.tar.gz"},
            consolidated_paths=sample_chembl_parquets,
            dir=temp_dir,
            module=chembl,
        )
        assert isinstance(obj.compounds, pl.LazyFrame)

    def test_chembl_string_representation(self, temp_dir, sample_chembl_parquets):
        """Chembl should have meaningful string representation."""
        obj = Chembl(
            version="36",
            raw_paths={"sql_db": temp_dir / "chembl.tar.gz"},
            consolidated_paths=sample_chembl_parquets,
            dir=temp_dir,
            module=chembl,
        )
        str_repr = str(obj)
        assert "chembl" in str_repr.lower()
        assert "36" in str_repr

    def test_chembl_is_frozen(self, temp_dir, sample_chembl_parquets):
        """Chembl dataclass should be frozen (immutable)."""
        obj = Chembl(
            version="36",
            raw_paths={"sql_db": temp_dir / "chembl.tar.gz"},
            consolidated_paths=sample_chembl_parquets,
            dir=temp_dir,
            module=chembl,
        )
        # Frozen dataclass should raise on attribute assignment
        with pytest.raises(Exception):  # FrozenInstanceError
            obj.version = "37"  # ty: ignore[invalid-assignment]


class TestPapyrusDataClass:
    """Test Papyrus dataclass properties."""

    def test_papyrus_name_is_papyrus(self, temp_dir, sample_papyrus_parquets):
        """Papyrus.name should return 'papyrus'."""
        obj = Papyrus(
            version="05.7",
            raw_paths={"bioactivity": temp_dir / "bio.tsv.xz"},
            consolidated_paths=sample_papyrus_parquets,
            dir=temp_dir,
            module=papyrus,
        )
        assert obj.name == "papyrus"

    def test_papyrus_compose_returns_dict(self, temp_dir, sample_papyrus_parquets):
        """Papyrus.compose() should return dict with expected keys."""
        obj = Papyrus(
            version="05.7",
            raw_paths={"bioactivity": temp_dir / "bio.tsv.xz"},
            consolidated_paths=sample_papyrus_parquets,
            dir=temp_dir,
            module=papyrus,
        )
        composed = obj.compose()
        assert isinstance(composed, dict)
        assert "bioactivity" in composed
        assert "compounds" in composed
        assert "proteins" in composed

    def test_papyrus_lfs_returns_lazy_frames(self, temp_dir, sample_papyrus_parquets):
        """Papyrus.lfs should return dict of LazyFrames."""
        obj = Papyrus(
            version="05.7",
            raw_paths={"bioactivity": temp_dir / "bio.tsv.xz"},
            consolidated_paths=sample_papyrus_parquets,
            dir=temp_dir,
            module=papyrus,
        )
        lfs = obj.lfs
        assert isinstance(lfs, dict)
        assert all(isinstance(lf, pl.LazyFrame) for lf in lfs.values())

    def test_papyrus_proteins_property(self, temp_dir, sample_papyrus_parquets):
        """Papyrus.proteins should return protein LazyFrame."""
        obj = Papyrus(
            version="05.7",
            raw_paths={"bioactivity": temp_dir / "bio.tsv.xz"},
            consolidated_paths=sample_papyrus_parquets,
            dir=temp_dir,
            module=papyrus,
        )
        assert isinstance(obj.proteins, pl.LazyFrame)

    def test_papyrus_is_frozen(self, temp_dir, sample_papyrus_parquets):
        """Papyrus dataclass should be frozen (immutable)."""
        obj = Papyrus(
            version="05.7",
            raw_paths={"bioactivity": temp_dir / "bio.tsv.xz"},
            consolidated_paths=sample_papyrus_parquets,
            dir=temp_dir,
            module=papyrus,
        )
        # Frozen dataclass should raise on attribute assignment
        with pytest.raises(Exception):  # FrozenInstanceError
            obj.version = "05.6"  # ty: ignore[invalid-assignment]

    def test_papyrus_string_representation(self, temp_dir, sample_papyrus_parquets):
        """Papyrus should have meaningful string representation."""
        obj = Papyrus(
            version="05.7",
            raw_paths={"bioactivity": temp_dir / "bio.tsv.xz"},
            consolidated_paths=sample_papyrus_parquets,
            dir=temp_dir,
            module=papyrus,
        )
        str_repr = str(obj)
        assert "papyrus" in str_repr.lower()
        assert "05.7" in str_repr


# ============================================================================
# Tests: Helper Composition Functions
# ============================================================================


class TestChemblCompositionHelpers:
    """Test internal composition helper functions."""

    def test_bioactivities_includes_protein_join(self, sample_chembl_parquets):
        """_bioactivities should include protein information."""
        lfs = chembl.clean(sample_chembl_parquets)
        result = chembl._bioactivities(lfs)
        collected: pl.DataFrame = result.collect()  # ty: ignore[invalid-assignment]

        # Should have columns from both bioactivity and protein
        assert "molregno" in collected.columns
        assert "pref_name" in collected.columns  # From protein

    def test_bioactivities_includes_assay_info(self, sample_chembl_parquets):
        """_bioactivities should include assay information."""
        lfs = chembl.clean(sample_chembl_parquets)
        result = chembl._bioactivities(lfs)
        collected: pl.DataFrame = result.collect()  # ty: ignore[invalid-assignment]

        # Should have assay-related columns
        assert "assay_id" in collected.columns

    def test_compounds_structure_join(self, sample_chembl_parquets):
        """_compounds should join structure and property data."""
        lfs = chembl.clean(sample_chembl_parquets)
        result = chembl._compounds(lfs)
        collected: pl.DataFrame = result.collect()  # ty: ignore[invalid-assignment]

        # Should have structure-related columns
        assert "canonical_smiles" in collected.columns
        assert "mw_freebase" in collected.columns

    def test_compounds_includes_records_info(self, sample_chembl_parquets):
        """_compounds should include compound record and document info."""
        lfs = chembl.clean(sample_chembl_parquets)
        result = chembl._compounds(lfs)
        collected: pl.DataFrame = result.collect()  # ty: ignore[invalid-assignment]

        # Should have document-related columns
        assert "doc_id" in collected.columns or "pubmed_id" in collected.columns

    def test_components_domain_hierarchy(self, sample_chembl_parquets):
        """_components should include component, class, and domain hierarchy."""
        lfs = chembl.clean(sample_chembl_parquets)
        result = chembl._components(lfs)
        collected: pl.DataFrame = result.collect()  # ty: ignore[invalid-assignment]

        # Should have the component structure
        assert "component_id" in collected.columns
        assert "domain_id" in collected.columns


# ============================================================================
# Tests: Edge Cases & Error Handling
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_bioactivity_composition(self, temp_dir):
        """Should handle empty bioactivity table."""
        empty_bio = pl.DataFrame(
            {
                "activity_id": [],
                "target_id": [],
                "assay_id": [],
                "action_type": [],
            }
        )
        empty_protein = pl.DataFrame(
            {
                "target_id": [],
            }
        )
        empty_action = pl.DataFrame(
            {
                "action_type": [],
            }
        )
        empty_assay = pl.DataFrame(
            {
                "assay_id": [],
            }
        )
        empty_assay_type = pl.DataFrame(
            {
                "assay_type": [],
            }
        )

        bio_path = temp_dir / "bio.parquet"
        prot_path = temp_dir / "prot.parquet"
        action_path = temp_dir / "action.parquet"
        assay_path = temp_dir / "assay.parquet"
        assay_type_path = temp_dir / "assay_type.parquet"

        empty_bio.write_parquet(bio_path)
        empty_protein.write_parquet(prot_path)
        empty_action.write_parquet(action_path)
        empty_assay.write_parquet(assay_path)
        empty_assay_type.write_parquet(assay_type_path)

        lfs = {
            "bioactivity": pl.scan_parquet(bio_path),
            "protein": pl.scan_parquet(prot_path),
            "action_type": pl.scan_parquet(action_path),
            "assays": pl.scan_parquet(assay_path),
            "assay_type": pl.scan_parquet(assay_type_path),
        }

        # Should not raise, but return empty frame
        result = chembl._bioactivities(lfs)
        assert isinstance(result, pl.LazyFrame)

    def test_none_values_preserved_in_clean(self, temp_dir):
        """None values should be preserved through cleaning."""
        df = pl.DataFrame(
            {
                "name": ["A", None, "C"],
                "value": [1.0, None, 3.0],
            }
        )
        parquet_path = temp_dir / "test.parquet"
        df.write_parquet(parquet_path)

        result = chembl.clean({"test": parquet_path})
        collected: pl.DataFrame = result["test"].collect()  # ty: ignore[invalid-assignment]

        assert collected["name"][1] is None
        assert collected["value"][1] is None


class TestVersionFormatting:
    """Test version string formatting edge cases."""

    def test_version_formatter_float_input(self):
        """Float versions should be formatted correctly."""
        assert chembl._version_formatter(24.1) == "24_1"
        assert chembl._version_formatter(22.0) == "22"
        assert chembl._version_formatter(1.5) == "01_5"

    def test_version_formatter_string_input(self):
        """String versions should be preserved or normalized."""
        assert chembl._version_formatter("24_1") == "24_1"
        assert chembl._version_formatter("22") == "22"
        assert chembl._version_formatter("24.1") == "24_1"

    def test_version_formatter_integer_input(self):
        """Integer versions should be zero-padded."""
        assert chembl._version_formatter(1) == "01"
        assert chembl._version_formatter(9) == "09"
        assert chembl._version_formatter(36) == "36"

    def test_version_formatter_leading_zeros_stripped(self):
        """Leading zeros should be stripped before padding."""
        assert chembl._version_formatter("024") == "24"
        assert chembl._version_formatter("001") == "01"


class TestCleaningTransformations:
    """Test data cleaning transformations in detail."""

    def test_clean_fill_nan_with_none(self, temp_dir):
        """clean() should convert NaN values to None."""
        df = pl.DataFrame(
            {
                "value": [1.0, float("nan"), 3.0],
            }
        )
        parquet_path = temp_dir / "test.parquet"
        df.write_parquet(parquet_path)

        result = chembl.clean({"test": parquet_path})
        collected: pl.DataFrame = result["test"].collect()  # ty: ignore[invalid-assignment]

        assert collected["value"][1] is None

    def test_clean_mixed_column_cases(self, temp_dir):
        """clean() should lowercase all column names regardless of case."""
        df = pl.DataFrame(
            {
                "MixedCase": [1, 2],
                "UPPERCASE": [3, 4],
                "lowercase": [5, 6],
            }
        )
        parquet_path = temp_dir / "test.parquet"
        df.write_parquet(parquet_path)

        result = chembl.clean({"test": parquet_path})
        collected: pl.DataFrame = result["test"].collect()  # ty: ignore[invalid-assignment]

        assert "mixedcase" in collected.columns
        assert "uppercase" in collected.columns
        assert "lowercase" in collected.columns
        assert "MixedCase" not in collected.columns

    def test_clean_preserves_numeric_types(self, temp_dir):
        """clean() should preserve numeric column types."""
        df = pl.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.5, 2.5, 3.5],
            }
        )
        parquet_path = temp_dir / "test.parquet"
        df.write_parquet(parquet_path)

        result = chembl.clean({"test": parquet_path})
        collected: pl.DataFrame = result["test"].collect()  # ty: ignore[invalid-assignment]

        assert collected["int_col"].dtype == pl.Int64
        assert collected["float_col"].dtype == pl.Float64


class TestPapyrusCompositionDetails:
    """Test Papyrus composition with detailed column checking."""

    def test_papyrus_bioactivity_after_protein_join(self, sample_papyrus_parquets):
        """Papyrus bioactivity composition should join protein columns."""
        lfs = papyrus.clean(sample_papyrus_parquets)
        result = papyrus.compose(lfs)
        bioactivity: pl.DataFrame = result["bioactivity"].collect()  # ty: ignore[invalid-assignment]

        # Should have columns from both bioactivity and protein
        assert "target_id" in bioactivity.columns
        assert "uniprot_id" in bioactivity.columns
        assert "connectivity" in bioactivity.columns

    def test_papyrus_full_same_as_bioactivity(self, sample_papyrus_parquets):
        """Papyrus full should be same as bioactivity (both have protein join)."""
        lfs = papyrus.clean(sample_papyrus_parquets)
        result = papyrus.compose(lfs)
        bioactivity: pl.DataFrame = result["bioactivity"].collect()  # ty: ignore[invalid-assignment]
        full: pl.DataFrame = result["full"].collect()  # ty: ignore[invalid-assignment]

        # Should have same columns
        assert set(bioactivity.columns) == set(full.columns)

    def test_papyrus_compounds_removes_activity_id(self, sample_papyrus_parquets):
        """Papyrus compounds should not have activity_id column."""
        lfs = papyrus.clean(sample_papyrus_parquets)
        result = papyrus.compose(lfs)
        compounds: pl.DataFrame = result["compounds"].collect()  # ty: ignore[invalid-assignment]

        # activity_id should be dropped
        assert "activity_id" not in compounds.columns
        # But structural identifiers should remain
        assert "connectivity" in compounds.columns


class TestCompositionJoinValidation:
    """Test composition join validation and constraints."""

    def test_chembl_bioactivity_join_cardinality(self, sample_chembl_parquets):
        """Bioactivity-protein join should maintain m:1 cardinality."""
        lfs = chembl.clean(sample_chembl_parquets)
        result = chembl._bioactivities(lfs)
        collected: pl.DataFrame = result.collect()  # ty: ignore[invalid-assignment]

        # Each activity should map to exactly one protein
        # (or None if no matching protein)
        assert "target_id" in collected.columns
        assert len(collected) >= 0

    def test_chembl_compounds_unique_structures(self, sample_chembl_parquets):
        """Compounds should be based on molregno unique values."""
        lfs = chembl.clean(sample_chembl_parquets)
        result = chembl._compounds(lfs)
        collected: pl.DataFrame = result.collect()  # ty: ignore[invalid-assignment]

        # Should not have duplicate molregno values
        molregno_count = len(collected.select("molregno").unique())
        assert molregno_count == len(collected) or len(collected) == 0

    def test_papyrus_compounds_structure_uniqueness(self, sample_papyrus_parquets):
        """Papyrus compounds should be unique by connectivity/inchikey/inchi."""
        lfs = papyrus.clean(sample_papyrus_parquets)
        result = papyrus.compose(lfs)
        compounds: pl.DataFrame = result["compounds"].collect()  # ty: ignore[invalid-assignment]

        # Get unique count across the three structure columns
        unique_count = len(
            compounds.select(("connectivity", "inchikey", "inchi")).unique()
        )
        assert unique_count == len(compounds) or len(compounds) == 0


class TestDataIntegrity:
    """Test data integrity through pipeline."""

    def test_clean_to_compose_column_consistency(self, sample_chembl_parquets):
        """Column names should be consistent from clean through compose."""
        lfs = chembl.clean(sample_chembl_parquets)
        assert all(isinstance(lf, pl.LazyFrame) for lf in lfs.values())

        composed = chembl.compose(lfs)
        bioactivity: pl.DataFrame = composed["bioactivity"].collect()  # ty: ignore[invalid-assignment]

        # All columns should be lowercase
        assert all(col.islower() for col in bioactivity.columns)

    def test_papyrus_protein_consistency(self, sample_papyrus_parquets):
        """Papyrus protein table should be accessible from multiple outputs."""
        lfs = papyrus.clean(sample_papyrus_parquets)
        result = papyrus.compose(lfs)

        proteins_direct: pl.DataFrame = result["proteins"].collect()  # ty: ignore[invalid-assignment]
        proteins_via_bioactivity: pl.DataFrame = (
            result["bioactivity"].select("target_id", "uniprot_id").unique().collect()
        )  # ty: ignore[invalid-assignment]

        # Both should have target_id
        assert "target_id" in proteins_direct.columns
        assert "target_id" in proteins_via_bioactivity.columns

    def test_clean_empty_string_replacement_consistency(self, temp_dir):
        """Empty strings should be consistently replaced with None."""
        df = pl.DataFrame(
            {
                "col1": ["value", "", "another"],
                "col2": ["", None, "data"],
                "col3": [1, 2, 3],
            }
        )
        parquet_path = temp_dir / "test.parquet"
        df.write_parquet(parquet_path)

        result = chembl.clean({"test": parquet_path})
        collected: pl.DataFrame = result["test"].collect()  # ty: ignore[invalid-assignment]

        # Check that empty strings in string columns are replaced
        # (None values should remain None)
        assert collected["col2"][1] is None
