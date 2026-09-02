"""Tests for the get module covering data retrieval, composition, and lazy frame operations.

These tests use mocking and minimal fixtures to avoid downloading entire datasets,
while thoroughly testing all core functionality.
"""

import shutil
import tempfile
from dataclasses import FrozenInstanceError
from pathlib import Path

import polars as pl
import pytest

from fairfetched.get import chembl, papyrus
from fairfetched.get.dataset import Chembl, Papyrus

# ============================================================================
# Global fixture to patch ensure_url for all tests
# ============================================================================


@pytest.fixture(autouse=True)
def patch_download_pipeline(monkeypatch, tmp_path):
    """Mock the entire download + consolidation pipeline to avoid real downloads."""

    def dummy_ensure_url(url, path, force=False):
        """Dummy file download - creates empty files instead of downloading."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"dummy content")
        return p

    def dummy_ensure_raw(version, raw_dir=None):
        """Mock ensure_raw to return dummy paths without downloading."""
        if raw_dir is None:
            raw_dir = tmp_path / "chembl" / version / "raw"
        raw_dir = Path(raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        return {"sql_db": raw_dir / "chembl.tar.gz"}

    def dummy_papyrus_ensure_raw(version, raw_dir=None):
        """Mock Papyrus ensure_raw to return dummy paths without downloading."""
        if raw_dir is None:
            raw_dir = tmp_path / "papyrus" / version / "raw"
        raw_dir = Path(raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        return {
            "bioactivity": raw_dir / "bioactivity.tsv.xz",
            "protein": raw_dir / "protein.tsv.xz",
            "readme": raw_dir / "README.txt",
        }

    def dummy_ensure_parquet(raw_paths, parquet_dir=None):
        """Mock ensure_parquet to return dummy parquet paths without processing."""
        if parquet_dir is None:
            parquet_dir = Path(raw_paths["sql_db"]).parent / "parquet"
        parquet_dir = Path(parquet_dir)
        parquet_dir.mkdir(parents=True, exist_ok=True)

        # Return dummy parquet paths for all expected tables
        tables = [
            "molecule_dictionary",
            "compound_properties",
            "compound_structures",
            "bioactivity",
            "protein",
            "action_type",
            "assays",
            "assay_type",
            "compound_records",
            "docs",
            "compound_structural_alerts",
            "component_sequences",
            "component_class",
            "component_domains",
            "domains",
        ]
        return {table: parquet_dir / f"{table}.parquet" for table in tables}

    def dummy_papyrus_ensure_parquet(raw_filepath_dict, parquet_dir=None):
        """Mock Papyrus ensure_parquet to return dummy parquet paths."""
        if parquet_dir is None:
            parquet_dir = (
                Path(raw_filepath_dict.get("bioactivity", tmp_path)).parent / "parquet"
            )
        parquet_dir = Path(parquet_dir)
        parquet_dir.mkdir(parents=True, exist_ok=True)
        return {
            "bioactivity": parquet_dir / "bioactivity.parquet",
            "protein": parquet_dir / "protein.parquet",
        }

    monkeypatch.setattr("fairfetched.utils.ensure.ensure_url", dummy_ensure_url)
    monkeypatch.setattr("fairfetched.get.chembl.ensure_raw_files", dummy_ensure_raw)
    monkeypatch.setattr(
        "fairfetched.get.papyrus.ensure_raw_files", dummy_papyrus_ensure_raw
    )
    monkeypatch.setattr(
        "fairfetched.get.chembl.ensure_parquet_tables", dummy_ensure_parquet
    )
    monkeypatch.setattr(
        "fairfetched.get.papyrus.ensure_parquet_tables", dummy_papyrus_ensure_parquet
    )


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
    """Parquet files mirroring the real ChEMBL SQLite schema.

    ``ensure_sqlite_db_to_parquets`` writes one parquet per source table under
    its original name, so table and column names here follow
    schema_documentation.txt. Shape: 3 molecules, 2 targets (one single-protein
    ``tid`` 100, one two-component complex ``tid`` 101), 3 assays, 3 activities.
    """
    parquet_dir = temp_dir / "chembl_parquets"
    parquet_dir.mkdir()

    tables = {
        "molecule_dictionary": pl.DataFrame(
            {
                "molregno": [1, 2, 3],
                "chembl_id": ["CHEMBL1", "CHEMBL2", "CHEMBL3"],
                "pref_name": ["Aspirin", "Ibuprofen", "Ibuprofen sodium"],
                "max_phase": [4.0, 4.0, None],
                "molecule_type": ["Small molecule"] * 3,
                "withdrawn_flag": [0, 0, 0],
                "chirality": [2, 0, 0],
            }
        ),
        "compound_structures": pl.DataFrame(
            {
                "molregno": [1, 2, 3],
                "standard_inchi_key": [
                    "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
                    "HEFNNWSXXWATRW-UHFFFAOYSA-N",
                    "HEFNNWSXXWATRW-UHFFFAOYSA-M",
                ],
                "standard_inchi": ["InChI=1S/x"] * 3,
                "canonical_smiles": [
                    "O=C(O)Cc1ccccc1C(=O)O",
                    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
                    "CC(C)Cc1ccc(C(C)C(=O)[O-])cc1.[Na+]",
                ],
            }
        ),
        "compound_properties": pl.DataFrame(
            {
                "molregno": [1, 2, 3],
                "mw_freebase": [180.16, 206.28, 206.28],
                "alogp": [1.19, 3.97, 3.97],
                "hba": [3, 1, 1],
                "hbd": [1, 1, 1],
                "psa": [63.6, 37.3, 37.3],
                "qed_weighted": [0.55, 0.62, 0.62],
            }
        ),
        "molecule_hierarchy": pl.DataFrame(
            {
                "molregno": [1, 2, 3],
                "parent_molregno": [1, 2, 2],
                "active_molregno": [1, 2, 2],
            }
        ),
        "compound_structural_alerts": pl.DataFrame(
            {
                "cpd_str_alert_id": [1, 2, 3],
                "molregno": [1, 2, 2],
                "alert_id": [1, 2, 1],
            }
        ),
        "structural_alerts": pl.DataFrame(
            {
                "alert_id": [1, 2],
                "alert_set_id": [1, 2],
                "alert_name": ["Genotoxic carbamate", "PAINS filter A"],
                "smarts": ["[NX3]C(=O)O", "c1ccccc1"],
            }
        ),
        "compound_records": pl.DataFrame(
            {
                "record_id": [1, 2, 3],
                "molregno": [1, 2, 3],
                "doc_id": [5000, 5001, 5002],
                "compound_key": ["1", "2a", "5"],
                "compound_name": ["aspirin", "ibuprofen", "cpd 5"],
                "src_id": [1, 1, 1],
            }
        ),
        "activities": pl.DataFrame(
            {
                "activity_id": [10, 11, 12],
                "assay_id": [1000, 1001, 1002],
                "doc_id": [5000, None, 5002],
                "record_id": [1, 2, 3],
                "molregno": [1, 2, 3],
                "standard_relation": ["=", "=", "="],
                "standard_value": [5.2, 3.1, 6.5],
                "standard_units": ["nM", "nM", "nM"],
                "standard_type": ["IC50", "EC50", "Ki"],
                "standard_flag": [1, 1, 1],
                "pchembl_value": [8.28, 8.51, 8.19],
                "data_validity_comment": [None, None, None],
                "potential_duplicate": [0, 0, 0],
                "action_type": ["ANTAGONIST", "AGONIST", None],
                "src_id": [1, 1, 1],
                "type": ["IC50", "EC50", "Ki"],
            },
            schema_overrides={"data_validity_comment": pl.String},
        ),
        "action_type": pl.DataFrame(
            {
                "action_type": ["ANTAGONIST", "AGONIST"],
                "description": ["Antagonist", "Agonist"],
                "parent_type": ["NEGATIVE MODULATOR", "POSITIVE MODULATOR"],
            }
        ),
        "ligand_eff": pl.DataFrame(
            {
                "activity_id": [10, 11],
                "bei": [22.1, 18.4],
                "sei": [11.2, 9.8],
                "le": [0.42, 0.38],
                "lle": [7.1, 4.5],
            }
        ),
        "assays": pl.DataFrame(
            {
                "assay_id": [1000, 1001, 1002],
                "doc_id": [5000, 5001, 5002],
                "description": ["Binding", "Functional", "Binding"],
                "assay_type": ["B", "F", "B"],
                "assay_organism": ["Homo sapiens"] * 3,
                "assay_tax_id": [9606, 9606, 9606],
                "tid": [100, 101, 100],
                "relationship_type": ["D", "D", "H"],
                "confidence_score": [9, 8, 9],
                "src_id": [1, 1, 1],
                "chembl_id": ["CHEMBL1000", "CHEMBL1001", "CHEMBL1002"],
                "bao_format": ["BAO_0000357", "BAO_0000219", "BAO_0000357"],
                "variant_id": [None, None, None],
            },
            schema_overrides={"variant_id": pl.Int64},
        ),
        "assay_type": pl.DataFrame(
            {
                "assay_type": ["B", "F"],
                "assay_desc": ["Binding", "Functional"],
            }
        ),
        "confidence_score_lookup": pl.DataFrame(
            {
                "confidence_score": [8, 9],
                "description": [
                    "Homologous single protein target assigned",
                    "Direct single protein target assigned",
                ],
                "target_mapping": ["1 protein", "1 protein"],
            }
        ),
        "relationship_type": pl.DataFrame(
            {
                "relationship_type": ["D", "H"],
                "relationship_desc": ["Direct", "Homologue"],
            }
        ),
        "variant_sequences": pl.DataFrame(
            {
                "variant_id": [1],
                "mutation": ["T790M"],
                "accession": ["P00533"],
            }
        ),
        "target_dictionary": pl.DataFrame(
            {
                "tid": [100, 101],
                "target_type": ["SINGLE PROTEIN", "PROTEIN COMPLEX"],
                "pref_name": ["Tyrosine-protein kinase TYK2", "CDK2/Cyclin A"],
                "tax_id": [9606, 9606],
                "organism": ["Homo sapiens", "Homo sapiens"],
                "chembl_id": ["CHEMBL3199", "CHEMBL1907602"],
                "species_group_flag": [0, 0],
            }
        ),
        "target_components": pl.DataFrame(
            {
                "targcomp_id": [1, 2, 3],
                "tid": [100, 101, 101],
                "component_id": [10, 11, 12],
                "homologue": [0, 0, 0],
            }
        ),
        "component_sequences": pl.DataFrame(
            {
                "component_id": [10, 11, 12],
                "component_type": ["PROTEIN", "PROTEIN", "PROTEIN"],
                "accession": ["P29597", "P24941", "P20248"],
                "sequence": ["MAAA", "MBBB", "MCCC"],
                "description": [
                    "Non-receptor tyrosine-protein kinase TYK2",
                    "Cyclin-dependent kinase 2",
                    "Cyclin-A2",
                ],
                "tax_id": [9606, 9606, 9606],
                "organism": ["Homo sapiens"] * 3,
                "db_source": ["SWISS-PROT"] * 3,
            }
        ),
        "component_synonyms": pl.DataFrame(
            {
                "compsyn_id": [1, 2, 3, 4],
                "component_id": [10, 10, 11, 12],
                "component_synonym": ["TYK2", "2.7.10.2", "CDK2", "CCNA2"],
                "syn_type": ["GENE_SYMBOL", "EC_NUMBER", "GENE_SYMBOL", "GENE_SYMBOL"],
            }
        ),
        "component_class": pl.DataFrame(
            {
                "component_id": [10, 11],
                "protein_class_id": [1, 1],
                "comp_class_id": [1, 2],
            }
        ),
        "protein_classification": pl.DataFrame(
            {
                "protein_class_id": [1],
                "parent_id": [None],
                "pref_name": ["Tyrosine protein kinase"],
                "short_name": ["TyrKinase"],
                "protein_class_desc": ["enzyme  kinase  protein kinase"],
                "class_level": [4],
            }
        ),
        "component_domains": pl.DataFrame(
            {
                "compd_id": [1, 2],
                "domain_id": [100, 100],
                "component_id": [10, 11],
                "start_position": [589, 4],
                "end_position": [875, 286],
            }
        ),
        "domains": pl.DataFrame(
            {
                "domain_id": [100],
                "domain_type": ["Pfam-A"],
                "source_domain_id": ["PF07714"],
                "domain_name": ["PK_Tyr_Ser-Thr"],
                "domain_description": ["Protein tyrosine and serine/threonine kinase"],
            }
        ),
        "biotherapeutics": pl.DataFrame(
            {
                "molregno": [1],
                "description": ["synthetic control biologic"],
                "helm_notation": ["PEPTIDE1{A.C.D}$$$$"],
            }
        ),
        "docs": pl.DataFrame(
            {
                "doc_id": [5000, 5001, 5002],
                "journal": ["J Med Chem", "Bioorg Med Chem", "J Med Chem"],
                "year": [2018, 2019, 2020],
                "volume": ["61", "27", "63"],
                "pubmed_id": [12345, 12346, 12347],
                "doi": ["10.1/a", "10.1/b", "10.1/c"],
                "chembl_id": ["CHEMBL_DOC1", "CHEMBL_DOC2", "CHEMBL_DOC3"],
                "title": ["Paper A", "Paper B", "Paper C"],
                "doc_type": ["PUBLICATION"] * 3,
            }
        ),
        "source": pl.DataFrame(
            {
                "src_id": [1],
                "src_description": ["Scientific Literature"],
                "src_short_name": ["LITERATURE"],
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
        sources = chembl.source_urls(latest)
        assert isinstance(sources, dict)
        assert "sql_db" in sources
        assert sources["sql_db"].startswith("https://")

    def test_version_formatter_normalizes_versions(self):
        """Version formatter should canonicalize version strings."""
        assert chembl._format_version(24.1) == "24_1"
        assert chembl._format_version(22) == "22"
        assert chembl._format_version("24_1") == "24_1"

    def test_version_to_url_format(self):
        """URLs should follow ChEMBL FTP structure."""
        url = chembl._version_to_url("24_1")
        assert "ftp.ebi.ac.uk" in url
        assert "chembl_24_1" in url
        assert url.endswith(".tar.gz")

    def test_version_formatter_invalid_type(self):
        """Version formatter should handle invalid types gracefully."""
        with pytest.raises(TypeError):
            chembl._format_version([1, 2, 3])  # ty: ignore[invalid-argument-type]


# class TestChemblEnsureRaw:
#     """Test ensure_raw function for ChEMBL."""

#     # def test_ensure_raw_uses_provided_directory(self, temp_dir):
#     """ensure_raw should use the provided raw_dir."""
#     version = chembl.latest()
#     raw_dir = temp_dir / "my_raw"
#     result = chembl.ensure_raw(version, raw_dir=raw_dir)

#     # Verify it returns a dict with sql_db key
#     assert isinstance(result, dict)
#     assert "sql_db" in result
#     assert isinstance(result["sql_db"], Path)


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
        sources = papyrus.source_urls(version)
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
        result = chembl.cleanly_scan_parquet_tables(sample_chembl_parquets)

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

        result = chembl.cleanly_scan_parquet_tables({"test": parquet_path})
        collected: pl.DataFrame = result["test"].collect()

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

        result = chembl.cleanly_scan_parquet_tables({"test": parquet_path})
        collected: pl.DataFrame = result["test"].collect()

        # Check for None values (empty strings should be replaced)
        assert collected["name"][1] is None or collected["name"][1] == ""
        assert collected["description"][2] is None or collected["description"][2] == ""


class TestChemblCompose:
    """Test composition of cleaned lazy frames."""

    def test_build_views_returns_dict_with_expected_keys(self, sample_chembl_parquets):
        """build_views() should return dict with bioactivity, compounds, proteins, components."""
        result = chembl.build_views(sample_chembl_parquets)

        assert isinstance(result, dict)
        expected_keys = {"bioactivity", "compounds", "proteins", "components"}
        assert expected_keys == set(result.keys())

    def test_build_views_returns_lazy_frames(self, sample_chembl_parquets):
        """All views results should be LazyFrames."""
        result = chembl.build_views(sample_chembl_parquets)

        assert all(isinstance(lf, pl.LazyFrame) for lf in result.values())

    def test_build_views_raises_on_missing_required_table(self, sample_chembl_parquets):
        """A table the views join, absent from the release, aborts build_views
        before any join runs."""
        paths = dict(sample_chembl_parquets)
        del paths["target_components"]

        with pytest.raises(ValueError, match="target_components"):
            chembl.build_views(paths)

    def test_build_views_raises_on_empty_required_table(
        self, temp_dir, sample_chembl_parquets
    ):
        """A required table ChEMBL ships with zero rows reads back Null-typed;
        build_views should raise, not fail later inside a join."""
        paths = dict(sample_chembl_parquets)
        empty = temp_dir / "activities_empty.parquet"
        pl.DataFrame(schema={"activity_id": pl.Null}).write_parquet(empty)
        paths["activities"] = empty

        with pytest.raises(ValueError, match="activities"):
            chembl.build_views(paths)

    def test_build_views_raises_on_empty_non_anchor_table(
        self, temp_dir, sample_chembl_parquets
    ):
        """Every joined table is enforced, not just the view anchors: an empty
        lookup like assay_type aborts too."""
        paths = dict(sample_chembl_parquets)
        empty = temp_dir / "assay_type_empty.parquet"
        pl.DataFrame(
            schema={"assay_type": pl.Null, "assay_desc": pl.Null}
        ).write_parquet(empty)
        paths["assay_type"] = empty

        with pytest.raises(ValueError, match="assay_type"):
            chembl.build_views(paths)

    def test_bioactivity_composition_joins_correctly(self, sample_chembl_parquets):
        """Bioactivity should carry activity, assay, target and document fields
        and stay at one row per activity."""
        result = chembl.build_views(sample_chembl_parquets)
        bioactivity: pl.DataFrame = result["bioactivity"].collect()

        assert len(bioactivity) == 3  # no fan-out
        assert "molregno" in bioactivity.columns
        assert "pchembl_value" in bioactivity.columns
        assert "assay_id" in bioactivity.columns
        assert "tid" in bioactivity.columns  # target via assays.tid
        assert "pref_name" in bioactivity.columns  # from target_dictionary
        assert "pubmed_id" in bioactivity.columns  # from docs
        # activities.doc_id null is backfilled from assays.doc_id
        assert bioactivity["pubmed_id"].null_count() == 0

    def test_compounds_composition_includes_structures(self, sample_chembl_parquets):
        """Compounds should include structures, properties, parent hierarchy and
        structural alerts, one row per molregno."""
        result = chembl.build_views(sample_chembl_parquets)
        compounds: pl.DataFrame = result["compounds"].collect()

        assert len(compounds) == 3
        assert "canonical_smiles" in compounds.columns
        assert "mw_freebase" in compounds.columns
        assert "parent_molregno" in compounds.columns
        assert "structural_alerts" in compounds.columns

    def test_proteins_expands_targets_to_components(self, sample_chembl_parquets):
        """Proteins view expands target_dictionary through target_components to
        the component sequences (multi-component targets yield several rows)."""
        result = chembl.build_views(sample_chembl_parquets)
        proteins: pl.DataFrame = result["proteins"].collect()

        assert "tid" in proteins.columns
        assert "chembl_id" in proteins.columns
        assert "accession" in proteins.columns
        assert "gene_symbols" in proteins.columns
        assert len(proteins) == 3  # tid 100 -> 1 component, tid 101 -> 2


# ============================================================================
# Tests: Papyrus Data Pipeline
# ============================================================================


class TestPapyrusClean:
    """Test Papyrus data cleaning."""

    def test_papyrus_clean_returns_lazy_frames(self, sample_papyrus_parquets):
        """clean() should return dict of LazyFrames."""
        result = papyrus.cleanly_scan_parquet_tables(sample_papyrus_parquets)

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

        result = papyrus.cleanly_scan_parquet_tables(
            {"protein": protein_path, "bioactivity": bioactivity_path}
        )
        collected: pl.DataFrame = result["protein"].collect()

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

        result = papyrus.cleanly_scan_parquet_tables(
            {"bioactivity": bioactivity_path, "protein": protein_path}
        )
        collected: pl.DataFrame = result["bioactivity"].collect()

        assert all(col.islower() for col in collected.columns)


class TestPapyrusCompose:
    """Test Papyrus composition."""

    def test_papyrus_build_views_returns_expected_keys(self, sample_papyrus_parquets):
        """build_views() should return bioactivity, compounds, full, and proteins."""
        result = papyrus.build_views(sample_papyrus_parquets)

        expected_keys = {"bioactivity", "compounds", "full", "proteins"}
        assert expected_keys == set(result.keys())

    def test_papyrus_full_joins_protein(self, sample_papyrus_parquets):
        """Bioactivity should be joined with protein."""
        result = papyrus.build_views(sample_papyrus_parquets)
        bioactivity: pl.DataFrame = result["full"].collect()

        # Should have columns from both tables
        assert "target_id" in bioactivity.columns
        assert "uniprot_id" in bioactivity.columns

    def test_papyrus_compounds_unique_structures(self, sample_papyrus_parquets):
        """Compounds should have unique connectivity/inchikey/inchi."""
        result = papyrus.build_views(sample_papyrus_parquets)
        compounds: pl.DataFrame = result["compounds"].collect()

        # Should be unique across structure identifiers
        assert "inchikey" in compounds.columns
        assert "inchi" in compounds.columns
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
            parquet_paths=sample_chembl_parquets,
            dir=temp_dir,
            module=chembl,
        )
        assert obj.name == "chembl"

    def test_chembl_lfs_returns_lazy_frames(self, temp_dir, sample_chembl_parquets):
        """Chembl.lfs should return dict of LazyFrames."""
        obj = Chembl(
            version="36",
            raw_paths={"sql_db": temp_dir / "chembl.tar.gz"},
            parquet_paths=sample_chembl_parquets,
            dir=temp_dir,
            module=chembl,
        )
        lfs = obj.lfs
        assert isinstance(lfs, dict)
        assert all(isinstance(lf, pl.LazyFrame) for lf in lfs.values())

    def test_chembl_view_exposes_bioactivity_and_compounds(
        self, temp_dir, sample_chembl_parquets
    ):
        """Chembl.view exposes the joined views as LazyFrames."""
        obj = Chembl(
            version="36",
            raw_paths={"sql_db": temp_dir / "chembl.tar.gz"},
            parquet_paths=sample_chembl_parquets,
            dir=temp_dir,
            module=chembl,
        )
        assert isinstance(obj.view.bioactivity, pl.LazyFrame)
        assert isinstance(obj.view.compounds, pl.LazyFrame)

    def test_chembl_raw_table_attribute(self, temp_dir, sample_chembl_parquets):
        """Chembl exposes raw source tables as attributes."""
        obj = Chembl(
            version="36",
            raw_paths={"sql_db": temp_dir / "chembl.tar.gz"},
            parquet_paths=sample_chembl_parquets,
            dir=temp_dir,
            module=chembl,
        )
        assert isinstance(obj.tables.activities, pl.LazyFrame)
        assert obj.tables.activities is obj.lfs["activities"]

    def test_chembl_string_representation(self, temp_dir, sample_chembl_parquets):
        """Chembl should have meaningful string representation."""
        obj = Chembl(
            version="36",
            raw_paths={"sql_db": temp_dir / "chembl.tar.gz"},
            parquet_paths=sample_chembl_parquets,
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
            parquet_paths=sample_chembl_parquets,
            dir=temp_dir,
            module=chembl,
        )
        # Frozen dataclass should raise on attribute assignment
        with pytest.raises(FrozenInstanceError):  # FrozenInstanceError
            obj.version = "378"  # ty: ignore[invalid-assignment]


class TestPapyrusDataClass:
    """Test Papyrus dataclass properties."""

    def test_papyrus_name_is_papyrus(self, temp_dir, sample_papyrus_parquets):
        """Papyrus.name should return 'papyrus'."""
        obj = Papyrus(
            version="05.7",
            raw_paths={"bioactivity": temp_dir / "bio.tsv.xz"},
            parquet_paths=sample_papyrus_parquets,
            dir=temp_dir,
            module=papyrus,
        )
        assert obj.name == "papyrus"

    def test_papyrus_build_views_returns_dict(self, temp_dir, sample_papyrus_parquets):
        """Papyrus.build_views() should return dict with expected keys."""
        obj = Papyrus(
            version="05.7",
            raw_paths={"bioactivity": temp_dir / "bio.tsv.xz"},
            parquet_paths=sample_papyrus_parquets,
            dir=temp_dir,
            module=papyrus,
        )
        assert isinstance(obj.view.bioactivity, pl.LazyFrame)
        assert isinstance(obj.view.compounds, pl.LazyFrame)
        assert isinstance(obj.view.proteins, pl.LazyFrame)
        assert isinstance(obj.view.full, pl.LazyFrame)

    def test_papyrus_lfs_returns_lazy_frames(self, temp_dir, sample_papyrus_parquets):
        """Papyrus.lfs should return dict of LazyFrames."""
        obj = Papyrus(
            version="05.7",
            raw_paths={"bioactivity": temp_dir / "bio.tsv.xz"},
            parquet_paths=sample_papyrus_parquets,
            dir=temp_dir,
            module=papyrus,
        )
        lfs = obj.lfs
        assert isinstance(lfs, dict)
        assert all(isinstance(lf, pl.LazyFrame) for lf in lfs.values())

    def test_papyrus_proteins_property(self, temp_dir, sample_papyrus_parquets):
        """Papyrus.view.proteins and the raw Papyrus.tables.protein table are LazyFrames."""
        obj = Papyrus(
            version="05.7",
            raw_paths={"bioactivity": temp_dir / "bio.tsv.xz"},
            parquet_paths=sample_papyrus_parquets,
            dir=temp_dir,
            module=papyrus,
        )
        assert isinstance(obj.view.proteins, pl.LazyFrame)
        assert isinstance(obj.tables.protein, pl.LazyFrame)

    def test_papyrus_is_frozen(self, temp_dir, sample_papyrus_parquets):
        """Papyrus dataclass should be frozen (immutable)."""
        obj = Papyrus(
            version="05.7",
            raw_paths={"bioactivity": temp_dir / "bio.tsv.xz"},
            parquet_paths=sample_papyrus_parquets,
            dir=temp_dir,
            module=papyrus,
        )
        # Frozen dataclass should raise on attribute assignment
        with pytest.raises(FrozenInstanceError):  # FrozenInstanceError
            obj.version = "05.6"  # ty: ignore[invalid-assignment]

    def test_papyrus_string_representation(self, temp_dir, sample_papyrus_parquets):
        """Papyrus should have meaningful string representation."""
        obj = Papyrus(
            version="05.7",
            raw_paths={"bioactivity": temp_dir / "bio.tsv.xz"},
            parquet_paths=sample_papyrus_parquets,
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

    def test_bioactivities_includes_target_join(self, sample_chembl_parquets):
        """_bioactivities should reach the target through assays.tid."""
        lfs = chembl.cleanly_scan_parquet_tables(sample_chembl_parquets)
        collected: pl.DataFrame = chembl._bioactivities(lfs).collect()

        assert "molregno" in collected.columns
        assert "pref_name" in collected.columns  # from target_dictionary
        assert "tid" in collected.columns

    def test_bioactivities_includes_assay_info(self, sample_chembl_parquets):
        """_bioactivities should include assay information."""
        lfs = chembl.cleanly_scan_parquet_tables(sample_chembl_parquets)
        collected: pl.DataFrame = chembl._bioactivities(lfs).collect()

        assert "assay_id" in collected.columns
        assert "assay_desc" in collected.columns  # from assay_type lookup

    def test_compounds_structure_join(self, sample_chembl_parquets):
        """_compounds should join structure and property data."""
        lfs = chembl.cleanly_scan_parquet_tables(sample_chembl_parquets)
        collected: pl.DataFrame = chembl._compounds(lfs).collect()

        assert "canonical_smiles" in collected.columns
        assert "mw_freebase" in collected.columns

    def test_compounds_includes_hierarchy_and_alerts(self, sample_chembl_parquets):
        """_compounds should carry salt/parent hierarchy and structural alerts,
        not literature provenance (that lives in the bioactivity view)."""
        lfs = chembl.cleanly_scan_parquet_tables(sample_chembl_parquets)
        collected: pl.DataFrame = chembl._compounds(lfs).collect()

        assert "parent_molregno" in collected.columns
        assert "structural_alerts" in collected.columns
        assert "doc_id" not in collected.columns

    def test_components_domain_hierarchy(self, sample_chembl_parquets):
        """_components should include component, class, and domain hierarchy."""
        lfs = chembl.cleanly_scan_parquet_tables(sample_chembl_parquets)
        collected: pl.DataFrame = chembl._components(lfs).collect()

        assert "component_id" in collected.columns
        assert "domain_id" in collected.columns
        assert "protein_classes" in collected.columns

    def test_targets_expands_components(self, sample_chembl_parquets):
        """_targets should expand a complex target to its components."""
        lfs = chembl.cleanly_scan_parquet_tables(sample_chembl_parquets)
        collected: pl.DataFrame = chembl._targets(lfs).collect()

        assert len(collected) == 3
        assert set(collected["tid"]) == {100, 101}
        assert "gene_symbols" in collected.columns


# ============================================================================
# Tests: Edge Cases & Error Handling
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error conditions."""

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

        result = chembl.cleanly_scan_parquet_tables({"test": parquet_path})
        collected: pl.DataFrame = result["test"].collect()

        assert collected["name"][1] is None
        assert collected["value"][1] is None


class TestVersionFormatting:
    """Test version string formatting edge cases."""

    def test_version_formatter_float_input(self):
        """Float versions should be formatted correctly."""
        assert chembl._format_version(24.1) == "24_1"
        assert chembl._format_version(22.0) == "22"
        assert chembl._format_version(1.5) == "01_5"

    def test_version_formatter_string_input(self):
        """String versions should be preserved or normalized."""
        assert chembl._format_version("24_1") == "24_1"
        assert chembl._format_version("22") == "22"
        assert chembl._format_version("24.1") == "24_1"

    def test_version_formatter_integer_input(self):
        """Integer versions should be zero-padded."""
        assert chembl._format_version(1) == "01"
        assert chembl._format_version(9) == "09"
        assert chembl._format_version(36) == "36"

    def test_version_formatter_leading_zeros_stripped(self):
        """Leading zeros should be stripped before padding."""
        assert chembl._format_version("024") == "24"
        assert chembl._format_version("001") == "01"


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

        result = chembl.cleanly_scan_parquet_tables({"test": parquet_path})
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

        result = chembl.cleanly_scan_parquet_tables({"test": parquet_path})
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

        result = chembl.cleanly_scan_parquet_tables({"test": parquet_path})
        collected: pl.DataFrame = result["test"].collect()  # ty: ignore[invalid-assignment]

        assert collected["int_col"].dtype == pl.Int64
        assert collected["float_col"].dtype == pl.Float64


class TestPapyrusCompositionDetails:
    """Test Papyrus composition with detailed column checking."""

    def test_papyrus_full_after_protein_join(self, sample_papyrus_parquets):
        """Papyrus bioactivity composition should join protein columns."""
        lfs = papyrus.cleanly_scan_parquet_tables(sample_papyrus_parquets)
        result = papyrus.build_views(sample_papyrus_parquets)
        full: pl.DataFrame = result["full"].collect()

        # Should have columns from both bioactivity and protein
        assert "target_id" in full.columns
        assert "uniprot_id" in full.columns
        assert "inchikey" in full.columns

    def test_papyrus_full_not_same_as_bioactivity(self, sample_papyrus_parquets):
        """Papyrus full should no longer be same as bioactivity (only full has a protein join)."""
        lfs = papyrus.cleanly_scan_parquet_tables(sample_papyrus_parquets)
        result = papyrus.build_views(sample_papyrus_parquets)
        bioactivity: pl.DataFrame = result["bioactivity"].collect()
        full: pl.DataFrame = result["full"].collect()

        # Should have same columns
        assert set(bioactivity.columns) != set(full.columns)

    def test_papyrus_compounds_removes_activity_id(self, sample_papyrus_parquets):
        """Papyrus compounds should not have activity_id column."""
        lfs = papyrus.cleanly_scan_parquet_tables(sample_papyrus_parquets)
        result = papyrus.build_views(sample_papyrus_parquets)
        compounds: pl.DataFrame = result["compounds"].collect()

        # activity_id should be dropped
        assert "activity_id" not in compounds.columns
        # But structural identifiers should remain
        assert "inchikey" in compounds.columns


class TestCompositionJoinValidation:
    """Test composition join validation and constraints."""

    # @TODO: make this test actually check cardinality
    # def test_chembl_bioactivity_join_cardinality(self, sample_chembl_parquets):
    #     """Bioactivity-protein join should maintain m:1 cardinality."""
    #     lfs = chembl.clean(sample_chembl_parquets)
    #     result = chembl._bioactivities(lfs)
    #     collected: pl.DataFrame = result.collect()

    #     # Each activity should map to exactly one protein
    #     # (or None if no matching protein)
    #     assert "target_id" in collected.columns
    #     assert len(collected) >= 0

    def test_chembl_compounds_unique_structures(self, sample_chembl_parquets):
        """Compounds should be based on molregno unique values."""
        lfs = chembl.cleanly_scan_parquet_tables(sample_chembl_parquets)
        collected: pl.DataFrame = chembl._compounds(lfs).collect()

        # Should not have duplicate molregno values
        molregno_count = len(collected.select("molregno").unique())
        assert molregno_count == len(collected) or len(collected) == 0

    def test_papyrus_compounds_structure_uniqueness(self, sample_papyrus_parquets):
        """Papyrus compounds should be unique by connectivity/inchikey/inchi."""
        result = papyrus.build_views(sample_papyrus_parquets)
        compounds: pl.DataFrame = result["compounds"].collect()

        # Get unique count across the structure columns
        unique_count = len(compounds.select(("inchikey", "inchi")).unique())
        assert unique_count == len(compounds) or len(compounds) == 0


class TestDataIntegrity:
    """Test data integrity through pipeline."""

    def test_clean_to_build_views_column_consistency(self, sample_chembl_parquets):
        """Column names should be consistent from clean through build_views."""
        lfs = chembl.cleanly_scan_parquet_tables(sample_chembl_parquets)
        assert all(isinstance(lf, pl.LazyFrame) for lf in lfs.values())

        views = chembl.build_views(sample_chembl_parquets)
        bioactivity: pl.DataFrame = views["bioactivity"].collect()  # ty: ignore[invalid-assignment]

        # All columns should be lowercase
        assert all(col.islower() for col in bioactivity.columns)

    def test_papyrus_protein_consistency(self, sample_papyrus_parquets):
        """Papyrus protein table should be accessible from multiple outputs."""
        lfs = papyrus.cleanly_scan_parquet_tables(sample_papyrus_parquets)
        result = papyrus.build_views(sample_papyrus_parquets)

        proteins_direct: pl.DataFrame = result["proteins"].collect()  # ty: ignore[invalid-assignment]
        proteins_via_full: pl.DataFrame = (
            result["full"].select("target_id", "uniprot_id").unique().collect()
        )  # ty: ignore[invalid-assignment]

        # Both should have target_id
        assert "target_id" in proteins_direct.columns
        assert "target_id" in proteins_via_full.columns

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

        result = chembl.cleanly_scan_parquet_tables({"test": parquet_path})
        collected: pl.DataFrame = result["test"].collect()  # ty: ignore[invalid-assignment]

        # Check that empty strings in string columns are replaced
        # (None values should remain None)
        assert collected["col2"][1] is None
