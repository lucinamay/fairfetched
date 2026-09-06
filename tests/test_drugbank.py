"""DrugBank source: offline checks on register/pin, the streaming XML parse and
the joins. No download. The XML fixture is invented (DrugBank is licensed); it
only has to exercise the schema paths the parser walks."""

import gzip
import hashlib
import json
import zipfile
from pathlib import Path

import polars as pl
import pytest

from fairfetched.get import drugbank

# Two top-level drugs. Drug 2 carries a <pathways> block with a *nested* <drug>
# that must not be counted as a third drug.
_XML = """<?xml version="1.0" encoding="UTF-8"?>
<drugbank xmlns="http://www.drugbank.ca" version="5.1" exported-on="2025-01-01">
  <drug type="biotech">
    <drugbank-id primary="true">DB90001</drugbank-id>
    <drugbank-id>BTD90001</drugbank-id>
    <name>Fakezumab</name>
    <description>An invented biologic.</description>
    <cas-number>000-00-0</cas-number>
    <unii>FAKE123</unii>
    <state>liquid</state>
    <groups>
      <group>approved</group>
      <group>investigational</group>
    </groups>
    <synonyms>
      <synonym language="english">Fake-zumab</synonym>
      <synonym language="">FZB</synonym>
    </synonyms>
    <external-identifiers>
      <external-identifier>
        <resource>PubChem Compound</resource>
        <identifier>99999999</identifier>
      </external-identifier>
      <external-identifier>
        <resource>UniProtKB</resource>
        <identifier>P00000</identifier>
      </external-identifier>
    </external-identifiers>
    <calculated-properties>
      <property><kind>logP</kind><value>1.23</value><source>FakeALOGPS</source></property>
    </calculated-properties>
    <experimental-properties>
      <property><kind>Melting Point</kind><value>100 C</value><source>FakeRef</source></property>
    </experimental-properties>
    <atc-codes>
      <atc-code code="X01XX99">
        <level code="X01XX">Fake subgroup</level>
        <level code="X">FAKE ORGANS</level>
      </atc-code>
    </atc-codes>
    <categories>
      <category><category>Fake Agents</category><mesh-id>D000000</mesh-id></category>
    </categories>
    <targets>
      <target>
        <id>BE9000001</id>
        <name>Fake receptor</name>
        <organism>Humans</organism>
        <actions><action>inhibitor</action></actions>
        <known-action>yes</known-action>
        <polypeptide id="P00000" source="Swiss-Prot">
          <gene-name>FAKE1</gene-name>
        </polypeptide>
      </target>
    </targets>
    <enzymes>
      <enzyme>
        <id>BE9000002</id>
        <name>Fake enzyme</name>
        <organism>Humans</organism>
        <actions><action>substrate</action></actions>
        <polypeptide id="P11111"><gene-name>FAKE2</gene-name></polypeptide>
      </enzyme>
    </enzymes>
    <drug-interactions>
      <drug-interaction>
        <drugbank-id>DB90002</drugbank-id>
        <name>Placebix</name>
        <description>Risk increased.</description>
      </drug-interaction>
    </drug-interactions>
  </drug>
  <drug type="small molecule">
    <drugbank-id primary="true">DB90002</drugbank-id>
    <name>Placebix</name>
    <groups><group>approved</group></groups>
    <pathways>
      <pathway>
        <smpdb-id>SMP0000001</smpdb-id>
        <name>Fake pathway</name>
        <drugs>
          <drug>
            <drugbank-id>DB90001</drugbank-id>
            <name>Fakezumab</name>
          </drug>
        </drugs>
      </pathway>
    </pathways>
  </drug>
</drugbank>
"""


@pytest.fixture
def xml_file(tmp_path: Path) -> Path:
    p = tmp_path / "full database.xml"
    p.write_text(_XML)
    return p


@pytest.fixture
def raw_paths(tmp_path: Path, xml_file: Path) -> dict[str, Path]:
    return drugbank.register(xml_file, version="5.1.13", raw_dir=tmp_path / "raw")


@pytest.fixture
def parquet_paths(tmp_path: Path, raw_paths: dict[str, Path]) -> dict[str, Path]:
    return drugbank.ensure_parquet_tables(raw_paths, tmp_path / "parquet")


class TestRegister:
    def test_gzip_and_manifest_written(self, raw_paths):
        gz = raw_paths["full_database"]
        assert gz.suffix == ".gz"
        assert gzip.decompress(gz.read_bytes()) == _XML.encode()  # byte-identical
        m = json.loads((gz.parent / "_manifest.json").read_text())
        assert m["version"] == "5.1.13"
        assert m["files"]["full_database"]["sha256"] == hashlib.sha256(
            gz.read_bytes()
        ).hexdigest()

    def test_idempotent_and_verifies(self, tmp_path, xml_file):
        a = drugbank.register(xml_file, "5.1.13", tmp_path / "raw")
        b = drugbank.register(xml_file, "5.1.13", tmp_path / "raw")  # no force: verify path
        assert a == b

    def test_tampered_cache_raises(self, raw_paths):
        gz = raw_paths["full_database"]
        with gzip.open(gz, "wb") as fh:
            fh.write(b"<drugbank/>")
        with pytest.raises(ValueError, match="changed on disk"):
            drugbank.ensure_raw_files("5.1.13", raw_dir=gz.parent)

    def test_accepts_release_zip(self, tmp_path):
        z = tmp_path / "drugbank_all_full_database.xml.zip"
        with zipfile.ZipFile(z, "w") as zf:
            zf.writestr("full database.xml", _XML)
        out = drugbank.register(z, "5.1.13", tmp_path / "raw")
        assert gzip.decompress(out["full_database"].read_bytes()) == _XML.encode()

    def test_missing_cache_without_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not registered"):
            drugbank.ensure_raw_files("5.1.13", raw_dir=tmp_path / "raw")

    def test_version_read_from_xml_root(self, tmp_path, xml_file):
        assert drugbank.version_of(xml_file) == "5.1"  # <drugbank version="5.1">
        out = drugbank.register(xml_file, raw_dir=tmp_path / "raw")  # version=None
        m = json.loads((out["full_database"].parent / "_manifest.json").read_text())
        assert m["version"] == "5.1"

    def test_version_missing_from_xml_root_raises(self, tmp_path):
        p = tmp_path / "noversion.xml"
        p.write_text('<drugbank xmlns="http://www.drugbank.ca"></drugbank>')
        with pytest.raises(ValueError, match="no version attribute"):
            drugbank.version_of(p)


class TestParse:
    def test_all_tables_written(self, parquet_paths):
        assert set(parquet_paths) == set(drugbank._COLUMNS)
        assert all(p.exists() for p in parquet_paths.values())

    def test_nested_pathway_drug_not_counted(self, parquet_paths):
        drugs = pl.read_parquet(parquet_paths["drugs"])
        assert sorted(drugs["drugbank_id"]) == ["DB90001", "DB90002"]

    def test_primary_id_and_groups(self, parquet_paths):
        row = pl.read_parquet(parquet_paths["drugs"]).filter(drugbank_id="DB90001").row(0, named=True)
        assert row["name"] == "Fakezumab"
        assert row["type"] == "biotech"
        assert row["groups"] == "approved;investigational"

    def test_targets_carry_polypeptide_and_kind(self, parquet_paths):
        t = pl.read_parquet(parquet_paths["targets"])
        assert set(t["kind"]) == {"target", "enzyme"}
        recep = t.filter(target_id="BE9000001").row(0, named=True)
        assert recep["uniprot_id"] == "P00000"
        assert recep["gene_name"] == "FAKE1"
        assert recep["actions"] == "inhibitor"

    def test_properties_split_by_source_kind(self, parquet_paths):
        p = pl.read_parquet(parquet_paths["properties"])
        assert set(p["source_kind"]) == {"calculated", "experimental"}

    def test_external_identifiers(self, parquet_paths):
        x = pl.read_parquet(parquet_paths["external_identifiers"])
        assert dict(zip(x["resource"], x["identifier"])) == {
            "PubChem Compound": "99999999",
            "UniProtKB": "P00000",
        }

    def test_reparse_skipped_when_tables_exist(self, tmp_path, raw_paths):
        first = drugbank.ensure_parquet_tables(raw_paths, tmp_path / "pq")
        mtimes = {t: p.stat().st_mtime_ns for t, p in first.items()}
        again = drugbank.ensure_parquet_tables(raw_paths, tmp_path / "pq")
        assert {t: p.stat().st_mtime_ns for t, p in again.items()} == mtimes


class TestViews:
    def test_targets_view_carries_drug_name(self, parquet_paths):
        t = drugbank.build_views(parquet_paths)["targets"].collect()
        assert set(t.filter(pl.col("target_id") == "BE9000001")["name"]) == {"Fakezumab"}

    def test_interactions_view_carries_subject_name(self, parquet_paths):
        i = drugbank.build_views(parquet_paths)["interactions"].collect()
        assert i.row(0, named=True)["name"] == "Fakezumab"
        assert i.row(0, named=True)["interacts_with_id"] == "DB90002"

    def test_empty_string_nulled_on_scan(self, parquet_paths):
        # drug 2 has no description -> stored "", nulled on scan
        drugs = drugbank.cleanly_scan_parquet(parquet_paths["drugs"]).collect()
        assert drugs.filter(drugbank_id="DB90002").row(0, named=True)["description"] is None
