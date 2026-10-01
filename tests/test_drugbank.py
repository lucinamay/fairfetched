"""DrugBank source: offline checks on register/pin, the streaming XML parse and
the joins. No download. The XML fixture is invented (DrugBank is licensed); it
only has to exercise the schema paths the parser walks."""

import gzip
import hashlib
import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import polars as pl
import pytest

from fairfetched.get import drugbank

# Two top-level drugs. Drug 2 carries a <pathways> block with a *nested* <drug>
# that must not be counted as a third drug, shares bio-entity BE9000001 with drug 1
# under a different kind, and binds BE9000009 "DNA", which has no <polypeptide>.
# Drug 3 binds nothing at all, joins no pathway, and has an empty type="".
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
          <go-classifiers>
            <go-classifier><category>function</category><description>protein binding</description></go-classifier>
            <go-classifier><category>function</category><description>ATP binding</description></go-classifier>
            <go-classifier><category>process</category><description>signal transduction</description></go-classifier>
            <go-classifier><category>component</category><description>plasma membrane</description></go-classifier>
          </go-classifiers>
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
    <carriers>
      <carrier>
        <id>BE9000003</id>
        <name>Fake carrier</name>
        <organism>Humans</organism>
        <polypeptide id="P22222">
          <gene-name>FAKE3</gene-name>
          <go-classifiers>
            <go-classifier><category>function</category><description>lipid binding</description></go-classifier>
          </go-classifiers>
        </polypeptide>
      </carrier>
    </carriers>
    <salts>
      <salt>
        <drugbank-id primary="true">DBSALT90001</drugbank-id>
        <name>Fakezumab hydrochloride</name>
        <unii>SALT123</unii>
      </salt>
    </salts>
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
    <enzymes>
      <enzyme>
        <id>BE9000001</id>
        <name>Fake receptor</name>
        <organism>Humans</organism>
        <actions><action>substrate</action></actions>
        <polypeptide id="P00000" source="Swiss-Prot">
          <gene-name>FAKE1</gene-name>
          <go-classifiers>
            <go-classifier><category>function</category><description>protein binding</description></go-classifier>
            <go-classifier><category>function</category><description>ATP binding</description></go-classifier>
            <go-classifier><category>process</category><description>signal transduction</description></go-classifier>
            <go-classifier><category>component</category><description>plasma membrane</description></go-classifier>
          </go-classifiers>
        </polypeptide>
      </enzyme>
    </enzymes>
    <targets>
      <target>
        <id>BE9000009</id>
        <name>DNA</name>
        <organism>Humans</organism>
        <actions><action>intercalation</action></actions>
      </target>
    </targets>
    <drug-interactions>
      <drug-interaction>
        <drugbank-id>DB90001</drugbank-id>
        <name>Fakezumab</name>
        <description>Risk increased.</description>
      </drug-interaction>
    </drug-interactions>
    <pathways>
      <pathway>
        <smpdb-id>SMP0000001</smpdb-id>
        <name>Fake pathway</name>
        <drugs>
          <drug>
            <drugbank-id>DB90001</drugbank-id>
            <name>Fakezumab</name>
          </drug>
          <drug>
            <drugbank-id>DB90002</drugbank-id>
            <name>Placebix</name>
          </drug>
        </drugs>
      </pathway>
    </pathways>
  </drug>
  <drug type="">
    <drugbank-id primary="true">DB90003</drugbank-id>
    <name>Nullaxin</name>
    <groups><group>experimental</group></groups>
  </drug>
</drugbank>
"""


def _drug(parquet_paths, drugbank_id: str) -> dict:
    return (
        pl.read_parquet(parquet_paths["drug"])
        .filter(drugbank_id=drugbank_id)
        .row(0, named=True)
    )


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
        assert (
            m["files"]["full_database"]["sha256"]
            == hashlib.sha256(gz.read_bytes()).hexdigest()
        )

    def test_idempotent_and_verifies(self, tmp_path, xml_file):
        a = drugbank.register(xml_file, "5.1.13", tmp_path / "raw")
        b = drugbank.register(
            xml_file, "5.1.13", tmp_path / "raw"
        )  # no force: verify path
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
    def test_four_tables_written(self, parquet_paths):
        assert set(parquet_paths) == {"drug", "biomolecule", "pathway", "drug_drug"}
        assert all(p.exists() for p in parquet_paths.values())
        # drug_drug is a directory of part-files; the others are single files
        assert parquet_paths["drug_drug"].is_dir()

    def test_nested_pathway_drug_not_counted(self, parquet_paths):
        drugs = pl.read_parquet(parquet_paths["drug"])
        assert sorted(drugs["drugbank_id"]) == ["DB90001", "DB90002", "DB90003"]

    def test_primary_id_extracted_secondary_kept(self, parquet_paths):
        row = _drug(parquet_paths, "DB90001")
        assert row["drugbank_id"] == "DB90001"
        assert [(i["@primary"], i["$"]) for i in row["drugbank-id"]] == [
            ("true", "DB90001"),
            (None, "BTD90001"),
        ]

    def test_blocks_stay_nested_under_their_tag_names(self, parquet_paths):
        row = _drug(parquet_paths, "DB90001")
        assert row["groups"] == ["approved", "investigational"]
        assert row["cas-number"] == "000-00-0"
        assert row["@type"] == "biotech"
        assert [s["$"] for s in row["synonyms"]] == ["Fake-zumab", "FZB"]
        assert row["atc-codes"][0]["@code"] == "X01XX99"
        assert [lv["@code"] for lv in row["atc-codes"][0]["level"]] == ["X01XX", "X"]
        assert [p["kind"] for p in row["calculated-properties"]] == ["logP"]

    def test_salt_keeps_its_own_primary_id(self, parquet_paths):
        # <salt> carries a second <drugbank-id primary="true">; it is not the drug's
        salt = _drug(parquet_paths, "DB90001")["salts"][0]
        assert salt["drugbank-id"] == {"@primary": "true", "$": "DBSALT90001"}
        assert salt["name"] == "Fakezumab hydrochloride"

    def test_binds_carries_the_edge_not_the_entity(self, parquet_paths):
        binds = {b["be_id"]: b for b in _drug(parquet_paths, "DB90001")["binds"]}
        assert {k: b["kind"] for k, b in binds.items()} == {
            "BE9000001": "target",
            "BE9000002": "enzyme",
            "BE9000003": "carrier",
        }
        assert binds["BE9000001"]["actions"] == ["inhibitor"]
        assert binds["BE9000001"]["known-action"] == "yes"
        assert binds["BE9000003"]["actions"] is None

    def test_biomolecule_deduped_across_drugs_and_kinds(self, parquet_paths):
        # BE9000001 is DB90001's target and DB90002's enzyme: one entity row, two edges
        bio = pl.read_parquet(parquet_paths["biomolecule"])
        assert sorted(bio["be_id"]) == [
            "BE9000001",
            "BE9000002",
            "BE9000003",
            "BE9000009",
        ]
        assert bio.filter(be_id="BE9000001").height == 1
        kinds = {
            (d, b["kind"])
            for d in ("DB90001", "DB90002")
            for b in _drug(parquet_paths, d)["binds"]
            if b["be_id"] == "BE9000001"
        }
        assert kinds == {("DB90001", "target"), ("DB90002", "enzyme")}

    def test_non_protein_biomolecule_kept(self, parquet_paths):
        # BE9000009 "DNA" has no <polypeptide>; a table keyed on uniprot_id loses it
        dna = pl.read_parquet(parquet_paths["biomolecule"]).filter(be_id="BE9000009")
        assert dna.row(0, named=True)["name"] == "DNA"
        assert dna.row(0, named=True)["polypeptide"] is None

    def test_polypeptide_nested_under_the_entity(self, parquet_paths):
        bio = pl.read_parquet(parquet_paths["biomolecule"]).filter(be_id="BE9000001")
        poly = bio.row(0, named=True)["polypeptide"]
        assert len(poly) == 1
        assert poly[0]["@id"] == "P00000"
        assert poly[0]["gene-name"] == "FAKE1"
        assert len(poly[0]["go-classifiers"]) == 4

    def test_pathway_hoisted_and_reachable_from_the_drug(self, parquet_paths):
        pw = pl.read_parquet(parquet_paths["pathway"])
        assert pw.height == 1
        row = pw.row(0, named=True)
        assert row["smpdb-id"] == "SMP0000001"
        assert [d["name"] for d in row["drugs"]] == ["Fakezumab", "Placebix"]
        assert _drug(parquet_paths, "DB90002")["pathway_ids"] == ["SMP0000001"]
        assert _drug(parquet_paths, "DB90001")["pathway_ids"] == []

    def test_drug_drug_stored_directed(self, parquet_paths):
        ddi = pl.read_parquet(parquet_paths["drug_drug"]).sort("drugbank_id")
        assert list(zip(ddi["drugbank_id"], ddi["interacts_with_id"])) == [
            ("DB90001", "DB90002"),
            ("DB90002", "DB90001"),
        ]
        assert "interacts_with_name" not in ddi.columns  # join-recoverable, not stored

    def test_reparse_skipped_when_tables_exist(self, tmp_path, raw_paths):
        first = drugbank.ensure_parquet_tables(raw_paths, tmp_path / "pq")
        mtimes = {t: p.stat().st_mtime_ns for t, p in first.items()}
        again = drugbank.ensure_parquet_tables(raw_paths, tmp_path / "pq")
        assert {t: p.stat().st_mtime_ns for t, p in again.items()} == mtimes

    @pytest.mark.parametrize("left_in", ["drug_drug", "_partial/drug_drug"])
    def test_reparse_drops_stale_parts(self, tmp_path, raw_paths, left_in):
        stale = tmp_path / "pq" / left_in / "part-9999.parquet"
        stale.parent.mkdir(parents=True)
        pl.DataFrame({"drugbank_id": ["DBSTALE"]}).write_parquet(stale)
        paths = drugbank.ensure_parquet_tables(raw_paths, tmp_path / "pq")
        assert "DBSTALE" not in pl.read_parquet(paths["drug_drug"])["drugbank_id"]
        assert not (tmp_path / "pq" / "_partial").exists()

    def test_failed_parse_leaves_no_tables(self, tmp_path):
        bad = tmp_path / "bad.xml"
        bad.write_text(_XML.replace("<unii>FAKE123</unii>", "<unii>a</unii><unii>b</unii>"))
        raw = drugbank.register(bad, version="5.1.13", raw_dir=tmp_path / "raw")
        with pytest.raises(ValueError, match="repeats <unii>") as err:
            drugbank.ensure_parquet_tables(raw, tmp_path / "pq")
        assert "in drug DB90001" in err.value.__notes__
        assert not (tmp_path / "pq" / "drug.parquet").exists()

    def test_conflicting_biomolecule_raises(self, tmp_path):
        # drug 2's copy of BE9000001 renamed, so the two occurrences disagree
        bad = tmp_path / "bad.xml"
        i = _XML.index("<enzymes>", _XML.index("DB90002"))
        bad.write_text(_XML[:i] + _XML[i:].replace("Fake receptor", "Other", 1))
        raw = drugbank.register(bad, version="5.1.13", raw_dir=tmp_path / "raw")
        with pytest.raises(ValueError, match="BE9000001 under DB90002 differs"):
            drugbank.ensure_parquet_tables(raw, tmp_path / "pq")


class TestDeclaredShape:
    """The three declared sets exist so a schema change in a new DrugBank release
    stops the parse instead of silently writing rows that disagree on shape."""

    def test_undeclared_repeat_raises(self):
        e = ET.fromstring("<drug><name>a</name><name>b</name></drug>")
        with pytest.raises(ValueError, match="repeats <name>"):
            drugbank._xml_element(e)

    def test_undeclared_attributed_leaf_raises(self):
        e = ET.fromstring('<drug><state code="s">solid</state></drug>')
        with pytest.raises(ValueError, match="_ATTR_LEAF"):
            drugbank._xml_element(e)

    def test_container_gaining_attributes_raises(self):
        e = ET.fromstring('<groups format="x"><group>approved</group></groups>')
        with pytest.raises(ValueError, match="container <groups>"):
            drugbank._xml_element(e)

    def test_whitespace_only_text_is_null_not_a_string(self):
        # an unpopulated <references/> must not become a string where its populated
        # siblings are structs
        e = ET.fromstring("<drug><references>\n  </references></drug>")
        assert drugbank._xml_element(e) == {"references": None}


class TestViews:
    def test_drugs_view_flattens_groups(self, parquet_paths):
        d = drugbank.build_views(parquet_paths)["drugs"].collect()
        row = d.filter(drugbank_id="DB90001").row(0, named=True)
        assert row["groups"] == "approved;investigational"
        assert row["type"] == "biotech"
        assert row["cas_number"] == "000-00-0"

    def test_targets_view_one_row_per_bound_polypeptide(self, parquet_paths):
        t = drugbank.build_views(parquet_paths)["targets"].collect()
        assert t.height == 5  # 3 for DB90001, 2 for DB90002
        recep = t.filter(drugbank_id="DB90001", target_id="BE9000001").row(
            0, named=True
        )
        assert recep["name"] == "Fakezumab"
        assert recep["target_name"] == "Fake receptor"
        assert recep["uniprot_id"] == "P00000"
        assert recep["gene_name"] == "FAKE1"
        assert recep["actions"] == "inhibitor"
        assert recep["known_action"] == "yes"

    def test_targets_view_skips_a_drug_that_binds_nothing(self, parquet_paths):
        # DB90003 has no _BIOMOL_TERM block; its empty list must not explode to a null row
        t = drugbank.build_views(parquet_paths)["targets"].collect()
        assert t.filter(drugbank_id="DB90003").height == 0

    def test_targets_view_keeps_an_entry_that_binds_no_polypeptide(self, parquet_paths):
        t = drugbank.build_views(parquet_paths)["targets"]
        dna = t.filter(target_id="BE9000009").collect().row(0, named=True)
        assert dna["target_name"] == "DNA"
        assert dna["uniprot_id"] is None

    def test_go_classifiers_view_is_per_drug(self, parquet_paths):
        g = drugbank.build_views(parquet_paths)["go_classifiers"].collect()
        assert g.height == 9  # DB90001: 4 + 1, DB90002: 4
        assert sorted(
            g.filter(drugbank_id="DB90001", kind="target", category="function")[
                "description"
            ]
        ) == ["ATP binding", "protein binding"]
        assert g.filter(kind="carrier")["description"].to_list() == ["lipid binding"]
        assert g.filter(uniprot_id="P11111").height == 0  # P11111 has no classifiers

    def test_go_kind_follows_the_edge_not_the_entity(self, parquet_paths):
        # the same BE9000001/P00000 terms appear as target terms for DB90001 and as
        # enzyme terms for DB90002
        g = drugbank.build_views(parquet_paths)["go_classifiers"].collect()
        assert set(
            zip(
                g.filter(uniprot_id="P00000")["drugbank_id"],
                g.filter(uniprot_id="P00000")["kind"],
            )
        ) == {("DB90001", "target"), ("DB90002", "enzyme")}

    def test_interactions_view_names_both_drugs(self, parquet_paths):
        i = drugbank.build_views(parquet_paths)["interactions"].collect()
        assert set(zip(i["name"], i["interacts_with_name"])) == {
            ("Fakezumab", "Placebix"),
            ("Placebix", "Fakezumab"),
        }

    def test_empty_string_nulled_on_scan(self, parquet_paths):
        # drug 3 has type="": stored as "", nulled on scan
        raw = _drug(parquet_paths, "DB90003")
        assert raw["@type"] == ""
        drugs = drugbank.cleanly_scan_parquet_tables(parquet_paths)["drug"].collect()
        assert drugs.filter(drugbank_id="DB90003").row(0, named=True)["@type"] is None
