"""PubChem sources: offline checks on snapshot handling, consolidation, cleaning
and the views. No download."""

import errno
import gzip
from datetime import date
from functools import partial
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from fairfetched.get import _pubchem, pubchem_bioassay, pubchem_compound

# RFC 1321 test vectors and a two-file PubChem mini-snapshot.
_MD5_ABC = "900150983cd24fb0d6963f7d28e17f72"
_MD5_EMPTY = "d41d8cd98f00b204e9800998ecf8427e"

_URLS = {"a": "https://x.org/A.gz", "b": "https://x.org/B.gz"}
_MD5S = {"A.gz": _MD5_ABC, "B.gz": _MD5_EMPTY}


def _write_gz(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as fh:
        fh.write(text)
    return path


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    served = MagicMock(return_value="20260929")
    checksums = MagicMock(return_value=_MD5S)
    content = {_URLS["a"]: b"abc", _URLS["b"]: b""}

    def download(url, path, force=False):
        path.write_bytes(content[url])

    monkeypatch.setattr(_pubchem, "snapshot_date", served)
    monkeypatch.setattr(_pubchem, "upstream_md5s", checksums)
    monkeypatch.setattr(_pubchem, "ensure_url", download)
    ensure = partial(
        _pubchem.ensure_snapshot,
        "20260929",
        raw_dir=raw,
        version_url=_URLS["a"],
        urls=_URLS,
        md5_urls=("https://x.org/md5",),
        root_dir=tmp_path,
    )
    return raw, ensure, content, served, checksums


class TestSnapshot:
    @pytest.mark.parametrize("state", ["new", "legacy", "repair", "force"])
    def test_verified_lifecycle(self, snapshot, state):
        raw, ensure, _, served, checksums = snapshot
        ensure()
        if state == "legacy":
            (raw / "_manifest.json").unlink()
        elif state == "repair":
            (raw / "b.tsv.gz").unlink()
        elif state == "force":
            (raw / "a.tsv.gz").write_bytes(b"stale")
        paths = ensure(force=state == "force")

        assert paths == {name: raw / f"{name}.tsv.gz" for name in _URLS}

        assert paths["a"].read_bytes() == b"abc"
        served.side_effect = AssertionError("network")
        checksums.side_effect = AssertionError("network")
        assert ensure()["b"].read_bytes() == b""
        paths["a"].write_bytes(b"corrupt")
        with pytest.raises(ValueError):
            ensure()

    @pytest.mark.parametrize(
        "pinned,body,dates,md5s",
        [
            (False, b"bad", ["20260929"], [_MD5S]),
            (False, b"abc", ["20261013"], []),
            (False, b"abc", ["20260929", "20261013"], [_MD5S]),
            (False, b"abc", ["20260929"] * 2, [_MD5S, {}]),
            (False, b"abc", ["20260929"] * 2, [_MD5S, OSError("offline")]),
            (True, b"new", ["20260929"], []),
        ],
    )
    def test_failed_acquisition(self, snapshot, pinned, body, dates, md5s):
        raw, ensure, content, served, checksums = snapshot
        if pinned:
            ensure()
        before = {path.name: path.read_bytes() for path in raw.glob("*")}
        content[_URLS["a"]] = body
        served.side_effect = dates
        checksums.side_effect = md5s
        with pytest.raises((ValueError, OSError)):
            ensure(force=True)
        assert {path.name: path.read_bytes() for path in raw.glob("*")} == before


class TestVersions:
    def test_live_head_date_and_available_versions(self):
        with patch("urllib.request.urlopen") as opened:
            opened.return_value.__enter__.return_value.headers = {
                "Last-Modified": "Tue, 29 Sep 2026 18:32:00 GMT"
            }
            assert pubchem_bioassay.latest() == "20260929"
            assert pubchem_bioassay.available_versions() == ("20260929",)
            assert pubchem_compound.latest() == "20260929"
        assert opened.call_args.args[0].get_method() == "HEAD"

    def test_upstream_md5_names_and_source_urls(self):
        with patch(
            "urllib.request.urlopen",
            side_effect=[
                BytesIO(f"{_MD5_ABC}  A.gz\n{_MD5_EMPTY}  B.gz\n".encode()),
                BytesIO(f"{_MD5_EMPTY}  C.gz\n".encode()),
            ],
        ):
            assert _pubchem.upstream_md5s(("one", "two")) == {
                "A.gz": _MD5_ABC,
                "B.gz": _MD5_EMPTY,
                "C.gz": _MD5_EMPTY,
            }
        assert pubchem_bioassay.source_urls("any")["bioactivities"].endswith(
            "bioactivities.tsv.gz"
        )
        assert pubchem_compound.source_urls("any")["cid_smiles"].endswith(
            "CID-SMILES.gz"
        )


_ACTIVITY_HEADER = (
    "AID\tSID\tSID Group\tCID\tActivity Outcome\tActivity Name\tActivity Qualifier\t"
    "Activity Value\tActivity Unit\tProtein Accession\tGene ID\tTarget TaxID\tPMID\n"
)


class TestConsolidation:
    @pytest.mark.parametrize("upstream_names", [False, True])
    def test_small_gzip_tables_preserve_quotes_without_disk_check(
        self, tmp_path, upstream_names
    ):
        target = (
            "Aid2GeneidAccessionUniProt.gz" if upstream_names else "aid_target.tsv.gz"
        )
        raw = {
            "aid_target": _write_gz(
                tmp_path / "raw" / target,
                'AID\tGeneid\tAccession\tUniProtKB_AC/ID\n1\t\t"odd\tP1\n',
            ),
            "sid_cid_smiles": _write_gz(
                tmp_path / "raw" / "Sid2CidSMILES.gz",
                "SID\tCID\tIsomeric SMILES\n1\t2\tCC\n",
            ),
        }
        usage = MagicMock(total=10**12, free=0)
        with patch("fairfetched.utils.ensure.shutil.disk_usage", return_value=usage):
            out = pubchem_bioassay.ensure_parquet_tables(raw)
        assert pl.read_parquet(out["aid_target"])["Accession"].to_list() == ['"odd']
        assert pl.read_parquet(out["sid_cid_smiles"])["Isomeric SMILES"].to_list() == [
            "CC"
        ]

    def test_types_come_from_the_schema_not_from_the_first_rows(self, tmp_path):
        rows = "1\t10\t1\t\tInactive\t\t\t\t\t\t\t\t\n" * 3
        rows += "2\t11\t1\t5\tActive\tIC50\t=\t0.5\tuM\tNP_1\t7\t9606\t123\n"
        raw = _write_gz(
            tmp_path / "raw" / "bioactivities.tsv.gz", _ACTIVITY_HEADER + rows
        )
        table = pl.read_parquet(
            pubchem_bioassay.ensure_parquet_tables({"bioactivities": raw})[
                "bioactivities"
            ]
        )
        assert table.schema["Activity Value"] == pl.Float64
        assert table.schema["CID"] == pl.Int64
        assert table["Activity Value"].to_list() == [None, None, None, 0.5]
        assert table["Activity Name"].to_list() == [None, None, None, "IC50"]

    @pytest.mark.parametrize(
        "filename", ["aid_target.tsv.gz", "Aid2GeneidAccessionUniProt.gz"]
    )
    def test_renamed_upstream_column_raises(self, tmp_path, filename):
        raw = _write_gz(
            tmp_path / "raw" / filename,
            "AID\tGeneID\tAccession\tUniProtKB_AC/ID\n1\t2\tA\tP1\n",
        )
        with pytest.raises(ValueError, match="upstream columns changed"):
            pubchem_bioassay.ensure_parquet_tables({"aid_target": raw})

    @pytest.mark.parametrize("upstream_names", [False, True])
    def test_headerless_compound_files_take_schema_names(
        self, tmp_path, upstream_names
    ):
        smiles = "CID-SMILES.gz" if upstream_names else "cid_smiles.tsv.gz"
        inchi = "CID-InChI-Key.gz" if upstream_names else "cid_inchi_key.tsv.gz"
        raw = {
            "cid_smiles": _write_gz(tmp_path / "raw" / smiles, "1\tCC\n2\tC\n"),
            "cid_inchi_key": _write_gz(
                tmp_path / "raw" / inchi,
                "1\tInChI=1S/C2H6/c1-2/h1-2H3\tOTMSDBZUPAUEDD-UHFFFAOYSA-N\n",
            ),
        }
        out = pubchem_compound.ensure_parquet_tables(raw)
        assert pl.read_parquet(out["cid_smiles"]).to_dict(as_series=False) == {
            "CID": [1, 2],
            "SMILES": ["CC", "C"],
        }
        assert pl.read_parquet(out["cid_inchi_key"]).columns == [
            "CID",
            "InChI",
            "InChI Key",
        ]


class TestDecompressionDiskCheck:
    """``bioactivities`` is estimated at 5.9 times its ``.gz`` size."""

    @pytest.mark.parametrize("free_minus_estimate", [-1, 0])
    def test_decompression_disk_check(self, tmp_path, monkeypatch, free_minus_estimate):
        raw = _write_gz(
            tmp_path / "raw" / "bioactivities.tsv.gz",
            _ACTIVITY_HEADER + "1\t10\t1\t5\tActive\t\t\t\t\t\t\t\t\n" * 50,
        )
        temp = tmp_path / "tmp"
        temp.mkdir()
        monkeypatch.setattr("tempfile.tempdir", str(temp))
        usage = MagicMock(
            total=10**12, free=int(raw.stat().st_size * 5.9) + free_minus_estimate
        )
        with patch("fairfetched.utils.ensure.shutil.disk_usage", return_value=usage):
            if free_minus_estimate < 0:
                with pytest.raises(
                    OSError, match="decompressing bioactivities.tsv.gz"
                ) as exc:
                    pubchem_bioassay.ensure_parquet_tables({"bioactivities": raw})
                assert exc.value.errno == errno.ENOSPC
                assert list((tmp_path / "parquet").iterdir()) == []
            else:
                out = pubchem_bioassay.ensure_parquet_tables({"bioactivities": raw})
                assert pl.read_parquet(out["bioactivities"]).height == 50
        assert list(temp.iterdir()) == []


def _table(name: str, rows: list[dict], module=pubchem_bioassay) -> pl.DataFrame:
    """Raw-layer table: the source's column names and types, nulls where a row
    names no value."""
    return pl.DataFrame(rows, schema=module._SCHEMAS[name])


def _write_tables(tmp_path: Path, tables: dict[str, pl.DataFrame]) -> dict[str, Path]:
    paths = {}
    for name, table in tables.items():
        paths[name] = tmp_path / f"{name}.parquet"
        table.write_parquet(paths[name])
    return paths


def _bioassay_tables(**overrides: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Assay 10 (kept, two targets) and 11 (kept, no target); one assay per
    excluded depositor (20-24)."""
    excluded = [
        "ChEMBL",
        "Tox21",
        "EPA ToxCast",
        "EPA DSSTox",
        "NIEHS Division of Translational Toxicology (DTT)",
    ]
    tables = {
        "bioassays": _table(
            "bioassays",
            [
                {
                    "AID": 10,
                    "BioAssay Name": "kinase screen",
                    "Deposit Date": "20040815",
                    "Modify Date": "20260710",
                    "Source Name": "Broad Institute",
                    "Outcome Type": "Confirmatory",
                    "BioAssay Types": "Biochemical|In vitro",
                    "Protein Accessions": "NP_1|1ABC_A",
                    "UniProts IDs": "P11111",
                    "Gene IDs": "5|7",
                    "Taxonomy IDs": "9606|10090",
                },
                {
                    "AID": 11,
                    "BioAssay Name": "cell viability",
                    "Deposit Date": "20100102",
                    "Modify Date": "20100102",
                    "Source Name": "NMMLSC",
                    "Outcome Type": "Screening",
                },
            ]
            + [
                {
                    "AID": 20 + i,
                    "BioAssay Name": f"excluded {i}",
                    "Deposit Date": "20200101",
                    "Modify Date": "20200101",
                    "Source Name": source,
                }
                for i, source in enumerate(excluded)
            ],
        ),
        "bioactivities": _table(
            "bioactivities",
            [
                # substance 100 on assay 10: one row per target
                {
                    "AID": 10,
                    "SID": 100,
                    "SID Group": 1,
                    "CID": 1,
                    "Activity Outcome": "Active",
                    "Activity Value": 0.5,
                    "Protein Accession": "NP_1",
                },
                {
                    "AID": 10,
                    "SID": 100,
                    "SID Group": 1,
                    "CID": 1,
                    "Activity Outcome": "Active",
                    "Activity Value": 0.5,
                    "Protein Accession": "1ABC_A",
                },
                # no CID, no structure, no target
                {"AID": 11, "SID": 101, "SID Group": 1, "Activity Outcome": "Inactive"},
                {"AID": 20, "SID": 100, "SID Group": 1, "CID": 1},
                {"AID": 22, "SID": 100, "SID Group": 1, "CID": 1},
            ],
        ),
        "aid_target": _table(
            "aid_target",
            [
                {"AID": 10, "Geneid": 5, "Accession": "NP_1", "UniProtKB_AC/ID": "P1"},
                {"AID": 10, "Geneid": 6, "Accession": "NP_1", "UniProtKB_AC/ID": "P1"},
                {"AID": 10, "Accession": "1ABC_A", "UniProtKB_AC/ID": "P3"},
                {"AID": 10, "Accession": "1ABC_A", "UniProtKB_AC/ID": "P2"},
                {"AID": 10, "Geneid": 7},
                {"AID": 10, "Geneid": 8, "Accession": "NP_1"},  # no UniProt entry
                {"AID": 20, "Accession": "NP_1", "UniProtKB_AC/ID": "P9"},
            ],
        ),
        "sid_cid_smiles": _table(
            "sid_cid_smiles",
            [
                {"SID": 100, "CID": 1, "Isomeric SMILES": "CC"},
                {"SID": 102, "CID": 1, "Isomeric SMILES": "CC"},
                {"SID": 103, "CID": 2, "Isomeric SMILES": "C"},
            ],
        ),
    }
    return tables | overrides


class TestBioassayClean:
    @pytest.fixture
    def lfs(self, tmp_path):
        return pubchem_bioassay.cleanly_scan_parquet_tables(
            _write_tables(tmp_path, _bioassay_tables())
        )

    def test_column_names_are_snake_case(self, lfs):
        assert lfs["bioactivities"].collect_schema().names() == [
            "aid",
            "sid",
            "sid_group",
            "cid",
            "activity_outcome",
            "activity_name",
            "activity_qualifier",
            "activity_value",
            "activity_unit",
            "protein_accession",
            "gene_id",
            "target_taxid",
            "pmid",
        ]
        assert lfs["aid_target"].collect_schema().names() == [
            "aid",
            "gene_id",
            "protein_accession",
            "uniprot_id",
        ]
        assert lfs["sid_cid_smiles"].collect_schema().names() == [
            "sid",
            "cid",
            "smiles",
        ]
        assert "uniprot_ids" in lfs["bioassays"].collect_schema().names()

    def test_dates_are_parsed(self, lfs):
        assay = lfs["bioassays"].filter(aid=10).collect()
        assert assay["deposit_date"].to_list() == [date(2004, 8, 15)]
        assert assay["modify_date"].to_list() == [date(2026, 7, 10)]

    def test_pipe_joined_fields_become_lists(self, lfs):
        assay = lfs["bioassays"].filter(aid=10).collect()
        assert assay["bioassay_types"].to_list() == [["Biochemical", "In vitro"]]
        assert assay["protein_accessions"].to_list() == [["NP_1", "1ABC_A"]]
        assert assay["uniprot_ids"].to_list() == [["P11111"]]
        assert assay["gene_ids"].to_list() == [[5, 7]]
        assert assay["taxonomy_ids"].to_list() == [[9606, 10090]]

    def test_absent_list_field_stays_null(self, lfs):
        assay = lfs["bioassays"].filter(aid=11).collect()
        assert assay["gene_ids"].to_list() == [None]


class TestBioassayViews:
    @pytest.fixture
    def views(self, tmp_path):
        return pubchem_bioassay.build_views(_write_tables(tmp_path, _bioassay_tables()))

    def test_assays_leave_out_the_excluded_depositors(self, views):
        assert sorted(views["assays"].collect()["aid"]) == [10, 11]

    def test_bioactivity_leaves_out_the_excluded_depositors(self, views):
        rows = views["bioactivity"].collect()
        assert sorted(rows["aid"]) == [10, 10, 11]

    def test_bioactivity_carries_assay_fields(self, views):
        row = views["bioactivity"].filter(aid=11).collect()
        assert row["bioassay_name"].to_list() == ["cell viability"]
        assert row["source_name"].to_list() == ["NMMLSC"]
        assert row["outcome_type"].to_list() == ["Screening"]

    def test_uniprot_ids_are_a_sorted_deduplicated_list_per_target(self, views):
        rows = views["bioactivity"].filter(aid=10).collect()
        by_accession = dict(
            zip(rows["protein_accession"], rows["uniprot_ids"], strict=True)
        )
        assert by_accession["NP_1"].to_list() == ["P1"]
        assert by_accession["1ABC_A"].to_list() == ["P2", "P3"]

    def test_row_without_target_or_structure_keeps_nulls(self, views):
        row = views["bioactivity"].filter(aid=11).collect()
        assert row["uniprot_ids"].to_list() == [None]
        assert row["smiles"].to_list() == [None]

    def test_smiles_comes_from_the_substance(self, views):
        rows = views["bioactivity"].filter(aid=10).collect()
        assert rows["smiles"].to_list() == ["CC", "CC"]

    def test_compounds_has_one_row_per_cid(self, views):
        compounds = views["compounds"].sort("cid").collect()
        assert compounds.to_dict(as_series=False) == {
            "cid": [1, 2],
            "smiles": ["CC", "C"],
        }

    def test_proteins_leave_out_the_excluded_depositors(self, views):
        assert views["proteins"].collect()["aid"].unique().to_list() == [10]
        assert views["proteins"].collect().height == 6

    def test_renamed_excluded_depositor_raises(self, tmp_path):
        tables = _bioassay_tables()
        tables["bioassays"] = tables["bioassays"].with_columns(
            pl.col("Source Name").replace("Tox21", "Tox21 (NCATS)")
        )
        with pytest.raises(ValueError, match=r"\['Tox21'\] deposit no assay"):
            pubchem_bioassay.build_views(_write_tables(tmp_path, tables))

    def test_repeated_sid_raises(self, tmp_path):
        tables = _bioassay_tables()
        tables["sid_cid_smiles"] = pl.concat(
            [tables["sid_cid_smiles"], tables["sid_cid_smiles"].head(1)]
        )
        with pytest.raises(ValueError, match="sid_cid_smiles.sid is not unique"):
            pubchem_bioassay.build_views(_write_tables(tmp_path, tables))

    def test_repeated_aid_raises(self, tmp_path):
        tables = _bioassay_tables()
        tables["bioassays"] = pl.concat(
            [tables["bioassays"], tables["bioassays"].head(1)]
        )
        with pytest.raises(ValueError, match="bioassays.aid is not unique"):
            pubchem_bioassay.build_views(_write_tables(tmp_path, tables))


class TestCompoundViews:
    @pytest.fixture
    def views(self, tmp_path):
        tables = {
            "cid_smiles": _table(
                "cid_smiles",
                [{"CID": 1, "SMILES": "CC"}, {"CID": 2, "SMILES": "C"}],
                pubchem_compound,
            ),
            "cid_inchi_key": _table(
                "cid_inchi_key",
                [{"CID": 1, "InChI": "InChI=1S/C2H6", "InChI Key": "KEY-1"}],
                pubchem_compound,
            ),
        }
        return pubchem_compound.build_views(_write_tables(tmp_path, tables))

    def test_smiles_columns(self, views):
        assert views["smiles"].collect().to_dict(as_series=False) == {
            "cid": [1, 2],
            "smiles": ["CC", "C"],
        }

    def test_inchi_columns(self, views):
        assert views["inchi"].collect().to_dict(as_series=False) == {
            "cid": [1],
            "inchi": ["InChI=1S/C2H6"],
            "inchikey": ["KEY-1"],
        }

    def test_two_step_join_keeps_a_compound_without_inchi(self, views):
        mine = pl.LazyFrame({"cid": [2, 1]})
        joined = (
            mine.join(views["smiles"], on="cid", how="left")
            .join(views["inchi"], on="cid", how="left")
            .sort("cid")
            .collect()
        )
        assert joined["smiles"].to_list() == ["CC", "C"]
        assert joined["inchikey"].to_list() == ["KEY-1", None]
