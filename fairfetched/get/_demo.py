"""Tiny offline sample datasets for docstrings and tests.

``Chembl.demo()`` / ``Papyrus.demo()`` write these frames to a temp dir once
per process and return a wrapper you can query without any download. Table and
column names mirror the real releases (ChEMBL ``schema_documentation.txt`` /
Papyrus); the values are invented. Shape: 3 molecules, 3 activities, 2 targets
(ChEMBL); 3 activities, 2 proteins (Papyrus).
"""

import tempfile
from functools import lru_cache
from pathlib import Path

import polars as pl

DEMO_DIR = Path(tempfile.gettempdir()) / "fairfetched-demo"


def chembl_frames() -> dict[str, pl.DataFrame]:
    """One frame per raw ChEMBL table needed to build every view."""
    return {
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


def papyrus_frames() -> dict[str, pl.DataFrame]:
    """The two raw Papyrus source tables."""
    return {
        "bioactivity": pl.DataFrame(
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
        ),
        "protein": pl.DataFrame(
            {
                "target_id": [100, 101],
                "uniprotid": ["P12345", "P12346"],
                "target_chembl_id": ["CHEMBL100", "CHEMBL101"],
                "pref_name": ["Kinase 1", "Kinase 2"],
            }
        ),
    }


def write_parquets(frames: dict[str, pl.DataFrame], directory: Path) -> dict[str, Path]:
    """Write each frame to ``directory/<name>.parquet``; return the path map."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, df in frames.items():
        path = directory / f"{name}.parquet"
        df.write_parquet(path)
        paths[name] = path
    return paths


@lru_cache(maxsize=1)
def chembl_parquets() -> dict[str, Path]:
    return write_parquets(chembl_frames(), DEMO_DIR / "chembl")


@lru_cache(maxsize=1)
def papyrus_parquets() -> dict[str, Path]:
    return write_parquets(papyrus_frames(), DEMO_DIR / "papyrus")
