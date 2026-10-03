"""PubChem BioAssay: rolling latest bulk assay, activity and target tables.

``view.bioactivity`` leaves out the assays deposited by ``EXCLUDED_SOURCES``
(ChEMBL and the ToxCast/Tox21 programme), which fairfetched serves from their
own sources. The raw tables keep every depositor. Structures cover the tested
substances only; :mod:`fairfetched.get.pubchem_compound` has every compound.
"""

import gzip
from functools import partial
from pathlib import Path

import polars as pl
import polars.selectors as cs

from fairfetched.utils import BASE_DIR, raw

from . import _pubchem

PUBCHEM_BIOASSAY_DIR = BASE_DIR / "pubchem_bioassay"

_BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pubchem/Bioassay/Extras/"

_URLS: dict[str, str] = {
    "bioassays": _BASE_URL + "bioassays.tsv.gz",
    "bioactivities": _BASE_URL + "bioactivities.tsv.gz",
    "aid_target": _BASE_URL + "Aid2GeneidAccessionUniProt.gz",
    "sid_cid_smiles": _BASE_URL + "Sid2CidSMILES.gz",
}
_MD5_URLS = (_BASE_URL + "checksum.md5",)

# polars holds a .gz in memory whole. Table -> uncompressed size as a multiple
# of the .gz size: 17.9 GB from 3.0 GB in snapshot 20260929
_DECOMPRESS_FIRST = {"bioactivities": 5.9}

EXCLUDED_SOURCES: tuple[str, ...] = (
    "ChEMBL",
    "Tox21",
    "EPA ToxCast",
    "EPA DSSTox",
    "NIEHS Division of Translational Toxicology (DTT)",
)

_I, _S, _F = pl.Int64, pl.String, pl.Float64

# bioactivities is sorted by AID, so inference would see one assay only
_SCHEMAS: dict[str, dict[str, type[pl.DataType]]] = {
    "bioassays": {
        "AID": _I,
        "BioAssay Name": _S,
        "Deposit Date": _S,
        "Modify Date": _S,
        "Source Name": _S,
        "Source ID": _S,
        "Substance Type": _S,
        "Outcome Type": _S,
        "Project Category": _S,
        "BioAssay Group": _S,
        "BioAssay Types": _S,
        "Protein Accessions": _S,
        "UniProts IDs": _S,
        "Gene IDs": _S,
        "Target TaxIDs": _S,
        "Taxonomy IDs": _S,
        "Cell IDs": _S,
        "Number of Tested SIDs": _I,
        "Number of Active SIDs": _I,
        "Number of Tested CIDs": _I,
        "Number of Active CIDs": _I,
    },
    "bioactivities": {
        "AID": _I,
        "SID": _I,
        "SID Group": _I,
        "CID": _I,
        "Activity Outcome": _S,
        "Activity Name": _S,
        "Activity Qualifier": _S,
        "Activity Value": _F,
        "Activity Unit": _S,
        "Protein Accession": _S,
        "Gene ID": _I,
        "Target TaxID": _I,
        "PMID": _I,
    },
    "aid_target": {
        "AID": _I,
        "Geneid": _I,
        "Accession": _S,
        "UniProtKB_AC/ID": _S,
    },
    "sid_cid_smiles": {"SID": _I, "CID": _I, "Isomeric SMILES": _S},
}

# snake_cased source names that need more than lowercasing
_RENAMES = {
    "uniprots_ids": "uniprot_ids",
    "uniprotkb_ac/id": "uniprot_id",
    "geneid": "gene_id",
    "accession": "protein_accession",
    "isomeric_smiles": "smiles",
}
_STRING_LISTS = ("bioassay_types", "protein_accessions", "uniprot_ids")
_INTEGER_LISTS = ("gene_ids", "target_taxids", "taxonomy_ids", "cell_ids")


latest = partial(_pubchem.snapshot_date, _URLS["bioactivities"])
available_versions = partial(_pubchem.available_versions, _URLS["bioactivities"])


def source_urls(version: str) -> dict[str, str]:
    """The versionless URLs; the same for every ``version``."""
    return _URLS


ensure_raw_files = partial(
    _pubchem.ensure_snapshot,
    version_url=_URLS["bioactivities"],
    urls=_URLS,
    md5_urls=_MD5_URLS,
    root_dir=PUBCHEM_BIOASSAY_DIR,
)


def _scan_checked_header(name: str, path: Path, **kwargs) -> pl.LazyFrame:
    # A full Polars schema is positional; reject renamed or reordered columns.
    with (gzip.open if path.suffix == ".gz" else open)(path, "rt") as fh:
        found = fh.readline().rstrip("\n").split("\t")
    expected = list(kwargs["schema"])
    if found != expected:
        raise ValueError(
            f"{path.name}: upstream columns changed. Expected {expected}, found {found}"
        )
    return raw.scan_raw(path, **kwargs)


ensure_parquet_tables = partial(
    raw.ensure_parquet_tables,
    scan_kwargs={
        name: {"schema": schema, "quote_char": None, "separator": "\t"}
        for name, schema in _SCHEMAS.items()
    },
    scanner=_scan_checked_header,
    decompress_first=_DECOMPRESS_FIRST,
)


def _snake_case(column: str) -> str:
    snake = column.lower().replace(" ", "_")
    return _RENAMES.get(snake, snake)


def _clean(lf: pl.LazyFrame) -> pl.LazyFrame:
    """snake_case column names; ``deposit_date``/``modify_date`` to ``pl.Date``;
    the ``|``-joined ``bioassays`` fields to lists."""
    return lf.rename(_snake_case).with_columns(
        cs.by_name("deposit_date", "modify_date", require_all=False).str.to_date(
            "%Y%m%d"
        ),
        cs.by_name(*_STRING_LISTS, require_all=False).str.split("|"),
        cs.by_name(*_INTEGER_LISTS, require_all=False)
        .str.split("|")
        .cast(pl.List(pl.Int64)),
    )


cleanly_scan_parquet_tables = partial(raw.scan_parquets, clean=_clean)


def build_views(parquet_paths: dict[str, Path]) -> dict[str, pl.LazyFrame]:
    """Joined views over the cleaned tables. Raises if a name in
    ``EXCLUDED_SOURCES`` is absent from ``bioassays``: a depositor renamed
    upstream would otherwise return to the views unnoticed. Raises too if
    ``bioassays.aid`` or ``sid_cid_smiles.sid`` repeats, since either would
    multiply ``bioactivity`` rows.

    - ``assays``: ``bioassays`` without the ``EXCLUDED_SOURCES`` depositors
    - ``bioactivity``: ``bioactivities`` for those assays, one row per
      (``aid``, ``sid``, ``sid_group``, target), with ``bioassay_name``,
      ``source_name`` and ``outcome_type`` from ``assays``, ``smiles`` from
      ``sid_cid_smiles`` and ``uniprot_ids`` from ``aid_target``. ``uniprot_ids``
      is a list: 85 (``aid``, ``protein_accession``) pairs map to several
      UniProt entries
    - ``compounds``: ``cid`` and ``smiles`` of the tested compounds
    - ``proteins``: ``aid_target`` for those assays
    """
    lfs = cleanly_scan_parquet_tables(parquet_paths)
    sources = lfs["bioassays"].select(pl.col("source_name").unique()).collect()
    if absent := sorted(set(EXCLUDED_SOURCES) - set(sources["source_name"])):
        raise ValueError(
            f"{absent} deposit no assay in this snapshot; update EXCLUDED_SOURCES"
        )
    # checked here, not with join(validate=): that takes the 304M-row joins
    # off the streaming engine
    for table, key in (("bioassays", "aid"), ("sid_cid_smiles", "sid")):
        if lfs[table].select(pl.col(key).is_duplicated().any()).collect().item():
            raise ValueError(f"{table}.{key} is not unique in this snapshot")
    assays = lfs["bioassays"].remove(pl.col("source_name").is_in(EXCLUDED_SOURCES))
    uniprot_ids = (
        lfs["aid_target"]
        .drop_nulls(["protein_accession", "uniprot_id"])
        .group_by("aid", "protein_accession")
        .agg(uniprot_ids=pl.col("uniprot_id").unique().sort())
    )
    bioactivity = (
        lfs["bioactivities"]
        .join(
            assays.select("aid", "bioassay_name", "source_name", "outcome_type"),
            on="aid",
        )
        .join(lfs["sid_cid_smiles"].select("sid", "smiles"), on="sid", how="left")
        .join(uniprot_ids, on=["aid", "protein_accession"], how="left")
    )
    return {
        "assays": assays,
        "bioactivity": bioactivity,
        "compounds": lfs["sid_cid_smiles"].select("cid", "smiles").unique(),
        "proteins": lfs["aid_target"].join(assays.select("aid"), on="aid", how="semi"),
    }


def help() -> None:
    print(build_views.__doc__)
