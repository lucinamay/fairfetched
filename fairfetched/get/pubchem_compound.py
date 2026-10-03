"""PubChem Compound: rolling latest structures for every CID.

Join structures onto sources that carry a PubChem CID (SIDER,
:mod:`fairfetched.get.pubchem_bioassay`).
"""

from functools import partial
from pathlib import Path

import polars as pl

from fairfetched.utils import BASE_DIR, raw

from . import _pubchem

PUBCHEM_COMPOUND_DIR = BASE_DIR / "pubchem_compound"

_BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/"

_URLS: dict[str, str] = {
    "cid_smiles": _BASE_URL + "CID-SMILES.gz",
    "cid_inchi_key": _BASE_URL + "CID-InChI-Key.gz",
}
_MD5_URLS = tuple(url + ".md5" for url in _URLS.values())

# neither file has a header; names follow README-Extras
_SCHEMAS: dict[str, dict[str, type[pl.DataType]]] = {
    "cid_smiles": {"CID": pl.Int64, "SMILES": pl.String},
    "cid_inchi_key": {"CID": pl.Int64, "InChI": pl.String, "InChI Key": pl.String},
}

_RENAMES = {"CID": "cid", "SMILES": "smiles", "InChI": "inchi", "InChI Key": "inchikey"}


latest = partial(_pubchem.snapshot_date, _URLS["cid_smiles"])
available_versions = partial(_pubchem.available_versions, _URLS["cid_smiles"])


def source_urls(version: str) -> dict[str, str]:
    """The versionless URLs; the same for every ``version``."""
    return _URLS


ensure_raw_files = partial(
    _pubchem.ensure_snapshot,
    version_url=_URLS["cid_smiles"],
    urls=_URLS,
    md5_urls=_MD5_URLS,
    root_dir=PUBCHEM_COMPOUND_DIR,
)


ensure_parquet_tables = partial(
    _pubchem.ensure_parquet_tables,
    schemas=_SCHEMAS,
    headerless=True,
    # uncompressed size as a multiple of the .gz size, snapshot 20260927:
    # 8.9 GB from 1.5 GB and 23.5 GB from 7.4 GB
    decompress_first={"cid_smiles": 6.0, "cid_inchi_key": 3.2},
)


def _clean(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Column names to ``cid``, ``smiles``, ``inchi``, ``inchikey``."""
    return lf.rename(_RENAMES, strict=False)


cleanly_scan_parquet_tables = partial(raw.scan_parquets, clean=_clean)


def build_views(parquet_paths: dict[str, Path]) -> dict[str, pl.LazyFrame]:
    """- ``smiles``: ``cid``, ``smiles``; 124.7M compounds
    - ``inchi``: ``cid``, ``inchi``, ``inchikey``; 3 compounds fewer

    The two are not joined into one view: that join builds a hash table over
    23 GB of InChI strings and does not finish in 17 GB of RAM. Join each onto
    the frame that needs structures, which keeps that frame as the build side
    (5.2M CIDs: 26 s)::

        mine.join(view["smiles"], on="cid", how="left").join(
            view["inchi"], on="cid", how="left"
        )
    """
    lfs = cleanly_scan_parquet_tables(parquet_paths)
    return {"smiles": lfs["cid_smiles"], "inchi": lfs["cid_inchi_key"]}


def help() -> None:
    print(build_views.__doc__)
