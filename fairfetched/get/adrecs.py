"""ADReCS (Adverse Drug Reaction Classification System) source.

Prefer the ``dataset.Adrecs`` wrapper; standalone use::

    raw = ensure_raw_files("3.3")
    tables = ensure_parquet_tables(raw)
    views = build_views(tables)
    views["drug_adr"].sink_parquet("adrecs_drug_adr.parquet")

``pl.read_excel`` needs an Excel engine (``fastexcel``); add it to the project
dependencies before running.

ADReCSTarget license is https://creativecommons.org/licenses/by-nc-sa/4.0/, (see https://bioinf.xmu.edu.cn/ADReCS-Target/download.jsp)
ADReCS itself does not state any license.
"""

import logging as lg
import re
from functools import partial
from pathlib import Path

import polars as pl

from fairfetched.utils import BASE_DIR, ensure_url, raw

_lg = lg.getLogger(__name__)

ADRECS_DIR = BASE_DIR / "adrecs"

_NULL_TOKENS = {
    "Not Available": None,
    "---": None,
    "null": None,  # adr_gene / threeLever
    "-": None,  # gene / variation / association
    "N/A": None,
}


_COL_RENAME = {
    "badd_did": "drug_id",
    "badd_tid": "rid",
    "ditop2_id": "rid",
    "adrecs_id": "adr_hierarchical",
    "pubchem_id": "pubchem",
    "orgaism": "organism",
    "geneid": "gene_id",
    "string": "original_string",
}


def _files(version: str) -> dict[str, str]:
    base = f"https://www.bio-add.org/ADReCS/download/v{version}/"
    v = version
    return {
        "drug": f"{base}Drug_information_v{v}.xlsx",
        "adr": f"{base}ADR_ontology_v{v}.xlsx",
        "drug_adr": f"{base}Drug_ADR_v{v}.txt.gz",
        "drug_adr_matrix": f"{base}Drug_ADR_Matrix_v{v}.txt.gz",
        "adr_severity": f"{base}ADReCS_ADR_Severity_Grade_v{v}.txt.gz",
        "adr_frequency": f"{base}ADReCS_ADR_Frequency_v{v}.txt.gz",
        "drug_adr_quant": f"{base}ADReCS_Drug_ADR_relations_quantification_v{v}.txt.gz",
    }


ADRECS_VERSIONS: dict[str, dict[str, str]] = {v: _files(v) for v in ("3.2", "3.3")}


def available_versions() -> tuple[str, ...]:
    return tuple(ADRECS_VERSIONS.keys())


def latest() -> str:
    return available_versions()[-1]


def source_urls(version: str) -> dict[str, str]:
    return ADRECS_VERSIONS[str(version)]


def ensure_raw_files(
    version: str, raw_dir: Path | str | None = None, force: bool = False
) -> dict[str, Path]:
    """Download each raw file under its original name; skip if already present."""
    if raw_dir is None:
        raw_dir = ADRECS_DIR / version / "raw"
    raw_dir = Path(raw_dir)
    return {
        name: ensure_url(url, raw_dir / url.split("/")[-1], force=force)
        for name, url in source_urls(version).items()
    }


def _clean(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Fold headers to snake_case, apply :data:`_COL_RENAME`, null out ADReCS's
    placeholder tokens. Applied lazily on scan, never written to Parquet."""
    cols = lf.collect_schema().names()
    folded = {
        c: re.sub(r"[.\-\s]+", "_", c).replace("﻿", "").strip("_").lower() for c in cols
    }
    lf = lf.rename(folded)
    lf = lf.rename({k: v for k, v in _COL_RENAME.items() if k in folded.values()})
    return lf.with_columns(pl.col(pl.String).replace(_NULL_TOKENS))


# the names ``dataset.Adrecs`` expects of a source module
ensure_parquet_tables = raw.ensure_parquet_tables  # xlsx: first sheet only; ADReCS core files are single-sheet
cleanly_scan_parquet_tables = partial(raw.scan_parquets, clean=_clean)


def build_views(parquet_paths: dict[str, Path]) -> dict[str, pl.LazyFrame]:
    """Joined views over the cleaned tables.

    - ``drugs``: drug identifiers/xrefs (``drug`` table verbatim)
    - ``adrs``: ADR ontology (``adr`` table verbatim)
    - ``drug_adr``: every drug-ADR pair with drug xrefs, ADR ontology row and
      the FAERS severity/frequency numbers where ADReCS quantified them
    """
    lfs = cleanly_scan_parquet_tables(parquet_paths)
    quant = lfs["drug_adr_quant"].select(
        "drug_id", "adr_id", "adr_severity_grade_faers", "adr_frequency_faers"
    )
    drug_adr = (
        lfs["drug_adr"]
        .join(lfs["drug"].drop("drug_name"), on="drug_id", how="left")
        .join(lfs["adr"].drop("adr_term"), on="adr_id", how="left")
        .join(quant, on=["drug_id", "adr_id"], how="left")
    )
    return {"drugs": lfs["drug"], "adrs": lfs["adr"], "drug_adr": drug_adr}


def help() -> None:
    print(build_views.__doc__)
