"""ADReCS-Target source: drug targets (proteins, genes, genetic variations)
implicated in adverse drug reactions.

One undated release; the files are ``.xlsx`` plus one ``#``-separated
``threeLever.txt`` with no header. Header folding and null-token handling are
shared with :mod:`fairfetched.get.adrecs` via :func:`~fairfetched.get.adrecs._clean`.

Homepage: https://www.bio-add.org/ADReCS-Target/ .

Prefer the ``dataset.AdrecsTarget`` wrapper; standalone use::

    raw = ensure_raw_files("1.0")
    tables = ensure_parquet_tables(raw)
    build_views(tables)["drug_adr_protein"].sink_parquet("out.parquet")

``pl.read_excel`` needs an Excel engine (``fastexcel``); add it to the project
dependencies before running.
"""

import logging as lg
from pathlib import Path

import polars as pl

from fairfetched.utils import BASE_DIR, ensure_url

from .adrecs import cleanly_scan_parquet

_lg = lg.getLogger(__name__)

ADRECS_TARGET_DIR = BASE_DIR / "adrecs_target"

_BASE_URL = "https://www.bio-add.org/ADReCS-Target/files/download/"

# 'drug' first: the others reference drugs only by name.
_FILES = {
    "drug": "ALLDRUG_INFO.xlsx",
    "protein": "ADReCS_Target_INFO.xlsx",
    "gene": "ADRAlert_LINCS_Gene_inf.xlsx",
    "genetic_variation": "SNP_Variation_INFO.xlsx",
    "adr": "ALLTOXI_INFO.xlsx",
    "adr_all_associations": "threeLever.txt",
    "adr_protein": "P_D_A.xlsx",
    "adr_gene": "ADRAlert2GENE2ID.xlsx",
    "adr_drug": "V_D_A.xlsx",
}

# threeLever.txt ships with no header row; columns per the ADReCS-Target docs.
_THREELEVER_COLUMNS = [
    "adr_hierarchical",
    "adr_term",
    "drug_name",
    "uniprot_ac",
    "grade",  # GA/GB/GC, GC most common; meaning undocumented upstream
]

# The release is unversioned; a single pseudo-version keeps the module API
# identical to get.chembl / get.papyrus.
ADRECS_TARGET_VERSIONS: dict[str, dict[str, str]] = {
    "1.0": {name: _BASE_URL + fname for name, fname in _FILES.items()}
}


def available_versions() -> tuple[str, ...]:
    return tuple(ADRECS_TARGET_VERSIONS.keys())


def latest() -> str:
    return available_versions()[-1]


def source_urls(version: str) -> dict[str, str]:
    return ADRECS_TARGET_VERSIONS[str(version)]


def ensure_raw_files(
    version: str = "1.0", raw_dir: Path | str | None = None, force: bool = False
) -> dict[str, Path]:
    if raw_dir is None:
        raw_dir = ADRECS_TARGET_DIR / version / "raw"
    raw_dir = Path(raw_dir)
    return {
        name: ensure_url(url, raw_dir / url.split("/")[-1], force=force)
        for name, url in source_urls(version).items()
    }


def _read_raw(name: str, path: Path) -> pl.DataFrame:
    if name == "adr_all_associations":
        return pl.read_csv(
            path, separator="#", has_header=False, new_columns=_THREELEVER_COLUMNS
        )
    if path.name.lower().endswith(".xlsx"):
        return pl.read_excel(path)  # ponytail: first sheet; ALLTOXI_INFO's extra sheets are dropped
    return pl.read_csv(path, separator="\t", infer_schema_length=10000)


def ensure_parquet_tables(
    raw_paths: dict[str, Path], table_dir: Path | str | None = None
) -> dict[str, Path]:
    if table_dir is None:
        table_dir = next(iter(raw_paths.values())).parent.parent / "parquet"
    table_dir = Path(table_dir)
    table_dir.mkdir(exist_ok=True, parents=True)

    out: dict[str, Path] = {}
    for name, path_ in raw_paths.items():
        dest = table_dir / f"{name}.parquet"
        out[name] = dest
        if dest.exists():
            continue
        _lg.info(f"parsing {path_} -> {dest}")
        _read_raw(name, Path(path_)).write_parquet(dest)
    return out


def cleanly_scan_parquet_tables(
    parquet_paths: dict[str, Path],
) -> dict[str, pl.LazyFrame]:
    """Scan raw Parquet paths and apply :func:`~fairfetched.get.adrecs._clean` lazily."""
    return {name: cleanly_scan_parquet(p) for name, p in parquet_paths.items()}


def build_views(parquet_paths: dict[str, Path]) -> dict[str, pl.LazyFrame]:
    """Joined views over the cleaned tables.

    - ``proteins``: ADReCS target proteins with their gene annotation
    - ``drug_adr_protein``: drug-protein-ADR triples (``P_D_A``) enriched with
      protein detail (on ``rid``) and the ADR ontology row (on ``adr_id``)
    - ``drug_adr_gene``: drug-gene-ADR triples (``ADRAlert2GENE2ID``) enriched
      with gene detail (on ``gene_id``)
    """
    lfs = cleanly_scan_parquet_tables(parquet_paths)
    gene = lfs["gene"]
    adr = lfs["adr"].select("adr_id", "toxicity_detail", "organism", "data_source")
    return {
        "proteins": lfs["protein"].join(gene, on="gene_id", how="left"),
        "drug_adr_protein": (
            lfs["adr_protein"]
            .join(lfs["protein"].drop("uniprot_ac"), on="rid", how="left")
            .join(adr, on="adr_id", how="left")
        ),
        "drug_adr_gene": (
            lfs["adr_gene"].join(gene, on="gene_id", how="left").join(adr, on="adr_id", how="left")
        ),
    }


def help() -> None:
    print(build_views.__doc__)
