import logging as lg
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl

from fairfetched.utils import (
    BASE_DIR,
    ensure_sqlite_db_to_parquets,
    ensure_untarred_sqlite,
    ensure_url,
    file_suffix_from_url,
    lowercase_columns,
)
from fairfetched.utils.typing import BioactivityDBViews

_lg = lg.getLogger(__name__)

CHEMBL_DIR = BASE_DIR / "chembl"


def _format_version(version: float | str) -> str:
    if isinstance(version, int | float):
        version = str(version)
    if isinstance(version, Sequence) and not isinstance(version, str):
        raise TypeError(f"invalid version type: {type(version)}")
    if not isinstance(version, str):
        try:
            version = str(version)
        except Exception:
            raise TypeError(f"invalid version type: {type(version)}")

    version = version.lstrip("0")
    if "." in version:
        version = version.split(".")[0].zfill(2) + "." + version.split(".")[1]
    # for canonicalize the version number 22.1 and 24.1 and left pad with a zero if needed
    return version.replace(".", "_").replace("_0", "").zfill(2)


def _version_to_url(version: str):
    base = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases"
    return f"{base}/chembl_{version}/chembl_{version}_sqlite.tar.gz"


CHEMBL_VERSIONS: dict[str, dict[str, str]] = {
    version: {"sql_db": _version_to_url(version)}
    for version in sorted(map(_format_version, list(range(1, 38)) + ["24_1", "22_1"]))
}


def available_versions() -> tuple[str, ...]:
    return tuple(CHEMBL_VERSIONS.keys())


def latest() -> str:
    return available_versions()[-1]


def source_urls(version: str) -> dict[str, str]:
    return CHEMBL_VERSIONS[str(version)]


def ensure_raw_files(
    version: str, raw_dir: Path | str | None = None, force=False
) -> dict[str, Path]:
    """Download the original SQL database with its original name and compression."""
    if raw_dir is None:
        raw_dir = CHEMBL_DIR / version
    raw_dir = Path(raw_dir)
    return {
        name: ensure_url(
            url=url, path=raw_dir / f"{name}{file_suffix_from_url(url)}", force=force
        )
        for name, url in source_urls(version).items()
    }


def ensure_parquet_tables(
    raw_paths: dict[str, Path], table_dir: Path | str | Any | None = None
) -> dict[str, Path]:
    sql_tar_gz_path = raw_paths["sql_db"]
    if table_dir is None:
        table_dir = Path(sql_tar_gz_path).parent / "extracted"
    table_dir = Path(table_dir)
    table_dir.mkdir(exist_ok=True, parents=True)

    raw_sql = ensure_untarred_sqlite(sql_tar_gz_path)
    # the untarred should also stay so that we have access.....
    # #@TODO: perhaps make tables deterministic for chembl to circumvent
    parquets = ensure_sqlite_db_to_parquets(raw_sql, cache_dir=table_dir, force=False)

    return parquets


def cleanly_scan_parquet(path_: Path | str) -> pl.LazyFrame:
    """scans parquet paths and lazily handles null value conversion to None"""
    return (
        pl.scan_parquet(path_)
        .pipe(lowercase_columns)
        .fill_nan(None)
        .with_columns(
            pl.col(pl.String).replace({"": None}),
        )
    )


def cleanly_scan_parquet_tables(
    parquet_paths: dict[str, Path],
) -> dict[str, pl.LazyFrame]:
    """scans parquet paths and lazily handles null value conversion to None"""
    return {name: cleanly_scan_parquet(path_) for name, path_ in parquet_paths.items()}


# should be present v20+ @TODO: devise better system for pre-v20 if needed
_REQUIRED = (
    # bioactivity
    "activities",
    "compound_records",
    "action_type",
    "ligand_eff",
    "assays",
    "assay_type",
    "confidence_score_lookup",
    "relationship_type",
    "variant_sequences",
    "docs",
    "source",
    # compounds
    "molecule_dictionary",
    "compound_structures",
    "compound_properties",
    "molecule_hierarchy",
    "biotherapeutics",
    "compound_structural_alerts",
    "structural_alerts",
    # proteins / components
    "target_dictionary",
    "target_components",
    "component_sequences",
    "component_synonyms",
    "component_class",
    "protein_classification",
    "component_domains",
    "domains",
)


def _required_tables(parquet_paths: dict[str, Path]) -> dict[str, pl.LazyFrame]:
    """Scan the joined tables and return them, or raise if any is unusable.

    ``ensure_sqlite_db_to_parquets`` writes a parquet for every schema table, so
    a table an old release ships with zero rows still produces a file whose join
    keys read back as Null dtype and would raise ``SchemaError`` mid-join. Every
    ``_REQUIRED`` table must be present and non-empty; anything else raises
    ``ValueError`` here, before a single join runs.
    """
    scanned = cleanly_scan_parquet_tables(parquet_paths)
    empty = {
        name
        for name, lf in scanned.items()
        if lf.select(pl.len()).collect().item() == 0
    }
    unusable = [t for t in _REQUIRED if t not in scanned or t in empty]
    if unusable:
        raise ValueError(
            f"chembl release cannot build views; tables absent or empty: {unusable}"
        )
    return {name: scanned[name] for name in _REQUIRED}


def build_views(parquet_paths: dict[str, Path]) -> BioactivityDBViews:
    """Build joined domain views from the raw ChEMBL SQLite tables.

    Table and column names are those of the ChEMBL schema
    (schema_documentation.txt); ``ensure_sqlite_db_to_parquets`` writes one
    parquet per source table under its original name, so the views join those
    names directly. ``_required_tables`` checks every joined table is present
    and non-empty up front, so each view below is a plain join chain.
    """
    lfs = _required_tables(parquet_paths)
    return {
        "bioactivity": _bioactivities(lfs),
        "compounds": _compounds(lfs),
        "proteins": _targets(lfs),
        "components": _components(lfs),
    }


def _gene_symbols(lfs: dict[str, pl.LazyFrame]) -> pl.LazyFrame:
    """One row per ``component_id`` with its GENE_SYMBOL synonyms as a list."""
    syn = lfs["component_synonyms"].filter(syn_type="GENE_SYMBOL")
    return syn.group_by("component_id").agg(
        pl.col("component_synonym").unique().alias("gene_symbols")
    )


def _protein_classes(lfs: dict[str, pl.LazyFrame]) -> pl.LazyFrame:
    """One row per ``component_id`` with its protein-family class descriptions
    as a list."""
    cc = lfs["component_class"].join(
        lfs["protein_classification"], on="protein_class_id", how="left", suffix="_pc"
    )
    return cc.group_by("component_id").agg(
        pl.col("protein_class_desc").unique().alias("protein_classes")
    )


def _structural_alerts(lfs: dict[str, pl.LazyFrame]) -> pl.LazyFrame:
    """One row per ``molregno`` with its structural-alert names as a list."""
    csa = lfs["compound_structural_alerts"].join(
        lfs["structural_alerts"], on="alert_id", how="left", suffix="_sa"
    )
    return csa.group_by("molregno").agg(
        pl.col("alert_name").unique().alias("structural_alerts")
    )


def _bioactivities(lfs: dict[str, pl.LazyFrame]) -> pl.LazyFrame:
    """One row per activity: the measurement plus its assay, target, document
    and data-source context.

    Path to the target is ``activities.assay_id -> assays.tid ->
    target_dictionary`` (``activities`` itself carries no target). Kept at
    ``target_dictionary`` (tid) granularity; join the ``proteins`` view on
    ``tid`` for sequence/accession-level detail (that join fans out for
    multi-component targets).
    """
    out = lfs["activities"].join(
        lfs["compound_records"].select("record_id", "compound_key", "compound_name"),
        on="record_id",
        how="left",
        suffix="_rec",
        validate="m:1",
    )
    out = out.join(
        lfs["action_type"],
        on="action_type",
        how="left",
        suffix="_action",
        validate="m:1",
    )
    out = out.join(
        lfs["ligand_eff"].select("activity_id", "bei", "sei", "le", "lle"),
        on="activity_id",
        how="left",
        suffix="_le",
        validate="m:1",
    )
    out = out.join(
        lfs["assays"], on="assay_id", how="left", suffix="_assay", validate="m:1"
    )
    # activities.doc_id / src_id are nullable; assays always has them.
    out = out.with_columns(
        pl.col("doc_id").fill_null(pl.col("doc_id_assay")),
        pl.col("src_id").fill_null(pl.col("src_id_assay")),
    )
    out = out.join(
        lfs["assay_type"],
        on="assay_type",
        how="left",
        suffix="_atype",
        validate="m:1",
    )
    out = out.join(
        lfs["confidence_score_lookup"],
        on="confidence_score",
        how="left",
        suffix="_conf",
        validate="m:1",
    )
    out = out.join(
        lfs["relationship_type"],
        on="relationship_type",
        how="left",
        suffix="_rel",
        validate="m:1",
    )
    out = out.join(
        lfs["variant_sequences"].select("variant_id", "mutation", "accession"),
        on="variant_id",
        how="left",
        suffix="_var",
        validate="m:1",
    )
    # target reached through assays.tid (activities carries no target).
    out = out.join(
        lfs["target_dictionary"].select(
            "tid",
            "pref_name",
            "target_type",
            "organism",
            "tax_id",
            "chembl_id",
            "species_group_flag",
        ),
        on="tid",
        how="left",
        suffix="_tgt",
        validate="m:1",
    )
    out = out.join(
        lfs["docs"].select(
            "doc_id",
            "journal",
            "year",
            "pubmed_id",
            "doi",
            "chembl_id",
            "title",
            "doc_type",
        ),
        on="doc_id",
        how="left",
        suffix="_doc",
        validate="m:1",
    )
    out = out.join(
        lfs["source"].select("src_id", "src_description", "src_short_name"),
        on="src_id",
        how="left",
        suffix="_src",
        validate="m:1",
    )
    return out


def _compounds(lfs: dict[str, pl.LazyFrame]) -> pl.LazyFrame:
    """One row per ``molregno``: structure, calculated properties, salt/parent
    hierarchy, biotherapeutic sequence and a list of structural alerts.

    Literature provenance lives in the ``bioactivity`` view, not here, so the
    per-molecule row is not fanned out by ``compound_records``.
    """
    out = lfs["molecule_dictionary"]
    for name, suffix in (
        ("compound_structures", "_struct"),
        ("compound_properties", "_props"),
        ("molecule_hierarchy", "_hier"),
    ):
        out = out.join(
            lfs[name], on="molregno", how="left", suffix=suffix, validate="1:1"
        )
    out = out.join(
        lfs["biotherapeutics"].select("molregno", "helm_notation"),
        on="molregno",
        how="left",
        suffix="_bio",
        validate="1:1",
    )
    out = out.join(
        _structural_alerts(lfs),
        on="molregno",
        how="left",
        suffix="_alert",
        validate="1:1",
    )
    return out


def _targets(lfs: dict[str, pl.LazyFrame]) -> pl.LazyFrame:
    """One row per (target, component): ``target_dictionary`` expanded through
    ``target_components`` to the protein components, with gene symbols and
    protein-family classes attached as lists.

    Multi-component targets (complexes, protein families) yield several rows.
    """
    out = lfs["target_dictionary"].join(
        lfs["target_components"], on="tid", how="left", suffix="_tc"
    )
    out = out.join(
        lfs["component_sequences"], on="component_id", how="left", suffix="_cs"
    )
    out = out.join(
        _gene_symbols(lfs),
        on="component_id",
        how="left",
        suffix="_syn",
        validate="m:1",
    )
    out = out.join(
        _protein_classes(lfs),
        on="component_id",
        how="left",
        suffix="_cls",
        validate="m:1",
    )
    return out


def _components(lfs: dict[str, pl.LazyFrame]) -> pl.LazyFrame:
    """Component-centric protein view: ``component_sequences`` with its
    family classes, gene symbols and structural domains. Independent of
    ``target_dictionary`` — join on ``component_id`` to reach targets.
    """
    out = lfs["component_sequences"].join(
        _protein_classes(lfs),
        on="component_id",
        how="left",
        suffix="_cls",
        validate="1:1",
    )
    out = out.join(
        _gene_symbols(lfs),
        on="component_id",
        how="left",
        suffix="_syn",
        validate="1:1",
    )
    cd = lfs["component_domains"].join(
        lfs["domains"], on="domain_id", how="left", suffix="_dom", validate="m:1"
    )
    out = out.join(cd, on="component_id", how="left", suffix="_cd")
    return out
