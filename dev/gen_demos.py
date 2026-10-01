# run with `uv run python -m dev.gen_demos`
"""Regenerate ``fairfetched/get/_demo/<dataset>/*.parquet`` as row slices of the
real releases cached under ``BASE_DIR`` (ChEMBL, Papyrus, ToxCast).

Each slice is seeded on a few entities and closed over the foreign keys the
views join on, so every view of ``<Dataset>.demo()`` is non-empty and every
row is verbatim from the release. Sider/Adrecs/Drugbank have no demo.
"""

from pathlib import Path

import polars as pl

from fairfetched.get import chembl, papyrus, toxcast
from fairfetched.utils import BASE_DIR

OUT = Path(__file__).parents[1] / "fairfetched" / "get" / "_demo"

_VERSIONS = {"chembl": "37", "papyrus": "05.7", "toxcast": "4.3"}

# aspirin, ibuprofen, carbachol (a salt) and its parent, a peptide
_CHEMBL_SEED_MOLREGNOS = [1280, 11674, 892, 94790, 197]
_ACTIVITIES_PER_MOLECULE = 2

# one human, one rat protein, 5 bioactivity rows each
_PAPYRUS_SEED_TARGET_IDS = ["Q16719_WT", "P15709_WT"]

# acetamide, acetaminophen, acifluorfen on two endpoints (ER agonism, HepG2 cell cycle)
_TOXCAST_SEED_CHIDS = [20005, 20006, 20022]
_TOXCAST_SEED_AEIDS = [2, 4]
_TOXCAST_BLANK_ROWS = 2
_TOXCAST_QC_ROWS_PER_CHEMICAL = 2

# (table, key, tables whose already-sliced ``key`` values select its rows), in
# dependency order; ``activities`` is the seed.
_CHEMBL_CLOSURE = [
    ("compound_records", "record_id", ["activities"]),
    ("action_type", "action_type", ["activities"]),
    ("ligand_eff", "activity_id", ["activities"]),
    ("assays", "assay_id", ["activities"]),
    ("assay_type", "assay_type", ["assays"]),
    ("confidence_score_lookup", "confidence_score", ["assays"]),
    ("relationship_type", "relationship_type", ["assays"]),
    ("variant_sequences", "variant_id", ["assays"]),
    ("docs", "doc_id", ["activities", "assays"]),
    ("source", "src_id", ["activities", "assays"]),
    ("molecule_dictionary", "molregno", ["activities"]),
    ("compound_structures", "molregno", ["activities"]),
    ("compound_properties", "molregno", ["activities"]),
    ("molecule_hierarchy", "molregno", ["activities"]),
    ("biotherapeutics", "molregno", ["activities"]),
    ("compound_structural_alerts", "molregno", ["activities"]),
    ("structural_alerts", "alert_id", ["compound_structural_alerts"]),
    ("target_dictionary", "tid", ["assays"]),
    ("target_components", "tid", ["assays"]),
    ("component_sequences", "component_id", ["target_components"]),
    ("component_synonyms", "component_id", ["target_components"]),
    ("component_class", "component_id", ["target_components"]),
    ("component_domains", "component_id", ["target_components"]),
    ("protein_classification", "protein_class_id", ["component_class"]),
    ("domains", "domain_id", ["component_domains"]),
]


def _scan(dataset: str, table: str) -> pl.LazyFrame:
    return pl.scan_parquet(
        BASE_DIR / dataset / _VERSIONS[dataset] / "parquet" / f"{table}.parquet"
    )


def _rows_with(lf: pl.LazyFrame, key: str, values: pl.Series) -> pl.LazyFrame:
    """Rows of ``lf`` whose ``key`` is among ``values``."""
    return lf.join(values.alias(key).to_frame().lazy().unique(), on=key, how="semi")


def chembl_slice() -> dict[str, pl.DataFrame]:
    """All raw ChEMBL tables the views join, restricted to ``_CHEMBL_SEED_MOLREGNOS``
    and what they reach: activities -> assays/docs/records/targets -> components,
    classification, domains, alerts, hierarchy. Per molecule, the activities that
    carry an assay variant, an action type and a pChEMBL value come first, so
    ``variant_sequences`` and ``action_type`` are not empty."""
    activities = (
        _scan("chembl", "activities")
        .filter(pl.col("molregno").is_in(_CHEMBL_SEED_MOLREGNOS))
        .join(_scan("chembl", "assays").select("assay_id", "variant_id"), on="assay_id")
        .sort(
            pl.col("variant_id").is_not_null(),
            pl.col("action_type").is_not_null(),
            pl.col("pchembl_value").is_not_null(),
            "activity_id",
            descending=[True, True, True, False],
        )
        .group_by("molregno", maintain_order=True)
        .head(_ACTIVITIES_PER_MOLECULE)
        .drop("variant_id")
        .collect()
    )
    out = {"activities": activities}
    for table, key, parents in _CHEMBL_CLOSURE:
        values = pl.concat([out[p][key] for p in parents])
        out[table] = _rows_with(_scan("chembl", table), key, values).collect()
    return out


def papyrus_slice() -> dict[str, pl.DataFrame]:
    """``bioactivity`` rows of ``_PAPYRUS_SEED_TARGET_IDS`` and their ``protein`` rows."""
    keep = pl.Series("target_id", _PAPYRUS_SEED_TARGET_IDS)
    return {
        t: _rows_with(_scan("papyrus", t), "target_id", keep).collect()
        for t in ("bioactivity", "protein")
    }


def toxcast_slice() -> dict[str, pl.DataFrame]:
    """``mc5_mc6`` rows of ``_TOXCAST_SEED_CHIDS`` on ``_TOXCAST_SEED_AEIDS`` (plus a
    few ``chid``-null rows) and the ``cytotox``, ``assay_annotations``,
    ``assay_target_mappings``, ``analytical_qc`` rows their ``chid``/``aeid`` reach."""
    endpoint_rows = _scan("toxcast", "mc5_mc6").filter(
        pl.col("aeid").is_in(_TOXCAST_SEED_AEIDS)
    )
    mc5_mc6 = pl.concat(
        [
            endpoint_rows.filter(pl.col("chid").is_in(_TOXCAST_SEED_CHIDS)),
            endpoint_rows.filter(pl.col("chid").is_null()).head(_TOXCAST_BLANK_ROWS),
        ]
    ).collect()
    cytotox = _rows_with(
        _scan("toxcast", "cytotox"), "chid", mc5_mc6["chid"].drop_nulls()
    ).collect()
    aeids = mc5_mc6["aeid"]
    return {
        "mc5_mc6": mc5_mc6,
        "cytotox": cytotox,
        "assay_annotations": _rows_with(
            _scan("toxcast", "assay_annotations"), "aeid", aeids
        ).collect(),
        "assay_target_mappings": _rows_with(
            _scan("toxcast", "assay_target_mappings"), "aeid", aeids
        ).collect(),
        "analytical_qc": _rows_with(
            _scan("toxcast", "analytical_qc"),
            "dsstox_substance_id",
            cytotox["dsstox_substance_id"],
        )
        .group_by("dsstox_substance_id", maintain_order=True)
        .head(_TOXCAST_QC_ROWS_PER_CHEMICAL)
        .collect(),
    }


def write_all() -> None:
    """Write the three slices to ``OUT/<dataset>/<table>.parquet``, print row counts and
    bytes, and collect every view built from the written files (fails if one is
    unbuildable)."""
    for dataset, module, slices in (
        ("chembl", chembl, chembl_slice()),
        ("papyrus", papyrus, papyrus_slice()),
        ("toxcast", toxcast, toxcast_slice()),
    ):
        directory = OUT / dataset
        directory.mkdir(parents=True, exist_ok=True)
        for old in directory.glob("*.parquet"):
            old.unlink()
        paths = {}
        for table, df in slices.items():
            paths[table] = directory / f"{table}.parquet"
            df.write_parquet(paths[table])
            print(
                f"{dataset}/{table:28} {df.height:>5} rows {paths[table].stat().st_size:>9,} B"
            )
        for view, lf in module.build_views(paths).items():
            print(f"  view {view:12} {lf.collect().shape}")


if __name__ == "__main__":
    write_all()
