"""ToxCast (EPA CompTox invitrodb v4.3)
``pl.read_excel`` needs ``fastexcel`` (``pip install fairfetched[toxcast]``).
"""

import logging as lg
import shutil
import tempfile
import zipfile
from functools import partial
from pathlib import Path

import polars as pl
import polars.selectors as cs

from fairfetched.utils import BASE_DIR, ensure_url, manifest, raw
from fairfetched.utils.polars import _TEMP_FILES
from fairfetched.utils.raw import scan_raw

_lg = lg.getLogger(__name__)

TOXCAST_DIR = BASE_DIR / "toxcast"
_MANIFEST_PATH = Path(__file__).parent / "manifests" / "toxcast.json"

_BASE_URL = "https://clowder.edap-cluster.com/files/{}/blob"

_CLOWDER_FILES: dict[str, dict[str, str]] = {
    "4.3": {
        "assay_annotations": "68af6bd3e4b02565fc7c3aa8",
        "assay_target_mappings": "68af6bd3e4b02565fc7c3aa0",
        "cytotox": "68af6bd3e4b02565fc7c3aa4",
        "analytical_qc": "68af6bd3e4b02565fc7c3ab8",
        "summary_zip": "68af6b70e4b02565fc7c3a98",
    }
}

# everything is an xlsx except the zip, which holds the big mc5-6 table
_SUFFIX = {"summary_zip": ".zip"}
_TABLE_NAME = {"summary_zip": "mc5_mc6"}

_QC_FLOAT_COLS = (
    "average_mass",
    "log10_vapor_pressure_OPERA_pred",
    "logKow_octanol_water_OPERA_pred",
)

_MODEL_ID_COLS = ("m4id", "m5id")

_ANNOTATION_COLS = [
    "aeid",
    "assay_component_endpoint_name",
    "assay_component_endpoint_desc",
    "assay_function_type",
    "signal_direction",
    "organism",
    "tissue",
    "cell_short_name",
    "intended_target_type",
    "intended_target_type_sub",
    "intended_target_family",
    "intended_target_family_sub",
]


def available_versions() -> tuple[str, ...]:
    return tuple(_CLOWDER_FILES)


def latest() -> str:
    return available_versions()[-1]


def source_urls(version: str) -> dict[str, str]:
    return {k: _BASE_URL.format(v) for k, v in _CLOWDER_FILES[str(version)].items()}


def write_manifest(version: str = "4.3") -> dict:
    """Pin what Clowder serves now. Run by hand, then commit the manifest."""
    raw_paths = ensure_raw_files(version, force=True, verify=False)
    return manifest.write(raw_paths, _MANIFEST_PATH, toxcast_version=version)


# -- fetch / consolidate / view ---------------------------------------------


def ensure_raw_files(
    version: str = "4.3",
    raw_dir: Path | str | None = None,
    force: bool = False,
    verify: bool = True,
) -> dict[str, Path]:
    """Download the four xlsx files and the summary zip as
    ``<name>_v<version>.<ext>``; skip if present."""
    raw_dir = Path(raw_dir or TOXCAST_DIR / version / "raw")
    raw_paths = {
        name: ensure_url(
            url, raw_dir / f"{name}_v{version}{_SUFFIX.get(name, '.xlsx')}", force=force
        )
        for name, url in source_urls(version).items()
    }
    if verify:
        manifest.verify(raw_paths, _MANIFEST_PATH)
    return raw_paths


def _zip_member(zf: zipfile.ZipFile, *substrings: str) -> str:
    """The one member of ``zf`` whose name contains every substring; raises on 0
    or >1 matches. The README's member names disagree with each other (AUG2024
    stamps, ``flags_`` vs ``flagsv``), so none is hardcoded."""
    hits = [n for n in zf.namelist() if all(s in n for s in substrings)]
    if len(hits) != 1:
        raise ValueError(f"expected 1 zip member with {substrings}, got {hits}")
    return hits[0]


def _scan(name: str, path: Path) -> pl.LazyFrame:
    """xlsx via :func:`scan_raw`; the zip yields its ``mc5-6`` member, extracted
    to system temp (not ``BASE_DIR``) and removed at exit."""
    if name == "assay_annotations":
        return scan_raw(path, sheet_name="annotations_combined")
    if name != "summary_zip":
        return scan_raw(path)
    with zipfile.ZipFile(path) as zf:
        member = _zip_member(zf, "mc5-6", "winning_model_fits", ".csv")
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            _TEMP_FILES.append(tmp.name)
            with zf.open(member) as src:
                shutil.copyfileobj(src, tmp)
    # R's NA literal; without it numeric model-fit columns read as String.
    # Full-file inference: R writes round numbers as 1.8e+07 and decimals appear
    # late, so a prefix infers Int64 for columns that fail further down.
    return scan_raw(tmp.name, null_values=["NA"], infer_schema_length=None)


def ensure_parquet_tables(
    raw_paths: dict[str, Path], table_dir: Path | str | None = None
) -> dict[str, Path]:
    """Consolidate each raw file into a Parquet table, untouched. The summary
    zip yields ``mc5_mc6``; the xlsx files keep their names."""
    return raw.ensure_parquet_tables(
        raw_paths, table_dir, scanner=_scan, table_names=_TABLE_NAME
    )


def _clean(lf: pl.LazyFrame) -> pl.LazyFrame:
    """``"NA"`` strings to null; the R merge duplicate ``<col>.x``/``<col>.y`` to
    one ``<col>``; remaining dots in column names to underscores; the numeric
    ``analytical_qc`` columns xlsx stores as text to Float64; the ``mc5_mc6``
    model ids, which R writes as ``1.8e+07`` and so read as Float64, to Int64."""
    return (
        lf.select(~cs.ends_with(".y"))
        .rename(lambda c: c.removesuffix(".x").replace(".", "_"))
        .with_columns(cs.string().replace("NA", None))
        .with_columns(cs.by_name(*_QC_FLOAT_COLS, require_all=False).cast(pl.Float64))
        .with_columns(cs.by_name(*_MODEL_ID_COLS, require_all=False).cast(pl.Int64))
    )


cleanly_scan_parquet_tables = partial(raw.scan_parquets, clean=_clean)


def build_views(parquet_paths: dict[str, Path]) -> dict[str, pl.LazyFrame]:
    """Joined views over the tables.

    - ``compounds``: ``cytotox`` verbatim, one row per chemical (``chid``)
    - ``bioactivity``: ``mc5_mc6`` (winning model fit per sample-endpoint) with
      endpoint name/description, organism, tissue and intended target from
      ``assay_annotations`` on ``aeid``; rows without a chemical (``chid`` null:
      media blanks and reference spids) are dropped
    - ``full``: ``bioactivity`` with the ``cytotox`` burst columns, joined m:1 on
      ``chid``
    - ``targets``: ``assay_target_mappings`` verbatim; long format, one row per
      (``aeid``, ``target_type``), so not joined into ``bioactivity``
    - ``assay_annotations``: all 63 assay metadata columns
    """
    lfs = cleanly_scan_parquet_tables(parquet_paths)
    bioactivity = (
        lfs["mc5_mc6"]
        .drop_nulls("chid")
        .join(lfs["assay_annotations"].select(_ANNOTATION_COLS), on="aeid", how="left")
    )
    full = bioactivity.join(
        lfs["cytotox"].drop("casn", "chnm", "dsstox_substance_id"),
        on="chid",
        how="left",
        validate="m:1",
    )
    return {
        "compounds": lfs["cytotox"],
        "bioactivity": bioactivity,
        "full": full,
        "targets": lfs["assay_target_mappings"],
        "assay_annotations": lfs["assay_annotations"],
    }


def help() -> None:
    print(build_views.__doc__)


if __name__ == "__main__":
    written = write_manifest()
    for name, entry in written["files"].items():
        print(f"{name:22} {entry['bytes']:>14,} B  {entry['sha256'][:16]}..")
    print(f"\nwrote {_MANIFEST_PATH}")
