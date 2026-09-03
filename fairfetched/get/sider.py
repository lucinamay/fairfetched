"""SIDER (Side Effect Resource) source: drug -> adverse-reaction pairs, with
label-extracted frequencies.

SIDER's download URLs carry no version and always serve whatever is
current, so there is nothing in the URL to pin a release. ``_sider_manifest.json``
is the pin instead (via :mod:`fairfetched.utils.manifest`): download raises if a
file's sha256 no longer matches, which means upstream moved and every count
derived from SIDER has to be rechecked. Regenerate it with
``python -m fairfetched.get.sider`` and commit the result.

Prefer the ``dataset.Sider`` wrapper; standalone use::

    raw = ensure_raw_files("4.1")
    tables = ensure_parquet_tables(raw)
    build_views(tables)["side_effects"].sink_parquet("sider_side_effects.parquet")
"""

import logging as lg
from pathlib import Path

import polars as pl

from fairfetched.utils import BASE_DIR, ensure_url, manifest

_lg = lg.getLogger(__name__)

SIDER_DIR = BASE_DIR / "sider"
_MANIFEST_PATH = Path(__file__).parent / "_sider_manifest.json"

_BASE_URL = "https://sideeffects.embl.de/media/download/"


_FILES = {
    "meddra_freq": "meddra_freq.tsv.gz",
    "meddra_all_se": "meddra_all_se.tsv.gz",
    "meddra": "meddra.tsv.gz",
    "drug_names": "drug_names.tsv",
    "readme": "README",  # for column names
}


_COLUMNS: dict[str, list[str]] = {
    "meddra_freq": [
        "stitch_flat",
        "stitch_stereo",
        "umls_label",
        "placebo",
        "freq_desc",
        "lower",
        "upper",
        "term_type",
        "umls_meddra",
        "term_name",
    ],
    "meddra_all_se": [
        "stitch_flat",
        "stitch_stereo",
        "umls_label",
        "term_type",
        "umls_meddra",
        "term_name",
    ],
    "meddra": ["umls_id", "term_type", "meddra_id", "term_name"],  # reordered
    "drug_names": ["stitch_flat", "drug_name"],
}

# not versioned officially, but generally called 4.1
SIDER_VERSIONS: dict[str, dict[str, str]] = {
    "4.1": {name: _BASE_URL + fname for name, fname in _FILES.items()}
}


def available_versions() -> tuple[str, ...]:
    return tuple(SIDER_VERSIONS.keys())


def latest() -> str:
    return available_versions()[-1]


def source_urls(version: str) -> dict[str, str]:
    return SIDER_VERSIONS[str(version)]


def write_manifest(version: str = "4.1") -> dict:
    """Pin whatever SIDER currently serves (:mod:`fairfetched.utils.manifest`);
    SIDER's URLs carry no version. Run by hand, then commit ``_sider_manifest.json``."""
    raw_paths = ensure_raw_files(version, force=True, verify=False)
    return manifest.write(raw_paths, _MANIFEST_PATH, sider_version=version)


# -- fetch / consolidate / view ---------------------------------------------


def ensure_raw_files(
    version: str = "4.1",
    raw_dir: Path | str | None = None,
    force: bool = False,
    verify: bool = True,
) -> dict[str, Path]:
    """Download each raw file under its original name; skip if already present."""
    if raw_dir is None:
        raw_dir = SIDER_DIR / version / "raw"
    raw_dir = Path(raw_dir)
    raw_paths = {
        name: ensure_url(url, raw_dir / url.split("/")[-1], force=force)
        for name, url in source_urls(version).items()
    }
    if verify:
        manifest.verify(raw_paths, _MANIFEST_PATH)
    return raw_paths


def _read_raw(name: str, path: Path) -> pl.DataFrame:
    """Headerless TSV (polars auto-decompresses gzip) with the README columns."""
    return pl.read_csv(
        path,
        separator="\t",
        has_header=False,
        new_columns=_COLUMNS[name],
        # SIDER's term names carry unescaped double quotes ('"Ventilation"
        # pneumonitis'), so quote parsing has to be off.
        quote_char=None,
        infer_schema_length=10000,
    )


def ensure_parquet_tables(
    raw_paths: dict[str, Path], table_dir: Path | str | None = None
) -> dict[str, Path]:
    """Consolidate each raw file into a Parquet table, untouched: README columns,
    original values. Cleaning happens later, on scan. README is skipped."""
    if table_dir is None:
        table_dir = next(iter(raw_paths.values())).parent.parent / "parquet"
    table_dir = Path(table_dir)
    table_dir.mkdir(exist_ok=True, parents=True)

    out: dict[str, Path] = {}
    for name, path_ in raw_paths.items():
        if name == "readme":
            continue
        dest = table_dir / f"{name}.parquet"
        out[name] = dest
        if dest.exists():
            continue
        _lg.info(f"parsing {path_} -> {dest}")
        _read_raw(name, Path(path_)).write_parquet(dest)
    return out


def stitch_id_to_cid(stitch_id: pl.Expr) -> pl.Expr:
    """STITCH compound id ('CID1########' / 'CID0########') -> PubChem CID.

    'CID100000085' -> 85 (carnitine). The digit after 'CID' is a flag
    (1 flattened, 0 stereo-specific), not part of the id.
    """
    return stitch_id.str.slice(4).cast(pl.Int64)


def _clean(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Null SIDER's empty-string placeholder; add ``pubchem_cid`` next to any
    STITCH id column. Applied lazily on scan, never written to Parquet."""
    cols = lf.collect_schema().names()
    lf = lf.with_columns(pl.col(pl.String).replace({"": None}))
    if "stitch_flat" in cols:
        lf = lf.with_columns(pubchem_cid=stitch_id_to_cid(pl.col("stitch_flat")))
    return lf


def cleanly_scan_parquet(path_: Path | str) -> pl.LazyFrame:
    return _clean(pl.scan_parquet(path_))


def cleanly_scan_parquet_tables(
    parquet_paths: dict[str, Path],
) -> dict[str, pl.LazyFrame]:
    return {name: cleanly_scan_parquet(p) for name, p in parquet_paths.items()}


def build_views(parquet_paths: dict[str, Path]) -> dict[str, pl.LazyFrame]:
    """Joined views over the cleaned tables.

    - ``drugs``: STITCH flat id -> generic drug name (``drug_names`` verbatim)
    - ``side_effects``: every drug-ADR pair (``meddra_all_se``) with the drug name
    - ``frequencies``: label-extracted frequencies (``meddra_freq``) with the drug
      name; ``placebo`` is null unless the row came from a placebo arm
    """
    lfs = cleanly_scan_parquet_tables(parquet_paths)
    drugs = lfs["drug_names"]
    named = drugs.select("stitch_flat", "drug_name")
    return {
        "drugs": drugs,
        "side_effects": lfs["meddra_all_se"].join(named, on="stitch_flat", how="left"),
        "frequencies": lfs["meddra_freq"].join(named, on="stitch_flat", how="left"),
    }


def help() -> None:
    print(build_views.__doc__)


if __name__ == "__main__":
    written = write_manifest()
    for name, entry in written["files"].items():
        print(f"{name:14} {entry['bytes']:>12,} B  {entry['sha256'][:16]}..")
    print(f"\nwrote {_MANIFEST_PATH}")
