"""Regenerate CAPRICHO's references for `tests/test_capricho.py` with a given
CAPRICHO version, into `capricho/<name>/<recipe>_data_*`; a recipe is skipped when
its `_recipe.json` (written last) exists. Keeps the rows CAPRICHO outputs
(`_not_aggregated`) and those it removes (`_removed_subset`), as parquet, and
CAPRICHO's `aggregate_data` on the former (`_aggregated`) and its unit
conversions (`unit_conversions.parquet`), run in-process with the installed
CAPRICHO, which must be the given version.

ADORA3 runs the command of CAPRICHO's `tests/resources/ADORA3_data_recipe.json`;
CHEMBL281 the options ADORA3 leaves off (mixtures, undefined stereo, assay
overlap; CAPRICHO's default overlap is 0, 5 is ours); CHEMBL281_nondefault the
switches CHEMBL281 leaves at their default, and non-default selections;
CHEMBL232 real cases for "Log Ki" types (admitted by `--bioactivity-type` with
`--calculate-pchembl`), inactivity comments and non-molar units. CAPRICHO
reads ChEMBL through chembl_downloader (pystow); fairfetched's ChEMBL archive is
linked into `pystow_home`, so nothing is downloaded, but CAPRICHO extracts the
~30 GB `.db` there. Delete `pystow_home` afterwards.

    uv run python tests/reference/generate_capricho.py 1.0.4 /tmp/capricho-pystow
    uv run python tests/reference/generate_capricho.py \
        "capricho @ git+https://github.com/David-Araripe/Capricho@c3b505a2" /tmp/capricho-pystow
"""

import argparse
import json
import os
import shutil
import subprocess
from importlib.metadata import version as package_version
from pathlib import Path

import polars as pl
import rdkit

from fairfetched.get import Chembl

OUT = Path(__file__).with_name("capricho")
CHEMBL_VERSION = "36"
RECIPES = {  # name -> `capricho get` options
    "ADORA3": [
        *("--target-ids", "CHEMBL256", "--calculate-pchembl"),
        *("--chembl-release", "31", "--chembl-version", CHEMBL_VERSION),
        "--require-doc-date",
        *("--max-assay-size", "100", "--min-assay-size", "20"),
        "--strict-mutant-removal",
    ],
    "CHEMBL281": [
        *("--target-ids", "CHEMBL281", "--calculate-pchembl"),
        *("--chembl-version", CHEMBL_VERSION),
        *("--min-assay-overlap", "5", "--drop-unassigned-chiral"),
    ],
    "CHEMBL281_nondefault": [
        *("--target-ids", "CHEMBL281", "--calculate-pchembl"),
        *("--chembl-version", CHEMBL_VERSION, "--chembl-release", "30"),
        *("--bioactivity-type", "Ki,IC50", "--standard-units", "nM,uM"),
        *("--standard-relation", "=,<,>", "--confidence-scores", "8,9"),
        *("--require-doc-date", "--min-assay-overlap", "3"),
        *(
            "--no-chirality",
            "--drop-unassigned-chiral",
            "--dont-curate-annotation-errors",
        ),
    ],
    "CHEMBL232": [
        *("--target-ids", "CHEMBL232", "--calculate-pchembl"),
        *("--chembl-version", CHEMBL_VERSION, "--bioactivity-type", "Ki,Inhibition"),
    ],
}


def _name(spec: str) -> str:
    """'1.0.4' -> '1.0.4'; '... @ git+...@<rev>' -> '<rev>'."""
    return spec.rsplit("@", 1)[-1].strip()


def run_capricho(spec: str, pystow_home: Path, name: str, args: list[str]) -> Path:
    """`capricho get <args>` as `spec`, writing `<name>_data*.csv` into the
    returned folder; fairfetched's ChEMBL archive is linked in, not downloaded.
    RDKit is pinned to this environment's version, as its stereo perception
    changes between releases (`find_undefined_stereocenters`)."""
    tar = (
        pystow_home
        / "chembl"
        / CHEMBL_VERSION
        / f"chembl_{CHEMBL_VERSION}_sqlite.tar.gz"
    )
    tar.parent.mkdir(parents=True, exist_ok=True)
    if not tar.exists():  # chembl_downloader unlinks it after extracting
        tar.symlink_to(Chembl.from_version(CHEMBL_VERSION).raw_paths["sql_db"])
    work = pystow_home / "run"
    work.mkdir(exist_ok=True)
    from_ = spec if "@" in spec else f"capricho=={spec}"
    subprocess.run(
        ["uvx", "--from", from_, "--with", f"rdkit=={rdkit.__version__}"]
        + ["capricho", "get", *args]
        + ["--output-path", f"{name}_data.csv"],
        cwd=work,
        env=os.environ | {"PYSTOW_HOME": str(pystow_home)},
        check=True,
    )
    return work


def read_capricho(work: Path, name: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """CAPRICHO's output and removed subset, with the same dtypes; all-null
    columns (pchembl_value) would otherwise infer as String."""
    kept = pl.read_csv(
        work / f"{name}_data_not_aggregated.csv", infer_schema_length=None
    )
    removed = pl.read_csv(work / f"{name}_data_removed_subset.csv", schema=kept.schema)
    return kept, removed


def aggregate_reference(out: Path, recipe: str) -> None:
    """`aggregate_data` with the recipe's options on `<recipe>_data_not_aggregated`,
    which is CAPRICHO's `pre_agg_df` (`capricho get`); pipe-joined values stay
    strings."""
    from Capricho.cli.chembl_data_pipeline import aggregate_data

    options = json.loads((out / f"{recipe}_data_recipe.json").read_text())
    if options["capricho version"] != package_version("capricho"):
        raise RuntimeError(f"installed CAPRICHO is not {options['capricho version']}")
    pre_agg = pl.read_parquet(out / f"{recipe}_data_not_aggregated.parquet")
    df = aggregate_data(
        pre_agg.to_pandas(),
        chirality=options["chirality"],
        extra_multival_cols=options["metadata_columns"],
        extra_id_cols=options["id_columns"],
        aggregate_mutants=options["aggregate_mutants"],
        compound_equality=options["compound_equality"],
        value_col=options["value_column"],
    )
    strings = {c: "string" for c in df.select_dtypes("object").columns}
    pl.from_pandas(df.astype(strings)).write_parquet(
        out / f"{recipe}_data_aggregated.parquet"
    )


def unit_reference(out: Path) -> None:
    """CAPRICHO's five `convert_*_units`, each followed by `flag_unit_conversion`
    as in `get_standardize_and_clean_workflow`, on every distinct
    `standard_units` of ChEMBL 36 with `standard_value` 1.0 and null, into
    `unit_conversions.parquet` (`units_in`, `value_in` before conversion)."""
    from Capricho.chembl import unit_conversions as uc
    from Capricho.chembl.data_flag_functions import flag_unit_conversion

    if out.name != package_version("capricho"):
        raise RuntimeError(f"installed CAPRICHO is not {out.name}")
    units = (
        Chembl.from_version(CHEMBL_VERSION)
        .view.bioactivity.select(pl.col("standard_units").unique())
        .collect()
    )
    df = (
        units.join(pl.DataFrame({"standard_value": [1.0, None]}), how="cross")
        .with_columns(
            units_in="standard_units",
            value_in="standard_value",
            data_processing_comment=pl.lit(""),
        )
        .to_pandas()
    )
    for convert in (
        uc.convert_permeability_units,
        uc.convert_molar_concentration_units,
        uc.convert_mass_concentration_units,
        uc.convert_dose_units,
        uc.convert_time_units,
    ):
        df = flag_unit_conversion(convert(df))
    pl.from_pandas(df).write_parquet(out / "unit_conversions.parquet")


def generate(spec: str, pystow_home: Path) -> Path:
    out = OUT / _name(spec)
    if not (out / "unit_conversions.parquet").exists():
        unit_reference(out)
    for recipe in RECIPES:
        if (out / f"{recipe}_data_recipe.json").exists() and not (
            out / f"{recipe}_data_aggregated.parquet"
        ).exists():
            aggregate_reference(out, recipe)
    todo = [r for r in RECIPES if not (out / f"{r}_data_recipe.json").exists()]
    out.mkdir(parents=True, exist_ok=True)
    for recipe in todo:
        work = run_capricho(spec, pystow_home, recipe, RECIPES[recipe])
        kept, removed = read_capricho(work, recipe)
        kept.write_parquet(out / f"{recipe}_data_not_aggregated.parquet")
        removed.write_parquet(out / f"{recipe}_data_removed_subset.parquet")
        shutil.copy(work / f"{recipe}_data_recipe.json", out)
        aggregate_reference(out, recipe)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", help="CAPRICHO version on PyPI, or a pip requirement")
    parser.add_argument("pystow_home", type=Path, help="scratch dir for ChEMBL")
    args = parser.parse_args()
    print(generate(args.spec, args.pystow_home))
