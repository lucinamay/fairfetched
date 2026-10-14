"""Regenerate CAPRICHO's references for `tests/test_capricho.py` with a given
CAPRICHO version, into `capricho/<name>/<recipe>_data_*`; a recipe is skipped when
its `_recipe.json` (written last) exists. Keeps the rows CAPRICHO outputs
(`_not_aggregated`) and those it removes (`_removed_subset`), as parquet.

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
import os
import shutil
import subprocess
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


def generate(spec: str, pystow_home: Path) -> Path:
    out = OUT / _name(spec)
    todo = [r for r in RECIPES if not (out / f"{r}_data_recipe.json").exists()]
    out.mkdir(parents=True, exist_ok=True)
    for recipe in todo:
        work = run_capricho(spec, pystow_home, recipe, RECIPES[recipe])
        kept, removed = read_capricho(work, recipe)
        kept.write_parquet(out / f"{recipe}_data_not_aggregated.parquet")
        removed.write_parquet(out / f"{recipe}_data_removed_subset.parquet")
        shutil.copy(work / f"{recipe}_data_recipe.json", out)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", help="CAPRICHO version on PyPI, or a pip requirement")
    parser.add_argument("pystow_home", type=Path, help="scratch dir for ChEMBL")
    args = parser.parse_args()
    print(generate(args.spec, args.pystow_home))
