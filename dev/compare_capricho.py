"""Compare `fairfetched.get._capricho.flag` with CAPRICHO 1.0.4 on random ChEMBL 36
targets: activity sets, removed subset, pChEMBL and every flag. Both outputs are
cached in `out_dir`; delete `pystow_home` (~30 GB) afterwards.

    taskpolicy -c background env POLARS_MAX_THREADS=2 uv run --group capricho \\
        python dev/compare_capricho.py 300 /tmp/claude-capricho-pystow /tmp/claude-compare
"""

import argparse
from pathlib import Path

import polars as pl

from fairfetched.get import Chembl, _capricho
from tests.reference.generate_capricho import read_capricho, run_capricho
from tests.test_capricho import CAPRICHO_COMMENTS

SPEC = "1.0.4"
MAX_ROWS = 3000  # per target under the default selection, to bound the run
ARGS = [
    *("--chembl-version", "36", "--calculate-pchembl", "--strict-mutant-removal"),
    *("--min-assay-size", "5", "--max-assay-size", "1000", "--min-assay-overlap", "2"),
    "--drop-unassigned-chiral",
]
RECIPE = {
    "calculate_pchembl": True,
    "strict_mutant_removal": True,
    "min_assay_size": 5,
    "max_assay_size": 1000,
    "min_assay_overlap": 2,
    "mol_flags": True,
    "drop_unassigned_chiral": True,
}
COMMENTS = CAPRICHO_COMMENTS | {"drop_missing_year": "Missing document date"}
DISCARDED = ("drop_missing_structure", "drop_standardization_failed")
SELECTED = (
    "drop_confidence",
    "drop_assay_type",
    "drop_relation",
    "drop_missing_standard_value",
)  # CAPRICHO's SQL WHERE


def pick_targets(chembl: Chembl, n: int, path: Path, seed: int = 0) -> list[str]:
    """`n` random targets with 1 to `MAX_ROWS` rows under CAPRICHO's default
    selection; cached in `path`, as both outputs are computed for this list."""
    if not path.exists():
        sizes = (
            chembl.view.bioactivity.filter(
                pl.col("confidence_score").is_in([7, 8, 9]),
                pl.col("assay_type").is_in(["B", "F"]),
                standard_relation="=",
            )
            .group_by("target_chembl_id")
            .len()
            .filter(pl.col("len") <= MAX_ROWS)
            .sort("target_chembl_id")  # group_by order varies between runs
            .collect()
        )
        targets = sizes.sample(n, seed=seed)["target_chembl_id"].sort()
        path.write_text("\n".join(targets))
    return path.read_text().split()


def ours(chembl: Chembl, targets: list[str], path: Path) -> pl.DataFrame:
    if not path.exists():
        bioactivity = chembl.view.bioactivity.filter(
            pl.col("target_chembl_id").is_in(targets)
        )
        lf = _capricho.flag(bioactivity, chembl.lfs, **RECIPE)
        lf.remove(pl.any_horizontal(SELECTED)).sink_parquet(path)
    return pl.read_parquet(path)


def theirs(targets: list[str], pystow_home: Path, path: Path) -> pl.DataFrame:
    if not path.exists():
        work = run_capricho(
            SPEC, pystow_home, "random", ["--target-ids", ",".join(targets), *ARGS]
        )
        kept, removed = read_capricho(work, "random")
        pl.concat(
            [kept.with_columns(removed=False), removed.with_columns(removed=True)],
            how="diagonal_relaxed",
        ).write_parquet(path)
    return pl.read_parquet(path)


def compare(o: pl.DataFrame, t: pl.DataFrame) -> pl.DataFrame:
    """One row per check: rows flagged by both, only by us, only by CAPRICHO."""
    kept = o.remove(pl.any_horizontal(DISCARDED))
    ids_o, ids_t = set(kept["activity_id"]), set(t["activity_id"])
    print(
        f"activities: ours {len(ids_o)}, CAPRICHO {len(ids_t)}, "
        f"only ours {len(ids_o - ids_t)}, only CAPRICHO {len(ids_t - ids_o)}"
    )
    comments = pl.concat_str(
        "data_dropping_comment", "data_processing_comment", ignore_nulls=True
    )
    j = kept.join(
        t.select(
            "activity_id",
            "removed",
            comments=comments.fill_null(""),
            pchembl_theirs=pl.col("pchembl_value").fill_nan(None),
        ),
        on="activity_id",
    )
    d = (j["pchembl_value"] - j["pchembl_theirs"]).abs()
    print(
        f"pchembl: null differs {(j['pchembl_value'].is_null() != j['pchembl_theirs'].is_null()).sum()}, "
        f"max |diff| {d.max()}"
    )
    checks = {
        f: (j[f], j["comments"].str.contains(c, literal=True))
        for f, c in COMMENTS.items()
        if f in j.columns
    }
    checks["removed"] = (j["drop_missing_value"] | j["drop_mixture"], j["removed"])
    return pl.DataFrame(
        [
            {
                "check": k,
                "both": (a & b).sum(),
                "only_ours": (a & ~b).sum(),
                "only_capricho": (~a & b).sum(),
            }
            for k, (a, b) in checks.items()
        ]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("n", type=int, help="number of random targets")
    parser.add_argument("pystow_home", type=Path, help="scratch dir for ChEMBL")
    parser.add_argument("out_dir", type=Path, help="cache for both outputs")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    chembl = Chembl.from_version("36")
    targets = pick_targets(chembl, args.n, args.out_dir / "targets.txt")
    print(f"{len(targets)} targets", flush=True)
    o = ours(chembl, targets, args.out_dir / "ours.parquet")
    print("ours done", flush=True)
    t = theirs(targets, args.pystow_home, args.out_dir / "capricho.parquet")
    pl.Config.set_tbl_rows(30)
    print(compare(o, t))
