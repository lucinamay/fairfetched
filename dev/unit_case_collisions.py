"""ChEMBL 36 `standard_units` whose lowercased spelling is a key of
`de.UNIT_CONVERSIONS`: the counts behind `de.CASE_SENSITIVE_UNITS`,
`de.UNIT_TYPE_MISMATCH` and the `convert_units` docstring. Rows with a
`standard_value`; which quantity a spelling stands for was judged from
`standard_type` and `description_assay`.

    uv run python dev/unit_case_collisions.py
"""

import polars as pl

from fairfetched.get import Chembl
from fairfetched.prep import data_expr as de

RESPELLED = ["mm", "um", "nm", "m", "nM/s", "nM s-1", "10'-8M/s"]  # SI spelling differs


def converting_spellings(b: pl.LazyFrame) -> pl.DataFrame:
    """Every spelling `convert_units()` notes, with its target and top types."""
    spelling = pl.col("note_unit_converted").struct.field("standard_units")
    return (
        b.with_columns(**de.convert_units())
        .filter(pl.col("note_unit_converted").is_not_null())
        .group_by(spelling=spelling, target="standard_units")
        .agg(
            n=pl.len(),
            types=pl.col("standard_type").value_counts(sort=True).head(5),
        )
        .sort("n", descending=True)
        .collect()
    )


def respelled_by_type(b: pl.LazyFrame) -> pl.DataFrame:
    """Rows per respelled unit and type, with three assay descriptions."""
    return (
        b.filter(pl.col("standard_units").is_in(RESPELLED))
        .group_by("standard_units", "standard_type")
        .agg(
            n=pl.len(),
            n_assays=pl.col("assay_id").n_unique(),
            desc=pl.col("description_assay").unique().head(3),
        )
        .sort("standard_units", "n", descending=[False, True])
        .collect()
    )


def types_per_target(b: pl.LazyFrame) -> pl.DataFrame:
    """Types per conversion target with their median converted value: the search
    for `de.UNIT_TYPE_MISMATCH` candidates in every table."""
    return (
        b.with_columns(**de.convert_units())
        .filter(pl.col("note_unit_converted").is_not_null())
        .group_by("standard_units", "standard_type")
        .agg(n=pl.len(), median=pl.col("standard_value").median())
        .sort("standard_units", "n", descending=[False, True])
        .collect()
    )


def type_mismatches(b: pl.LazyFrame) -> pl.DataFrame:
    """Rows `de.has_unit_type_mismatch` flags, per unit, with the matched types
    and the share of values within [-15, 15] (log scale)."""
    v = pl.col("standard_value")
    return (
        b.filter(de.has_unit_type_mismatch())
        .group_by("standard_units")
        .agg(
            n=pl.len(),
            in_log_range=v.is_between(-15, 15).mean(),
            types=pl.col("standard_type").value_counts(sort=True).head(8),
        )
        .sort("n", descending=True)
        .collect()
    )


def raw_um(b: pl.LazyFrame) -> pl.DataFrame:
    """Raw `units` == "um": `standard_units` and standard_value / value, which
    is 1000 where ChEMBL read it as uM."""
    return (
        b.filter(units="um")
        .group_by("standard_units")
        .agg(n=pl.len(), ratio=(pl.col("standard_value") / pl.col("value")).median())
        .sort("n", descending=True)
        .collect()
    )


if __name__ == "__main__":
    b = Chembl.from_version("36").view.bioactivity.filter(
        pl.col("standard_value").is_not_null()
    )
    with pl.Config(tbl_rows=-1, fmt_str_lengths=120, tbl_width_chars=250):
        print(converting_spellings(b))
        print(respelled_by_type(b))
        print(types_per_target(b))
        print(type_mismatches(b))
        print(raw_um(b))
