import polars as pl

# import fairfetched.standardization.mol_expr  # ensure namespace registration #ty: ignore[ruff-f401]
from fairfetched.standardization import mol_expr as me


def basic_test():
    df = pl.DataFrame({"name": "mymol", "smiles": "CCCCCO"})

    df.with_columns(
        pl.col("smiles")  # ty: ignore[unresolved-attribute]
        .mol.from_smiles()
        .mol.standardise(*me.CHEMBL_PIPELINE)  # ty: ignore[unresolved-attribute]
        .mol.to_kekulised_smiles()
        .alias("kekulised_smiles")
    )


def test_basic_kekulised_smiles_not_null():
    df = pl.DataFrame({"name": ["mymol"], "smiles": ["CCCCCO"]})

    out = df.with_columns(
        pl.col("smiles")  # ty: ignore[unresolved-attribute]
        .mol.from_smiles()
        .mol.standardise(*me.CHEMBL_PIPELINE)  # ty: ignore[unresolved-attribute]
        .mol.to_kekulised_smiles()
        .alias("kekulised_smiles")
    )

    assert out.select(pl.col("kekulised_smiles").is_not_null().all()).item()
