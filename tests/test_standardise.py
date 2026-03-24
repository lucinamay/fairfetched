import polars as pl
from rdkit.Chem import Mol
from rdkit.Chem.Draw.rdMolDraw2D import MolToSVG
from rdkit.Chem.rdmolops import RemoveAllHs

# import fairfetched.standardization.mol_expr  # ensure namespace registration #ty: ignore[ruff-f401]
from fairfetched.standardization import mol_expr as me
from fairfetched.standardization.mol_expr import MolExpr
from fairfetched.standardization.mol_functions import remove_stereo


def basic_test():
    df = pl.DataFrame({"name": "mymol", "smiles": "CCCCCO"})

    df.with_columns(
        pl.col("smiles")  # ty: ignore[unresolved-attribute]
        .mol.from_smiles()
        .mol.standardise(*me.PIPELINE_CHEMBL)
        .mol.to_kekulised_smiles()
        .alias("kekulised_smiles")
    )


def test_basic_kekulised_smiles_not_null_polars():
    df = pl.DataFrame({"name": ["mymol"] * 10, "smiles": ["CCCCCO"] * 10})
    parallel = True
    out = df.with_columns(
        pl.col("smiles")  # ty: ignore[unresolved-attribute]
        .mol.from_smiles(parallel=False)
        .mol.standardise(*me.PIPELINE_CHEMBL, parallel=parallel)
        .mol.to_kekulised_smiles(parallel=False)
        .alias("kekulised_smiles")
    )

    assert out.select(pl.col("kekulised_smiles").is_not_null().all()).item()
    assert out.select(pl.col("kekulised_smiles").eq(pl.col("smiles")).all()).item()


def test_standardised_mol_as_binary():
    df = pl.DataFrame(
        {"name": ["mymol"] * 5, "smiles": ["CCCCCO", "CC", "CC@OC", "CCC", "CC(C)OC"]}
    )
    parallel = True
    out = df.with_columns(
        pl.col("smiles")  # ty: ignore[unresolved-attribute]
        .mol.from_smiles(parallel=False)
        .mol.standardise(*me.PIPELINE_CHEMBL, parallel=parallel)
        .alias("mol")
    )

    assert out.get_column("mol").dtype == pl.Binary


def test_basic_kekulised_smiles_not_null():
    df = pl.DataFrame({"name": ["mymol"] * 10, "smiles": ["CCCCCO"] * 10})

    out = df.with_columns(
        MolExpr.from_smiles("smiles")
        .standardise(*me.PIPELINE_CHEMBL, parallel=True)
        .to_kekulised_smiles()
        .alias("kekulised_smiles")
    )

    assert out.select(pl.col("kekulised_smiles").is_not_null().all()).item()
    assert out.select(pl.col("kekulised_smiles").eq(pl.col("smiles")).all()).item()


def test_basic_kekulised_smiles_not_null_lazy():
    df = pl.DataFrame({"name": ["mymol"] * 10, "smiles": ["CCCCCO"] * 10}).lazy()
    parallel = True
    out: pl.DataFrame = df.with_columns(  # ty: ignore[invalid-assignment]
        MolExpr.from_smiles("smiles", parallel)
        .standardise(*me.PIPELINE_CHEMBL, parallel=parallel)
        .to_kekulised_smiles(parallel)
        .alias("kekulised_smiles")
    ).collect()

    assert out.select(pl.col("kekulised_smiles").is_not_null().all()).item()
    assert out.select(pl.col("kekulised_smiles").eq(pl.col("smiles")).all()).item()


def test_intermediate_fine():
    df = pl.DataFrame({"name": ["mymol"] * 10, "smiles": ["CCCCCO"] * 10})
    out = df.with_columns(
        MolExpr.from_smiles("smiles")
        .standardise(*me.PIPELINE_CHEMBL)
        .alias("intermediate")
    )

    assert out.select(pl.col("intermediate").is_not_null().all()).item()
    assert out.get_column("intermediate").dtype == pl.Binary


def test_to_mol_objects():
    df = pl.DataFrame({"name": ["mymol"] * 10, "smiles": ["CCCCCO"] * 10})
    out = df.with_columns(
        MolExpr.from_smiles("smiles")
        .standardise(*me.PIPELINE_CHEMBL)
        .to_mol_objects()
        .alias("mol")
    )

    assert out.select(pl.col("mol").is_not_null().all()).item()
    assert out.get_column("mol").dtype == pl.Object
    assert all(isinstance(i, Mol) for i in out.get_column("mol"))


def test_to_mol_objects():
    df = pl.DataFrame({"name": ["mymol"] * 10, "smiles": ["CCCCCO"] * 10})
    out = df.with_columns(MolExpr.from_smiles("smiles").alias("mol"))

    assert out.select(pl.col("mol").is_not_null().all()).item()


def test_all_parallel():
    df = pl.DataFrame(
        {
            "name": ["mymol"] * 5,
            "smiles": [
                "CCCCCO",
                "O=C(O)[C@@H](N)C",
                "CC(=O)OC1=CC=CC=C1C(=O)O",
                "CCC",
                "CC(C)OC",
            ],
        }
    )
    parallel = True
    out = df.with_columns(MolExpr.from_smiles("smiles", parallel=parallel).alias("mol"))
    assert out.select(pl.col("mol").is_not_null().all()).item()
    assert out.get_column("mol").dtype == pl.Binary

    out = out.with_columns(
        MolExpr.from_col("mol").standardise(
            remove_stereo, *me.PIPELINE_CHEMBL, parallel=parallel
        )
    )
    assert out.select(pl.col("mol").is_not_null().all()).item()
    assert out.get_column("mol").dtype == pl.Binary

    out_ = out.select(
        MolExpr.from_col("mol").to_inchi(parallel=parallel).alias("inchi"),
        MolExpr.from_col("mol")
        .to_inchi_and_auxinfo(parallel=parallel)
        .alias("inchi_aux"),
        MolExpr.from_col("mol").to_kekulised_smiles(parallel=parallel).alias("smiles"),
        MolExpr.from_col("mol").to_inchikey(parallel=parallel).alias("inchikey"),
    )
    assert (
        not out_.get_column("smiles").str.contains("@").any()
    )  # proper removal of stereochemistry

    for i in out_.columns:
        assert out_.select(pl.col(i).is_not_null().all()).item()
        if i == "inchi_aux":
            assert out_.get_column(i).dtype.is_(
                pl.Struct({"inchi": pl.String, "inchi_auxinfo": pl.String})
            )
        else:
            assert out_.get_column(i).dtype == pl.String


def my_func(mol):
    return RemoveAllHs(mol)


def test_custom_pipe():
    df = pl.DataFrame(
        {
            "name": ["mymol"] * 5,
            "smiles": [
                "CCCCCO",
                "O=C(O)[C@@H](N)C",
                "CC(=O)OC1=CC=CC=C1C(=O)O",
                "CCC",
                "CC(C)OC",
            ],
        }
    )

    # out = df.with_columns(
    #     MolExpr.from_smiles("smiles")
    #     .molpipe(RemoveAllHs, parallel=True, return_dtype=pl.Binary)
    #     .alias("mol")
    # )
    # assert out.select(pl.col("mol").is_not_null().all()).item()

    out = df.with_columns(
        MolExpr.from_smiles("smiles").standardise(my_func, parallel=True).alias("mol")
    )
    assert out.select(pl.col("mol").is_not_null().all()).item()
