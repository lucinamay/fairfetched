import polars as pl
import pytest
from rdkit.Chem import Mol
from rdkit.Chem.rdmolops import RemoveAllHs

from fairfetched.prep import mol_expr as me
from fairfetched.prep.mol_expr import MolExpr
from fairfetched.prep.mol_functions import remove_stereo

TEST_DF = pl.DataFrame(
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


def basic_test():
    df = pl.DataFrame({"name": "mymol", "smiles": "CCCCCO"})

    _ = df.with_columns(
        pl.col("smiles")  # ty: ignore[unresolved-attribute]
        .mol.from_smiles()  # pyright: ignore[reportAttributeAccessIssue]
        .mol.standardise(*me.STEPS_CHEMBL)
        .mol.to_kekulized_smiles()
        .alias("kekulised_smiles")
    )


def test_namespace_on_binary_mol():
    # namespace methods work on already-binary mol columns, not entry points
    df = TEST_DF
    parallel = True
    out = df.with_columns(
        MolExpr.from_smiles("smiles", parallel=False)
        .standardize(*me.STEPS_CHEMBL, parallel=parallel)
        .alias("mol")
    )

    # now use namespace methods on the binary mol column
    out = out.with_columns(
        pl.col("mol")  # ty: ignore[unresolved-attribute]
        .mol.to_kekulized_smiles(parallel=False)  # pyright: ignore[reportAttributeAccessIssue]
        .alias("kekulised_smiles")
    )

    assert out.select(pl.col("kekulised_smiles").is_not_null().all()).item()
    assert out.get_column("mol").dtype == pl.Binary


def test_basic_kekulised_smiles_not_null():
    df = pl.DataFrame({"name": ["mymol"] * 10, "smiles": ["CCCCCO"] * 10})

    out = df.with_columns(
        MolExpr.from_smiles("smiles")
        .standardize(*me.STEPS_CHEMBL, parallel=True)
        .to_kekulized_smiles()
        .alias("kekulised_smiles")
    )

    assert out.select(pl.col("kekulised_smiles").is_not_null().all()).item()
    assert out.select(pl.col("kekulised_smiles").eq(pl.col("smiles")).all()).item()


def test_basic_kekulised_smiles_not_null_lazy():
    df = pl.DataFrame({"name": ["mymol"] * 10, "smiles": ["CCCCCO"] * 10}).lazy()
    parallel = True
    out: pl.DataFrame = df.with_columns(
        MolExpr.from_smiles("smiles", parallel)
        .standardize(*me.STEPS_CHEMBL, parallel=parallel)
        .to_kekulized_smiles(parallel)
        .alias("kekulised_smiles")
    ).collect()

    assert out.select(pl.col("kekulised_smiles").is_not_null().all()).item()
    assert out.select(pl.col("kekulised_smiles").eq(pl.col("smiles")).all()).item()


@pytest.mark.parametrize("fp_size", [1024, 2048])
def test_morgan_fp(fp_size):
    df = TEST_DF
    out = df.with_columns(
        MolExpr.from_smiles(parallel=False)
        .to_morgan_fp(fp_size=fp_size)
        .alias("morgan")
    )

    assert out.get_column("morgan").dtype == pl.Array
    assert out.get_column("morgan").dtype == pl.Array(pl.UInt8, shape=fp_size)
    assert out.get_column("morgan").n_unique() == df.n_unique(["smiles"])


def test_intermediate_fine():
    df = pl.DataFrame({"name": ["mymol"] * 10, "smiles": ["CCCCCO"] * 10})
    out = df.with_columns(
        MolExpr.from_smiles("smiles")
        .standardize(*me.STEPS_CHEMBL)
        .alias("intermediate")
    )

    assert out.select(pl.col("intermediate").is_not_null().all()).item()
    assert out.get_column("intermediate").dtype == pl.Binary


def test_to_mol_objects():
    df = pl.DataFrame({"name": ["mymol"] * 10, "smiles": ["CCCCCO"] * 10})
    out = df.with_columns(
        MolExpr.from_smiles("smiles")
        .standardize(*me.STEPS_CHEMBL)
        .to_mol_objects()
        .alias("mol")
    )

    assert out.select(pl.col("mol").is_not_null().all()).item()
    assert out.get_column("mol").dtype == pl.Object
    assert all(isinstance(i, Mol) for i in out.get_column("mol"))


def test_intermediate():
    df = TEST_DF
    out = df.with_columns(MolExpr.from_smiles("smiles").alias("mol"))

    assert out.select(pl.col("mol").is_not_null().all()).item()


def test_all_parallel():
    df = TEST_DF
    parallel = True
    out = df.with_columns(MolExpr.from_smiles("smiles", parallel=parallel).alias("mol"))
    assert out.select(pl.col("mol").is_not_null().all()).item()
    assert out.get_column("mol").dtype == pl.Binary

    out = out.with_columns(
        MolExpr.col("mol").standardize(
            remove_stereo, *me.STEPS_CHEMBL, parallel=parallel
        )
    )
    assert out.select(pl.col("mol").is_not_null().all()).item()
    assert out.get_column("mol").dtype == pl.Binary

    out_ = out.select(
        MolExpr.col("mol").to_inchi(parallel=parallel).alias("inchi_separate"),
        MolExpr.col("mol").to_inchi_and_auxinfo(parallel=parallel),
        MolExpr.col("mol").to_kekulized_smiles(parallel=parallel).alias("smiles"),
        MolExpr.col("mol").to_inchikey(parallel=parallel).alias("inchikey"),
    )
    assert (
        not out_.get_column("smiles").str.contains("@").any()
    )  # proper removal of stereochemistry

    for i in out_.columns:
        assert out_.select(pl.col(i).is_not_null().all()).item()
        assert out_.get_column(i).dtype == pl.String


def my_func(mol):
    return RemoveAllHs(mol)


def test_custom_pipe():
    df = TEST_DF

    # out = df.with_columns(
    #     MolExpr.from_smiles("smiles")
    #     .molpipe(RemoveAllHs, parallel=True, return_dtype=pl.Binary)
    #     .alias("mol")
    # )
    # assert out.select(pl.col("mol").is_not_null().all()).item()

    out = df.with_columns(
        MolExpr.from_smiles("smiles").standardize(my_func, parallel=True).alias("mol")
    )
    assert out.select(pl.col("mol").is_not_null().all()).item()


class TestCaprichoMolFlags:
    """Expected values follow from the structures; CAPRICHO's own functions
    are compared when the `capricho` group is installed."""

    smiles = ("C[C@H](N)O", "CC(N)O", "C/C=C/C", "CC.O", "CC(N)C(O)Cl")
    # (R)-1-aminoethanol: defined center; racemic: one undefined center;
    # (E)-but-2-ene: bond stereo only; ethane + water: 2 fragments;
    # 2-amino-1-chloropropan-1-ol: two undefined centers
    has_stereo = (True, False, True, False, False)
    undefined = (0, 1, 0, 0, 2)
    fragments = (1, 1, 1, 2, 1)

    @pytest.fixture
    def out(self) -> pl.DataFrame:
        mol = MolExpr.col("mol")
        return (
            pl.DataFrame({"smiles": [*self.smiles, None]})
            .with_columns(mol=MolExpr.from_smiles())
            .select(
                stereo=mol.has_stereo(),
                undefined=mol.num_undefined_stereocenters(),
                fragments=mol.num_fragments(),
            )
        )

    def test_values(self, out):
        assert out["stereo"].to_list() == [*self.has_stereo, None]
        assert out["undefined"].to_list() == [*self.undefined, None]
        assert out["fragments"].to_list() == [*self.fragments, None]

    def test_against_capricho(self, out):
        pytest.importorskip("Capricho")
        import pandas as pd  # pyright: ignore[reportMissingImports]
        from Capricho.chembl.data_flag_functions import (  # pyright: ignore[reportMissingImports]
            flag_stereochemistry_removal,
        )
        from Capricho.core.stereo import (  # pyright: ignore[reportMissingImports]
            find_undefined_stereocenters,
        )

        undefined = [len(find_undefined_stereocenters(s)) for s in self.smiles]
        pdf = flag_stereochemistry_removal(pd.DataFrame({"s": list(self.smiles)}), "s")
        stereo = pdf["data_processing_comment"].str.contains("Stereochemistry").tolist()
        assert out["undefined"].to_list()[:-1] == undefined
        assert out["stereo"].to_list()[:-1] == stereo
