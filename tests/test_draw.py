"""`fairfetched.prep.draw`: `show_mols` renders the head of a frame as
polars' HTML table with the mol column drawn; `marimo_table` wraps
`mo.ui.table` with a drawing `format_mapping`.

Expected SVGs come from `rdMolDraw2D` called directly in the test, so the
assertions compare against RDKit, not against `mol_svg`.
"""

import polars as pl
import pytest
from rdkit.Chem import Mol, MolFromSmiles
from rdkit.Chem.Draw import rdMolDraw2D

from fairfetched.prep.draw import (
    Html,
    marimo_table,
    mol_png,
    mol_svg,
    show_mols,
)
from fairfetched.prep.mol_expr import MolExpr

SMILES = ["CCO", "c1ccccc1", None, "CC(=O)O"]
DF = pl.DataFrame({"id": list(range(len(SMILES))), "smiles": SMILES}).with_columns(
    MolExpr.from_smiles("smiles").to_binary().alias("mol")
)


def rdkit_svg(mol: Mol, size=(200, 150)) -> str:
    drawer = rdMolDraw2D.MolDraw2DSVG(*size)
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


class TestShowMols:
    def test_draws_every_non_null_mol_and_keeps_other_cells(self) -> None:
        """Oracle draws `Mol(binary)`: RDKit's drawing of a binary-roundtripped
        mol can differ from the SMILES-parsed one (CC(=O)O does)."""
        html = show_mols(DF, "mol").html
        assert html.count("<svg") == 3
        for b in DF["mol"].drop_nulls():
            assert rdkit_svg(Mol(b)) in html  # pyright: ignore[reportCallIssue, reportArgumentType]
        assert "<td>3</td>" in html and "__mol" not in html

    def test_null_mol_shows_as_null(self) -> None:
        html = show_mols(DF.select("mol"), "mol").html
        assert html.count("<td>null</td>") == 1

    def test_smiles_column_is_drawn_too(self) -> None:
        html = show_mols(DF.select("smiles"), "smiles").html
        assert html.count("<svg") == 3

    def test_pipe_on_lazyframe_takes_head(self) -> None:
        out = DF.lazy().pipe(show_mols, "mol", n=2)
        assert isinstance(out, Html)
        assert out._repr_html_().count("<svg") == 2
        assert "<td>3</td>" not in out.html

    def test_n_above_polars_default_is_not_elided(self) -> None:
        big = pl.concat([DF.select("smiles")] * 4)  # 16 rows, 12 mols
        html = show_mols(big, "smiles", n=16).html
        assert html.count("<svg") == 12 and "&hellip;" not in html

    def test_size_is_passed_to_rdkit(self) -> None:
        html = show_mols(DF.head(1), "mol", size=(80, 60)).html
        assert rdkit_svg(Mol(DF["mol"][0]), (80, 60)) in html  # pyright: ignore[reportCallIssue, reportArgumentType]

    def test_single_mol_helpers(self) -> None:
        assert mol_svg("CCO") == rdkit_svg(MolFromSmiles("CCO"))
        assert mol_png("CCO").startswith(b"\x89PNG")
        assert mol_svg(None) == "" and mol_png(None) == b""


@pytest.fixture
def mo():
    return pytest.importorskip("marimo")  # skips these tests only, not the module


class TestMarimoTable:
    def test_returns_table_with_drawing_format_mapping(self, mo) -> None:
        table = DF.pipe(marimo_table, "mol", page_size=2)
        assert isinstance(table, mo.ui.table)
        cell = table._format_mapping["mol"]
        assert str(cell(DF["mol"][0]).text).startswith(
            "<img src='data:image/png;base64,"
        )
        assert (
            cell(None).text == ""
        )  # not a broken <img>, which an empty PNG would give

    def test_accepts_lazyframe(self, mo) -> None:
        assert isinstance(DF.lazy().pipe(marimo_table), mo.ui.table)
