"""Structure images for frames holding mol binary or SMILES, the polars
counterpart of `PandasTools.ChangeMoleculeRendering`. Polars has no
per-column HTML hook, so `show_mols` renders a head of the frame itself:

    >>> df.pipe(show_mols, "mol", n=5)          # Jupyter: displays as-is
    >>> mo.as_html(df.pipe(show_mols, "mol"))   # marimo
    >>> df.pipe(marimo_table, "mol")            # marimo: paginated, lazy per page
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl
from rdkit.Chem import Mol, MolFromSmiles
from rdkit.Chem.Draw import rdMolDraw2D

Size = tuple[int, int]
_PLACEHOLDER = "__mol{}__"


def _mol(x: bytes | str | None) -> Mol | None:
    if x is None:
        return None
    return Mol(x) if isinstance(x, bytes) else MolFromSmiles(x)  # pyright: ignore[reportCallIssue, reportArgumentType]


def mol_svg(x: bytes | str | None, size: Size = (200, 150)) -> str:
    """SVG text of one mol binary or SMILES; empty string for None or unparseable."""
    mol = _mol(x)
    if mol is None:
        return ""
    drawer = rdMolDraw2D.MolDraw2DSVG(*size)
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def mol_png(x: bytes | str | None, size: Size = (200, 150)) -> bytes:
    """PNG bytes of one mol binary or SMILES; empty bytes for None or unparseable."""
    mol = _mol(x)
    if mol is None:
        return b""
    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


@dataclass(frozen=True)
class Html:
    html: str

    def _repr_html_(self) -> str:
        return self.html


def show_mols(
    df: pl.DataFrame | pl.LazyFrame,
    mol_col: str = "mol",
    n: int = 10,
    size: Size = (200, 150),
) -> Html:
    """The first `n` rows as polars' own HTML table with `mol_col` drawn.

    Pipe-able: `df.pipe(show_mols, "mol")`. Only the head is collected and drawn.
    """
    head = df.lazy().head(n).collect()
    mols = head[mol_col].to_list()
    placeholders = [
        None if x is None else _PLACEHOLDER.format(i) for i, x in enumerate(mols)
    ]
    with pl.Config(tbl_rows=n, tbl_cols=-1):
        html = head.with_columns(pl.Series(mol_col, placeholders))._repr_html_()
    for placeholder, x in zip(placeholders, mols, strict=True):
        if placeholder is not None:
            html = html.replace(f"&quot;{placeholder}&quot;", mol_svg(x, size))
    return Html(html)


def marimo_table(
    df: pl.DataFrame | pl.LazyFrame,
    mol_col: str = "mol",
    size: Size = (200, 150),
    **table_kwargs,
):
    """`mo.ui.table(df, ...)` drawing `mol_col`; marimo renders a page at a time,
    so the whole frame can be passed. Pipe-able: `df.pipe(marimo_table, "mol")`.
    """
    import marimo as mo  # ty: ignore[unresolved-import]  # pyright: ignore[reportMissingImports]

    def cell(x):  # one type for every row: marimo builds a frame from the outputs
        return mo.Html("") if x is None else mo.image(src=mol_png(x, size))

    return mo.ui.table(df, format_mapping={mol_col: cell}, **table_kwargs)
