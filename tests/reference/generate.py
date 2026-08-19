"""Regenerate `standardize_reference.csv`, the expected-value table used by
`tests/test_standardize_correctness.py`.

Uses  RDKit + `chembl_structure_pipeline` / `papyrus_structure_pipeline`
directly as 'ground truth' to catch any unwanted fairfetched wrapper behaviour.

The only identity claim written by hand is a PubChem CID per molecule. Title and
InChIKey are fetched from PubChem here and joined onto the CID, so nothing says
"this name has this key" without a URL you can open. The run aborts if a
computed key disagrees with the fetched one.

`papyrus_*` is empty where Papyrus filtered the molecule out by weight; `mol_weight`
(RDKit exact MW, the same descriptor Papyrus checks) is the column to compare it
against, not a number in a comment.
`published_*` is empty where PubChem has no entry.

Needs network. Run after upgrading RDKit or either structure pipeline.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from typing import Any

import polars as pl
import rdkit
from chembl_structure_pipeline import standardize_mol
from papyrus_structure_pipeline import standardize as papyrus_standardize
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")

OUT = Path(__file__).with_name("standardize_reference.csv")
COMPOUND = "https://pubchem.ncbi.nlm.nih.gov/compound"
PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid"

# name, input SMILES, PubChem CID of the compound this input must standardize
# *to* (None: nothing published to check against), what the entry demonstrates.
#
# For the inputs ChEMBL rewrites, the CID is the *product*: CID 1032 under
# `propionate_anion` is propanoic acid, and the fetched title says so. Every
# such row spells that out in its note.
MOLECULES: list[tuple[str, str, int | None, str]] = [
    # --- the same compound written several ways; standardization should make
    # --- these converge on one InChIKey
    (
        "aspirin_kekule",
        "CC(=O)OC1=CC=CC=C1C(=O)O",
        2244,
        "aspirin drawn with alternating single/double ring bonds",
    ),
    (
        "aspirin_aromatic",
        "CC(=O)Oc1ccccc1C(=O)O",
        2244,
        "aspirin drawn with an aromatic ring",
    ),
    (
        "aspirin_anion",
        "CC(=O)Oc1ccccc1C(=O)[O-]",
        2244,
        "aspirin drawn deprotonated; CID is neutral aspirin, what ChEMBL protonates it back to",
    ),
    (
        "nitrobenzene_pentavalent",
        "O=[N+]([O-])c1ccccc1",
        7416,
        "nitro group written charge-separated",
    ),
    (
        "nitrobenzene_reordered",
        "[O-][N+](=O)c1ccccc1",
        7416,
        "same nitro group, atoms written in the other order",
    ),
    # --- distinct compounds that must NOT be merged
    (
        "l_alanine",
        "O=C(O)[C@@H](N)C",
        5950,
        "(S)-alanine; must stay distinct from the (R) form",
    ),
    (
        "d_alanine",
        "O=C(O)[C@H](N)C",
        71080,
        "(R)-alanine; must stay distinct from the (S) form",
    ),
    (
        "maleic_acid",
        "OC(=O)/C=C\\C(=O)O",
        444266,
        "(Z) diacid; must stay distinct from fumaric acid",
    ),
    (
        "fumaric_acid",
        "OC(=O)/C=C/C(=O)O",
        444972,
        "(E) diacid; must stay distinct from maleic acid",
    ),
    (
        "naproxen_s",
        "COc1ccc2cc([C@H](C)C(=O)O)ccc2c1",
        156391,
        "(S)-naproxen, the marketed drug",
    ),
    (
        "naproxen_r",
        "COc1ccc2cc([C@@H](C)C(=O)O)ccc2c1",
        169118,
        "(R)-naproxen, the inactive enantiomer",
    ),
    # --- inputs where ChEMBL standardization changes the structure; these are
    # --- what catches a pipeline that has silently stopped running
    (
        "sodium_acetate_covalent",
        "CC(=O)O[Na]",
        31372,
        "Na drawn covalently bonded; CID is the ionic salt ChEMBL splits it into",
    ),
    (
        "propionate_anion",
        "CCC(=O)[O-]",
        1032,
        "bare carboxylate; CID is propanoic acid, what ChEMBL protonates it to",
    ),
    (
        "ethylammonium",
        "CC[NH3+]",
        6341,
        "bare ammonium cation; CID is ethylamine, what ChEMBL deprotonates it to",
    ),
    (
        "ethanethiolate",
        "CC[S-]",
        6343,
        "bare thiolate; CID is ethanethiol, what ChEMBL protonates it to",
    ),
    (
        "iron_diacetate",
        "CC(=O)O[Fe]OC(C)=O",
        None,
        "Fe drawn covalently bonded; ChEMBL leaves it alone, and PubChem has no entry for this drawing",
    ),
    # --- salts and zwitterions ChEMBL standardization passes through unchanged
    (
        "sodium_acetate_ionic",
        "[Na+].CC(=O)[O-]",
        31372,
        "already-ionic salt; standardize_mol does not strip it",
    ),
    (
        "betaine",
        "C[N+](C)(C)CC(=O)[O-]",
        247,
        "zwitterion whose charges must survive uncharging",
    ),
    # --- ordinary compounds spanning the Papyrus 200-800 Da window (mol_weight
    # --- column is computed, not asserted here -- see build())
    ("ethanol", "CCO", 702, "small molecule, well under the Papyrus floor"),
    ("caffeine", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C", 2519, "just under the Papyrus floor"),
    ("paracetamol", "CC(=O)Nc1ccc(O)cc1", 1983, "under the Papyrus floor"),
    (
        "ibuprofen",
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        3672,
        "just over the Papyrus floor",
    ),
    (
        "clonazepam",
        "Clc1ccccc1C1=NCC(=O)Nc2ccc(cc12)[N+](=O)[O-]",
        2802,
        "comfortably inside the Papyrus window",
    ),
    (
        "cholesterol",
        "C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)O)C)C",
        5997,
        "stereochemistry and skeleton both survive Papyrus intact",
    ),
    (
        "testosterone",
        "C[C@]12CC[C@H]3[C@H]([C@@H]1CC[C@@H]2O)CCC4=CC(=O)CC[C@]34C",
        6013,
        "Papyrus keeps the stereocentres but shifts the enol tautomer",
    ),
    (
        "dexamethasone",
        "C[C@@H]1C[C@H]2[C@@H]3CCC4=CC(=O)C=C[C@]4(C)[C@@]3(F)[C@@H](O)C[C@]2(C)[C@@]1(O)C(=O)CO",
        5743,
        "Papyrus returns a different skeleton than ChEMBL",
    ),
    (
        "phenylboronic_acid",
        "OB(O)c1ccccc1",
        66827,
        "boron; rejected by only_organic's element set",
    ),
]


def key(mol: Chem.Mol | None) -> str | None:
    return None if mol is None else Chem.MolToInchiKey(mol) or None


def nostereo(mol: Chem.Mol) -> Chem.Mol:
    copy = Chem.Mol(mol)
    Chem.RemoveStereochemistry(copy)
    return copy


def papyrus(mol: Chem.Mol) -> Chem.Mol | None:
    """Papyrus raises rather than returns for some of what its filters reject
    (mixtures, inorganics, and by default anything outside 200-800 Da)."""
    try:
        return papyrus_standardize(Chem.Mol(mol))
    except Exception:
        return None


def per_mol(fn: Callable[[Chem.Mol], Any], return_dtype: pl.DataType = pl.String) -> pl.Expr:
    """`fn` applied to each input SMILES parsed into a fresh Mol -- fresh
    because both pipelines mutate what they are handed."""

    def run(smiles: str) -> Any:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise SystemExit(f"RDKit cannot parse {smiles!r}")
        return fn(mol)

    return pl.col("input_smiles").map_elements(run, return_dtype=return_dtype)


def published() -> pl.DataFrame:
    """Title and InChIKey per CID, in one PUG-REST call, as a frame to join on.
    PubChem collapses the CIDs listed twice in MOLECULES, so this has to be a
    join rather than a column pasted alongside."""
    cids = sorted({cid for _, _, cid, _ in MOLECULES if cid})
    url = f"{PUG}/{','.join(map(str, cids))}/property/Title,InChIKey/JSON"
    with urllib.request.urlopen(url, timeout=60) as response:
        properties = json.load(response)["PropertyTable"]["Properties"]
    return pl.DataFrame(properties).rename(
        {
            "CID": "published_cid",
            "Title": "published_title",
            "InChIKey": "published_inchikey",
        }
    )


def build() -> pl.DataFrame:
    table = (
        pl.DataFrame(
            MOLECULES,
            schema={
                "name": pl.String,
                "input_smiles": pl.String,
                "published_cid": pl.Int64,
                "note": pl.String,
            },
            orient="row",
        )
        .with_columns(
            # exact MW of the input, same descriptor Papyrus filters on
            # (small_molecule_min_mw/max_mw in standardizer.py, default 200-800)
            mol_weight=per_mol(Descriptors.ExactMolWt, return_dtype=pl.Float64),
            # RDKit alone: the baseline a no-op pipeline would give
            rdkit_smiles=per_mol(Chem.MolToSmiles),
            rdkit_inchikey=per_mol(key),
            chembl_smiles=per_mol(lambda m: Chem.MolToSmiles(standardize_mol(m))),
            chembl_inchikey=per_mol(lambda m: key(standardize_mol(m))),
            chembl_nostereo_inchikey=per_mol(
                lambda m: key(standardize_mol(nostereo(m)))
            ),
            papyrus_inchikey=per_mol(lambda m: key(papyrus(m))),
            papyrus_nostereo_inchikey=per_mol(lambda m: key(papyrus(nostereo(m)))),
        )
        .join(published(), on="published_cid", how="left")
    )

    # ne_missing so a CID PubChem returned nothing for (retired, merged) fails
    # here too, rather than silently leaving the row unchecked
    wrong = table.filter(
        pl.col("published_cid").is_not_null()
        & pl.col("chembl_inchikey").ne_missing(pl.col("published_inchikey"))
    )
    if not wrong.is_empty():
        raise SystemExit(
            "computed keys disagree with PubChem:\n"
            + "\n".join(
                f"  {r['name']}: {r['chembl_inchikey']} != {r['published_inchikey']} "
                f"({COMPOUND}/{r['published_cid']} {r['published_title']})"
                for r in wrong.to_dicts()
            )
        )
    return table


if __name__ == "__main__":
    versions = " ".join(
        f"{pkg}={v}"
        for pkg, v in [("rdkit", rdkit.__version__)]
        + [
            (p, version(p))
            for p in ("chembl_structure_pipeline", "papyrus_structure_pipeline")
        ]
    )
    with OUT.open("wb") as handle:
        handle.write(
            f"# generated by {Path(__file__).name} -- do not hand-edit\n".encode()
        )
        handle.write(f"# computed with {versions}\n".encode())
        handle.write(
            f"# published_* fetched from {COMPOUND}/<published_cid>\n".encode()
        )
        build().write_csv(handle)
    print(f"wrote {OUT}")
