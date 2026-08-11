"""Scientific-correctness tests for `fairfetched.standardize`.

The existing `test_standardize.py` asserts not-null and dtype only, so it passes
against a no-op `standardize()`. This file pins actual chemistry: reference
InChIKeys, idempotence, convergence, and the null-provenance of every silent
failure path.

Tests marked `xfail(strict=True)` document confirmed bugs. They flip to failing
(and so must be unmarked) once the bug is fixed.
"""

import logging

import polars as pl
import pytest
from rdkit import RDLogger
from rdkit.Chem import AddHs, MolFromSmiles, MolToSmiles

from fairfetched.standardize import mol_expr as me
from fairfetched.standardize.mol_expr import MolExpr
from fairfetched.standardize.mol_functions import (
    _binary_to_inchikey,
    chembl_standardize,
    no_mixtures,
    only_organic,
    papyrus_standardize,
    remove_stereo,
    safe_step,
    valid_inchi,
    via_inchi,
)
from fairfetched.standardize.pipeline import STEPS_PAPYRUS, STEPS_PAPYRUS_NOSTEREO

RDLogger.DisableLog("rdApp.*")

chembl = pytest.importorskip("chembl_structure_pipeline")
papyrus = pytest.importorskip("papyrus_structure_pipeline")


# name, input smiles, expected InChIKey after STEPS_CHEMBL, after STEPS_PAPYRUS
# (None = the pipeline rejects it). Keys cross-checked against published values.
REFERENCE = [
    ("aspirin_kekule", "CC(=O)OC1=CC=CC=C1C(=O)O", "BSYNRYMUTXBXSQ-UHFFFAOYSA-N", None),
    ("aspirin_aromatic", "CC(=O)Oc1ccccc1C(=O)O", "BSYNRYMUTXBXSQ-UHFFFAOYSA-N", None),
    ("aspirin_anion", "CC(=O)Oc1ccccc1C(=O)[O-]", "BSYNRYMUTXBXSQ-UHFFFAOYSA-N", None),
    ("caffeine", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C", "RYYVLZVUVIJVGH-UHFFFAOYSA-N", None),
    ("paracetamol", "CC(=O)Nc1ccc(O)cc1", "RZVAJINKPMORJF-UHFFFAOYSA-N", None),
    ("ethanol", "CCO", "LFQSCWFLJHTTHZ-UHFFFAOYSA-N", None),
    ("l_alanine", "O=C(O)[C@@H](N)C", "QNAYBMKLOCPYGJ-REOHCLBHSA-N", None),
    ("d_alanine", "O=C(O)[C@H](N)C", "QNAYBMKLOCPYGJ-UWTATZPHSA-N", None),
    ("sodium_acetate", "[Na+].CC(=O)[O-]", "VMHLLURERBWHNL-UHFFFAOYSA-M", None),
    ("betaine", "C[N+](C)(C)CC(=O)[O-]", "KWIUHFFTVRNATP-UHFFFAOYSA-N", None),
    ("nitrobenzene_a", "O=[N+]([O-])c1ccccc1", "LQNUZADURLCDLV-UHFFFAOYSA-N", None),
    ("nitrobenzene_b", "[O-][N+](=O)c1ccccc1", "LQNUZADURLCDLV-UHFFFAOYSA-N", None),
    ("maleic_acid", "OC(=O)/C=C\\C(=O)O", "VZCYOOQTPOCHFL-UPHRSURJSA-N", None),
    ("fumaric_acid", "OC(=O)/C=C/C(=O)O", "VZCYOOQTPOCHFL-OWOJBTEDSA-N", None),
    ("phenylboronic_acid", "OB(O)c1ccccc1", "HXITXNWTGFUOAU-UHFFFAOYSA-N", None),
    # >= 200 Da, so these two survive Papyrus' small-molecule filter
    (
        "ibuprofen",
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "HEFNNWSXXWATRW-UHFFFAOYSA-N",
        "HEFNNWSXXWATRW-UHFFFAOYSA-N",
    ),
    (
        "clonazepam",
        "Clc1ccccc1C1=NCC(=O)Nc2ccc(cc12)[N+](=O)[O-]",
        "DGBIGWXXNGSACT-UHFFFAOYSA-N",
        "DGBIGWXXNGSACT-UHFFFAOYSA-N",
    ),
]

IDS = [c[0] for c in REFERENCE]

# Inputs where ChEMBL standardization is observable on top of what RDKit's SMILES
# parser already normalizes: metal disconnection and uncharging. Most of
# REFERENCE is aromaticity/tautomer-invariant through InChI, so these are the
# rows that actually detect a pipeline that silently stopped running.
# name, input, canonical SMILES unstandardized, canonical SMILES standardized
NORMALIZATIONS = [
    ("metal_disconnect", "CC(=O)O[Na]", "CC(=O)[O][Na]", "CC(=O)[O-].[Na+]"),
    ("uncharge_carboxylate", "CCC(=O)[O-]", "CCC(=O)[O-]", "CCC(=O)O"),
    ("uncharge_ammonium", "CC[NH3+]", "CC[NH3+]", "CCN"),
    ("uncharge_thiolate", "CC[S-]", "CC[S-]", "CCS"),
    (
        "uncharge_aspirin_anion",
        "CC(=O)Oc1ccccc1C(=O)[O-]",
        "CC(=O)Oc1ccccc1C(=O)[O-]",
        "CC(=O)Oc1ccccc1C(=O)O",
    ),
]


def keys(smiles, *steps):
    """InChIKey column for `smiles` put through `steps`."""
    expr = MolExpr.from_smiles("smiles")
    if steps:
        expr = expr.standardize(*steps)
    df = pl.DataFrame({"smiles": list(smiles)})
    return df.with_columns(expr.to_inchikey().alias("k"))["k"].to_list()


def smis(smiles, *steps):
    """Canonical SMILES column for `smiles` put through `steps`."""
    expr = MolExpr.from_smiles("smiles")
    if steps:
        expr = expr.standardize(*steps)
    df = pl.DataFrame({"smiles": list(smiles)})
    return df.with_columns(expr.to_smiles().alias("s"))["s"].to_list()


# --- reference values: the one thing that catches a no-op pipeline ---


@pytest.mark.parametrize("name,smi,chembl_key,_", REFERENCE, ids=IDS)
def test_chembl_reference_inchikey(name, smi, chembl_key, _):
    """Regression guard against a chembl_structure_pipeline upgrade changing
    behaviour. Note most rows are aromaticity- and tautomer-invariant through
    InChI, so they do not detect a pipeline that stopped running -- that is what
    test_chembl_normalizations_are_observable is for."""
    assert keys([smi], *me.STEPS_CHEMBL) == [chembl_key]


@pytest.mark.parametrize(
    "name,smi,raw,std", NORMALIZATIONS, ids=[c[0] for c in NORMALIZATIONS]
)
def test_chembl_normalizations_are_observable(name, smi, raw, std):
    """Pins both sides: RDKit's parser alone gives `raw`, standardization gives
    `std`. Fails if the step becomes a no-op."""
    assert smis([smi]) == [raw]
    assert smis([smi], *me.STEPS_CHEMBL) == [std]
    assert raw != std


def test_chembl_does_not_disconnect_all_metals():
    """Sodium is disconnected, iron is not. Pinned because 'metal disconnection'
    reads as universal and is not."""
    assert smis(["CC(=O)O[Fe]OC(C)=O"], *me.STEPS_CHEMBL) == ["CC(=O)[O][Fe][O]C(C)=O"]


@pytest.mark.parametrize("name,smi,_,papyrus_key", REFERENCE, ids=IDS)
def test_papyrus_reference_inchikey(name, smi, _, papyrus_key):
    """Papyrus nulls anything outside 200-800 Da. Pinned so an all-null column
    from a genuinely broken pipeline is distinguishable from the MW filter."""
    assert keys([smi], *STEPS_PAPYRUS) == [papyrus_key]


def test_standardize_is_not_a_noop():
    """Kekule aspirin must come out aromatic and the anion must come out
    neutral. Both fail if the standardization step never runs."""
    df = pl.DataFrame({"smiles": ["CC(=O)OC1=CC=CC=C1C(=O)O", "CC(=O)Oc1ccccc1C(=O)[O-]"]})
    out = df.with_columns(
        MolExpr.from_smiles("smiles").standardize(*me.STEPS_CHEMBL).to_smiles().alias("s")
    )
    assert out["s"].to_list() == ["CC(=O)Oc1ccccc1C(=O)O"] * 2
    assert out["s"].to_list() != df["smiles"].to_list()


def test_chembl_steps_do_not_strip_salts():
    """STEPS_CHEMBL is standardize_mol only. get_parent_mol is imported in
    _optional.py but never exposed, so salts survive. Pinned, not endorsed."""
    assert keys(["[Na+].CC(=O)[O-]"], *me.STEPS_CHEMBL) == ["VMHLLURERBWHNL-UHFFFAOYSA-M"]


# --- properties that catch a silently broken pipeline ---


@pytest.mark.parametrize("name,smi,chembl_key,_", REFERENCE, ids=IDS)
def test_chembl_idempotent(name, smi, chembl_key, _):
    twice = me.STEPS_CHEMBL + me.STEPS_CHEMBL
    assert keys([smi], *twice) == keys([smi], *me.STEPS_CHEMBL)


@pytest.mark.parametrize(
    "group",
    [
        ["CC(=O)OC1=CC=CC=C1C(=O)O", "CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Oc1ccccc1C(=O)[O-]"],
        ["O=[N+]([O-])c1ccccc1", "[O-][N+](=O)c1ccccc1"],
    ],
    ids=["aspirin_forms", "nitro_forms"],
)
def test_representations_converge(group):
    """Same compound written differently -> one InChIKey after standardization."""
    out = keys(group, *me.STEPS_CHEMBL)
    assert len(set(out)) == 1, out


def test_stereoisomers_stay_distinct():
    """Standardization must not silently flatten stereochemistry."""
    assert len(set(keys(["O=C(O)[C@@H](N)C", "O=C(O)[C@H](N)C"], *me.STEPS_CHEMBL))) == 2
    assert len(set(keys(["OC(=O)/C=C\\C(=O)O", "OC(=O)/C=C/C(=O)O"], *me.STEPS_CHEMBL))) == 2


def test_remove_stereo_collapses_double_bond_geometry():
    """The existing suite only checks for '@', which misses cis/trans."""
    pair = ["OC(=O)/C=C\\C(=O)O", "OC(=O)/C=C/C(=O)O"]
    assert len(set(keys(pair, remove_stereo, *me.STEPS_CHEMBL))) == 1


def test_remove_stereo_collapses_tetrahedral():
    pair = ["O=C(O)[C@@H](N)C", "O=C(O)[C@H](N)C"]
    assert len(set(keys(pair, remove_stereo, *me.STEPS_CHEMBL))) == 1


STEROID = "CC(=O)O[C@H]1C[C@@H]2CC[C@H]3[C@@H](CCC4=CC(=O)CC[C@]34C)[C@@H]2CC1"


def test_papyrus_nostereo_differs_from_papyrus():
    """STEPS_PAPYRUS_NOSTEREO must actually drop stereo, not just alias PAPYRUS."""
    with_stereo = keys([STEROID], *STEPS_PAPYRUS)
    without = keys([STEROID], *STEPS_PAPYRUS_NOSTEREO)
    assert with_stereo == ["PJWOJESQZYYXIO-GXZVXFIFSA-N"]
    assert without == ["PJWOJESQZYYXIO-UHFFFAOYSA-N"]


def test_papyrus_silently_removes_stereo_on_its_own():
    """Papyrus defaults to tautomer_allow_stereo_removal=True, so plain
    STEPS_PAPYRUS already destroys stereocentres for many molecules. Pinned so
    nobody assumes STEPS_PAPYRUS is stereo-preserving."""
    naproxen = "COc1ccc2cc([C@H](C)C(=O)O)ccc2c1"  # (S), the marketed drug
    assert keys([naproxen], *me.STEPS_CHEMBL) == ["CMWTZPSULFXXJA-VIFPVBQESA-N"]
    assert keys([naproxen], *STEPS_PAPYRUS) == ["CMWTZPSULFXXJA-UHFFFAOYSA-N"]


# --- null provenance: every silent failure path ---


def test_invalid_smiles_null_count():
    """Unparseable input yields null, and only the unparseable rows."""
    smi = ["CCO", "not_a_smiles", "c1ccccc", "CCC", "CC(C"]
    out = keys(smi)
    assert [k is None for k in out] == [False, True, True, False, True]


def test_null_input_stays_null():
    df = pl.DataFrame({"smiles": ["CCO", None, "CCC"]})
    out = df.with_columns(
        MolExpr.from_smiles("smiles").standardize(*me.STEPS_CHEMBL).to_smiles().alias("s")
    )
    assert out["s"].to_list() == ["CCO", None, "CCC"]


def test_step_returning_non_mol_raises():
    """A step with a wrong return type must not be silently swallowed.
    MolPipeline.__call__ is not itself wrapped in safe_step, so it surfaces."""
    df = pl.DataFrame({"smiles": ["CCO"]})
    with pytest.raises(AttributeError):
        df.with_columns(
            MolExpr.from_smiles("smiles").standardize(lambda m: "not a mol").alias("m")
        )


def test_wrapped_and_unwrapped_steps_fail_differently():
    """STEPS_PAPYRUS holds the raw _papyrus_standardize, while STEPS_CHEMBL holds
    the safe_step-wrapped chembl_standardize -- so a papyrus error kills the whole
    query and a chembl error nulls the column. The wrapped papyrus_standardize in
    mol_functions.py exists but nothing uses it."""
    assert me.STEPS_CHEMBL[0] is chembl_standardize
    assert STEPS_PAPYRUS[-1] is not papyrus_standardize

    def boom(mol):
        raise RuntimeError("standardization failed")

    df = pl.DataFrame({"smiles": ["CCO"]})
    with pytest.raises(RuntimeError):
        df.with_columns(MolExpr.from_smiles("smiles").standardize(boom).alias("m"))

    logging.disable(logging.CRITICAL)
    try:
        out = df.with_columns(
            MolExpr.from_smiles("smiles")
            .standardize(safe_step(boom))
            .to_smiles()
            .alias("s")
        )
    finally:
        logging.disable(logging.NOTSET)
    assert out["s"].to_list() == [None]


@pytest.mark.xfail(
    strict=True,
    reason="BUG: safe_step swallows the ImportError from the _optional.py "
    "placeholder, so a missing chembl_structure_pipeline yields a column of "
    "nulls instead of an error",
)
def test_missing_optional_dep_raises(monkeypatch):
    import fairfetched.standardize.mol_functions as mf

    def boom(mol, *a, **k):
        raise ImportError("chembl_structure_pipeline not installed")

    monkeypatch.setattr(mf, "_chembl_standardize", boom)
    logging.disable(logging.CRITICAL)
    try:
        df = pl.DataFrame({"smiles": ["CCO"] * 3})
        with pytest.raises(Exception):
            df.with_columns(
                MolExpr.from_smiles("smiles")
                .standardize(mf.chembl_standardize)
                .to_smiles()
                .alias("s")
            )
    finally:
        logging.disable(logging.NOTSET)


# --- column contracts ---


@pytest.mark.xfail(
    strict=True,
    reason="BUG: to_descriptors() unnests a field named 'smiles', overwriting "
    "the caller's input column with kekulized non-isomeric SMILES",
)
def test_to_descriptors_preserves_input_smiles():
    df = pl.DataFrame({"id": [1, 2], "smiles": ["O=C(O)[C@@H](N)C", "CCO"]})
    out = df.with_columns(MolExpr.from_smiles("smiles").to_descriptors())
    assert out["smiles"].to_list() == df["smiles"].to_list()


def test_to_descriptors_fields_agree_with_scalar_sinks():
    df = pl.DataFrame({"smi": ["CC(=O)Oc1ccccc1C(=O)O", "CCO"]})
    out = df.with_columns(
        MolExpr.from_smiles("smi").to_descriptors(),
        MolExpr.from_smiles("smi").to_inchikey().alias("k2"),
        MolExpr.from_smiles("smi").to_inchi().alias("i2"),
    )
    assert out["inchikey"].to_list() == out["k2"].to_list()
    assert out["inchi"].to_list() == out["i2"].to_list()


@pytest.mark.parametrize("empty", [True, False], ids=["empty", "all_null"])
@pytest.mark.xfail(
    strict=True,
    reason="BUG: from_col_infer returns the input String series untouched when "
    "it finds no non-null value, violating its declared pl.Binary return_dtype",
)
def test_from_col_infer_handles_no_usable_value(empty):
    data = [] if empty else [None, None]
    df = pl.DataFrame({"s": data}, schema={"s": pl.String})
    out = df.with_columns(MolExpr.from_col_infer("s").to_smiles().alias("out"))
    assert out["out"].to_list() == data


def test_from_col_infer_picks_the_right_parser():
    inchi = "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"
    for col, val in [("s", "CCO"), ("s", inchi)]:
        df = pl.DataFrame({col: [val]})
        out = df.with_columns(MolExpr.from_col_infer(col).to_inchikey().alias("k"))
        assert out["k"].to_list() == ["LFQSCWFLJHTTHZ-UHFFFAOYSA-N"]


def test_from_inchi_roundtrip():
    df = pl.DataFrame({"inchi": ["InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"]})
    out = df.with_columns(MolExpr.from_inchi("inchi").to_inchi().alias("back"))
    assert out["back"].to_list() == df["inchi"].to_list()


# --- execution-mode equivalence ---

EQUIV_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "CCO",
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
    "CCO",
    "c1ccccc1",
    "O=C(O)[C@@H](N)C",
] * 4


def test_parallel_equals_serial():
    """_map_nodedup swallows any exception from the pool and falls back to
    serial, so a broken parallel path shows up only as a slow run."""
    df = pl.DataFrame({"smiles": EQUIV_SMILES})
    serial = df.with_columns(
        MolExpr.from_smiles("smiles").standardize(*me.STEPS_CHEMBL).to_inchikey().alias("k")
    )
    par = df.with_columns(
        MolExpr.from_smiles("smiles", parallel=True)
        .standardize(*me.STEPS_CHEMBL, parallel=True)
        .to_inchikey(parallel=True)
        .alias("k")
    )
    assert serial["k"].to_list() == par["k"].to_list()


def test_parallel_fallback_is_logged(caplog):
    """An unpicklable step makes the pool fail; _map_nodedup catches everything
    and silently reruns serially. The answer stays right, but the fallback must
    at least be visible in the log -- otherwise a broken parallel path is
    indistinguishable from a slow one."""
    df = pl.DataFrame({"smiles": ["CCO"] * 4})
    with caplog.at_level(logging.ERROR):
        out = df.with_columns(
            MolExpr.from_smiles("smiles")
            .standardize(lambda m: m, parallel=True)
            .to_smiles()
            .alias("s")
        )
    assert out["s"].to_list() == ["CCO"] * 4
    assert "resorting to native polars" in caplog.text


@pytest.mark.parametrize(
    "sink",
    [
        # to_inchikey/to_smiles/to_inchi take no dedup kwarg, unlike their
        # siblings; to_custom is the only String-returning sink that does
        lambda e, **kw: e.to_custom(_binary_to_inchikey, pl.String, **kw).alias("v"),
        lambda e, **kw: e.to_morgan_fp(fp_size=256, **kw).alias("v"),
        lambda e, **kw: e.num_heavy_atoms(**kw).alias("v"),
    ],
    ids=["string", "array", "int"],
)
def test_dedup_equals_nodedup(sink):
    """_map re-expands deduplicated results with a left join; polars gives no
    row-order guarantee without maintain_order=."""
    df = pl.DataFrame({"smiles": EQUIV_SMILES})
    plain = df.with_columns(sink(MolExpr.from_smiles("smiles")))
    dedup = df.with_columns(sink(MolExpr.from_smiles("smiles", dedup=True), dedup=True))
    assert plain["v"].to_list() == dedup["v"].to_list()
    assert plain["v"].dtype == dedup["v"].dtype


def test_dedup_struct_equals_nodedup():
    df = pl.DataFrame({"smi": EQUIV_SMILES})
    plain = df.with_columns(MolExpr.from_smiles("smi").to_descriptors())
    dedup = df.with_columns(MolExpr.from_smiles("smi").to_descriptors(dedup=True))
    assert plain.equals(dedup)


def test_dedup_with_nulls():
    df = pl.DataFrame({"smiles": ["CCO", None, "CCC", None, "CCO"]})
    out = df.with_columns(
        MolExpr.from_smiles("smiles", dedup=True).to_smiles().alias("s")
    )
    assert out["s"].to_list() == ["CCO", None, "CCC", None, "CCO"]


# --- filter steps ---


def mol(smi):
    return MolFromSmiles(smi)


@pytest.mark.parametrize(
    "smi,kept",
    [("CCO", True), ("CC(=O)Oc1ccccc1C(=O)O", True), ("[Na+].CC(=O)[O-]", False)],
)
def test_no_mixtures(smi, kept):
    assert (no_mixtures(mol(smi)) is not None) == kept


@pytest.mark.parametrize(
    "smi,kept",
    [
        ("CCO", True),
        ("ClCCBr", True),
        ("[Fe]", False),
        ("OB(O)c1ccccc1", False),  # boron: rejected, whether or not that is intended
        ("C[Si](C)C", False),
    ],
)
def test_only_organic(smi, kept):
    assert (only_organic(mol(smi)) is not None) == kept


@pytest.mark.xfail(
    strict=True,
    reason="BUG: only_organic's element set omits atomic number 1, so any mol "
    "carrying explicit hydrogens is rejected wholesale",
)
def test_only_organic_accepts_explicit_hydrogens():
    assert only_organic(AddHs(mol("CCO"))) is not None


def test_valid_inchi_keeps_representable_mols():
    assert valid_inchi(mol("CC(=O)Oc1ccccc1C(=O)O")) is not None


def test_via_inchi_preserves_identity():
    m = via_inchi(mol("CC(=O)Oc1ccccc1C(=O)O"))
    assert m is not None
    assert MolToSmiles(m) == "CC(=O)Oc1ccccc1C(=O)O"
