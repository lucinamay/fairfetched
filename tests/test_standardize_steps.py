"""The decomposed step lists in `pipeline.py` reproduce the library functions
they are built from. The oracle is the library's whole-function call on the
same input, so these tests say nothing about chemistry: only that
`STEPS_CHEMBL_STANDARDIZE_MOL`, `STEPS_CHEMBL_GET_PARENT_MOL` and
`STEPS_PAPYRUS_STANDARDIZE` equal `standardize_mol`, `get_parent_mol` and
Papyrus `standardize` on every input, and that each step with a visible effect
on SMILES input is actually run (dropping it changes a named input).

Steps with no effect on SMILES input (valence update, S-group removal,
kekulization, H removal, drawing cleanup, final sanitize; Papyrus `_uncharge`
after ChEMBL's uncharger) have no drop-test: nothing parsed from SMILES can
show them.
"""

import ast
import hashlib
import inspect
import pickle
from collections.abc import Callable
from functools import partial
from importlib.metadata import version
from pathlib import Path

import polars as pl
import pytest
from rdkit import RDLogger
from rdkit.Chem import Mol, MolFromSmiles, MolToSmiles

from fairfetched.prep import papyrus
from fairfetched.prep.mol_expr import MolExpr
from fairfetched.prep.mol_functions import (
    MolFn,
    chembl_exclude,
    chembl_fragment_parent,
    papyrus_remove_salts,
)
from fairfetched.prep.pipeline import (
    STEPS_CHEMBL_GET_PARENT_MOL,
    STEPS_CHEMBL_PARENT,
    STEPS_CHEMBL_STANDARDIZE_MOL,
    STEPS_PAPYRUS_STANDARDIZE,
)

RDLogger.DisableLog("rdApp.*")  # pyright: ignore[reportAttributeAccessIssue]
csp = pytest.importorskip("chembl_structure_pipeline")
psp = pytest.importorskip("papyrus_structure_pipeline")

# name: SMILES, chosen so that every step with a SMILES-visible effect acts on
# at least one of them (see the drop tests for which)
INPUTS: dict[str, str] = {
    "nitro": "CN(=O)=O",
    "ammonium": "CC[NH3+]",
    "tartaric": "O[C@H]([C@@H](O)C(=O)O)C(=O)O",
    "alkoxide": "CCO[Na]",
    "carbon13": "[13CH3]C(=O)O",
    "deuterated_chiral": "[2H][C@](C)(F)Cl",
    "hcl_salt": "CC[NH+](C)C.[Cl-]",
    "dmso_solvate": "CC(=O)Oc1ccccc1C(=O)O.CS(C)=O",
    "warfarin": "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",
    "enol": "OC(=CC(=O)c1ccccc1)c1ccccc1",
    "silane": "C[Si](C)(C)c1ccc(cc1)C(=O)Nc1ccc(cc1)C(C)(C)C",
    "cisplatin": "N.N.Cl[Pt]Cl",
    "two_organics": "CCCCCCCCCCc1ccccc1.CCCCCCCCCCCc1ccccc1",
    "pyridine_salt": "CCCCCCCCCCCCCCCCN.c1ccncc1",
    "tetraphenylborate": "CCCCCCCCCCCCCCCC[NH3+].c1ccc([B-](c2ccccc2)(c2ccccc2)c2ccccc2)cc1",
    "sulfonamide_anion": "CCCCCCCCCCCCCCC[N-]S(C)(=O)=O",
    "sodium_alkoxide": "CCCCCCCCCCCCCCCC[O-].[Na+]",
    "iron_mix": "[Fe].c1ccccc1CCCCCCCCCCCCCCC",
    "tartaric_iron": "O[C@H]([C@@H](O)C(=O)O)C(=O)O.[Fe]",  # excluded, yet changed by a step
    "sulfoxide_iron": "CS(=O)CCCCc1ccccc1.[Fe]",  # excluded, changed, not a salt
}
NAMES = list(INPUTS)
DF = pl.DataFrame({"name": NAMES, "smiles": list(INPUTS.values())})


def run(steps: list[MolFn], parallel: bool = False) -> dict[str, str | None]:
    """Canonical SMILES per input name after `steps`, one batch query."""
    expr = MolExpr.from_smiles("smiles").standardize(*steps, parallel=parallel)
    out = DF.with_columns(expr.to_smiles().alias("out"))
    return dict(zip(out["name"], out["out"], strict=True))


def oracle(fn: Callable[[Mol], Mol | None]) -> dict[str, str | None]:
    """Canonical SMILES per input name from the library function itself; None
    where it returns None or raises, as `safe_step` nulls raising steps."""
    out: dict[str, str | None] = {}
    for name, smiles in INPUTS.items():
        try:
            mol = fn(MolFromSmiles(smiles))
        except Exception:  # noqa: BLE001  # mirrors safe_step's blanket catch
            mol = None
        out[name] = None if mol is None else MolToSmiles(mol)
    return out


def step_name(step: MolFn) -> str:
    return getattr(step, "__name__", None) or step.func.__name__  # type: ignore[attr-defined]


def dropped(steps: list[MolFn], index: int) -> list[MolFn]:
    return steps[:index] + steps[index + 1 :]


def drop_cases(steps: list[MolFn], expected: dict[str, list[str]]) -> list:
    """(index, input name) pairs, parametrize ids `<step>-<input>`."""
    by_name = {step_name(s): i for i, s in enumerate(steps)}
    return [
        pytest.param(by_name[step], name, id=f"{step}-{name}")
        for step, names in expected.items()
        for name in names
    ]


class TestChemblStandardizeMol:
    """`STEPS_CHEMBL_STANDARDIZE_MOL` is `standardize_mol(check_exclusion=False)`."""

    full = run(STEPS_CHEMBL_STANDARDIZE_MOL)
    expected = oracle(lambda m: csp.standardize_mol(m, check_exclusion=False))

    @pytest.mark.parametrize("name", NAMES)
    def test_matches_library(self, name: str) -> None:
        assert self.full[name] == self.expected[name]

    def test_exclude_step_nulls_what_the_library_flags(self) -> None:
        flagged = oracle(lambda m: m if csp.exclude_flag.exclude_flag(m) else None)
        out = run([chembl_exclude])
        assert flagged["warfarin"] is None and flagged["cisplatin"] is not None
        for name in NAMES:
            assert (out[name] is None) == (flagged[name] is not None), name

    def test_library_default_gate_is_not_a_step(self) -> None:
        """`standardize_mol(check_exclusion=True)` returns a flagged mol
        untouched; the step list standardizes it. Equal everywhere else."""
        gated = oracle(lambda m: csp.standardize_mol(m))
        flagged = {n for n in NAMES if run([chembl_exclude])[n] is None}
        assert gated["tartaric_iron"] != self.full["tartaric_iron"]
        for name in set(NAMES) - flagged:
            assert gated[name] == self.full[name], name

    @pytest.mark.parametrize(
        "index,name",
        drop_cases(
            STEPS_CHEMBL_STANDARDIZE_MOL,
            {
                "chembl_normalize": ["alkoxide", "dmso_solvate"],
                "chembl_uncharge": ["ammonium", "hcl_salt", "sulfonamide_anion"],
                "chembl_flatten_tartrate": ["tartaric"],
            },
        ),
    )
    def test_dropping_step_changes_input(self, index: int, name: str) -> None:
        assert (
            run(dropped(STEPS_CHEMBL_STANDARDIZE_MOL, index))[name] != self.full[name]
        )


class TestChemblGetParentMol:
    """`STEPS_CHEMBL_GET_PARENT_MOL` is `get_parent_mol(check_exclusion=False)`,
    the `get_parent` default; `exclude_by_chembl_standards=True` nulls what the
    library flags instead of returning the solvent-stripped species it would."""

    full = run(STEPS_CHEMBL_GET_PARENT_MOL)
    expected = oracle(lambda m: csp.get_parent_mol(m, check_exclusion=False)[0])

    @pytest.mark.parametrize("name", NAMES)
    def test_matches_library(self, name: str) -> None:
        assert self.full[name] == self.expected[name]

    def test_exclusion_nulls_flagged(self) -> None:
        steps = [
            STEPS_CHEMBL_GET_PARENT_MOL[0],
            chembl_fragment_parent(exclude_by_chembl_standards=True),
        ]
        flagged = oracle(lambda m: m if csp.get_parent_mol(m)[1] else None)
        out = run(steps)
        assert flagged["cisplatin"] is not None  # the case this test is about
        for name in NAMES:
            assert (out[name] is None) == (flagged[name] is not None), name

    def test_after_standardize_equals_steps_chembl_parent(self) -> None:
        """Equal to `STEPS_CHEMBL_PARENT` except where `standardize_mol`'s
        exclusion gate skips a step the list runs (sulfoxide normalization
        here; the flattened tartrate is stripped as a salt by both)."""
        both = run(STEPS_CHEMBL_STANDARDIZE_MOL + STEPS_CHEMBL_GET_PARENT_MOL)
        parent = run(STEPS_CHEMBL_PARENT)
        assert both.pop("sulfoxide_iron") != parent.pop("sulfoxide_iron")
        assert both == parent

    @pytest.mark.parametrize(
        "index,name",
        drop_cases(
            STEPS_CHEMBL_GET_PARENT_MOL,
            {
                "chembl_isotope_parent": ["carbon13", "deuterated_chiral"],
                "_chembl_fragment_parent": ["hcl_salt", "dmso_solvate", "cisplatin"],
            },
        ),
    )
    def test_dropping_step_changes_input(self, index: int, name: str) -> None:
        assert run(dropped(STEPS_CHEMBL_GET_PARENT_MOL, index))[name] != self.full[name]


class TestPapyrusStandardize:
    """`STEPS_PAPYRUS_STANDARDIZE` is Papyrus `standardize()` with its defaults."""

    full = run(STEPS_PAPYRUS_STANDARDIZE)
    expected = oracle(lambda m: psp.standardize(m, raise_error=False))

    @pytest.mark.parametrize("name", NAMES)
    def test_matches_library(self, name: str) -> None:
        assert self.full[name] == self.expected[name]

    def test_filters_and_keeps(self) -> None:
        """Sanity on the oracle itself: the list filters and keeps something."""
        assert self.full["warfarin"] is not None
        assert self.full["silane"] is None  # inorganic
        assert self.full["two_organics"] is None  # mixture
        assert self.full["ammonium"] is None  # under 200 Da

    def test_parallel_pickles_partial_steps(self) -> None:
        assert run(STEPS_PAPYRUS_STANDARDIZE, parallel=True) == self.full

    def test_include_metals_false_matches_library_and_keeps_metals(self) -> None:
        """Each library call with `include_metals=False` overwrites its SALTS
        list in place, so the oracle restores it after every call."""

        def library(m: Mol) -> Mol:
            saved = list(psp.standardizer.SALTS)
            try:
                return psp.standardizer._remove_supplementary_salts(
                    m, include_metals=False
                )
            finally:
                psp.standardizer.SALTS[:] = saved

        expected = oracle(library)
        out = run([papyrus_remove_salts(include_metals=False)])
        assert out == expected
        assert out["iron_mix"] != run([papyrus_remove_salts()])["iron_mix"]

    def test_include_metals_false_leaves_the_salt_list_intact(self) -> None:
        """The library overwrites its SALTS with Mol objects on this path, so
        the second call raises there; the copy must stay a list of SMILES."""
        first = run([papyrus_remove_salts(include_metals=False)])
        assert run([papyrus_remove_salts(include_metals=False)]) == first
        assert all(isinstance(s, str) for s in papyrus.SALTS)

    def test_all_lists_pickle(self) -> None:
        """`parallel=True` pickles the pipeline; lambdas and closures would fail."""
        for steps in (
            STEPS_CHEMBL_STANDARDIZE_MOL,
            STEPS_CHEMBL_GET_PARENT_MOL,
            STEPS_PAPYRUS_STANDARDIZE,
        ):
            pickle.dumps(steps)

    @pytest.mark.parametrize(
        "index,name",
        drop_cases(
            STEPS_PAPYRUS_STANDARDIZE,
            {
                "_papyrus_remove_salts": [
                    "pyridine_salt",
                    "tetraphenylborate",
                    "iron_mix",
                ],
                "papyrus_no_mixtures": ["two_organics"],
                "papyrus_only_organic": ["silane"],
                "_mw_between": ["ammonium", "tartaric"],
                "_papyrus_canonical_tautomer": ["enol"],
            },
        ),
    )
    def test_dropping_step_changes_input(self, index: int, name: str) -> None:
        assert run(dropped(STEPS_PAPYRUS_STANDARDIZE, index))[name] != self.full[name]


class TestVendoredPapyrus:
    """`prep/papyrus.py` function by function against papyrus_structure_pipeline
    0.0.5 on every input; `oracle` nulls where the library raises."""

    @pytest.mark.parametrize("name", NAMES)
    def test_chembl_roundtrip(self, name: str) -> None:
        assert (
            oracle(papyrus.chembl_roundtrip)[name]
            == oracle(psp.standardizer._apply_chembl_standardization)[name]
        )

    @pytest.mark.parametrize("include_metals", [True, False])
    def test_remove_supplementary_salts(self, include_metals: bool) -> None:
        def library(m: Mol) -> Mol:
            saved = list(psp.standardizer.SALTS)
            try:
                return psp.standardizer._remove_supplementary_salts(
                    m, include_metals=include_metals
                )
            finally:
                psp.standardizer.SALTS[:] = saved

        ours = partial(
            papyrus.remove_supplementary_salts, include_metals=include_metals
        )
        assert oracle(ours) == oracle(library)

    def test_is_mixture_and_is_organic(self) -> None:
        for smiles in INPUTS.values():
            mol = MolFromSmiles(smiles)
            assert papyrus.is_mixture(mol) == psp.standardizer.is_mixture(mol), smiles
            assert papyrus.is_organic(mol) == psp.standardizer.is_organic(mol), smiles

    @pytest.mark.parametrize("name", NAMES)
    def test_uncharge(self, name: str) -> None:
        assert (
            oracle(papyrus.uncharge)[name] == oracle(psp.standardizer._uncharge)[name]
        )

    @pytest.mark.parametrize("allow_stereo_removal", [True, False])
    def test_canonicalize_tautomer(self, allow_stereo_removal: bool) -> None:
        ours = oracle(
            lambda m: papyrus.canonicalize_tautomer(
                m, allow_stereo_removal=allow_stereo_removal
            )[0]
        )
        library = oracle(
            lambda m: psp.standardizer._canonicalize_tautomer(
                m, allow_stereo_removal=allow_stereo_removal
            )
        )
        assert ours == library
        assert ours["enol"] != INPUTS["enol"]  # the input is not its canonical tautomer

    def test_constants_match_the_library(self) -> None:
        """The three module lists, and the neutralization SMARTS that the
        library keeps as a local inside `_uncharge` (read from its source)."""
        lib = psp.standardizer
        assert papyrus.SALTS == lib.SALTS
        assert [f"[{m}]" for m in papyrus.METALS] == lib.METALS
        assert papyrus.ORGANIC_ATOMS == lib.ORGANIC_ATOMS
        (assignment,) = [
            node
            for node in ast.walk(ast.parse(inspect.getsource(lib._uncharge)))
            if isinstance(node, ast.Assign)
            and any(
                getattr(t, "id", None) == "neutralizing_smiles" for t in node.targets
            )
        ]
        assert papyrus._NEUTRALIZATIONS == ast.literal_eval(assignment.value)

    def test_installed_library_is_the_copied_version(self) -> None:
        """A newer papyrus_structure_pipeline in the dev environment fails here
        first: re-diff `prep/papyrus.py` against upstream, then update
        `papyrus.UPSTREAM`. The equality tests above then say whether the
        behaviour moved."""
        source = Path(psp.standardizer.__file__).read_bytes()
        installed = (
            version("papyrus-structure-pipeline"),
            hashlib.sha256(source).hexdigest(),
        )
        assert installed == papyrus.UPSTREAM
