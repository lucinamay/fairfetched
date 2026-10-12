"""`chembl_structure_pipeline` is optional: without it every name below raises
the ImportError when called or dereferenced, so a ChEMBL step fails per
molecule like any other step error."""

try:
    from chembl_structure_pipeline import (  # ty:ignore[unresolved-import]
        get_parent_mol as chembl_get_parent_mol,
    )
    from chembl_structure_pipeline import (
        standardize_mol as _chembl_standardize,
    )
    from chembl_structure_pipeline import (
        standardizer as chembl_standardizer,
    )
    from chembl_structure_pipeline.exclude_flag import (  # ty:ignore[unresolved-import]
        exclude_flag as chembl_exclude_flag,
    )
except ImportError as e:

    class _Missing:
        def __init__(self, error: ImportError) -> None:
            self._error = error

        def __getattr__(self, name: str):
            raise self._error

        def __call__(self, *args, **kwargs):
            raise self._error

    chembl_get_parent_mol = _chembl_standardize = _Missing(e)  # pyright: ignore[reportAssignmentType, reportConstantRedefinition]
    chembl_standardizer = chembl_exclude_flag = _Missing(e)  # pyright: ignore[reportAssignmentType, reportConstantRedefinition]
