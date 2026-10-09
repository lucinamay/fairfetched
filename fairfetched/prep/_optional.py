import logging as lg

logger = lg.getLogger(__name__)

try:
    from chembl_structure_pipeline import (  # ty:ignore[unresolved-import] #ty:ignore[unused-ignore-comment] #ty:ignore[unused-ignore-comment]
        get_parent_mol as chembl_get_parent_mol,  # pyright: ignore[reportAssignmentType]
    )
    from chembl_structure_pipeline import (  # ty:ignore[unresolved-import] #ty:ignore[unused-ignore-comment] #ty:ignore[unused-ignore-comment]
        standardize_mol as _chembl_standardize,  # pyright: ignore[reportAssignmentType]
    )
except ImportError as e:
    logger.warning("""
        you should install chembl_structure_pipeline if you want
        to use chembl standardisation
        """)
    CHEMBL_IMPORT_ERROR = e

    def chembl_get_parent_mol(m, neutralize=True, check_exclusion=True, verbose=False):
        """placeholder for chembl_structure_pipeline.get_parent_mol in case
        the import doesn't work.
        raises respective import error when called"""
        logger.warning("""
            please install `chembl_structure_pipeline` if you want to use
            ChEMBL standardisation
            """)
        raise CHEMBL_IMPORT_ERROR

    def _chembl_standardize(mol):
        """placeholder for chembl_structure_pipeline.chembl_standardize in case
        the import doesn't work.

        raises respective import error when called.
        """
        logger.warning("""
            please install `chembl_structure_pipeline` if you want to use
            ChEMBL standardisation
            """)
        raise CHEMBL_IMPORT_ERROR


class _MissingPackage:
    """Stands in for an uninstalled package's module; any attribute access
    raises the ImportError seen at import time."""

    def __init__(self, error: ImportError) -> None:
        self._error = error

    def __getattr__(self, name: str):
        raise self._error


try:
    from chembl_structure_pipeline import (  # ty:ignore[unresolved-import] #ty:ignore[unused-ignore-comment]
        standardizer as chembl_standardizer,
    )
    from chembl_structure_pipeline.exclude_flag import (  # ty:ignore[unresolved-import] #ty:ignore[unused-ignore-comment]
        exclude_flag as chembl_exclude_flag,
    )
except ImportError as e:
    chembl_standardizer = _MissingPackage(e)

    def chembl_exclude_flag(mol, includeRDKitSanitization=True) -> bool:
        raise CHEMBL_IMPORT_ERROR
