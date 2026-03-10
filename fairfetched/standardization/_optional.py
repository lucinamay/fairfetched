import logging as lg

try:
    from chembl_structure_pipeline import (  # ty:ignore[unresolved-import] #ty:ignore[unused-ignore-comment] #ty:ignore[unused-ignore-comment]
        get_parent_mol as chembl_get_parent_mol,
    )
    from chembl_structure_pipeline import (  # ty:ignore[unresolved-import] #ty:ignore[unused-ignore-comment] #ty:ignore[unused-ignore-comment]
        standardize_mol as chembl_standardize,
    )
except ImportError as e:
    lg.warning("""
        you should install chembl_structure_pipeline if you want
        to use chembl standardisation
        """)
    CHEMBL_IMPORT_ERROR = e

    def chembl_get_parent_mol(mol):
        """placeholder for chembl_structure_pipeline.get_parent_mol in case
        the import doesn't work.
        raises respective import error when called"""
        lg.warning("""
            please install `chembl_structure_pipeline` if you want to use
            ChEMBL standardisation
            """)
        raise CHEMBL_IMPORT_ERROR

    def chembl_standardize(mol):
        """placeholder for chembl_structure_pipeline.chembl_standardize in case
        the import doesn't work.

        raises respective import error when called.
        """
        lg.warning("""
            please install `chembl_structure_pipeline` if you want to use
            ChEMBL standardisation
            """)
        raise CHEMBL_IMPORT_ERROR


try:
    from papyrus_structure_pipeline import (  # ty:ignore[unresolved-import] #ty:ignore[unused-ignore-comment] #ty:ignore[unused-ignore-comment]
        standardize as _papyrus_standardize,
    )


except ImportError as e:
    PAPYRUS_IMPORT_ERROR = e

    def _papyrus_standardize(mol, **kwargs):
        """placeholder for papyrus_standardize in case imports dont work.
        raises respective import error when called"""
        lg.warning("""
            please install `papyrus_structure_pipeline` if you want to use
            Papyrus standardisation
            """)
        raise PAPYRUS_IMPORT_ERROR
