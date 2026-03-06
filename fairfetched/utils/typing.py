from pathlib import Path
from typing import Any, NotRequired, Protocol, Tuple, TypedDict

from polars import LazyFrame


class ComposedLFDict(TypedDict):
    """a dictionary with different lazyframes composed from raw cheminformatics
    databases. requires at minimum a bioactivity and compounds dataframe"""

    bioactivity: LazyFrame
    compounds: LazyFrame
    full: NotRequired[LazyFrame]
    proteins: NotRequired[LazyFrame]
    components: NotRequired[LazyFrame]


class DatasetGetModule(Protocol):
    """Protocol for the logic required in the get.papyrus, get.chembl files

    For referencing use in the general _Base API of get.api.
    """

    __name__: str
    __file__: str

    def available_versions(self) -> Tuple[str]:
        """returns available versions of database"""
        ...

    def latest(self) -> str:
        """returns latest version of database"""
        ...

    def get_sources(self, version: str) -> dict[str, str]:
        """returns a dict of URLs of the sources of database"""
        ...

    def ensure_raw(
        self, version: str, raw_dir: Path | str | Any | None = None
    ) -> dict[str, Path]:
        """downloads raw files for specified version from database sources
        to cache dir if not yet present, and returns a dictionary of the source
        names and the paths to their raw files."""
        ...

    def ensure_consolidated(
        self,
        raw_paths: dict[str, Path],
        consolidated_dir: Path | str | Any | None = None,
    ) -> dict[str, Path]:
        """from dictionary of raw files, makes sure that all files are in parquet
        format to allow lazy loading with pl.scan. assigns datatypes and
        consolidates null values in case of ambiguous formats such as csv"""
        ...

    def clean(self, paths: dict[str, Path]) -> dict[str, LazyFrame]:
        """does rest of standardisation, such as column renaming etc"""
        ...

    def compose(self, lfs: dict[str, LazyFrame]) -> ComposedLFDict:
        """composes into few, comprehensive lazyframes through joins and
        pivots."""
        ...


#     "ComposedDict",
#     {
#         "bioactivity": pl.LazyFrame,
#         "compounds": pl.LazyFrame,
#         "proteins": NotRequired[pl.LazyFrame],
#     },
#     total=False,
# )
