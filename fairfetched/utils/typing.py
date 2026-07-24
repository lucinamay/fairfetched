from pathlib import Path
from typing import Any, NotRequired, Protocol, TypedDict

from polars import LazyFrame


class BioactivityDBViews(TypedDict):
    """a dictionary with different 'views' from raw cheminformatics
    databases, as lazyframes.
    requires at minimum a bioactivity and compounds dataframe
    """

    bioactivity: LazyFrame
    compounds: LazyFrame
    bioactivity_nostereochemistry: NotRequired[LazyFrame]
    full: NotRequired[LazyFrame]
    full_nostereochemistry: NotRequired[LazyFrame]
    proteins: NotRequired[LazyFrame]
    components: NotRequired[LazyFrame]


class DatasetGetModule(Protocol):
    """Protocol for the logic required in the get.papyrus, get.chembl files

    For referencing use in the general _Base API of get.dataset.
    """

    __name__: str
    __file__: str

    def available_versions(self) -> tuple[str, ...]:
        """returns available versions of database"""
        ...

    def latest(self) -> str:
        """returns latest version of database"""
        ...

    def source_urls(self, version: str) -> dict[str, str]:
        """Return the source URLs for a database version."""
        ...

    def ensure_raw_files(
        self, version: str, raw_dir: Path | str | Any | None = None
    ) -> dict[str, Path]:
        """Download raw files for a version and return their paths."""
        ...

    def ensure_parquet_tables(
        self,
        raw_paths: dict[str, Path],
        table_dir: Path | str | None = None,
    ) -> dict[str, Path]:
        """Convert raw files to Parquet table files and return their paths."""
        ...

    def cleanly_scan_parquet_tables(
        self, parquet_paths: dict[str, Path]
    ) -> dict[str, LazyFrame]:
        """Scan table files and apply dataset-specific cleanup."""
        ...

    def build_views(self, parquet_paths: dict[str, Path]) -> BioactivityDBViews:
        """Build comprehensive lazy views through joins and pivots."""
        ...


#     "ComposedDict",
#     {
#         "bioactivity": pl.LazyFrame,
#         "compounds": pl.LazyFrame,
#         "proteins": NotRequired[pl.LazyFrame],
#     },
#     total=False,
# )
