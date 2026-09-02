from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from polars import LazyFrame

from fairfetched.get import chembl, papyrus
from fairfetched.get._chembl_tables import ChemblTables
from fairfetched.get._papyrus_tables import PapyrusTables
from fairfetched.utils import BASE_DIR
from fairfetched.utils.typing import DatasetGetModule


class _View:
    """Joined domain views, built once from the raw tables."""

    def __init__(self, owner: "_Base") -> None:
        self._views = owner.module.build_views(owner.parquet_paths)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} available: {', '.join(self._views)}>"

    @property
    def bioactivity(self) -> LazyFrame:
        return self._views["bioactivity"]

    @property
    def compounds(self) -> LazyFrame:
        return self._views["compounds"]


class _ChemblView(_View):
    @property
    def proteins(self) -> LazyFrame:
        return self._views["proteins"]

    @property
    def components(self) -> LazyFrame:
        return self._views["components"]


class _PapyrusView(_View):
    """``proteins`` and ``full`` mirror the raw Papyrus tables: Papyrus already
    ships a near-flat schema, so these joins are close to passthrough. Use the
    raw-table attributes (``Papyrus.protein``, ``Papyrus.bioactivity``) when you
    want the source columns without the view's renames."""

    @property
    def proteins(self) -> LazyFrame:
        return self._views["proteins"]

    @property
    def full(self) -> LazyFrame:
        """bioactivity + protein data as one flat LazyFrame."""
        return self._views["full"]


@dataclass(frozen=True)
class _Base:
    """lightweight wrapper serving as the main point for a united API per database"""

    version: str
    raw_paths: dict[str, Path]
    parquet_paths: dict[str, Path]
    dir: Path
    module: DatasetGetModule

    def __str__(self) -> str:
        return f"{self.name}_{self.version}"

    def __repr__(self) -> str:
        return f"<{self.name.capitalize()}_{self.version} at {self.dir}>"

    def __hash__(self):
        return hash(
            (self.version, str(self.sources), str(self.raw_paths), self.module.__name__)
        )

    @cached_property
    def name(self) -> str:
        return self.module.__name__.split(".")[-1]

    @cached_property
    def sources(self) -> dict[str, str]:

        sources = self.module.source_urls(self.version)
        return sources

    @cached_property
    def lfs(self) -> dict[str, LazyFrame]:
        return self.module.cleanly_scan_parquet_tables(self.parquet_paths)

    @cached_property
    def view(self) -> _View:
        return _View(self)

    @classmethod
    def available_versions(cls) -> tuple[str, ...]:
        return cls.module.available_versions()


@dataclass(frozen=True, repr=False)
class Chembl(_Base, ChemblTables):
    module: DatasetGetModule = chembl

    @staticmethod
    def get_available_versions():
        return chembl.available_versions()

    @cached_property
    def view(self) -> _ChemblView:
        return _ChemblView(self)

    @cached_property
    def raw_sql_db_path(self) -> Path:
        return self.raw_paths["sql_db"]

    @classmethod
    def from_version(
        cls,
        version: str | int | float,  # ruff: ignore[PYI041]
        root_dir: Path | str = f"{BASE_DIR}/chembl",
        force: bool = False,
    ) -> "Chembl":
        """Downloads Chembl for version if not yet present in the given cache directory"""
        version = chembl._format_version(version)
        dir = Path(root_dir) / version

        raw_paths: dict[str, Path] = chembl.ensure_raw_files(
            version, raw_dir=dir / "raw", force=force
        )

        parquet_paths = chembl.ensure_parquet_tables(
            raw_paths, table_dir=dir / "parquet"
        )
        return Chembl(
            version=version,
            raw_paths=raw_paths,
            parquet_paths=parquet_paths,
            dir=dir,
            module=cls.module,
        )

    @classmethod
    def from_latest(
        cls,
        root_dir: Path | str = f"{BASE_DIR}/chembl",
        force: bool = False,
    ) -> "Chembl":
        return cls.from_version(version=chembl.latest(), root_dir=root_dir, force=force)


@dataclass(frozen=True, repr=False)
class Papyrus(_Base, PapyrusTables):
    module: DatasetGetModule = papyrus

    @staticmethod
    def get_available_versions():
        return papyrus.available_versions()

    @cached_property
    def view(self) -> _PapyrusView:
        return _PapyrusView(self)

    @classmethod
    def from_version(
        cls,
        version: str,
        root_dir: Path | str = f"{BASE_DIR}/papyrus",
    ) -> "Papyrus":
        """Downloads Chembl for version if not yet present in the given cache directory"""
        dir = Path(root_dir) / version
        raw_paths: dict[str, Path] = papyrus.ensure_raw_files(
            version, raw_dir=dir / "raw"
        )
        parquet_paths: dict[str, Path] = papyrus.ensure_parquet_tables(
            raw_paths, table_dir=dir / "parquet"
        )
        return Papyrus(
            version=version,
            raw_paths=raw_paths,
            parquet_paths=parquet_paths,
            dir=dir,
            module=cls.module,
        )

    @classmethod
    def from_latest(
        cls,
        root_dir: Path | str = f"{BASE_DIR}/papyrus",
    ) -> "Papyrus":
        return cls.from_version(version=papyrus.latest(), root_dir=root_dir)


if __name__ == "__main__":
    p = Papyrus.from_latest()
    p.view.proteins
    p.protein
