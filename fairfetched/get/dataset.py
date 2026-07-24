from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from polars import LazyFrame

from fairfetched.get import chembl, papyrus
from fairfetched.utils import BASE_DIR
from fairfetched.utils.typing import BioactivityDBViews, DatasetGetModule


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
        return f"<{self.name.capitalize()}_{self.version} at {self.dir}"

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

    @property
    def lfs(self) -> dict[str, LazyFrame]:
        return self.module.cleanly_scan_parquet_tables(self.parquet_paths)

    @property
    def views(self) -> BioactivityDBViews:
        return self.module.build_views(self.parquet_paths)

    @property
    def bioactivity(self) -> LazyFrame:
        return self.views["bioactivity"]

    @property
    def compounds(self) -> LazyFrame:
        return self.views["compounds"]

    @classmethod
    def available_versions(cls) -> tuple[str, ...]:
        return cls.module.available_versions()


@dataclass(frozen=True)
class Chembl(_Base):
    module: DatasetGetModule = chembl

    @staticmethod
    def get_available_versions():
        return chembl.available_versions()

    @cached_property
    def activity(self) -> LazyFrame:
        return self.views["bioactivity"]

    @cached_property
    def raw_sql_db_path(self) -> Path:
        return self.raw_paths["sql_db"]

    @classmethod
    def from_version(
        cls,
        version: str | int | float,  # ruff: ignore[PYI041]
        root_dir: Path | str = f"{BASE_DIR}/chembl",
    ) -> "Chembl":
        """Downloads Chembl for version if not yet present in the given cache directory"""
        version = chembl._format_version(version)
        dir = Path(root_dir) / version

        raw_paths: dict[str, Path] = chembl.ensure_raw_files(
            version, raw_dir=dir / "raw"
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
    ) -> "Chembl":
        return cls.from_version(version=chembl.latest(), root_dir=root_dir)


@dataclass(frozen=True)
class Papyrus(_Base):
    module: DatasetGetModule = papyrus

    @staticmethod
    def get_available_versions():
        return papyrus.available_versions()

    @property
    def proteins(self) -> LazyFrame:
        return self.views["proteins"]

    @property
    def full_data(self) -> LazyFrame:
        """all data (bioactivity + protein data),
        composed into one flat tabular format LazyFrame"""
        return self.views["full"]

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
    p.views["proteins"]

    p.lfs["proteins"]
