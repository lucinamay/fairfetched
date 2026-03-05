from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from types import ModuleType
from typing import Tuple

from polars import LazyFrame

import fairfetched.get.chembl as chembl
import fairfetched.get.papyrus as papyrus
from fairfetched.utils import BASE_DIR


@dataclass(frozen=True)
class _Base:
    version: str
    raw_paths: dict[str, Path]
    lfs: dict[str, LazyFrame]
    dir: Path
    module: ModuleType

    def __hash__(self):
        return hash(
            (self.version, str(self.sources), str(self.raw_paths), self.module.__name__)
        )

    @cached_property
    def name(self) -> str:
        return self.module.__name__.split(".")[-1]

    @cached_property
    def sources(self) -> dict[str, str]:
        return self.module.get_sources(self.version)

    @property
    def composed(self) -> dict[str, LazyFrame]:
        lfs = self.module.compose(self.lfs)
        return lfs

    @property
    def bioactivity(self) -> LazyFrame:
        return self.composed["bioactivity"]

    @property
    def compounds(self) -> LazyFrame:
        return self.composed["compounds"]

    @classmethod
    @cached_property
    def available_versions(cls) -> Tuple[str]:
        return cls.module.available_versions()


@dataclass(frozen=True)
class Chembl(_Base):
    module: ModuleType = chembl
    extracted_table_paths: dict[str, Path] | None = None

    @cached_property
    def activity(self) -> LazyFrame:
        return self.composed["activity"]

    @cached_property
    def raw_sql_db_path(self) -> Path:
        return self.raw_paths["sql_db"]

    @classmethod
    def from_version(
        cls,
        version: str | int | float,
        root_dir: Path | str = f"{BASE_DIR}/chembl",
    ) -> "Chembl":
        """Downloads Chembl for version if not yet present in the given cache directory"""
        version = chembl.__version_formatter(version)
        dir = Path(root_dir) / version

        raw_paths: dict[str, Path] = cls.module.ensure_raw(
            version, cache_dir=dir / "raw"
        )
        extracted_table_paths = cls.module.extract_sqlite(
            raw_paths["sql_db"], cache_dir=dir / "extracted"
        )
        lfs = cls.module.clean(extracted_table_paths)
        return Chembl(
            version=version,
            raw_paths=raw_paths,
            lfs=lfs,
            dir=dir,
            module=cls.module,
            extracted_table_paths=extracted_table_paths,
        )

    @classmethod
    def from_latest(cls, root_dir) -> "Chembl":
        return cls.from_version(version=cls.module.latest(), root_dir=root_dir)


@dataclass(frozen=True)
class Papyrus(_Base):
    module: ModuleType = papyrus

    @classmethod
    def from_version(
        cls,
        version: str,
        root_dir: Path | str = f"{BASE_DIR}/papyrus",
    ) -> "Papyrus":
        """Downloads Chembl for version if not yet present in the given cache directory"""
        dir = Path(root_dir) / version
        raw_paths: dict[str, Path] = cls.module.ensure_raw(
            version, cache_dir=dir / "raw"
        )
        lfs = cls.module.clean(raw_paths)
        return Papyrus(
            version=version,
            raw_paths=raw_paths,
            lfs=lfs,
            dir=dir,
            module=cls.module,
        )

    @classmethod
    def from_latest(
        cls,
        root_dir: Path | str = f"{BASE_DIR}/papyrus",
    ) -> "Papyrus":
        return cls.from_version(version=cls.module.latest(), root_dir=root_dir)
