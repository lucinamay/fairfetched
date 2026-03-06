from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from types import ModuleType
from typing import Tuple

from polars import LazyFrame

import fairfetched.get.chembl as chembl
import fairfetched.get.papyrus as papyrus
from fairfetched.utils import BASE_DIR
from fairfetched.utils.typing import ComposedLFDict, DatasetGetModule


@dataclass(frozen=True)
class _Base:
    version: str
    raw_paths: dict[str, Path]
    consolidated_paths: dict[str, Path]
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

        sources = self.module.get_sources(self.version)
        return sources

    @property
    def lfs(self) -> dict[str, LazyFrame]:
        return self.module.clean(self.consolidated_paths)

    def compose(self) -> ComposedLFDict:
        return self.module.compose(self.lfs)

    @property
    def bioactivity(self) -> LazyFrame:
        return self.compose()["bioactivity"]

    @property
    def compounds(self) -> LazyFrame:
        return self.compose()["compounds"]

    @classmethod
    @cached_property
    def available_versions(cls) -> Tuple[str]:
        return cls.module.available_versions()


@dataclass(frozen=True)
class Chembl(_Base):
    module: ModuleType = chembl

    @cached_property
    def activity(self) -> LazyFrame:
        return self.compose()["bioactivity"]

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
        version = chembl._version_formatter(version)
        dir = Path(root_dir) / version

        raw_paths: dict[str, Path] = chembl.ensure_raw(version, raw_dir=dir / "raw")

        consolidated_paths = chembl.ensure_consolidated(
            raw_paths, consolidated_dir=dir / "consolidated"
        )
        return Chembl(
            version=version,
            raw_paths=raw_paths,
            consolidated_paths=consolidated_paths,
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
    module: ModuleType = papyrus

    @property
    def proteins(self) -> LazyFrame:
        return self.compose()["proteins"]

    @classmethod
    def from_version(
        cls,
        version: str,
        root_dir: Path | str = f"{BASE_DIR}/papyrus",
    ) -> "Papyrus":
        """Downloads Chembl for version if not yet present in the given cache directory"""
        dir = Path(root_dir) / version
        raw_paths: dict[str, Path] = papyrus.ensure_raw(version, raw_dir=dir / "raw")
        consolidated_paths: dict[str, Path] = papyrus.ensure_consolidated(
            raw_paths, consolidated_dir=dir / "consolidated"
        )
        return Papyrus(
            version=version,
            raw_paths=raw_paths,
            consolidated_paths=consolidated_paths,
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
    p.compose()

    p.lfs["proteins"]
