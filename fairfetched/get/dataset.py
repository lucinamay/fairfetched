from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from polars import LazyFrame

from fairfetched.get import _demo, adrecs, adrecs_target, chembl, papyrus, sider
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
    raw-table attributes (``Papyrus.tables.protein``, ``Papyrus.tables.bioactivity``) when you
    want the source columns without the view's renames."""

    @property
    def proteins(self) -> LazyFrame:
        return self._views["proteins"]

    @property
    def full(self) -> LazyFrame:
        """bioactivity + protein data as one flat LazyFrame."""
        return self._views["full"]


class _AdrecsView(_View):
    @property
    def drugs(self) -> LazyFrame:
        return self._views["drugs"]

    @property
    def adrs(self) -> LazyFrame:
        return self._views["adrs"]

    @property
    def drug_adr(self) -> LazyFrame:
        return self._views["drug_adr"]


class _AdrecsTargetView(_View):
    @property
    def proteins(self) -> LazyFrame:
        return self._views["proteins"]

    @property
    def drug_adr_protein(self) -> LazyFrame:
        return self._views["drug_adr_protein"]

    @property
    def drug_adr_gene(self) -> LazyFrame:
        return self._views["drug_adr_gene"]


class _SiderView(_View):
    @property
    def drugs(self) -> LazyFrame:
        return self._views["drugs"]

    @property
    def side_effects(self) -> LazyFrame:
        return self._views["side_effects"]

    @property
    def frequencies(self) -> LazyFrame:
        return self._views["frequencies"]


class _SourceTables:
    """Source tables as attributes; see fairfetched.get._tables."""

    def __init__(self, owner: "_Base") -> None:
        self._owner = owner

    @property
    def lfs(self) -> dict[str, LazyFrame]:
        return self._owner.lfs

    def __str__(self) -> str:
        return str(self._owner)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} available: {', '.join(sorted(self.lfs))}>"


class _ChemblSourceTables(_SourceTables, ChemblTables):
    pass


class _PapyrusSourceTables(_SourceTables, PapyrusTables):
    pass


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
class Chembl(_Base):
    """ChEMBL wrapper: download once, then read lazily.

    ``Chembl.from_latest()`` (or ``Chembl.from_version(35)``) downloads a real
    release. ``Chembl.demo()`` returns a tiny offline sample with the same API,
    used by the examples here:

    >>> from fairfetched.get import Chembl
    >>> db = Chembl.demo()
    >>> db.view.compounds.collect().shape                    # joined domain views
    (3, 20)
    >>> db.view.bioactivity.filter(assay_id=54505).sink_csv('my_bioactivity_data.csv')
    >>> db.tables.molecule_dictionary.collect()["pref_name"].to_list()
    ['Aspirin', 'Ibuprofen', 'Ibuprofen sodium']
    >>> db.tables.molecule_dictionary.collect_schema()       # column names + dtypes, no scan
    Schema({'molregno': Int64, 'chembl_id': String, 'pref_name': String, 'max_phase': Float64, 'molecule_type': String, 'withdrawn_flag': Int64, 'chirality': Int64})
    >>> len(db.lfs)                                          # every raw table
    26
    """

    module: DatasetGetModule = chembl

    @staticmethod
    def get_available_versions():
        return chembl.available_versions()

    @cached_property
    def view(self) -> _ChemblView:
        return _ChemblView(self)

    @cached_property
    def tables(self) -> _ChemblSourceTables:
        return _ChemblSourceTables(self)

    @classmethod
    def demo(cls) -> "Chembl":
        """Tiny offline sample (3 molecules, 3 activities, 2 targets). See fairfetched.get._demo."""
        return cls(
            version="demo",
            raw_paths={},
            parquet_paths=_demo.chembl_parquets(),
            dir=_demo.DEMO_DIR / "chembl",
            module=cls.module,
        )

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
class Papyrus(_Base):
    """Papyrus wrapper: download once, then read lazily.

    ``Papyrus.from_latest()`` (or ``Papyrus.from_version("05.7")``) downloads a
    real release. ``Papyrus.demo()`` returns a tiny offline sample with the same
    API, used by the examples here:

    >>> from fairfetched.get import Papyrus
    >>> db = Papyrus.demo()
    >>> db.view.full.collect().shape            # bioactivity + protein, one flat frame
    (3, 9)
    >>> db.view.proteins.collect()["pref_name"].to_list()
    ['Kinase 1', 'Kinase 2']
    >>> db.tables.bioactivity.collect().height  # raw source tables
    3
    >>> db.tables.protein.collect_schema()      # column names + dtypes, no scan
    Schema({'target_id': Int64, 'uniprot_id': String, 'target_chembl_id': String, 'pref_name': String})
    """

    module: DatasetGetModule = papyrus

    @staticmethod
    def get_available_versions():
        return papyrus.available_versions()

    @cached_property
    def view(self) -> _PapyrusView:
        return _PapyrusView(self)

    @cached_property
    def tables(self) -> _PapyrusSourceTables:
        return _PapyrusSourceTables(self)

    @classmethod
    def demo(cls) -> "Papyrus":
        """Tiny offline sample (3 activities, 2 proteins). See fairfetched.get._demo."""
        return cls(
            version="demo",
            raw_paths={},
            parquet_paths=_demo.papyrus_parquets(),
            dir=_demo.DEMO_DIR / "papyrus",
            module=cls.module,
        )

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


@dataclass(frozen=True, repr=False)
class Adrecs(_Base):
    """ADReCS wrapper: download once, then read lazily.

    ``tables`` holds the raw source tables (cleaned on scan); ``view`` holds the
    joined domain views ``drugs``, ``adrs``, ``drug_adr``::

        db = Adrecs.from_latest()
        db.view.drug_adr.sink_parquet("adrecs_drug_adr.parquet")
        db.tables["adr"].collect_schema()
    """

    module: DatasetGetModule = adrecs

    @staticmethod
    def get_available_versions():
        return adrecs.available_versions()

    @cached_property
    def view(self) -> _AdrecsView:
        return _AdrecsView(self)

    @cached_property
    def tables(self) -> dict[str, LazyFrame]:
        return self.lfs

    @classmethod
    def from_version(
        cls,
        version: str,
        root_dir: Path | str = f"{BASE_DIR}/adrecs",
        force: bool = False,
    ) -> "Adrecs":
        dir = Path(root_dir) / str(version)
        raw_paths = adrecs.ensure_raw_files(
            str(version), raw_dir=dir / "raw", force=force
        )
        parquet_paths = adrecs.ensure_parquet_tables(raw_paths, table_dir=dir / "parquet")
        return cls(
            version=str(version),
            raw_paths=raw_paths,
            parquet_paths=parquet_paths,
            dir=dir,
            module=cls.module,
        )

    @classmethod
    def from_latest(
        cls, root_dir: Path | str = f"{BASE_DIR}/adrecs", force: bool = False
    ) -> "Adrecs":
        return cls.from_version(adrecs.latest(), root_dir=root_dir, force=force)


@dataclass(frozen=True, repr=False)
class AdrecsTarget(_Base):
    """ADReCS-Target wrapper: download once, then read lazily.

    ``view`` holds the joined domain views ``proteins``, ``drug_adr_protein``,
    ``drug_adr_gene``::

        db = AdrecsTarget.from_latest()
        db.view.drug_adr_protein.sink_parquet("drug_adr_protein.parquet")
    """

    module: DatasetGetModule = adrecs_target

    @staticmethod
    def get_available_versions():
        return adrecs_target.available_versions()

    @cached_property
    def view(self) -> _AdrecsTargetView:
        return _AdrecsTargetView(self)

    @cached_property
    def tables(self) -> dict[str, LazyFrame]:
        return self.lfs

    @classmethod
    def from_version(
        cls,
        version: str = "1.0",
        root_dir: Path | str = f"{BASE_DIR}/adrecs_target",
        force: bool = False,
    ) -> "AdrecsTarget":
        dir = Path(root_dir) / str(version)
        raw_paths = adrecs_target.ensure_raw_files(
            str(version), raw_dir=dir / "raw", force=force
        )
        parquet_paths = adrecs_target.ensure_parquet_tables(
            raw_paths, table_dir=dir / "parquet"
        )
        return cls(
            version=str(version),
            raw_paths=raw_paths,
            parquet_paths=parquet_paths,
            dir=dir,
            module=cls.module,
        )

    @classmethod
    def from_latest(
        cls, root_dir: Path | str = f"{BASE_DIR}/adrecs_target", force: bool = False
    ) -> "AdrecsTarget":
        return cls.from_version(adrecs_target.latest(), root_dir=root_dir, force=force)


@dataclass(frozen=True, repr=False)
class Sider(_Base):
    """SIDER wrapper: download once, then read lazily.

    SIDER's URLs are unversioned, so the release is pinned by content hash
    (``fairfetched.get.sider._sider_manifest.json``); a changed upstream file
    raises on download. ``view`` holds the joined domain views ``drugs``,
    ``side_effects``, ``frequencies``::

        db = Sider.from_latest()
        db.view.frequencies.sink_parquet("sider_frequencies.parquet")
        db.tables["meddra"].collect_schema()
    """

    module: DatasetGetModule = sider

    @staticmethod
    def get_available_versions():
        return sider.available_versions()

    @cached_property
    def view(self) -> _SiderView:
        return _SiderView(self)

    @cached_property
    def tables(self) -> dict[str, LazyFrame]:
        return self.lfs

    @classmethod
    def from_version(
        cls,
        version: str = "4.1",
        root_dir: Path | str = f"{BASE_DIR}/sider",
        force: bool = False,
    ) -> "Sider":
        dir = Path(root_dir) / str(version)
        raw_paths = sider.ensure_raw_files(
            str(version), raw_dir=dir / "raw", force=force
        )
        parquet_paths = sider.ensure_parquet_tables(raw_paths, table_dir=dir / "parquet")
        return cls(
            version=str(version),
            raw_paths=raw_paths,
            parquet_paths=parquet_paths,
            dir=dir,
            module=cls.module,
        )

    @classmethod
    def from_latest(
        cls, root_dir: Path | str = f"{BASE_DIR}/sider", force: bool = False
    ) -> "Sider":
        return cls.from_version(sider.latest(), root_dir=root_dir, force=force)
