from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Self

from polars import LazyFrame

from fairfetched.get import (
    _demo,
    adrecs,
    adrecs_target,
    chembl,
    drugbank,
    papyrus,
    pubchem_bioassay,
    pubchem_compound,
    sider,
    toxcast,
)
from fairfetched.get._table_autocomplete import (
    AdrecsTables,
    AdrecsTargetTables,
    ChemblTables,
    DrugbankTables,
    PapyrusTables,
    PubchemBioassayTables,
    PubchemCompoundTables,
    SiderTables,
    ToxcastTables,
)
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

    def capricho(self, **params) -> LazyFrame:
        """`bioactivity` with CAPRICHO `drop_*` / `note_*` flags, scanned from a
        cache parquet written on first call (see `fairfetched.get._capricho.flag`
        for `params`)."""
        raise NotImplementedError


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


class _ToxcastView(_View):
    @property
    def targets(self) -> LazyFrame:
        return self._views["targets"]

    @property
    def assay_annotations(self) -> LazyFrame:
        return self._views["assay_annotations"]


class _DrugbankView(_View):
    @property
    def drugs(self) -> LazyFrame:
        return self._views["drugs"]

    @property
    def targets(self) -> LazyFrame:
        return self._views["targets"]

    @property
    def go_classifiers(self) -> LazyFrame:
        return self._views["go_classifiers"]

    @property
    def interactions(self) -> LazyFrame:
        return self._views["interactions"]


class _PubchemBioassayView(_View):
    @property
    def assays(self) -> LazyFrame:
        return self._views["assays"]

    @property
    def proteins(self) -> LazyFrame:
        return self._views["proteins"]


class _PubchemCompoundView(_View):
    @property
    def smiles(self) -> LazyFrame:
        return self._views["smiles"]

    @property
    def inchi(self) -> LazyFrame:
        return self._views["inchi"]


class _SourceTables:
    """Source tables as attributes (``db.tables.drug``) and by key (``db.tables["drug"]``);
    the attributes come from fairfetched.get._table_autocomplete."""

    def __init__(self, owner: "_Base") -> None:
        self._owner = owner

    @property
    def lfs(self) -> dict[str, LazyFrame]:
        return self._owner.lfs

    def __getitem__(self, name: str) -> LazyFrame:
        return self.lfs[name]

    def __str__(self) -> str:
        return str(self._owner)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} available: {', '.join(sorted(self.lfs))}>"


class _ChemblSourceTables(_SourceTables, ChemblTables):
    pass


class _PapyrusSourceTables(_SourceTables, PapyrusTables):
    pass


class _AdrecsSourceTables(_SourceTables, AdrecsTables):
    pass


class _AdrecsTargetSourceTables(_SourceTables, AdrecsTargetTables):
    pass


class _SiderSourceTables(_SourceTables, SiderTables):
    pass


class _ToxcastSourceTables(_SourceTables, ToxcastTables):
    pass


class _DrugbankSourceTables(_SourceTables, DrugbankTables):
    pass


class _PubchemBioassaySourceTables(_SourceTables, PubchemBioassayTables):
    pass


class _PubchemCompoundSourceTables(_SourceTables, PubchemCompoundTables):
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
        """Download URLs; empty for offline demos (``version == "demo"``)."""
        if self.version == "demo":
            return {}
        return self.module.source_urls(self.version)

    @cached_property
    def lfs(self) -> dict[str, LazyFrame]:
        return self.module.cleanly_scan_parquet_tables(self.parquet_paths)

    @cached_property
    def view(self) -> _View:
        return _View(self)

    @classmethod
    def available_versions(cls) -> tuple[str, ...]:
        """Versions the source can provide."""
        return cls.module.available_versions()

    @classmethod
    def _from_download_version(
        cls,
        version: str,
        root_dir: Path | str,
        force: bool,
        ensure_raw_files: Callable[[str, Path, bool], dict[str, Path]],
    ) -> Self:
        dir = Path(root_dir) / str(version)
        raw_paths = ensure_raw_files(str(version), dir / "raw", force)
        parquet_paths = cls.module.ensure_parquet_tables(
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
    def local_versions(cls, root_dir: Path | str | None = None) -> tuple[str, ...]:
        """Versions with Parquet tables under ``root_dir`` (default
        ``BASE_DIR/<dataset>``)."""
        root = Path(root_dir or BASE_DIR / cls.module.__name__.split(".")[-1])
        return tuple(sorted({p.parts[-3] for p in root.glob("*/parquet/*.parquet")}))


@dataclass(frozen=True, repr=False, eq=False)  # eq=False keeps _Base.__hash__
class Chembl(_Base):
    """ChEMBL wrapper: download once, then read lazily.

    ``Chembl.from_latest()`` (or ``Chembl.from_version(35)``) downloads a real
    release. ``Chembl.demo()`` returns a tiny offline sample with the same API.

    Examples:
        >>> from fairfetched.get import Chembl
        >>> db = Chembl.demo()
        >>> len(db.lfs["activities"].collect_schema())  # all columns
        29
        >>> db.lfs["activities"].collect_schema().names()[:4]  # column names
        ['molregno', 'activity_id', 'assay_id', 'doc_id']
        >>> db.tables.molecule_dictionary.filter(molregno=1280).collect()["pref_name"].to_list()
        ['ASPIRIN']
        >>> db.view.compounds.collect().shape
        (5, 50)
        >>> len(db.lfs)
        26

    Selecting and viewing table outputs:

        >>> db.lfs["activities"].filter(molregno=1280).select(  # doctest: +SKIP
        ...     "activity_id", "assay_id", "standard_value"
        ... ).head(3).collect()  # doctest: +SKIP
        shape: (3, 3)
        ┌─────────────┬──────────┬────────────────┐
        │ activity_id ┆ assay_id ┆ standard_value │
        │ ---         ┆ ---      ┆ ---            │
        │ i64         ┆ i64      ┆ f64            │
        ╞═════════════╪══════════╪════════════════╡
        │ ...         ┆ ...      ┆ ...            │
        └─────────────┴──────────┴────────────────┘

    Joining activities with compound structures:

        >>> result = db.lfs["activities"].join(  # doctest: +SKIP
        ...     db.lfs["compound_structures"], on="molregno", how="left", validate="m:1"
        ... ).head().collect()  # doctest: +SKIP
        >>> result.shape  # doctest: +SKIP
        (5, 32)
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
        """Tiny offline slice of ChEMBL 37 (5 molecules, 10 activities). See fairfetched.get._demo."""
        return cls(
            version="demo",
            raw_paths={},
            parquet_paths=_demo.parquets("chembl"),
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
        return cls._from_download_version(
            chembl._format_version(version), root_dir, force, chembl.ensure_raw_files
        )

    @classmethod
    def from_latest(
        cls,
        root_dir: Path | str = f"{BASE_DIR}/chembl",
        force: bool = False,
    ) -> "Chembl":
        return cls.from_version(version=chembl.latest(), root_dir=root_dir, force=force)


@dataclass(frozen=True, repr=False, eq=False)  # eq=False keeps _Base.__hash__
class Papyrus(_Base):
    """Papyrus wrapper: download once, then read lazily.

    ``Papyrus.from_latest()`` (or ``Papyrus.from_version("05.7")``) downloads a
    real release. ``Papyrus.demo()`` returns a tiny offline sample with the same
    API, used by the examples here:

    >>> from fairfetched.get import Papyrus
    >>> db = Papyrus.demo()
    >>> db.view.full.collect().shape            # bioactivity + protein, one flat frame
    (10, 38)
    >>> db.view.proteins.collect()["uniprot_id"].to_list()
    ['KYNU_HUMAN', 'ST2A1_RAT']
    >>> db.tables.bioactivity.collect().height  # raw source tables
    10
    >>> db.tables.protein.collect_schema().names()[:3]  # column names, no scan
    ['target_id', 'uniprot_id', 'status']
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
        """Tiny offline slice of Papyrus 05.7 (10 activities, 2 proteins). See fairfetched.get._demo."""
        return cls(
            version="demo",
            raw_paths={},
            parquet_paths=_demo.parquets("papyrus"),
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


@dataclass(frozen=True, repr=False, eq=False)  # eq=False keeps _Base.__hash__
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
    def tables(self) -> _AdrecsSourceTables:
        return _AdrecsSourceTables(self)

    @classmethod
    def from_version(
        cls,
        version: str,
        root_dir: Path | str = f"{BASE_DIR}/adrecs",
        force: bool = False,
    ) -> "Adrecs":
        return cls._from_download_version(
            version, root_dir, force, adrecs.ensure_raw_files
        )

    @classmethod
    def from_latest(
        cls, root_dir: Path | str = f"{BASE_DIR}/adrecs", force: bool = False
    ) -> "Adrecs":
        return cls.from_version(adrecs.latest(), root_dir=root_dir, force=force)


@dataclass(frozen=True, repr=False, eq=False)  # eq=False keeps _Base.__hash__
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
    def tables(self) -> _AdrecsTargetSourceTables:
        return _AdrecsTargetSourceTables(self)

    @classmethod
    def from_version(
        cls,
        version: str = "1.0",
        root_dir: Path | str = f"{BASE_DIR}/adrecs_target",
        force: bool = False,
    ) -> "AdrecsTarget":
        return cls._from_download_version(
            version, root_dir, force, adrecs_target.ensure_raw_files
        )

    @classmethod
    def from_latest(
        cls, root_dir: Path | str = f"{BASE_DIR}/adrecs_target", force: bool = False
    ) -> "AdrecsTarget":
        return cls.from_version(adrecs_target.latest(), root_dir=root_dir, force=force)


@dataclass(frozen=True, repr=False, eq=False)  # eq=False keeps _Base.__hash__
class Sider(_Base):
    """SIDER wrapper: download once, then read lazily.

    SIDER's URLs are unversioned, so the release is pinned by content hash
    (``fairfetched/get/manifests/sider.json``); a changed upstream file
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
    def tables(self) -> _SiderSourceTables:
        return _SiderSourceTables(self)

    @classmethod
    def from_version(
        cls,
        version: str = "4.1",
        root_dir: Path | str = f"{BASE_DIR}/sider",
        force: bool = False,
    ) -> "Sider":
        return cls._from_download_version(
            version, root_dir, force, sider.ensure_raw_files
        )

    @classmethod
    def from_latest(
        cls, root_dir: Path | str = f"{BASE_DIR}/sider", force: bool = False
    ) -> "Sider":
        return cls.from_version(sider.latest(), root_dir=root_dir, force=force)


@dataclass(frozen=True, repr=False, eq=False)  # eq=False keeps _Base.__hash__
class Toxcast(_Base):
    """ToxCast (EPA invitrodb) wrapper: download once, then read lazily.

    The 7.5 GB summary zip is fetched whole; only its ``mc5-6`` table is kept
    (as Parquet). Clowder file ids are not content pins, so the release is
    pinned by hash (``fairfetched/get/manifests/toxcast.json``).
    ``view`` holds ``compounds``, ``bioactivity``, ``targets``,
    ``assay_annotations``::

        db = Toxcast.from_latest()
        db.view.bioactivity.sink_parquet("toxcast_bioactivity.parquet")

    ``Toxcast.demo()`` returns a tiny offline sample with the same API:

    >>> from fairfetched.get import Toxcast
    >>> db = Toxcast.demo()
    >>> db.view.compounds.collect()["chnm"].to_list()
    ['Acetamide', 'Acetaminophen', 'Acifluorfen']
    >>> db.view.bioactivity.collect().shape     # rows without a chemical are dropped
    (4, 76)
    >>> sorted(db.view.bioactivity.collect()["aenm"].unique())
    ['ACEA_ER_80hr', 'APR_HepG2_CellCycleArrest_1hr']
    >>> db.view.targets.collect().shape         # long format: one row per (aeid, target_type)
    (4, 7)
    """

    module: DatasetGetModule = toxcast

    @staticmethod
    def get_available_versions():
        return toxcast.available_versions()

    @cached_property
    def view(self) -> _ToxcastView:
        return _ToxcastView(self)

    @cached_property
    def tables(self) -> _ToxcastSourceTables:
        return _ToxcastSourceTables(self)

    @classmethod
    def demo(cls) -> "Toxcast":
        """Tiny offline slice of ToxCast 4.3 (3 chemicals, 2 endpoints). See fairfetched.get._demo."""
        return cls(
            version="demo",
            raw_paths={},
            parquet_paths=_demo.parquets("toxcast"),
            dir=_demo.DEMO_DIR / "toxcast",
            module=cls.module,
        )

    @classmethod
    def from_version(
        cls,
        version: str = "4.3",
        root_dir: Path | str = f"{BASE_DIR}/toxcast",
        force: bool = False,
    ) -> "Toxcast":
        return cls._from_download_version(
            version, root_dir, force, toxcast.ensure_raw_files
        )

    @classmethod
    def from_latest(
        cls, root_dir: Path | str = f"{BASE_DIR}/toxcast", force: bool = False
    ) -> "Toxcast":
        return cls.from_version(toxcast.latest(), root_dir=root_dir, force=force)


@dataclass(frozen=True, repr=False, eq=False)  # eq=False keeps _Base.__hash__
class Drugbank(_Base):
    """DrugBank wrapper. DrugBank is licensed, so there is no download: register
    your own ``full database.xml`` (or the release ``.zip``) once with
    ``from_xml``, then reload it with ``from_version``.

        db = Drugbank.from_xml("~/Downloads/drugbank_all_full_database.xml.zip",
                               version="5.1.13")
        db.view.targets.sink_parquet("drugbank_targets.parquet")
        db.tables["drug"].collect_schema()

    ``view`` holds the flat ``drugs``, ``targets``, ``go_classifiers`` and
    ``interactions``; ``tables`` holds the four nested raw tables (``drug``,
    ``biomolecule``, ``pathway``, ``drug_drug``). See
    :mod:`fairfetched.get.drugbank` for the layout and how the XML is pinned.
    """

    module: DatasetGetModule = drugbank

    @staticmethod
    def get_available_versions():
        return drugbank.available_versions()

    @cached_property
    def view(self) -> _DrugbankView:
        return _DrugbankView(self)

    @cached_property
    def tables(self) -> _DrugbankSourceTables:
        return _DrugbankSourceTables(self)

    @classmethod
    def _build(
        cls,
        version: str,
        root_dir: Path | str,
        xml_path: Path | str | None = None,
        force: bool = False,
    ) -> "Drugbank":
        dir = Path(root_dir) / version
        raw_paths = drugbank.ensure_raw_files(
            version, raw_dir=dir / "raw", xml_path=xml_path, force=force
        )
        parquet_paths = drugbank.ensure_parquet_tables(
            raw_paths, table_dir=dir / "parquet", force=force
        )
        return cls(
            version=version,
            raw_paths=raw_paths,
            parquet_paths=parquet_paths,
            dir=dir,
            module=cls.module,
        )

    @classmethod
    def from_xml(
        cls,
        xml_path: Path | str,
        version: str | None = None,
        root_dir: Path | str = f"{BASE_DIR}/drugbank",
        force: bool = False,
    ) -> "Drugbank":
        """Register a licensed DrugBank XML/zip and build its Parquet tables.
        Without ``version``, the XML root's ``version`` attribute names the
        directory."""
        version = str(version) if version else drugbank.version_of(xml_path)
        return cls._build(version, root_dir, xml_path=xml_path, force=force)

    @classmethod
    def from_version(
        cls,
        version: str,
        root_dir: Path | str = f"{BASE_DIR}/drugbank",
    ) -> "Drugbank":
        """Reload an already-registered version; raises if it was never registered."""
        return cls._build(str(version), root_dir)


@dataclass(frozen=True, repr=False, eq=False)  # eq=False keeps _Base.__hash__
class PubchemBioassay(_Base):
    """PubChem BioAssay wrapper: download once, then read lazily.

    PubChem serves its current build only, so the version is that build's date
    and ``from_version`` reopens a snapshot already on disk
    (``PubchemBioassay.local_versions()``). A snapshot is 3.2 GB to download and
    2.7 GB as Parquet; conversion needs 18 GB of temp space.

    ``view.bioactivity`` and ``view.assays`` leave out the assays deposited by
    ChEMBL and the ToxCast/Tox21 programme
    (``pubchem_bioassay.EXCLUDED_SOURCES``); ``tables`` keeps them::

        db = PubchemBioassay.from_latest()
        db.view.bioactivity.drop_nulls("activity_value").sink_parquet("potencies.parquet")

    ``PubchemBioassay.demo()`` returns a tiny offline sample with the same API:

    >>> from fairfetched.get import PubchemBioassay
    >>> db = PubchemBioassay.demo()
    >>> db.view.assays.collect()["aid"].to_list()
    [429, 1938]
    >>> db.tables.bioassays.collect().height    # 5 more, from the excluded depositors
    7
    >>> ec50 = db.view.bioactivity.filter(aid=1938).collect()
    >>> ec50["activity_name"].unique().to_list(), ec50["uniprot_ids"][0].to_list()
    (['EC50'], ['P08485'])
    >>> db.view.bioactivity.filter(aid=429).collect().shape   # 2 substances x 2 targets
    (4, 18)
    """

    module: DatasetGetModule = pubchem_bioassay

    @cached_property
    def view(self) -> _PubchemBioassayView:
        return _PubchemBioassayView(self)

    @cached_property
    def tables(self) -> _PubchemBioassaySourceTables:
        return _PubchemBioassaySourceTables(self)

    @classmethod
    def demo(cls) -> "PubchemBioassay":
        """Tiny offline slice of the 20260929 snapshot (7 assays, 13
        bioactivities). See fairfetched.get._demo."""
        return cls(
            version="demo",
            raw_paths={},
            parquet_paths=_demo.parquets("pubchem_bioassay"),
            dir=_demo.DEMO_DIR / "pubchem_bioassay",
            module=cls.module,
        )

    @classmethod
    def from_version(
        cls,
        version: str,
        root_dir: Path | str = f"{BASE_DIR}/pubchem_bioassay",
        force: bool = False,
    ) -> "PubchemBioassay":
        return cls._from_download_version(
            version, root_dir, force, pubchem_bioassay.ensure_raw_files
        )

    @classmethod
    def from_latest(
        cls, root_dir: Path | str = f"{BASE_DIR}/pubchem_bioassay", force: bool = False
    ) -> "PubchemBioassay":
        return cls.from_version(
            pubchem_bioassay.latest(), root_dir=root_dir, force=force
        )


@dataclass(frozen=True, repr=False, eq=False)  # eq=False keeps _Base.__hash__
class PubchemCompound(_Base):
    """PubChem Compound wrapper: SMILES, InChI and InChIKey for every CID.

    Versioned by build date like :class:`PubchemBioassay`. A snapshot is 8.9 GB
    to download and 8.4 GB as Parquet; conversion needs 24 GB of temp space.

    ``view.smiles`` and ``view.inchi`` are separate on purpose: join each onto
    the frame that needs structures (see ``pubchem_compound.build_views``)::

        db = PubchemCompound.from_latest()
        mine.join(db.view.smiles, on="cid", how="left").join(
            db.view.inchi, on="cid", how="left"
        )

    ``PubchemCompound.demo()`` returns a tiny offline sample with the same API:

    >>> from fairfetched.get import PubchemCompound
    >>> db = PubchemCompound.demo()
    >>> db.view.smiles.filter(cid=2244).collect()["smiles"].to_list()
    ['CC(=O)OC1=CC=CC=C1C(=O)O']
    >>> db.view.inchi.filter(cid=2244).collect()["inchikey"].to_list()
    ['BSYNRYMUTXBXSQ-UHFFFAOYSA-N']
    >>> db.view.smiles.collect().height, db.view.inchi.collect().height
    (3, 2)
    """

    module: DatasetGetModule = pubchem_compound

    @cached_property
    def view(self) -> _PubchemCompoundView:
        return _PubchemCompoundView(self)

    @cached_property
    def tables(self) -> _PubchemCompoundSourceTables:
        return _PubchemCompoundSourceTables(self)

    @classmethod
    def demo(cls) -> "PubchemCompound":
        """Tiny offline slice of the 20260927 snapshot (3 compounds, one without
        an InChI row). See fairfetched.get._demo."""
        return cls(
            version="demo",
            raw_paths={},
            parquet_paths=_demo.parquets("pubchem_compound"),
            dir=_demo.DEMO_DIR / "pubchem_compound",
            module=cls.module,
        )

    @classmethod
    def from_version(
        cls,
        version: str,
        root_dir: Path | str = f"{BASE_DIR}/pubchem_compound",
        force: bool = False,
    ) -> "PubchemCompound":
        return cls._from_download_version(
            version, root_dir, force, pubchem_compound.ensure_raw_files
        )

    @classmethod
    def from_latest(
        cls, root_dir: Path | str = f"{BASE_DIR}/pubchem_compound", force: bool = False
    ) -> "PubchemCompound":
        return cls.from_version(
            pubchem_compound.latest(), root_dir=root_dir, force=force
        )
