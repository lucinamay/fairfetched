"""DrugBank source: the full-database XML as four nested Parquet tables.

``drug`` is the hub -- one row per drug, every block that hangs off exactly one
drug kept as a ``List(Struct)`` column, plus a foreign key to each other table.
``biomolecule`` and ``pathway`` hold the two blocks measured to repeat verbatim
across drugs; ``drug_drug`` holds the 2,855,848 interactions. Nothing in the XML
is dropped but the two paths that are empty on every drug (``ahfs-codes``,
``products/ndc-id``). ``view`` flattens this back to the columns consumers use.

DrugBank is licensed. fairfetched ships no data and cannot download it. Register
your own licensed copy once -- the release ``.zip`` or an extracted
``full database.xml``::

    from fairfetched.get import Drugbank
    db = Drugbank.from_xml("~/Downloads/drugbank_all_full_database.xml.zip", version="5.1.13")
    db.view.targets.sink_parquet("drugbank_targets.parquet")
    db.tables["drug"].collect_schema()

``from_xml`` streams the file into
``$FAIRFETCHED_HOME/drugbank/<version>/raw/full_database.xml.gz`` (~10x smaller
than the raw XML; the parser reads it straight through ``gzip``) and writes a
local ``_manifest.json``with gzip's sha256 (:mod:`fairfetched.utils.manifest`, as SIDER uses).
Later checks hash and raises if the cache changed on disk. The manifest is not committed.

Once the Parquet tables exist the raw XML is never read again; re-parse by
deleting ``$FAIRFETCHED_HOME/drugbank/<version>/parquet``.
"""

import gzip
import logging as lg
import shutil
import xml.etree.ElementTree as ET
import zipfile
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import polars as pl

from fairfetched.utils import BASE_DIR, manifest, raw

_lg = lg.getLogger(__name__)

DRUGBANK_DIR = BASE_DIR / "drugbank"

_MANIFEST_NAME = "_manifest.json"
_DRIFT_MSG = (
    "the registered DrugBank cache changed on disk. Re-register from your "
    "licensed source with Drugbank.from_xml(path, version=..., force=True)."
)

# ponytail: one namespace constant, DrugBank has used it since 4.x. A release
# that changes it parses to zero drugs and ``ensure_parquet_tables`` raises.
# Everywhere else the parser strips the namespace off the tag rather than map it.
_NS = "{http://www.drugbank.ca}"
_DRUG_TAG = _NS + "drug"

DRUGBANK_VERSIONS: tuple[str, ...] = ("5.1.10", "5.1.11", "5.1.12", "5.1.13")

# licence required; not auto-fetchable
_RELEASES_URL = "https://go.drugbank.com/releases"


def available_versions() -> tuple[str, ...]:
    return DRUGBANK_VERSIONS


def latest() -> str:
    return DRUGBANK_VERSIONS[-1]


def source_urls(version: str) -> dict[str, str]:
    return {"full_database": _RELEASES_URL}


# -- register (bring your own licensed file) -------------------------------


@contextmanager
def _uncompressed_stream(path: Path):
    """Binary stream of the XML, whether given the release ``.zip``, a ``.gz`` or
    a plain ``.xml``."""
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            inner = next(n for n in z.namelist() if n.endswith(".xml"))
            with z.open(inner) as src:
                yield src
    elif path.suffix == ".gz":
        with gzip.open(path, "rb") as src:
            yield src
    else:
        with open(path, "rb") as src:
            yield src


def version_of(xml_path: Path | str) -> str:
    """The ``version`` attribute on the XML root, e.g. ``<drugbank version="5.1.13">``.
    Reads only the root's start tag, so it costs nothing on a multi-GB file."""
    with _uncompressed_stream(Path(xml_path).expanduser()) as src:
        version = next(ET.iterparse(src, events=("start",)))[1].get("version")
    if version is None:
        raise ValueError(f"no version attribute on the root element of {xml_path}")
    return version


def register(
    xml_path: Path | str,
    version: str | None = None,
    raw_dir: Path | str | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """Gzip the licensed XML into the data dir and pin it with a local manifest.
    Returns ``{"full_database": <gz path>}``. Idempotent: an existing cache is
    verified and returned unless ``force``."""
    xml_path = Path(xml_path).expanduser()
    version = str(version) if version else version_of(xml_path)
    raw_dir = Path(raw_dir or DRUGBANK_DIR / version / "raw")
    gz = raw_dir / "full_database.xml.gz"
    paths = {"full_database": gz}
    if gz.exists() and not force:
        manifest.verify(paths, raw_dir / _MANIFEST_NAME, _DRIFT_MSG)
        return paths

    raw_dir.mkdir(parents=True, exist_ok=True)
    with _uncompressed_stream(xml_path) as src, gzip.open(gz, "wb") as out:
        shutil.copyfileobj(src, out, 1 << 20)

    manifest.write(paths, raw_dir / _MANIFEST_NAME, version=version)
    _lg.info(
        f"registered DrugBank {version}: {xml_path} -> {gz} ({gz.stat().st_size} B gz)"
    )
    return paths


def ensure_raw_files(
    version: str,
    raw_dir: Path | str | None = None,
    xml_path: Path | str | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """Return the registered ``full_database.xml.gz``. Registers it first if
    ``xml_path`` is given; raises with instructions if the cache is absent and no
    path is supplied."""
    if xml_path is not None:
        return register(xml_path, version, raw_dir, force=force)
    raw_dir = Path(raw_dir or DRUGBANK_DIR / version / "raw")
    paths = {"full_database": raw_dir / "full_database.xml.gz"}
    if not paths["full_database"].exists():
        raise FileNotFoundError(
            f"DrugBank {version} is not registered. Download the release from "
            f"{_RELEASES_URL} under your licence and register it with "
            f"Drugbank.from_xml(path, version='{version}')."
        )
    manifest.verify(paths, raw_dir / _MANIFEST_NAME, _DRIFT_MSG)
    return paths


# -- parse XML -> Parquet -------------------------------------------------

_BOX = frozenset(
    {
        "actions",
        "affected-organisms",
        "articles",
        "atc-codes",  # has @code => record
        "attachments",
        "calculated-properties",
        "carriers",
        "categories",
        "dosages",
        "drug-interactions",
        "drugs",
        "enzymes",
        "experimental-properties",
        "external-identifiers",
        "external-links",
        "food-interactions",
        "go-classifiers",
        "groups",
        "international-brands",
        "links",
        "manufacturers",
        "mixtures",
        "packagers",
        "patents",
        "pathways",
        "pdb-entries",
        "pfams",
        "prices",
        "products",
        "reactions",
        "salts",
        "sequences",
        "snp-adverse-drug-reactions",
        "snp-effects",
        "synonyms",
        "targets",
        "textbooks",
        "transporters",
    }
)


_PARENT_CHILD_PAIRS = frozenset(
    {
        ("drug", "drugbank-id"),
        ("classification", "alternative-parent"),
        ("classification", "substituent"),
        ("atc-code", "level"),
        ("target", "polypeptide"),
        ("enzyme", "polypeptide"),
        ("carrier", "polypeptide"),
        ("transporter", "polypeptide"),
    }
)

# (parent, leaf) pairs whose text carries attributes, rendered {"@attr":…, "$": text};
# an attribute on any other leaf raises. @primary is present on a drug's primary
# <drugbank-id> and absent on its secondary ones.
_ATTR_LEAF = frozenset(
    {
        ("drug", "drugbank-id"),
        ("salt", "drugbank-id"),
        ("synonyms", "synonym"),
        ("atc-code", "level"),
        ("sequences", "sequence"),
        ("manufacturers", "manufacturer"),
        ("price", "cost"),
        ("polypeptide", "organism"),
        ("polypeptide", "gene-sequence"),
        ("polypeptide", "amino-acid-sequence"),
    }
)

_BIOMOL_TERMS = ("targets", "enzymes", "carriers", "transporters")

# drug's-bio-entity pair-specific
_DRUG_BIOMOL_ATTR = (
    "@position",
    "actions",
    "known-action",
    "inhibition-strength",
    "induction-strength",
    "references",
)

_DDI_SCHEMA = {
    "drugbank_id": pl.String,
    "interacts_with_id": pl.String,
    "description": pl.String,
}
_CHUNK = 2000  # drugs per drug_drug part-file


def _tag(elem) -> str:
    return elem.tag.split("}")[-1]


def _xml_element(elem, parent: str = ""):
    """One XML element as nested dict/list/str, faithful to the tree.

    Whitespace-only text becomes ``None`` so that an unpopulated ``<references/>``
    is a null struct rather than a string colliding with its populated siblings.
    """
    tag = _tag(elem)
    if tag in _BOX:
        if elem.attrib:
            raise ValueError(f"container <{tag}> gained attributes {list(elem.attrib)}")
        return [_xml_element(k, tag) for k in elem]

    attrs = {f"@{k.split('}')[-1]}": v for k, v in elem.attrib.items()}
    if not len(elem):
        text = elem.text if elem.text and elem.text.strip() else None
        if (parent, tag) in _ATTR_LEAF:
            return {**attrs, "$": text}
        if attrs:
            raise ValueError(
                f"<{parent}>/<{tag}> carries {list(attrs)}; add it to _ATTR_LEAF"
            )
        return text

    out = dict(attrs)
    for child in elem:
        ct = _tag(child)
        if (tag, ct) in _PARENT_CHILD_PAIRS:
            out.setdefault(ct, []).append(_xml_element(child, tag))
        elif ct in out:
            raise ValueError(
                f"<{tag}> repeats <{ct}>; add ('{tag}', '{ct}') to _PARENT_CHILD_PAIRS, "
                f"or '{tag}' to _BOX if it is a container"
            )
        else:
            out[ct] = _xml_element(child, tag)
    return out


def _iter_top_drugs(fh):
    """Yield each top-level ``<drug>`` element, cleared after use. DrugBank nests
    ``<drug>`` inside ``<pathways>`` too, so match on depth, not tag alone."""
    root = None
    depth = 0
    for event, elem in ET.iterparse(fh, events=("start", "end")):
        if event == "start":
            if root is None:
                root = elem
            depth += 1
            continue
        depth -= 1
        if depth == 1 and elem.tag == _DRUG_TAG:
            yield elem
            root.clear()  # drop processed siblings; saves memory


def _set_default_safe(table: dict, key: str, row: dict, drugbank_id: str) -> None:
    if table.setdefault(key, row) != row:
        raise ValueError(
            f"{key} under {drugbank_id} differs from its earlier occurrence"
        )


def _split(elem, biomolecules: dict, pathways: dict, ddi: list) -> dict:
    """One ``<drug>`` into the drug row, adding to the shared entity tables.

    The four _BIOMOL_TERM subtrees collapse into one ``binds`` list because ``kind`` is a
    property of the edge, not of the bio-entity: 286 of 3,266 BE ids are a target
    for one drug and an enzyme or carrier for another. The entity half is hoisted
    verbatim -- 0 of 3,266 BE ids differ between occurrences -- which removes the
    6.9x polypeptide and 11.3x GO-classifier repetition.
    """
    try:
        drug = _xml_element(elem)
    except ValueError as e:
        primary = elem.findtext(_NS + "drugbank-id[@primary='true']")
        e.add_note(f"in drug {primary}")
        raise
    primary = next(i["$"] for i in drug["drugbank-id"] if i.get("@primary") == "true")
    drug["drugbank_id"] = primary

    binds = []
    for container in _BIOMOL_TERMS:
        for entry in drug.pop(container, []):
            be_id = entry.get("id")
            _set_default_safe(
                biomolecules,
                be_id,
                {
                    "be_id": be_id,
                    "name": entry.get("name"),
                    "organism": entry.get("organism"),
                    "polypeptide": entry.get("polypeptide"),
                },
                primary,
            )
            binds.append(
                {
                    "be_id": be_id,
                    "kind": container[:-1],
                    **{f: entry.get(f) for f in _DRUG_BIOMOL_ATTR},
                }
            )
    drug["binds"] = binds

    pathway_ids = []
    for pathway in drug.pop("pathways", []):
        smpdb_id = pathway.get("smpdb-id")
        _set_default_safe(pathways, smpdb_id, pathway, primary)
        pathway_ids.append(smpdb_id)
    drug["pathway_ids"] = pathway_ids

    for x in drug.pop("drug-interactions", []):
        ddi.append(
            {
                "drugbank_id": primary,
                "interacts_with_id": x.get("drugbank-id"),
                "description": x.get("description"),
            }
        )
    return drug


def ensure_parquet_tables(
    raw_paths: dict[str, Path], table_dir: Path | str | None = None
) -> dict[str, Path]:
    """Stream-parse the gzipped XML into four Parquet tables. Untouched values;
    cleaning happens on scan. Skipped entirely if all four already exist.

    ``drug`` is the main table: one row per drug carrying every block that hangs off
    exactly one drug as a ``List(Struct)`` column, plus a foreign key to each
    other table (``binds[].be_id``, ``pathway_ids``, ``drugbank_id``). Only the
    two blocks measured to repeat verbatim across drugs are hoisted, into
    ``biomolecule`` and ``pathway``. ``drug_drug`` is a *directory* of part-files:
    2,855,848 rows do not fit in memory alongside the rest, and its flat schema is
    declared so the parts stay mutually readable. Field names are the XML tag
    names verbatim (``cas-number``, ``@type``); only synthesised keys are snake_case.
    """
    gz = Path(raw_paths["full_database"])
    table_dir = Path(table_dir or gz.parent.parent / "parquet")
    table_dir.mkdir(parents=True, exist_ok=True)
    dests = {t: table_dir / f"{t}.parquet" for t in ("drug", "biomolecule", "pathway")}
    dests["drug_drug"] = table_dir / "drug_drug"  # a directory of part-files
    if all(d.exists() for d in dests.values()):
        return dests

    # written here, moved into place only once all four are complete
    staging = table_dir / "_partial"
    shutil.rmtree(staging, ignore_errors=True)  # parts left by an interrupted run
    staged = {t: staging / d.name for t, d in dests.items()}
    staged["drug_drug"].mkdir(parents=True)

    drugs: list[dict] = []
    biomolecules: dict[str, dict] = {}
    pathways: dict[str, dict] = {}
    ddi: list[dict] = []
    frame: pl.DataFrame | None = None
    parts = 0

    def flush() -> None:
        """Trade the chunk's Python dicts for Arrow, which holds the same drugs in
        roughly a tenth the memory. ``diagonal_relaxed`` unions both new columns and
        new fields inside a ``List(Struct)``, so a chunk whose drugs happen to lack a
        block widens to null rather than raising."""
        nonlocal frame, parts
        if drugs:
            chunk = pl.DataFrame(drugs, infer_schema_length=None, strict=False)
            frame = (
                chunk
                if frame is None
                else pl.concat([frame, chunk], how="diagonal_relaxed")
            )
            drugs.clear()
        pl.DataFrame(ddi, schema=_DDI_SCHEMA).write_parquet(
            staged["drug_drug"] / f"part-{parts:04d}.parquet", compression="zstd"
        )
        ddi.clear()
        parts += 1

    i = 0
    with gzip.open(gz, "rb") as fh:
        for i, elem in enumerate(_iter_top_drugs(fh), 1):
            drugs.append(_split(elem, biomolecules, pathways, ddi))
            if i % _CHUNK == 0:
                flush()
                _lg.info("parsed %d drugs", i)
    if i == 0:
        raise ValueError(
            f"no <drug> elements in {gz}; not a DrugBank full-database XML"
        )
    flush()  # the tail, and the empty part-file that pins an empty table's schema
    _lg.info(
        f"parsed {i} drugs, {len(biomolecules)} bio-entities, {len(pathways)} pathways"
    )

    frame.write_parquet(staged["drug"], compression="zstd")
    for name, rows in (
        ("biomolecule", list(biomolecules.values())),
        ("pathway", list(pathways.values())),
    ):
        pl.DataFrame(rows, infer_schema_length=None, strict=False).write_parquet(
            staged[name], compression="zstd"
        )

    # drug.parquet last: until it lands, the all-exist check above re-parses
    dests["drug"].unlink(missing_ok=True)
    shutil.rmtree(dests["drug_drug"], ignore_errors=True)
    for t in ("drug_drug", "biomolecule", "pathway", "drug"):
        staged[t].replace(dests[t])
    staging.rmdir()
    return dests


# -- scan / view --------------------------------------------------------


def _clean(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Null empty strings in top-level String columns; nested fields keep ``""``.
    Leaf text is already ``None`` when empty, so this reaches attribute columns
    such as ``@type``. Applied on scan, never written to Parquet."""
    return lf.with_columns(pl.col(pl.String).replace({"": None}))


cleanly_scan_parquet_tables = partial(raw.scan_parquets, clean=_clean)


def build_views(parquet_paths: dict[str, Path]) -> dict[str, pl.LazyFrame]:
    """Flat views over the nested tables, one row per fact.

    - ``drugs``: identifiers, type, state, groups (``;``-joined) and description
    - ``targets``: every drug-target/enzyme/carrier/transporter edge with the drug
      name; one row per bound polypeptide (``uniprot_id``, ``gene_name``), and one
      row with nulls for an entry that binds no polypeptide
    - ``go_classifiers``: one row per GO term on a bound polypeptide, per drug;
      ``category`` is ``function``/``process``/``component`` unfiltered, ``kind``
      distinguishes target from enzyme/carrier/transporter
    - ``interactions``: every drug-drug interaction as the XML directs it, with
      both drugs' names; the partner name is recovered by join, so the 1 partner
      id of 4,567 that has no drug row gets a null name rather than being dropped

    ``interactions`` stays directed. The 1,427,655 symmetric pairs are exactly
    duplicated, but halving them to ``min(id) < max(id)`` would make a lookup by
    ``drugbank_id`` miss half its partners; a caller wanting the canonical half
    filters ``drugbank_id < interacts_with_id``.
    """
    lfs = cleanly_scan_parquet_tables(parquet_paths)
    drug, bio = lfs["drug"], lfs["biomolecule"]
    named = drug.select("drugbank_id", "name")
    poly = pl.col("polypeptide")

    # a drug binding nothing has an empty list, which explodes to a null row
    edges = (
        drug.select("drugbank_id", "name", "binds")
        .explode("binds")
        .filter(pl.col("binds").is_not_null())
        .unnest("binds")
    )
    entity = bio.select(
        "be_id", pl.col("name").alias("target_name"), "organism", "polypeptide"
    )
    go = (
        bio.select("be_id", "polypeptide")
        .explode("polypeptide")
        .select(
            "be_id",
            poly.struct.field("@id").alias("uniprot_id"),
            poly.struct.field("go-classifiers").alias("go"),
        )
        .explode("go")
        .filter(pl.col("go").is_not_null())
        .select(
            "be_id",
            "uniprot_id",
            pl.col("go").struct.field("category"),
            pl.col("go").struct.field("description"),
        )
    )
    return {
        "drugs": drug.select(
            "drugbank_id",
            "name",
            pl.col("@type").alias("type"),
            "state",
            pl.col("cas-number").alias("cas_number"),
            "unii",
            pl.col("groups").list.join(";").alias("groups"),
            "description",
        ),
        "targets": (
            edges.join(entity, on="be_id", how="left")
            .explode("polypeptide")
            .select(
                "drugbank_id",
                "kind",
                pl.col("be_id").alias("target_id"),
                "target_name",
                "organism",
                pl.col("actions").list.join(";").alias("actions"),
                pl.col("known-action").alias("known_action"),
                poly.struct.field("@id").alias("uniprot_id"),
                poly.struct.field("gene-name").alias("gene_name"),
                "name",
            )
        ),
        "go_classifiers": (
            edges.select("drugbank_id", "name", "kind", "be_id")
            .join(go, on="be_id", how="inner")
            .select(
                "drugbank_id",
                "kind",
                pl.col("be_id").alias("target_id"),
                "uniprot_id",
                "category",
                "description",
                "name",
            )
        ),
        "interactions": (
            lfs["drug_drug"]
            .join(named, on="drugbank_id", how="left")
            .join(
                named.select(
                    pl.col("drugbank_id").alias("interacts_with_id"),
                    pl.col("name").alias("interacts_with_name"),
                ),
                on="interacts_with_id",
                how="left",
            )
            .select(
                "drugbank_id",
                "interacts_with_id",
                "interacts_with_name",
                "description",
                "name",
            )
        ),
    }


def help() -> None:
    print(build_views.__doc__)
