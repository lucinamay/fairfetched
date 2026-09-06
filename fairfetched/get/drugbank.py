"""DrugBank source: drug identifiers, targets/enzymes/carriers/transporters,
drug-drug interactions, ATC codes, categories and calculated properties, parsed
from the full-database XML.

DrugBank is licensed. fairfetched ships no data and cannot download it. Register
your own licensed copy once -- the release ``.zip`` or an extracted
``full database.xml``::

    from fairfetched.get import Drugbank
    db = Drugbank.from_xml("~/Downloads/drugbank_all_full_database.xml.zip", version="5.1.13")
    db.view.targets.sink_parquet("drugbank_targets.parquet")
    db.tables["drug_interactions"].collect_schema()

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
from pathlib import Path

import polars as pl

from fairfetched.utils import BASE_DIR, manifest

_lg = lg.getLogger(__name__)

DRUGBANK_DIR = BASE_DIR / "drugbank"

_MANIFEST_NAME = "_manifest.json"
_DRIFT_MSG = (
    "the registered DrugBank cache changed on disk. Re-register from your "
    "licensed source with Drugbank.from_xml(path, version=..., force=True)."
)

# ponytail: one namespace constant, DrugBank has used it since 4.x. A release
# that changes it parses to zero drugs and ``ensure_parquet_tables`` raises.
_NS = {"db": "http://www.drugbank.ca"}
_DRUG_TAG = "{http://www.drugbank.ca}drug"

# # @TODO: fill
DRUGBANK_VERSIONS: tuple[str, ...] = ("5.1.10", "5.1.11", "5.1.12", "5.1.13")

# licence required; not auto-fetchable
_RELEASES_URL = "https://go.drugbank.com/releases"

_COLUMNS: dict[str, list[str]] = {
    "drugs": [
        "drugbank_id",
        "name",
        "type",
        "state",
        "cas_number",
        "unii",
        "groups",
        "description",
    ],
    "synonyms": ["drugbank_id", "synonym"],
    "external_identifiers": ["drugbank_id", "resource", "identifier"],
    "properties": ["drugbank_id", "source_kind", "kind", "value", "source"],
    "atc_codes": ["drugbank_id", "atc_code"],
    "categories": ["drugbank_id", "category", "mesh_id"],
    "targets": [
        "drugbank_id",
        "kind",
        "target_id",
        "target_name",
        "organism",
        "actions",
        "known_action",
        "uniprot_id",
        "gene_name",
    ],
    "drug_interactions": [
        "drugbank_id",
        "interacts_with_id",
        "interacts_with_name",
        "description",
    ],
}
_SCHEMA = {t: {c: pl.String for c in cols} for t, cols in _COLUMNS.items()}


def available_versions() -> tuple[str, ...]:
    return DRUGBANK_VERSIONS


def latest() -> str:
    return DRUGBANK_VERSIONS[-1]


def source_urls(version: str) -> dict[str, str]:
    return {"full_database": _RELEASES_URL}


# -- register (bring your own licensed file) -------------------------------


def _uncompressed_stream(path: Path):
    """Binary stream of the XML, whether given the release ``.zip``, a ``.gz`` or
    a plain ``.xml``."""
    if path.suffix == ".zip":
        z = zipfile.ZipFile(path)
        inner = next(n for n in z.namelist() if n.endswith(".xml"))
        return z.open(inner)
    if path.suffix == ".gz":
        return gzip.open(path, "rb")
    return open(path, "rb")


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
    gz = raw_dir / "full_database.xml.gz"
    if gz.exists():
        paths = {"full_database": gz}
        manifest.verify(paths, raw_dir / _MANIFEST_NAME, _DRIFT_MSG)
        return paths
    raise FileNotFoundError(
        f"DrugBank {version} is not registered. Download the release from "
        f"{_RELEASES_URL} under your licence and register it with "
        f"Drugbank.from_xml(path, version='{version}')."
    )


# -- parse XML -> Parquet -------------------------------------------------


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


def _flatten(drug, rows: dict[str, list]) -> None:
    ids = drug.findall("db:drugbank-id", _NS)
    primary = next((e.text for e in ids if e.get("primary") == "true"), None)
    if primary is None:
        primary = ids[0].text if ids else None
    if primary is None:
        return

    groups = [g.text for g in drug.findall("db:groups/db:group", _NS) if g.text]
    rows["drugs"].append(
        {
            "drugbank_id": primary,
            "name": drug.findtext("db:name", namespaces=_NS),
            "type": drug.get("type"),
            "state": drug.findtext("db:state", namespaces=_NS),
            "cas_number": drug.findtext("db:cas-number", namespaces=_NS),
            "unii": drug.findtext("db:unii", namespaces=_NS),
            "groups": ";".join(groups) or None,
            "description": drug.findtext("db:description", namespaces=_NS),
        }
    )

    for syn in drug.findall("db:synonyms/db:synonym", _NS):
        if syn.text:
            rows["synonyms"].append({"drugbank_id": primary, "synonym": syn.text})

    for xid in drug.findall("db:external-identifiers/db:external-identifier", _NS):
        rows["external_identifiers"].append(
            {
                "drugbank_id": primary,
                "resource": xid.findtext("db:resource", namespaces=_NS),
                "identifier": xid.findtext("db:identifier", namespaces=_NS),
            }
        )

    for tag, label in (
        ("calculated-properties", "calculated"),
        ("experimental-properties", "experimental"),
    ):
        for p in drug.findall(f"db:{tag}/db:property", _NS):
            rows["properties"].append(
                {
                    "drugbank_id": primary,
                    "source_kind": label,
                    "kind": p.findtext("db:kind", namespaces=_NS),
                    "value": p.findtext("db:value", namespaces=_NS),
                    "source": p.findtext("db:source", namespaces=_NS),
                }
            )

    for code in drug.findall("db:atc-codes/db:atc-code", _NS):
        rows["atc_codes"].append({"drugbank_id": primary, "atc_code": code.get("code")})

    for cat in drug.findall("db:categories/db:category", _NS):
        rows["categories"].append(
            {
                "drugbank_id": primary,
                "category": cat.findtext("db:category", namespaces=_NS),
                "mesh_id": cat.findtext("db:mesh-id", namespaces=_NS),
            }
        )

    for kind in ("targets", "enzymes", "carriers", "transporters"):
        for tgt in drug.findall(f"db:{kind}/db:{kind[:-1]}", _NS):
            actions = ";".join(
                a.text for a in tgt.findall("db:actions/db:action", _NS) if a.text
            )
            common = {
                "drugbank_id": primary,
                "kind": kind[:-1],
                "target_id": tgt.findtext("db:id", namespaces=_NS),
                "target_name": tgt.findtext("db:name", namespaces=_NS),
                "organism": tgt.findtext("db:organism", namespaces=_NS),
                "actions": actions or None,
                "known_action": tgt.findtext("db:known-action", namespaces=_NS),
            }
            polys = tgt.findall("db:polypeptide", _NS)
            if not polys:
                rows["targets"].append(
                    {**common, "uniprot_id": None, "gene_name": None}
                )
            for p in polys:
                rows["targets"].append(
                    {
                        **common,
                        "uniprot_id": p.get("id"),
                        "gene_name": p.findtext("db:gene-name", namespaces=_NS),
                    }
                )

    for x in drug.findall("db:drug-interactions/db:drug-interaction", _NS):
        rows["drug_interactions"].append(
            {
                "drugbank_id": primary,
                "interacts_with_id": x.findtext("db:drugbank-id", namespaces=_NS),
                "interacts_with_name": x.findtext("db:name", namespaces=_NS),
                "description": x.findtext("db:description", namespaces=_NS),
            }
        )


def ensure_parquet_tables(
    raw_paths: dict[str, Path], table_dir: Path | str | None = None
) -> dict[str, Path]:
    """Stream-parse the gzipped XML into one Parquet table per entity. All-String,
    untouched values; cleaning happens on scan. Skipped entirely if every table
    already exists."""
    gz = Path(raw_paths["full_database"])
    table_dir = Path(table_dir or gz.parent.parent / "parquet")
    table_dir.mkdir(parents=True, exist_ok=True)
    dests = {t: table_dir / f"{t}.parquet" for t in _COLUMNS}
    if all(d.exists() for d in dests.values()):
        return dests

    rows: dict[str, list] = {t: [] for t in _COLUMNS}
    i = 0
    with gzip.open(gz, "rb") as fh:
        for i, drug in enumerate(_iter_top_drugs(fh), 1):
            _flatten(drug, rows)
            if i % 2000 == 0:
                _lg.info("parsed %d drugs", i)
    if i == 0:
        raise ValueError(
            f"no <drug> elements in {gz}; not a DrugBank full-database XML"
        )
    _lg.info(f"parsed {i} drugs total; writing {len(dests)} tables")

    for t, dest in dests.items():
        pl.DataFrame(rows[t], schema=_SCHEMA[t]).write_parquet(dest)
    return dests


# -- scan / view --------------------------------------------------------


def _clean(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Null empty strings. Applied on scan, never written to Parquet."""
    return lf.with_columns(pl.col(pl.String).replace({"": None}))


def cleanly_scan_parquet(path_: Path | str) -> pl.LazyFrame:
    return _clean(pl.scan_parquet(path_))


def cleanly_scan_parquet_tables(
    parquet_paths: dict[str, Path],
) -> dict[str, pl.LazyFrame]:
    return {name: cleanly_scan_parquet(p) for name, p in parquet_paths.items()}


def build_views(parquet_paths: dict[str, Path]) -> dict[str, pl.LazyFrame]:
    """Joined views over the cleaned tables.

    - ``drugs``: the ``drugs`` table verbatim (identifiers, groups, description)
    - ``targets``: every drug-target/enzyme/carrier/transporter row with the drug
      name; one row per bound polypeptide (``uniprot_id``, ``gene_name``)
    - ``interactions``: every drug-drug interaction with the subject drug's name
    """
    lfs = cleanly_scan_parquet_tables(parquet_paths)
    named = lfs["drugs"].select("drugbank_id", "name")
    return {
        "drugs": lfs["drugs"],
        "targets": lfs["targets"].join(named, on="drugbank_id", how="left"),
        "interactions": lfs["drug_interactions"].join(
            named, on="drugbank_id", how="left"
        ),
    }


def help() -> None:
    print(build_views.__doc__)
