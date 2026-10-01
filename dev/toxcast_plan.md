# ToxCast addition — plan for implementation

Handoff doc. Not committed. Written by a prior session that did the research;
this session's scratchpad (probe downloads, PDF) is gone — everything needed
is below or re-derivable from the URLs given.

## Decided (do not re-litigate)

- Source: **invitrodb v4.3** (EPA CompTox ToxCast), summary-zip tier — user
  picked this over "metadata only" and "full MySQL dump" when asked.
- Download location: `BASE_DIR` (`PYSTOW_HOME`, which happens to be a OneDrive
  folder on this machine) — user picked this over a local-only override.
  Consequence: any multi-GB intermediate (the unzipped CSV) must NOT touch
  `BASE_DIR`; only the final parquet does. Same reasoning as
  `fairfetched/utils/polars.py::decompress_tsvxz`, which defaults to system
  temp for exactly this reason (Papyrus' 46.9 GB TSV).
- Scope: v4.3 only. v4.2 is reachable via the same Clowder API pattern
  (space id `66858831e4b0a7c65d17841d`) if ever needed, left for later.
  v3.x uses a pre-tcplfit2 schema (different columns) — not a drop-in old
  version, don't claim `available_versions()` should include it.

## Source facts (verified against the live API, 2026-10-01)

EPA distributes ToxCast through Figshare → Clowder (figshare entries are
link-only pointers into a Clowder space). Useful API calls:

```
curl https://api.figshare.com/v2/articles/6062623   # -> clowder space id
curl https://clowder.edap-cluster.com/api/spaces/<space_id>/datasets
curl https://clowder.edap-cluster.com/api/datasets/<dataset_id>/files
```

Download pattern: `GET https://clowder.edap-cluster.com/files/<file_id>/blob`
— **`HEAD` returns 404 on this server; only `GET` works.** Range requests
*are* honored (`Range: bytes=...` -> `206 Partial Content`) but only cheaply
near the start of the object — a range near the end of the 7.5 GB zip timed
out after 60s, so the backing store appears to stream from byte 0 regardless
of the requested offset. **Conclusion: there is no cheap way to read the
zip's central directory remotely; the zip must be downloaded in full before
anything inside it can be listed or extracted.**

File IDs for v4.3 (`fairfetched.get.toxcast.source_urls("4.3")` should return
exactly these, keyed sensibly):

| name | file id | bytes | content type |
|---|---|---|---|
| assay_annotations (xlsx) | `68af6bd3e4b02565fc7c3aa8` | 1,088,201 | xlsx |
| assay_target_mappings (xlsx) | `68af6bd3e4b02565fc7c3aa0` | 97,652 | xlsx |
| cytotox (xlsx) | `68af6bd3e4b02565fc7c3aa4` | 943,302 | xlsx |
| analytical_qc (xlsx) | `68af6bd3e4b02565fc7c3ab8` | 2,915,329 | xlsx |
| summary zip | `68af6b70e4b02565fc7c3a98` | 7,502,075,199 | `multi/files-zipped` |
| mysql dump (NOT in scope) | `68c3365ce4b02565fc7cd3f3` | 17,651,807,605 | gzip |

Base URL: `https://clowder.edap-cluster.com/files/{id}/blob`.

These 4 xlsx files were downloaded and inspected for real in this session
(`uv run --with fastexcel python3 ...`, `pl.read_excel` / `fastexcel.read_excel`).
Confirmed shapes/columns/keys:

- **assay_annotations.xlsx** — sheets `['annotations_combined', 'assay',
  'assay_component', 'assay_component_endpoint']`. Use `annotations_combined`
  (the pre-joined flat one) — (1647, 63). Keys: `aid` > `acid` > `aeid`
  (assay > assay_component > assay_component_endpoint, one-to-many down the
  hierarchy per the README). `aeid` is what joins to bioactivity.
  Two columns fell back to string dtype on read (mixed content) — not
  investigated further, not load-bearing for the plan.
- **assay_target_mappings.xlsx** — single sheet, (2187, 8). **Long format**:
  one row per `(aeid, target_type)`, `target_type` ∈
  `{entrez_gene_id, aop, ke}`. Joining this straight into bioactivity
  multiplies rows — keep it as its own view keyed on `aeid`, the same way
  `drugbank.py`'s `go_classifiers` is kept separate from `targets`.
- **cytotox.xlsx** — single sheet, (10487, 14), one row per `chid`. Columns:
  `chid, casn, chnm, dsstox_substance_id, cytotox_median_raw, cytotox_mad,
  global_mad, cytotox_median_log, cytotox_median_um, cytotox_lower_bound_um,
  ntested, nhit, cytotox_lower_bound_log, created_date`. This is the natural
  **`compounds`** view (`BioactivityDBViews` requires one) — already deduped
  to one row per chemical, no need to derive dedup from the huge bioactivity
  CSV.
- **analytical_qc.xlsx** — single sheet, (42891, 15), keyed on
  `(dsstox_substance_id, spid)`. `spid` can be null (chemical-level QC rows
  exist without a sample).

## Not yet verified (next session must check before trusting)

- **The zip's actual member names and sizes.** The README text is internally
  inconsistent about them (`..._flags_invitrodbv4_3_AUG2024.csv` vs
  `..._flagsv4_3_AUG2024.Rdata` — note the missing underscore — and the
  README says AUG2024 while the standalone xlsx files are stamped AUG2025).
  **Do not hardcode a member filename.** After downloading the zip, match
  members by substring/suffix at runtime, e.g. find the one `.csv` member
  whose name contains `mc5-6` (or `mc5_6`) and `winning_model_fits`. Same for
  `mc4_all_model_fits` if it's included (see scope question below).
- **Whether R's `"NA"` literal needs `null_values=["NA"]` on `scan_csv`.**
  The xlsx reads above threw "could not determine dtype, falling back to
  string" on 2 of 63 columns, consistent with literal `"NA"` tokens sitting
  in numeric columns (R's default `write.csv`/`write.table` NA string). The
  real mc5-6 CSV has ~150 numeric model-fit columns per the README (hill_*,
  gnls_*, poly1_*, ac50, ...) — if even a few rows carry `"NA"`, every one of
  those columns silently becomes `String` instead of `Float64` unless
  `null_values=["NA"]` is passed when it's first scanned. This is parsing
  correctness (telling the reader the file's own null convention), not a
  cleaning transform, so it belongs in the raw-CSV scan call itself, not in
  a later `_clean()` step — but **confirm against the real downloaded file**
  before committing to it; this session never saw actual mc5-6 rows, only
  the README's column list.
- **Exact uncompressed size of the mc5-6 (and mc4, if included) CSV.** Not
  measurable without downloading the zip. Budget disk/time accordingly —
  the whole zip is 7.5 GB compressed; CSVs don't compress as well as the
  RData duplicates bundled alongside them, so expect the CSVs to be a
  meaningful fraction of that.

## Open questions — decide with the user, don't guess

1. **mc4 in scope for v1?** mc4 (`mc4_all_model_fits...csv`) is the
   per-model-per-series table (hill/gnls/poly1/poly2/pow/exp2-5 all fit, one
   row each) that mc5-6 already summarizes down to the single winning model.
   Recommend: **skip mc4 for v1** — mc5-6 alone gives `hitc`, `ac50`,
   `modl` (winning model name) and the winning model's own parameters,
   which is what a `bioactivity` view needs. mc4 is for someone who wants to
   second-guess the model selection. Revisit if asked.
2. **sc1_sc2 (single-concentration) and mc7 (administered-equivalent-dose)
   and the endocrine models (ar.er.*, ht.h295r, toxcast_*_pathway_model) —
   in scope for v1?** Recommend: no, same reasoning — mc5-6 is the
   multi-concentration core; these are secondary derived products. Flag as
   easy follow-up additions (`source_urls` already knows the zip is the
   container; extraction would be "find another member by name").
3. **Chemical structures.** None of the chosen files carry SMILES/InChIKey —
   only `dsstox_substance_id` (DTXSID), `casn` (CASRN), `chnm` (name). If the
   point of adding ToxCast is to join against ChEMBL/Papyrus/DrugBank
   compounds, that join needs a structure or at least a DTXSID/CASRN
   crosswalk — neither is in today's scope. The EPA "Chemical Information"
   DOI (`10.23645/epacomptox.6062524`) or the standalone
   `pubchem_invitrodb_v4_3_19AUG2026.zip` (193 MB, sits in the same
   Summary_Files Clowder dataset, id `6a85a260e4b0731a36bd04cc`) are the two
   candidates — not fetched or inspected this session. Ask the user whether
   this is wanted for v1 or deferred.
4. **AUG2025 standalone xlsx vs AUG2024 in-zip copies.** The 4 standalone
   files are a newer stamp than the README describes for the zip's contents.
   Assume the standalone ones are authoritative for assay/target/cytotox/QC
   metadata (this plan already uses them, not the zip's copies) — flag this
   assumption to the user once, don't silently resolve it.

## Shape of the code (names + one-line docstrings only — fill in once approved)

### `fairfetched/get/toxcast.py` (new, mirrors `sider.py`'s shape most closely:
flat files, standalone + one big archive, no XML/SQL parsing)

```python
"""ToxCast (EPA CompTox invitrodb v4.3) source: ... [fill after approval]"""

TOXCAST_DIR = BASE_DIR / "toxcast"
_CLOWDER_FILES: dict[str, dict[str, str]] = {
    "4.3": { "assay_annotations": "<id>", "assay_target_mappings": "<id>",
             "cytotox": "<id>", "analytical_qc": "<id>", "summary_zip": "<id>" }
}

def available_versions() -> tuple[str, ...]: ...   # ("4.3",)
def latest() -> str: ...
def source_urls(version: str) -> dict[str, str]:
    """file id -> https://clowder.edap-cluster.com/files/{id}/blob, per _CLOWDER_FILES"""

def ensure_raw_files(version, raw_dir=None, force=False) -> dict[str, Path]:
    """ensure_url() each of the 4 xlsx + the summary zip, under original-ish names"""

def _zip_member(zf: zipfile.ZipFile, *substrings: str) -> str:
    """the one name in zf.namelist() containing every substring; raises if 0 or >1 match
    -- this is how we avoid hardcoding the AUG2024/2025-stamped filename"""

def ensure_parquet_tables(raw_paths, table_dir=None) -> dict[str, Path]:
    """4 xlsx -> parquet directly (pl.read_excel, annotations_combined sheet
    for assay_annotations). mc5-6 member -> extract to system temp (NOT
    BASE_DIR -- it's OneDrive), scan_csv(null_values=["NA"], ...) ->
    sink_parquet, drop the temp extract. ensure_table_manifest() at the end,
    same as adrecs.py."""

def cleanly_scan_parquet(path_) -> pl.LazyFrame: ...
def cleanly_scan_parquet_tables(parquet_paths) -> dict[str, pl.LazyFrame]: ...

def build_views(parquet_paths) -> dict[str, pl.LazyFrame]:
    """
    - compounds: cytotox table verbatim (chid, casn, chnm, dsstox_substance_id,
      cytotox_*) -- already one row per chemical
    - bioactivity: mc5_mc6 joined to assay_annotations on aeid for
      assay_component_endpoint_name/desc, organism, tissue, intended_target_*
    - targets: assay_target_mappings verbatim, kept separate (long format,
      would multiply bioactivity rows if joined in)
    - assay_annotations: raw table passthrough, for anyone who wants the full
      63-column assay metadata
    """

def help() -> None: ...
```

### `fairfetched/get/dataset.py`

Add `_ToxcastView(_View)` with `.targets` property (mirrors `_DrugbankView` /
`_AdrecsView` shape — `bioactivity`/`compounds` already come from the base
`_View`). Add `Toxcast(_Base)` dataclass with `.demo()` skipped unless a demo
fixture is wanted (check `fairfetched/get/_demo.py` for the existing pattern
before adding one — don't build a new demo mechanism), `.from_version()`,
`.from_latest()`, `.tables` as `self.lfs` (flat, like Sider/Adrecs — no
nested-table wrapper needed, there's no biomolecule/pathway hoisting like
DrugBank).

### `fairfetched/get/__init__.py`

Add `Toxcast` to the import and `__all__`.

### `pyproject.toml`

`fastexcel` is currently only under the `adrecs` extra
(`adrecs = ["fastexcel>=0.12"]`). ToxCast's xlsx reads need it too — either
widen that extra's name/scope or add a second `toxcast = ["fastexcel>=0.12"]`
extra (lighter diff, consistent with how `rdkit`/`standardise` are already
split by concern rather than merged). Lean toward the second; ask if unsure
which the user prefers.

### `tests/test_toxcast.py`

Group into classes per [[group-tests-into-classes]] memory. Expectations
from independent oracles per [[test-expectations-from-independent-oracles]]
memory — the oracles available without a live download:
- file sizes from the table above (`Content-Length` / Clowder `size` field is
  an independent oracle for "did the right bytes arrive", check before
  computing a sha256)
- column lists transcribed from the README (`DB_release_README_SUMMARY.pdf`,
  re-fetchable from `https://clowder.edap-cluster.com/files/697b7530e4b0731a6170449e/blob`)
  and from this session's actual `pl.read_excel` output above — never record
  output of the code under test as the expectation.
- `cytotox` shape (10487, 14) and `assay_target_mappings` shape (2187, 8) are
  now known-good numbers for this exact release; a test can assert them
  without needing network if a tiny fixture slice is committed, or can be
  marked to skip without a live download (check how `test_drugbank.py`
  handles "licensed, can't fetch in CI" — same shape of problem).
Verify any mutation-sensitive logic (the `_zip_member` substring matcher
especially — it's the one piece of real logic in this module) actually goes
red when broken, per [[verify-tests-fail-by-mutation]] memory, before calling
the test suite done.

## Known gotchas to encode, not rediscover

- `ensure_url` (`fairfetched/utils/ensure.py`) writes straight to the final
  path with no `.tmp` staging — an interrupted 7.5 GB download leaves a
  truncated file that the next `path.exists()` check treats as complete.
  Either fix this once in `ensure_url` (root-cause, benefits every module)
  or pin the zip's size in a manifest and check it on every load (SIDER-style,
  `fairfetched/utils/manifest.py`). The Clowder `size` field (`7502075199`)
  is a cheap independent check even without a full sha256. Recommend doing
  the `ensure_url` fix — it's a root-cause fix per the ladder's "fix it once,
  where all callers route through" rule, and every other module downloading
  anything large benefits, not just ToxCast.
- Clowder file IDs are not immutable content pins: the README PDF (id
  `697b7530e4b0731a6170449e`) was re-uploaded 2026-01-29 and the pubchem zip
  was added 2026-08-19, both inside the *same* "v4.3" dataset that was
  published 2025-09-03. A SIDER-style sha256 manifest (not a drugbank-style
  "just trust the URL") is the right pin here, same reasoning as SIDER's
  versionless URLs even though these URLs do encode "v4.3" in the filename.
- `uv run --with fastexcel python3 -c ...` works as a one-off without
  touching `pyproject.toml`, useful for any further probing before the
  dependency question above is settled.
