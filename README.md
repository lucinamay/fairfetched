# fairfetched

data APIs for reproducible data fetching in cheminformatics in line with FAIR principles.
the tool is designed such that all raw data is downloaded and kept in one central place (i.e. CHEMBL 37 as the original .db files), and the user has a fast, lightweight, intuitive API to filter / query the data as required for any particular project.

# installation

you can install this package through
`uv add fairfetched` (recommended)

or if you do not use the uv package manager:
`pip install fairfetched`

# examples

you can download Chembl or Papyrus through:

```python
from fairfetched.get import Chembl, Papyrus

# Download the latest ChEMBL release (saves to ~/.data/chembl/<version>)
db = Chembl.from_latest()

# Raw source tables: db.tables.molecule_dictionary or db.lfs["molecule_dictionary"] (both LazyFrames)
# Joined domain views: db.view.bioactivity, db.view.compounds, db.view.proteins, db.view.components
# Parquet file paths: db.parquet_paths
# Raw download paths: db.raw_paths
```

### examples of how to use the LazyFrames:

#### checking columns and datatypes:

```python
from fairfetched.get import Chembl

db = Chembl.demo()
len(db.lfs["activities"].collect_schema())           # 29
db.lfs["activities"].collect_schema().names()[:4]    # ['molregno', 'activity_id', 'assay_id', 'doc_id']
```

#### accessing tables and joined views:

```python
db.tables.molecule_dictionary.filter(molregno=1280).collect()["pref_name"].to_list()  # ['ASPIRIN']
db.view.compounds.collect().shape                                                     # (5, 50)
```

#### selecting and joining tables:

```python
# select specific columns
db.lfs["activities"].filter(molregno=1280).select(
    "activity_id", "assay_id", "standard_value"
).head(3).collect()
# shape: (3, 3)
# ┌─────────────┬──────────┬────────────────┐
# │ activity_id ┆ assay_id ┆ standard_value │
# ├─────────────┼──────────┼────────────────┤
# │ ...         ┆ ...      ┆ ...            │
# └─────────────┴──────────┴────────────────┘

# join with compound structures
result = db.lfs["activities"].join(
    db.lfs["compound_structures"], on="molregno", how="left", validate="m:1"
).head().collect()
# result.shape → (5, 32)
```

#### convert to pandas (if needed):

Ideally at the end of all filtering, call `.collect().to_pandas()` (see polars documentation for more info):

```python
import pandas as pd

df = db.lfs["activities"].head().collect().to_pandas()
isinstance(df, pd.DataFrame)  # True
```

# data preparation: CAPRICHO, clustering, cross-validation folds

needs the `split` extra. `prep.capricho` ports CAPRICHO 1.0.4
(<https://github.com/David-Araripe/Capricho>): `flag` adds one boolean `drop_*` /
`note_*` column per quality criterion, `aggregate` collapses the measurements of one
compound on one target. both take any bioactivity LazyFrame, so filter and transform it
with polars first and chain them with `.pipe`:

```python
import polars as pl
import polars.selectors as cs

from fairfetched.get import Chembl
from fairfetched.prep import capricho
from fairfetched.prep.cluster_expr import cluster
from fairfetched.prep.mol_expr import MolExpr
from fairfetched.prep.split_expr import kfold

chembl = Chembl.from_version("36")
targets = ["CHEMBL226", "CHEMBL251", "CHEMBL256"]  # adenosine A1, A2A, A3

data = (
    chembl.view.bioactivity.filter(pl.col("target_chembl_id").is_in(targets))
    .pipe(
        capricho.flag,
        chembl.lfs["activities"],  # unfiltered: database-wide assay sizes
        calculate_pchembl=True,
        convert_units=True,
        mol_flags=True,  # standardises structures into `standard_smiles`
        parallel=True,
    )
    .remove(pl.any_horizontal(cs.starts_with("drop_")))
    .pipe(capricho.aggregate, parallel=True)  # one row per compound and target
    .with_columns(fp=MolExpr.from_smiles("smiles").to_morgan_fp(dedup=True))
    .collect()
)
data = data.with_columns(cluster=cluster("fp", method="bitbirch", threshold=0.3))
data = data.with_columns(
    fold=kfold(
        5,
        stratify=["target_chembl_id", "pchembl_value_mean"],
        groups="cluster",
        method="tricarico",
        time_limit_seconds=60,
    )
)

test = data.filter(fold=0)
valid = data.filter(fold=1)
train = data.remove(pl.col("fold").is_in([0, 1]))

data.select(
    "connectivity", "target_chembl_id", "pchembl_value_mean",
    "pchembl_value_counts", "cluster", "fold",
).head(4)
# shape: (4, 6)
# ┌────────────────┬──────────────────┬────────────────────┬──────────────────────┬─────────┬──────┐
# │ connectivity   ┆ target_chembl_id ┆ pchembl_value_mean ┆ pchembl_value_counts ┆ cluster ┆ fold │
# │ ---            ┆ ---              ┆ ---                ┆ ---                  ┆ ---     ┆ ---  │
# │ str            ┆ str              ┆ f64                ┆ u32                  ┆ i32     ┆ u8   │
# ╞════════════════╪══════════════════╪════════════════════╪══════════════════════╪═════════╪══════╡
# │ ZKUCFFYOQOJLGT ┆ CHEMBL226        ┆ 6.8                ┆ 1                    ┆ 7       ┆ 0    │
# │ ZKUCFFYOQOJLGT ┆ CHEMBL251        ┆ 6.95               ┆ 1                    ┆ 7       ┆ 0    │
# │ ZKUCFFYOQOJLGT ┆ CHEMBL256        ┆ 6.64               ┆ 1                    ┆ 7       ┆ 0    │
# │ SZTYGIVWLSLJRO ┆ CHEMBL226        ┆ 6.47               ┆ 1                    ┆ 7       ┆ 0    │
# └────────────────┴──────────────────┴────────────────────┴──────────────────────┴─────────┴──────┘

data.pivot("target_chembl_id", index="fold", values="fold",
           aggregate_function="len", sort_columns=True).sort("fold")
# shape: (5, 4)
# ┌──────┬───────────┬───────────┬───────────┐
# │ fold ┆ CHEMBL226 ┆ CHEMBL251 ┆ CHEMBL256 │
# │ ---  ┆ ---       ┆ ---       ┆ ---       │
# │ u8   ┆ u32       ┆ u32       ┆ u32       │
# ╞══════╪═══════════╪═══════════╪═══════════╡
# │ 0    ┆ 1030      ┆ 1315      ┆ 1032      │
# │ 1    ┆ 1036      ┆ 1349      ┆ 1015      │
# │ 2    ┆ 1042      ┆ 1371      ┆ 1047      │
# │ 3    ┆ 1016      ┆ 1350      ┆ 1009      │
# │ 4    ┆ 1025      ┆ 1324      ┆ 1037      │
# └──────┴───────────┴───────────┴───────────┘
```

on ChEMBL 36 this keeps 16,998 compound–target pairs (10,441 compounds) and runs in
about 2.5 minutes, most of it RDKit standardisation. `"tricarico"` balances the folds
on the target and on binned pChEMBL together, `"sklearn"` takes one `stratify` column.

one compound measured on several targets has one fingerprint per row. BitBIRCH depends
on row order and can put copies of one fingerprint in different clusters (here 610 of
10,441 compounds), so the compound lands in several folds. nanoom 0.1.0a20261002 and
earlier cluster every row; deduplicated clustering is on nanoom's `dedup-cluster`
branch, not yet released.

filtering before `flag` behaves as CAPRICHO's own `--target-ids` / `--assay-ids`
selection: duplicates across documents, unit annotation errors and assay overlap are
compared within the rows passed; assay size is counted over `chembl.lfs["activities"]`.
a filter on other columns (e.g. year) changes what those flags compare against.
`chembl.view.capricho(**params)` flags the whole database once and caches the result.

# roadmap

- [ ] papyrus database support
    - [x] papyrus latest version download
    - [x] simple nested filtering
    - [ ] efficient nested filtering
    - [ ] all-version support
    - [ ] built-in pivots
- [ ] chembl database support
    - [x] database to tables (parquet)
    - [ ] intuitive pre-merged flat files
    - [ ] database visualisation
    - [x] remove the need for storing uncompressed .db (`_tables.json` lists the finished tables)
- [ ] reproducion from downloaded raw file
- [ ] reproducible molecular (and protein?) standardisation
- [ ] automated time-url logging and manifest files
- [ ] well-organised logging
- [ ] dependency minimisation
- [ ] other database support
- [ ] preservation of api and parsing logic per major version
