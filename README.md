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
# splitting

needs the `split` extra: `uv add "fairfetched[split]"`.

splitting is done with polars expressions, so it composes with the rest of the query.
two functions produce a group id per row, and two assign rows to a partition:

| function | from | returns |
|---|---|---|
| `cluster_expr.cluster` | `nanoom` | cluster id per row (`pl.Int32`), from a descriptor array |
| `cluster_expr.groups` | `chemsplit` | group id per row (`pl.Int32`), e.g. a Murcko scaffold |
| `split_expr.kfold` | `nanoom` | fold number per row (`pl.UInt8`), stratified, groups kept whole |
| `split_expr.holdout` | `chemsplit` | `"train"` / `"valid"` / `"test"` per row, discarded rows null |

the two backends share the `groups` column, so any grouping feeds any assignment.

```python
import polars as pl

from fairfetched.prep.cluster_expr import groups
from fairfetched.prep.split_expr import holdout, kfold, splitters

df = pl.DataFrame(
    {
        "name": ["diazepam", "oxazepam", "lorazepam", "nitrazepam",
                 "ciprofloxacin", "norfloxacin", "ofloxacin", "pefloxacin"],
        "canonical_smiles": [
            "CN1c2ccc(Cl)cc2C(c2ccccc2)=NCC1=O",
            "OC1N=C(c2ccccc2)c2cc(Cl)ccc2NC1=O",
            "OC1N=C(c2ccccc2Cl)c2cc(Cl)ccc2NC1=O",
            "O=C1CN=C(c2ccccc2)c2cc([N+](=O)[O-])ccc2N1",
            "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",
            "CCn1cc(C(=O)O)c(=O)c2cc(F)c(N3CCNCC3)cc21",
            "CC1COc2c(N3CCN(C)CC3)c(F)cc3c(=O)c(C(=O)O)cn1c23",
            "CCn1cc(C(=O)O)c(=O)c2cc(F)c(N3CCN(C)CC3)cc21",
        ],
        "pchembl_value": [7.1, 6.8, 7.9, 6.2, 8.4, 7.7, 8.1, 7.5],
        "year": [1963, 1965, 1977, 1965, 1987, 1986, 1985, 1985],
    }
)

out = df.with_columns(
    scaffold=groups("canonical_smiles", splitter="murcko_scaffold")
).with_columns(
    fold=kfold(2, stratify="pchembl_value", groups="scaffold"),
    split=holdout(
        "canonical_smiles", splitter="temporal", dates="year",
        train_size=0.5, test_size=0.5,
    ),
)
out.select("name", "year", "scaffold", "fold", "split")
# shape: (8, 5)
# ┌───────────────┬──────┬──────────┬──────┬───────┐
# │ name          ┆ year ┆ scaffold ┆ fold ┆ split │
# │ ---           ┆ ---  ┆ ---      ┆ ---  ┆ ---   │
# │ str           ┆ i64  ┆ i32      ┆ u8   ┆ enum  │
# ╞═══════════════╪══════╪══════════╪══════╪═══════╡
# │ diazepam      ┆ 1963 ┆ 0        ┆ 1    ┆ train │
# │ oxazepam      ┆ 1965 ┆ 0        ┆ 1    ┆ train │
# │ lorazepam     ┆ 1977 ┆ 0        ┆ 1    ┆ train │
# │ nitrazepam    ┆ 1965 ┆ 0        ┆ 1    ┆ train │
# │ ciprofloxacin ┆ 1987 ┆ 1        ┆ 1    ┆ test  │
# │ norfloxacin   ┆ 1986 ┆ 2        ┆ 0    ┆ test  │
# │ ofloxacin     ┆ 1985 ┆ 3        ┆ 0    ┆ test  │
# │ pefloxacin    ┆ 1985 ┆ 2        ┆ 0    ┆ test  │
# └───────────────┴──────┴──────────┴──────┴───────┘
```

the four benzodiazepines share one Murcko scaffold; the quinolones do not, because
ciprofloxacin's cyclopropyl and ofloxacin's fused oxazine are rings, not side chains.

the two `with_columns` calls cannot be merged: polars evaluates the expressions of one
call against the input frame, so `fold` and `split` only see `scaffold` after it is
written. `with_columns_seq` does not change this.

`groups` and `holdout` run on the distinct structures and map the result back, so one
compound never straddles a partition, and `train_size` counts compounds, not rows.
pass already-standardised structures (see `prep.mol_expr`); a `pl.Array`
descriptor column or a `(compound, target)` pair of columns works in place of SMILES.

`splitters()` lists the 64 chemsplit splitters, with the family and the strictness of
each. ids where `group_forming` is true go to `groups`, the rest to `holdout`:

```python
splitters().filter(requires_dates=True).select("id", "family", "strictness")
# shape: (2, 3)
# ┌─────────────────┬──────────────┬────────────┐
# │ id              ┆ family       ┆ strictness │
# │ ---             ┆ ---          ┆ ---        │
# │ str             ┆ str          ┆ str        │
# ╞═════════════════╪══════════════╪════════════╡
# │ temporal        ┆ lineage      ┆ strict     │
# │ deposition_date ┆ biomolecular ┆ strict     │
# └─────────────────┴──────────────┴────────────┘
```

which splitter answers which question is covered by the chemsplit README's
"24 deployment scenarios": <https://github.com/OlivierBeq/chemsplit>.

# roadmap

- [ ] papyrus database support
    - [x] papyrus latest version download
    - [x] simple nested filtering
    - [ ] efficient nested filtering
    - [ ] all-version support
    - [x] built-in pivots
- [ ] chembl database support
    - [x] database to tables (parquet)
    - [x] intuitive pre-merged flat files
    - [ ] database visualisation
    - [x] remove the need for storing uncompressed .db (`_tables.json` lists the finished tables)
- [ ] reproducion from downloaded raw file
- [ ] reproducible molecular (and protein?) standardisation
- [ ] automated time-url logging and manifest files
- [ ] well-organised logging
- [ ] dependency minimisation
- [x] other database support
- [ ] preservation of api and parsing logic per major version
