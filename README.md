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
