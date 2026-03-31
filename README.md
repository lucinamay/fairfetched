# fairfetched
data APIs for reproducible data fetching in cheminformatics in line with FAIR principles

# installation 
you can install this package through
`uv add fairfetched` (recommended)

or if you do not use the uv package manager:
`pip install fairfetched`


# examples
you can download Chembl or Papyrus through:
```python
from fairfetched.get import Chembl, Papyrus
mychembl = Chembl.from_latest() # this downloads Chembl raw files + extracts parquet files to wherever you
                                # have set the environment variable FAIRFETCHED_HOME, PYSTOW_HOME,
                                # or <HOME>/.data if not in environment variables.
                                # from there, fairfetched saves it to a folder chembl/<version>

mychembl.lfs                  # a dictionary of all chembl files in polars LazyFrame format, scanned directly from the extracted .parquet files


mychembl.consolidated_paths   # the paths to the parquet-converted tabular data files in the Chembl .db file

mychembl.raw_paths            # the paths to the raw chembl file as downloaded from Chembl. currently does include an uncompressed .db file

mychembl.compounds            # NOT YET IMPLEMENTED !! convenience alias for mychembl.compose()["compounds"], which uses mychembl.lfs LazyFrame joins to obtain an intuitive join of the data.
                              # from there, you can 
```

### examples of how to use the LazyFrames:

#### checking which columns+datatypes are in the file, so that you can choose to join them:
```python
>>> mychembl.lfs["activities"].collect_schema()
Schema({'activity_id': Int64, 'assay_id': Int64, 'doc_id': Int64, 'record_id': Int64, 'molregno': Int64, 'standard_relation': String, 'standard_value': Float64, 'standard_units': String, 'standard_flag': Int64, 'standard_type': String, 'activity_comment': String, 'data_validity_comment': String, 'potential_duplicate': Int64, 'pchembl_value': Float64, 'bao_endpoint': String, 'uo_units': String, 'qudt_units': String, 'toid': Int64, 'upper_value': Float64, 'standard_upper_value': Null, 'src_id': Int64, 'type': String, 'relation': String, 'value': Float64, 'units': String, 'text_value': String, 'standard_text_value': String, 'action_type': String})
```
#### selecting all entries based on doc_id:
```python
>>> mychembl.lfs["activities"].filter(doc_id=89530).drop_nulls("units").collect()
shape: (107, 28)
┌─────────────┬──────────┬────────┬───────────┬───┬───────┬────────────┬─────────────────────┬─────────────┐
│ activity_id ┆ assay_id ┆ doc_id ┆ record_id ┆ … ┆ units ┆ text_value ┆ standard_text_value ┆ action_type │
│ ---         ┆ ---      ┆ ---    ┆ ---       ┆   ┆ ---   ┆ ---        ┆ ---                 ┆ ---         │
│ i64         ┆ i64      ┆ i64    ┆ i64       ┆   ┆ str   ┆ str        ┆ str                 ┆ str         │
╞═════════════╪══════════╪════════╪═══════════╪═══╪═══════╪════════════╪═════════════════════╪═════════════╡
│ 15120638    ┆ 1431503  ┆ 89530  ┆ 2256150   ┆ … ┆ uM    ┆ null       ┆ null                ┆ null        │
│ 15120639    ┆ 1431503  ┆ 89530  ┆ 2256151   ┆ … ┆ uM    ┆ null       ┆ null                ┆ null        │
│ 15120640    ┆ 1431503  ┆ 89530  ┆ 2256152   ┆ … ┆ uM    ┆ null       ┆ null                ┆ null        │
│ 15120641    ┆ 1431503  ┆ 89530  ┆ 2256153   ┆ … ┆ uM    ┆ null       ┆ null                ┆ null        │
│ 15120642    ┆ 1431503  ┆ 89530  ┆ 2256154   ┆ … ┆ uM    ┆ null       ┆ null                ┆ null        │
│ …           ┆ …        ┆ …      ┆ …         ┆ … ┆ …     ┆ …          ┆ …                   ┆ …           │
│ 15125200    ┆ 1431507  ┆ 89530  ┆ 2256167   ┆ … ┆ uM    ┆ null       ┆ null                ┆ null        │
│ 15125201    ┆ 1431507  ┆ 89530  ┆ 2256168   ┆ … ┆ uM    ┆ null       ┆ null                ┆ null        │
│ 15125202    ┆ 1431507  ┆ 89530  ┆ 2256169   ┆ … ┆ uM    ┆ null       ┆ null                ┆ null        │
│ 15125203    ┆ 1431507  ┆ 89530  ┆ 2256170   ┆ … ┆ uM    ┆ null       ┆ null                ┆ null        │
│ 15125204    ┆ 1431507  ┆ 89530  ┆ 2256171   ┆ … ┆ uM    ┆ null       ┆ null                ┆ null        │
└─────────────┴──────────┴────────┴───────────┴───┴───────┴────────────┴─────────────────────┴─────────────┘
```

#### adding compound structure info to the activities on molregno
```python
>>> mychembl.lfs["activities"].join(mychembl.lfs["compound_structures"],on="molregno",how="left",validate="m:1").head().collect()
shape: (5, 32)
┌─────────────┬──────────┬────────┬───────────┬───┬────────────────────────┬─────────────────────────────────┬─────────────────────────────┬─────────────────────────────────┐
│ activity_id ┆ assay_id ┆ doc_id ┆ record_id ┆ … ┆ molfile                ┆ standard_inchi                  ┆ standard_inchi_key          ┆ canonical_smiles                │
│ ---         ┆ ---      ┆ ---    ┆ ---       ┆   ┆ ---                    ┆ ---                             ┆ ---                         ┆ ---                             │
│ i64         ┆ i64      ┆ i64    ┆ i64       ┆   ┆ str                    ┆ str                             ┆ str                         ┆ str                             │
╞═════════════╪══════════╪════════╪═══════════╪═══╪════════════════════════╪═════════════════════════════════╪═════════════════════════════╪═════════════════════════════════╡
│ 31863       ┆ 54505    ┆ 6424   ┆ 206172    ┆ … ┆                        ┆ InChI=1S/C20H12N2O2/c1-2-7-13(… ┆ BEBACPIIZGRKGG-UHFFFAOYSA-N ┆ c1ccc(-c2nc3c(-c4nc5ccccc5o4)c… │
│             ┆          ┆        ┆           ┆   ┆      RDKit          2D ┆                                 ┆                             ┆                                 │
│             ┆          ┆        ┆           ┆   ┆                        ┆                                 ┆                             ┆                                 │
│             ┆          ┆        ┆           ┆   ┆  24 2…                 ┆                                 ┆                             ┆                                 │
│ 31864       ┆ 83907    ┆ 6432   ┆ 208970    ┆ … ┆                        ┆ InChI=1S/C23H14N2O5/c1-12-5-8-… ┆ SUKVIELCKKEBOJ-UHFFFAOYSA-N ┆ Cc1ccc2oc(-c3cccc(N4C(=O)c5ccc… │
│             ┆          ┆        ┆           ┆   ┆      RDKit          2D ┆                                 ┆                             ┆                                 │
│             ┆          ┆        ┆           ┆   ┆                        ┆                                 ┆                             ┆                                 │
│             ┆          ┆        ┆           ┆   ┆  30 3…                 ┆                                 ┆                             ┆                                 │
│ 31865       ┆ 88152    ┆ 6432   ┆ 208970    ┆ … ┆                        ┆ InChI=1S/C23H14N2O5/c1-12-5-8-… ┆ SUKVIELCKKEBOJ-UHFFFAOYSA-N ┆ Cc1ccc2oc(-c3cccc(N4C(=O)c5ccc… │
│             ┆          ┆        ┆           ┆   ┆      RDKit          2D ┆                                 ┆                             ┆                                 │
│             ┆          ┆        ┆           ┆   ┆                        ┆                                 ┆                             ┆                                 │
│             ┆          ┆        ┆           ┆   ┆  30 3…                 ┆                                 ┆                             ┆                                 │
│ 31866       ┆ 83907    ┆ 6432   ┆ 208987    ┆ … ┆                        ┆ InChI=1S/C30H20N2O7/c1-37-24-6… ┆ ZFJHZUAZBGPPQK-UHFFFAOYSA-N ┆ COc1ccccc1-c1ccc2oc(-c3ccc(OC)… │
│             ┆          ┆        ┆           ┆   ┆      RDKit          2D ┆                                 ┆                             ┆                                 │
│             ┆          ┆        ┆           ┆   ┆                        ┆                                 ┆                             ┆                                 │
│             ┆          ┆        ┆           ┆   ┆  39 4…                 ┆                                 ┆                             ┆                                 │
│ 31867       ┆ 88153    ┆ 6432   ┆ 208987    ┆ … ┆                        ┆ InChI=1S/C30H20N2O7/c1-37-24-6… ┆ ZFJHZUAZBGPPQK-UHFFFAOYSA-N ┆ COc1ccccc1-c1ccc2oc(-c3ccc(OC)… │
│             ┆          ┆        ┆           ┆   ┆      RDKit          2D ┆                                 ┆                             ┆                                 │
│             ┆          ┆        ┆           ┆   ┆                        ┆                                 ┆                             ┆                                 │
│             ┆          ┆        ┆           ┆   ┆  39 4…                 ┆                                 ┆                             ┆                                 │
└─────────────┴──────────┴────────┴───────────┴───┴────────────────────────┴─────────────────────────────────┴─────────────────────────────┴─────────────────────────────────┘
```

#### move it to pandas for direct drop-in use (if you really want pandas...)
ideally as far down the line after you complete all filtering, you call `.collect().to_pandas()` (see polars documentation for more info)
```
mychembl.lfs["activities"].collect().to_pandas()
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
  - [ ] remove the need for storing uncompressed .db
- [ ] reproducion from downloaded raw file 
- [ ] reproducible molecular (and protein?) standardisation
- [ ] automated time-url logging and manifest files
- [ ] well-organised logging
- [ ] dependency minimisation
- [ ] other database support
- [ ] preservation of api and parsing logic per major version
