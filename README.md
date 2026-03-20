# fairfetched
data APIs for reproducible data fetching in cheminformatics in line with FAIR principles

# installation 
you can install this package through
`uv add fairfetched` (recommended)

or if you do not use the uv package manager:
`pip install fairfetched`


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
- [ ] reproducion from downloaded raw file 
- [ ] reproducible molecular (and protein?) standardisation
- [ ] automated time-url logging and manifest files
- [ ] well-organised logging
- [ ] dependency minimisation
- [ ] other database support
- [ ] preservation of api and parsing logic per major version
