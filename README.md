# Data-integrity audit of a public census-income benchmark

Code and data for *"Data-Integrity Defects and Redistribution Divergence in
a Public Census-Income Benchmark: Prevalence, Provenance, and Severity."*

An eight-check data-integrity audit applied to the canonical UCI Adult /
Census Income archive and two independently obtained, actively used
redistributions, followed by a matched modelling study measuring whether
each divergence found actually changes model performance.

## Repository layout

```
src/
  auditlib.py               shared, domain-agnostic audit functions
  s10_census_extension.py   Stage 10: prevalence and provenance audit
  s11_census_severity.py    Stage 11: severity / modelling measurement
data/
  raw/adult_census_income/       canonical UCI files (adult.data, adult.test, adult.names)
  mirrors/adult_census_income/   two OpenML redistributions (did=179, did=43898)
  manifest.json                   SHA-256 / DOI / retrieval date for every file above
results/
  s10_census_extension.json  prevalence and provenance results
  s11_census_severity.json   severity measurement results
```

## Reproducing the results

```
pip install -r requirements.txt
python src/s10_census_extension.py
python src/s11_census_severity.py
```

Both stages are self-contained, read only from `data/`, and write their
results to `results/` as machine-readable JSON. Every random operation is
seeded (`random_state=42` throughout).

## Data sources

- Canonical: UCI Adult / Census Income (Becker and Kohavi, 1996), DOI
  10.24432/C5XW20, https://archive.ics.uci.edu/dataset/2/adult
- Mirror 1: OpenML dataset id 179 ("adult", version 1), the version
  returned by `sklearn.datasets.fetch_openml('adult')` at default settings
- Mirror 2: OpenML dataset id 43898 ("adult", version 3)

Full checksums, licenses, and retrieval dates for all files are in
`data/manifest.json`.
