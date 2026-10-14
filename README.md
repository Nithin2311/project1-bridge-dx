# project1-bridge-dx

CAI 5155 Network Analysis & ML with Graphs, Fall 2026. Project 1 (due **Fri Oct 23**), Option A:
**Bridge diagnoses in EHRShot** — which ICD-10 categories connect the big disease clusters, and do the
connectors change by sex and by age (<65 / ≥65). Project 2 reuses this graph for community-entry prediction.

Team: Nithin Palyam, Dylan Girrens.

**Code only.** No EHRShot data, caches, or figures with patient-level content are ever committed.
`.gitignore` enforces it; keep it that way.

## Layout

```
config/option_a.yaml      every locked parameter (era cut, thresholds, subgroups, seeds)
docs/OPTION-A-SPEC.md     the plan in one page + work split
docs/DATA.md              where the data lives and how the loader finds it
src/bridgedx/             library: data -> cohort -> edges -> graph -> communities/centrality/subgroups -> viz
scripts/NN_*.py           pipeline steps, run in order; each writes to outputs/
outputs/                  node list, edge list, metrics tables, figures (gitignored except .gitkeep)
report/                   LaTeX (later; Overleaf mirror)
tests/                    small unit tests on synthetic data
```

## Setup

```bash
uv sync
export EHRSHOT_DATA="/path/to/FinalProject/data"   # folder with the 13 CSVs
uv run python scripts/00_check_data.py
```

Default data path if `EHRSHOT_DATA` is unset: `../FinalProject/data` relative to this repo.

## Pipeline (status)

| Step | Script | Status |
|---|---|---|
| 0 | `00_check_data.py` — verify CSVs, row counts, DuckDB cache | works |
| 1 | `01_cohort.py` — ICD-10 era, dedupe (patient, category), drop post-death, ≥5 categories, nodes ≥1% | works (4,131 pts, 601 nodes) |
| 2 | `02_edges.py` — Fisher + Bonferroni + OR≥1.2 + ≥10 patients; phi, Jaccard | works (24,137 kept of 180,300 pairs) |
| 3 | `03_graph.py` — build graph, export node/edge CSVs | works (density 0.134, 1 isolate) |
| 4 | `04_communities.py` — Leiden (+Louvain), modularity vs 100 rewired nulls, naming | works (main 8 comms Q=0.295; noRZ 9 comms Q=0.367) |
| 5 | `05_centrality.py` — weighted betweenness w/ null, participation coef, within-module degree, prevalence residual | stub |
| 6 | `06_subgroups.py` — F/M, <65/≥65 graphs; ARI, top-20 overlap, edge-wise permutation | stub |
| 7 | `07_sensitivity.py` — drop R/Z chapters; ICD-9 bridged appendix | stub |
| 8 | `08_figures.py` — figures and tables for the report | stub |
| 9 | `09_p2_labels.py` — community-entry label counts for the Project 2 paragraph | stub |
