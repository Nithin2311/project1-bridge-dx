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
uv sync --extra dev                                # dev extra = pytest, ruff, jupyter
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
| 3 | `03_graph.py` — build main (R/Z/O excluded) and full graphs, export node/edge CSVs | works (main 483 nodes, 13,088 edges, density 0.112) |
| 4 | `04_communities.py` — Leiden (+Louvain), RB quality vs 100 rewired nulls, standard Q, naming | works (main: 20 comms at res 1.5, Q=0.278, z=152; full + res 1.0/2.0 sensitivity) |
| 5 | `05_centrality.py` — weighted betweenness w/ null, roles, prevalence residual, cluster-restricted betweenness w/ null, precedence | works (centrality.csv, bridges.csv, cluster_precedence.csv) |
| 6 | `06_subgroups.py` — F/M, <65/≥65 graphs; ARI (all + non-isolates), top-20 overlap, edge-wise permutation, sex-specific flags | works (5,000 perms; 74 sex-, 122 age-specific edges at FDR 0.05) |
| 7 | `07_sensitivity.py` — ICD-9 appendix (empirical SNOMED map), in-era 2019 split, follow-up tertiles, summary with full graph + resolution sweep | works (sensitivity_summary.csv) |
| 8 | `08_figures.py` — figures (pdf+png) and LaTeX tables | works (8 figures, 8 tables in outputs/tables) |
| 9 | `09_p2_labels.py` — community-entry label counts for the Project 2 paragraph (prior history incl. pre-2015 excluded) | works (p2_label_counts.csv; 2 index rules x 2 horizons) |

## Run everything from the raw CSVs

```bash
uv sync --extra dev
for s in scripts/0[0-9]_*.py; do uv run python "$s"; done   # ~45 min; steps 4 and 5 (null models) are the slow ones
uv run pytest -q                                           # unit tests on synthetic data, no EHRShot needed
```

Decisions made after the proposal (each is recorded with its reason in `config/option_a.yaml`; several were
revised after an independent code review on 2026-10-14, see `../project1-bridge-dx-review-clone/REVIEW.md`):
main graph drops R, Z and O chapter nodes (full graph kept as sensitivity); Leiden resolution 1.5 (1.0 and 2.0
as sensitivity); quality reported both as RB quality at the working resolution and standard Q; 5,000 label
permutations; age at index = age at first ICD-10-era diagnosis; target clusters defined by anchor codes;
bridges = cluster-restricted betweenness that beats a 100-rewire null; sex-specific categories flagged in the
sex test; ICD-9 appendix bridged with an empirical SNOMED map plus a hand map for the top unbridged codes.
