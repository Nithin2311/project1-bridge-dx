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
report/                   placeholder; the report is a separate repo (see below)
tests/                    small unit tests on synthetic data
```

## Report

The report is its own repo, `project1-bridge-dx-report` (IEEE conference layout, `main.tex`, built with `make`).
Figures and LaTeX tables are **not** edited there: `scripts/08_figures.py` writes `outputs/fig_*.pdf` and
`outputs/tables/*.tex`, and the report repo's `sync_from_pipeline.sh` copies them in. To refresh after a rerun:

```bash
uv run python scripts/08_figures.py            # figures + tables into outputs/ (needs steps 03-07; 11 for the stability figure)
cd ../project1-bridge-dx-report && ./sync_from_pipeline.sh && make
```

Everything synced is aggregate (diagnosis-category level); no patient-level content leaves `outputs/`.

## Setup

```bash
uv sync --extra dev                                # dev extra = pytest, ruff, jupyter
export EHRSHOT_DATA="/path/to/FinalProject/data"   # folder with the 13 CSVs
uv run python scripts/00_check_data.py
```

Default data path if `EHRSHOT_DATA` is unset: `../FinalProject/data` relative to this repo.

## Pipeline (status)

Every graph quantity is computed on the canonical EHRShot **train + val** patients (2,787); the canonical
**test** patients (1,344) never touch the graph, so Project 2 inherits it without leakage.

| Step | Script | Status |
|---|---|---|
| 0 | `00_check_data.py` — verify CSVs, row counts, DuckDB cache | works |
| 1 | `01_cohort.py` — ICD-10 era, dedupe, drop post-death, ≥5 categories; patient split (canonical file or own 70/15/15); nodes ≥1% of train+val | works (4,131 cohort; 617 nodes) |
| 2 | `02_edges.py` — Fisher + Bonferroni + OR≥1.2 + ≥10 patients on train+val; phi, Jaccard; all-patient edges as sensitivity | works (17,248 kept; all-patients 24,149) |
| 3 | `03_graph.py` — main graph (R/Z/O excluded), all-chapters and all-patients graphs; node/edge CSVs | works (main 496 nodes, 9,115 edges) |
| 4 | `04_communities.py` — Leiden (+Louvain), RB quality vs 100 rewired nulls, standard Q, 100-seed consensus, stable cores for 4 targets; sensitivity variants | works (25 comms, Q=0.308, z=146; cores 53/10/21/10) |
| 5 | `05_centrality.py` — betweenness w/ null, roles, prevalence residual, cluster-restricted betweenness w/ null and magnitude floor, precedence | works |
| 6 | `06_subgroups.py` — F/M, <65/≥65 graphs on train+val; ARI, overlap, edge-wise permutation, sex-specific flags | works |
| 7 | `07_sensitivity.py` — ICD-9 appendix (empirical SNOMED map), in-era temporal split, follow-up tertiles, summary | works |
| 8 | `08_figures.py` — figures (pdf+png) and LaTeX tables | works |
| 9 | `09_p2_labels.py` — Project 2 landmark label tables (6-month grid, 24-month washout, 3 target definitions, 2 horizons, censoring) | works |
| 10 | `10_p2_tables.py` — Project 2 node features (structural, consensus, bridge, inpatient share, HCUP chronic, demographics), top-k backbone, patient-index subgraph stats | works |
| 11 | `11_stability.py` — patient bootstrap: persistence of edges, of core members in the anchor community, and of bridge ranks | works |

## Run everything from the raw CSVs

```bash
uv sync --extra dev
for s in scripts/0[0-9]_*.py scripts/1[01]_*.py; do uv run python "$s"; done   # ~75 min; steps 4, 5, 11 are the slow ones
uv run pytest -q                                           # unit tests on synthetic data, no EHRShot needed
```

External reference files (public, not patient data): `../FinalProject/data/external/person_id_map.csv`
(EHRShot canonical split from the Redivis `EHRSHOT_ASSETS.zip`) and `CCIR_v2026-1.csv` (HCUP chronic
condition indicator). Without the split file, step 1 falls back to its own seeded 70/15/15 split.

Decisions made after the proposal (each is recorded with its reason in `config/option_a.yaml`; revised after
two independent code reviews on 2026-10-14/15 (`../project1-bridge-dx-review-clone/REVIEW*.md`) and the
Project 2 readiness research (`docs/P2-READINESS.md`)):
patient split fixed before graph building (graph on canonical train+val); main graph drops R, Z and O chapter
nodes; Leiden resolution 1.5 (1.0 and 2.0 as sensitivity); RB quality and standard Q both reported; four targets
(cardiometabolic, metastatic cancer, hematologic cancer, mental health) defined by 100-seed stable cores with
single-seed community and chapter block alongside; bridges = cluster-restricted betweenness beating a 100-rewire
null with a magnitude floor; 20,000 label permutations; sex-specific categories flagged; ICD-9 appendix bridged
with an empirical SNOMED map plus a hand map; Project 2 labels by landmark grid with 24-month washout.
