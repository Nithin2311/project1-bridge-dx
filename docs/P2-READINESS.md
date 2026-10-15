# Project 2 readiness: what Project 1 can still do, what helps, what does not

Compiled 2026-10-15 from two literature briefs (docs/research/, 90 sources) and empirical probes on our data
(notebooks/p2_probe_fast.py, notebooks/p2_probe_bootstrap.py; outputs in outputs/p2_probes/). Project 1 is
classical network analysis only; every addition below is non-ML.

## Executive summary

1. The graph itself is solid. Under 30 patient bootstraps the median edge persistence is 0.97, 62% of edges
   recur in at least 90% of resamples, and the cardiometabolic module keeps 53 of 60 members at 80% or
   better. The cardiometabolic to cancer bridges N17, J96, E87 are in the top 10 in every resample, F05 in 97%.
2. Three things decide whether Project 2 is clean or caveat-ridden, and all three can be fixed in Project 1
   now: (a) fix a patient split before any graph computation and rebuild edges and communities on training
   patients only; (b) define targets by a frozen stable core, not by one Leiden seed, and treat "cancer" as
   two targets or a chapter because it never forms one block; (c) use a landmark label design with a
   24-month washout, which gives 202 / 344 / 332 positives instead of 167 / 220 / 225 and removes the
   prevalent-case problem by construction.
3. Four smaller additions make the Project 2 feature set and ablations ready-made: an inpatient-share flag
   per node (bridges 43% vs 12% for other nodes), a top-10 backbone (3,296 edges; the disparity filter is
   unusable here, 49 edges), a per-node feature table with chapter and demographic descriptors, and
   precomputed patient-subgraph sizes (median 7 nodes, 13 edges; 15% have no edges).
4. Things that do not help and should not be pursued: disparity-filter backbones, subgroup graphs as model
   inputs, the ICD-9-era graph as an input, Guimera-Amaral roles as features, chapter-only labels on their
   own, lower prevalence thresholds, validating a target by its outcome association, and any single calendar
   index date.
5. Recommended new Project 1 steps: 10 (patient split and train-only graph), 11 (target definitions and
   landmark labels), 12 (Project 2 node and patient tables). About one working day. Everything else belongs
   in Project 2.

## A. Helps: do in Project 1 now

| # | Addition | Evidence from our data | Literature | Effort |
|---|---|---|---|---|
| 1 | Fixed patient split before graph building; edges, phi, Leiden, bridges recomputed on training patients only; report train-only vs full (edge Jaccard, ARI, bridge overlap) as a P1 sensitivity | Not yet run. Bootstrap says the structure is stable, so a 70% train graph should keep most edges (expect Jaccard about 0.6 to 0.7 by analogy with the 2019+ window, 0.58) | Kapoor & Narayanan 2023 (L1.2, L1.3, L3.1); Davis 2023; Chet AAAI 2022 builds the graph on training patients' pre-label visits; EHRShot ships a canonical 2,295 / 2,232 / 2,212 split (not in our extract; benchmark repo) | small: one script, 10 min run |
| 2 | Three target definitions shipped side by side: chapter block, single-seed community, stable core (co-assigned with the anchor in at least 90% of 100 seeds and present in at least 80% of bootstraps) | Seed consensus at 1.5: cardiometabolic core 57 of 60; mental health core 12 (F31 F32 F33 F39 F40 F41 F43 F45 F51 G20 G25 G47); cancer has no core when the two anchors are averaged: metastatic core 15 (C77 C78 C79 C80 C76 C49 C20 C18 C34 C33 K11 D63 I80 I26 I82), haematology core 41, breast cancer sits with gynaecology, prostate with urology. Bootstrap: cardiometabolic 53 of 60 at 80%, mental 9 of 15, cancer 33 of 73 with 18 below 50% | Lancichinetti & Fortunato 2012; Riolo & Newman 2020 (building blocks); Peixoto 2021 (report modes, not one estimate); Griffith 2024 and Ferris 2025 (comorbidity clusters are unstable); Ba Dhafari 2026 (do not validate by outcome) | small: probe code exists |
| 3 | Landmark label design: one row per (patient, index) on a 6-month calendar grid from 2017-10-01; eligibility at least 24 months of ICD-10-era history, an encounter in the prior 12 months, no target code anywhere in history (ICD-10 rows plus concept-mapped ICD-9 rows), follow-up at least the horizon; censored rows excluded | 12-month horizon: cardiometabolic 1,710 rows / 510 patients / 202 positives; cancer 2,836 / 802 / 344; mental health 3,741 / 969 / 332. Positives per index year 27 to 91, so a temporal backtest is feasible. The current single-row rule gives 167 / 220 / 225 with a 12-month washout only | Paige 2018 landmarking; EHRShot new-diagnosis tasks (1.6 labels per patient, 365-day horizon, prior cases dropped); Abbas 2013 (at least 24 months washout); Roberts 2015; Li 2020 (ignoring censoring biases risk down); Gao/Albu 2025 | small: replaces step 9 logic |
| 4 | Acute node flag: share of a category's diagnosis rows dated inside an inpatient stay (visit_concept_id 9201 spans); optional HCUP CCIR chronic flag | Bridges median 0.43 (N17 0.51, F05 0.70, J96 0.58, E87 0.36, I63 0.28, G93 0.36) vs 0.12 for other nodes; 16 nodes above 0.50, 67 above 0.30. Visit-type of the condition row itself is useless here (most rows attach to "Orders Only" or "History" pseudo-visits) | Ramadan 2025 (same-admission codes leak); EHRShot keeps acute codes and relies on incident labels; HCUP CCIR; Gardinazzi 2026 and Crisafulli 2026 (inpatient networks have chronic central nodes, so our acute bridges are utilisation effects) | small; CCIR needs one public CSV download |
| 5 | Top-k backbone (k = 10) as the sparsified graph variant for Project 2; full graph as the "edges kept" ablation | k=10: 3,296 edges, density 0.028, 4 components, Leiden ARI 0.65 vs main, all three target sets preserved (47/49/15 overlap). k=5: 1,779 edges, ARI 0.52. Disparity filter: 49 edges at alpha 0.05, 0 at 0.01 (phi weights have no heavy tail per node) | Chet uses a per-node share threshold; Beaney 2024 CkNN k=10; Hossain 2024 (pruning dense regions helps graph classification); GraphSAGE sampling and DropEdge are training-time regularisers, complementary to a backbone | trivial: files already written |
| 6 | Node feature table for Project 2: prevalence, degree, strength, betweenness with null percentile, closeness, eigenvector, PageRank, consensus community id, stable-core flag, bridge-survivor flag, inpatient share, chapter and block, mean age at first occurrence, female share, backbone degree | Most columns exist in centrality.csv and bridges.csv; missing: PageRank, backbone degree, mean age, female share, CCIR flag | Singh 2026 (structural plus demographic features, same dataset); nobody uses global-graph descriptors as patient-subgraph features (open, cheap) | small |
| 7 | Patient-subgraph statistics per landmark row: codes in 1-year, 3-year and all-history lookback, induced edges under full and backbone graph, isolates, 1-hop neighbour count; bins below 3, 3 to 9, 10 or more | 1-year lookback at first index: median 7 nodes (IQR 3 to 15), median 13 edges, 14.8% with no edges, 19.5% with fewer than 3 nodes, 66% with 5 or more | SCOPE 2023 and ADMISSION 2024 (longer lookback better); Chet 1-hop expansion for sparse visits; no paper stratifies performance by subgraph size | small |
| 8 | Bootstrap and seed stability as reported Project 1 results (edge persistence, core membership, bridge rank persistence) | Done: outputs/p2_probes/bootstrap_*.csv | Crowson 2022; Ferris 2025 (only 25% of studies bootstrap) | none |
| 9 | Temporal drift numbers for the Project 2 backtest | Done in step 7: 2015-18 vs 2019+ partitions ARI 0.48; edge Jaccard to main 0.34 and 0.58 | Jiang 2018 (year-to-year module Jaccard down to 0.36); Guo 2021 (calibration drifts first); TRIPOD+AI | none |

## B. Does not help: skip

| Idea | Why not | Evidence |
|---|---|---|
| Disparity-filter backbone | Removes 99.6% of edges; phi per node is too uniform for the Serrano test | 49 edges at alpha 0.05 |
| Subgroup graphs (female, male, under 65, 65+) as model inputs | Too sparse at fixed thresholds; the 65+ graph has 110 isolates | subgroup_comparison.csv; use the pooled training graph and evaluate by subgroup instead |
| ICD-9-era graph as an input | Coding system mismatch, 7% unbridged, mapping noise; ARI 0.40 to main | sensitivity_summary.csv; keep it only for washout history |
| Guimera-Amaral roles as node features | Uninformative on a dense graph: participation median 0.75, one hub | centrality.csv |
| Chapter-only labels as the only definition | Data-driven clusters disagree with chapters (NVI 0.55 to 0.75) and our cancer codes scatter across six communities | Beaney 2024; consensus probe |
| Lower prevalence threshold (more nodes) | Patient subgraphs are already sparse; rare nodes add isolates | subgraph probe |
| Validating the target community by its outcome association | Circular; simulation shows outcome association anticorrelates with cluster truth | Ba Dhafari 2026 |
| Single calendar index (2019-01-01) | Under 150 positives everywhere | p2_label_counts.csv |
| Resolution sweeps beyond 1.0 / 1.5 / 2.0 | Already characterised; the instability is in the mental module, not the resolution | step 4 outputs |
| Survival or competing-risk head as the primary model | Out of course scope; keep as optional secondary | Gao/Albu 2025 show binary and competing-risk agree when follow-up is complete |

## C. Belongs in Project 2 (cannot be prepared without ML)

GNN architecture choices, DropEdge and neighbour sampling, learnable sparsification, external knowledge graphs
(GraphCare), embeddings, calibration, the diagnosis-count baseline, the edge-removal ablation, the 90-day gap
sensitivity (Placido 2023), and the temporal backtest itself (train index up to 2019-12-31, test from
2020-07-01, held-out patients).

## D. Decisions this implies for Project 2 (write them down once)

- Split: 70/15/15 by person_id, seed 0, fixed in config and committed as a patient-id list; or the canonical
  EHRShot split if it can be obtained from the benchmark repository.
- Graph: edges, phi, Leiden and all node features from training patients' records only; test subgraphs inherit
  them (inductive).
- Targets: cardiometabolic stable core; mental-health stable core; cancer as two targets (metastatic core,
  haematology core) with the neoplasm chapter as a sensitivity. Frozen membership lists in config.
- Labels: landmark grid, 24-month washout, 12-month horizon primary, censored rows excluded; death as a third
  class in a secondary analysis.
- Evaluation: GroupKFold by patient, AUROC and AUPRC with patient-cluster bootstrap CIs, results stratified by
  subgraph-size bin and by age band; calibration per index year.
- Ablations: full graph vs top-10 backbone, weighted vs unweighted, with and without acute nodes, edges removed,
  2016-19 vs 2020-22 graphs.

## E. Open questions

1. Adopt the fixed split now and rerun the main pipeline on training patients as the Project 1 sensitivity, or
   leave it to Project 2? Doing it now removes the largest caveat from both reports.
2. Cancer: two targets or neoplasm chapter?
3. Fetch the canonical EHRShot split from the benchmark repository (public GitHub) and the HCUP CCIR file
   (public AHRQ download)? Both are external files, not patient data.
4. Timeline: the three new steps cost about one working day before the Project 1 report; the deadline is
   2026-10-23.
