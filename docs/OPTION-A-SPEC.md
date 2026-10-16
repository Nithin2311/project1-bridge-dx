# Option A spec (one page) and work split

Source: `../FinalProject/Project 1 Ideas proposal.docx` (Oct 8). This is the condensed, executable version.
Rubric weights that decide the grade: Network Analysis 25, Interpretation 20, Report 20.

## Question
Which diagnoses connect the big disease clusters (cardiometabolic, cancer, mental health), and does that
set of connectors change between patients under 65 and over 65, or between women and men?

## Data facts (from the proposal, verify in step 0/1)
| Item | Value |
|---|---|
| Patients with ≥1 dx row | 6,148 of 6,731 |
| Patients with ICD-10-era rows | 4,656 |
| ... with ≥5 distinct 3-char categories (cohort) | 4,131 |
| Categories ≥1% prevalence (nodes) | 476 |
| Candidate pairs with ≥20 patients | 53,604 |
| Exact duplicate dx rows | 749,034 (29%) |
| Events after death | 1,924 in 244 patients |
| ICD-9 rows bridging to SNOMED | 78% |

## Graph
- Node: ICD-10 3-char category, names from concept.csv.
- Edge: same patient has both in the era; keep if Fisher significant after Bonferroni, OR ≥ 1.2, ≥ 10 shared patients.
- Weight: phi. Betweenness distance = 1/phi. Jaccard as check.
- Subgroups: same node set; women, men, <65, ≥65.

## Measures → rubric
| Measure | Required-technique box it ticks |
|---|---|
| Leiden (+Louvain), modularity vs 100 rewired nulls | community detection |
| Chapter purity + top-5 nodes per community | interpretation (describe before naming) |
| Weighted betweenness + null percentile | centrality |
| Participation coefficient + within-module z (Guimerà & Amaral) | centrality, "bridge vs hub" |
| Degree vs prevalence residual | prevalence-hub check |
| Density, clustering, internal/external edge ratio per community | density & clustering |
| ARI between partitions, top-20 overlap, edge-wise phi permutation (1,000) | subgroup comparison |
| ICD-9-era appendix graph | temporal comparison |

## Traps → handling
Prevalence hubs → phi + tests + residual plot · Duplicates → distinct (patient, category) · Post-death events → drop ·
ICD-9/10 switch → ICD-10 primary, ICD-9 appendix (empirical SNOMED map) · R/Z/O chapters → excluded from the main graph, full graph as sensitivity (decided 2026-10-14) · Unequal follow-up → stratified sensitivity (power-limited).

## Decisions after the proposal (2026-10-14, two independent reviews; reasons in config/option_a.yaml)
Graph built on the canonical EHRShot train+val patients (2,787; test never touches it). Main graph excludes R, Z, O chapters (496 nodes, 9,115 edges). Leiden resolution 1.5 (1.0, 2.0 sensitivity); RB quality and
standard Q both reported. Bridges = cluster-restricted betweenness that beats a 100-rewire null with a magnitude
floor. Directed first-recording precedence between target clusters. 20,000 permutations. Sex-specific codes flagged.
Four targets by 100-seed stable cores (cardiometabolic 53, metastatic 10, hematologic 21, mental health 10); E11 joins the
cardiometabolic block in 87% of seeds (anchor-mean rule), reported with a 0.8 sensitivity. All-patient graph = replication.

## Related works already named
Monchka 2022, Qiu 2021, Guimerà & Amaral 2005, Traag 2019, Hidalgo 2009. Need ≥5; have 5, add 2–3 from `../FinalProject/research/`.

## Work split (proposal; fill in with Dylan)
| Block | Scripts | Owner |
|---|---|---|
| Data → cohort → edges → graph | 00–03 | |
| Communities, centrality, null models | 04–05 | |
| Subgroups + permutation tests | 06 | |
| Sensitivity runs | 07 | |
| Figures, tables | 08 | |
| Report: Intro, Related, Construction | LaTeX | |
| Report: Methods, Results, Interpretation | LaTeX | |
| P2 label counts | 09 | |

## Timeline (revised 2026-10-13; original Oct 11 milestone slipped)
| By | Done |
|---|---|
| Thu Oct 15 | cohort + node list frozen, edge list exported, first Leiden run, top-neighbours plot |
| Sun Oct 18 | betweenness/roles with null, residual, per-community stats, 4 subgroup graphs + permutation test |
| Tue Oct 20 | related works drafted, figures/tables final, LaTeX draft, P2 label counts |
| Thu Oct 22 | proofread, reproducibility check from raw CSVs |
| Fri Oct 23 | submit |
