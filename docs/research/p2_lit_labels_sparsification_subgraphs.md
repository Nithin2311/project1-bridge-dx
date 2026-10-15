# Literature brief 2: label stability, sparsification, patient subgraphs, proximal codes
Compiled 2026-10-15 by an independent research agent for Project 2 design. "Unverified" = single source or DOI not resolved in session.

## 1. Label stability
- Lancichinetti & Fortunato 2012 (Sci Rep): consensus matrix D_ij = share of runs co-clustering i and j; threshold tau (0.5 used in stability tests); re-cluster D; stability rises monotonically with number of runs. https://doi.org/10.1038/srep00336
- Jeub, Sporns, Fortunato 2018: threshold-free multiresolution consensus. https://doi.org/10.1038/s41598-018-21352-7 ; ECG (Poulin & Theberge 2019) https://doi.org/10.1007/s41109-019-0162-z ; FastEnsemble 2025 https://doi.org/10.1371/journal.pcsy.0000069
- Riolo & Newman 2020: high-scoring partitions share invariant building blocks. https://doi.org/10.1103/PhysRevE.101.052306 ; Peixoto 2021: characterize modes (consensus and dissensus) rather than one point estimate. https://doi.org/10.1103/PhysRevX.11.021003
- Comorbidity clusters are unstable: Brunson 2019 https://doi.org/10.1093/jamiaopen/ooz067 ; Griffith 2024 (5 methods x 11 measures gave 1-24 clusters; ARI vs clinician clusters 0.02-0.24) https://doi.org/10.1016/j.jclinepi.2024.111435 ; Crowson 2022 https://doi.org/10.1002/acr.24973 ; Ferris 2025 (Nat Commun meta-analysis: no cluster reaches Jaccard >= 0.75) https://doi.org/10.1038/s41467-025-67372-6 ; Ba Dhafari 2026 (do not validate a cluster by its outcome association) https://doi.org/10.1016/j.jclinepi.2026.112209
- Beaney 2024 (Commun Med; 10.6M patients; CkNN k=10 sparsification, Markov Stability with Leiden, 2,000 optimisations; data-driven clusters vs chapters NVI 0.55-0.75) https://doi.org/10.1038/s43856-024-00529-4
- Hybrid labelling: Crisafulli 2026 (EPJ Data Sci; communities characterised by chapter enrichment) https://doi.org/10.1140/epjds/s13688-026-00651-4 ; Dervic 2024 overlapping communities https://doi.org/10.1038/s41746-024-01015-w

## 2. Sparsification for GNN input
- Oversmoothing: Chen 2020 (AAAI, MAD/MADGap) https://doi.org/10.1609/aaai.v34i04.5747 ; Hossain 2024 (truss-based pruning of dense regions helps graph classification) https://doi.org/10.1007/978-3-031-70371-3_10 ; Keriven 2022 https://doi.org/10.52202/068431-0165 ; Jamadandi 2024 spectral pruning arXiv:2404.04612
- No sparsifier preserves everything: Chen 2023 (VLDB) https://doi.org/10.14778/3632093.3632106 ; Yassin 2025 https://doi.org/10.1371/journal.pone.0316141 ; netbone https://doi.org/10.1038/s41598-023-42076-3 ; backbone R https://doi.org/10.1371/journal.pone.0269137
- Disparity filter: Serrano 2009 https://doi.org/10.1073/pnas.0808904106 ; Fotouhi 2018 comorbidity null models https://doi.org/10.1007/s41109-018-0101-4
- Statistically validated comorbidity links keep < 5% of projected links (Crisafulli 2026); Chet drops pairs below a per-node share threshold (per-node top-fraction) https://doi.org/10.1609/aaai.v36i4.20380
- Sampling vs sparsifying: GraphSAGE samples 25/10 neighbours (arXiv:1706.02216); DropEdge (arXiv:1907.10903) is a training-time regulariser; GAT re-weights but never removes (arXiv:1710.10903); NeuralSparse (ICML 2020, DOI unverified). Backbone and sampling are complementary.

## 3. Patient subgraphs from a global code graph
- Chet (AAAI 2022): per-visit local graph of the patient's codes with global weights, plus 1-hop global neighbours; learned embeddings; MIMIC-III/IV.
- Ontology family: GRAM (KDD 2017, arXiv:1611.07012), KAME, MiME, HAP (DOIs unverified). Analogue here: chapter/block ancestors as node features.
- GraphCare (ICLR 2024, arXiv:2305.12788); KAT-GNN (arXiv:2511.01249, preprint)
- Bipartite precedent: Lu & Uddin 2021 https://doi.org/10.1038/s41598-021-01964-2 ; Sun 2020 (IEEE JBHI, DOI unverified); Aziz 2021 https://doi.org/10.1038/s41598-021-95802-0
- Node features: Singh 2026 (structural + demographic) https://doi.org/10.1007/978-3-032-26028-4_14 ; nobody found uses global-graph structural descriptors as patient-subgraph node features (open, cheap).
- Sparse subgraphs: no paper stratifies performance by subgraph size (gap). Surveys: Oss Boll 2024 https://doi.org/10.1016/j.jbi.2024.104616 ; Hancox 2024.
- Lookback: EHRShot full history; SCOPE 2023 (longer windows better, 3,650 d chosen) https://doi.org/10.1038/s41598-023-38257-9 ; ADMISSION 2024 (7-year lookback) https://doi.org/10.1136/bmjopen-2023-080678 ; Pungitore 2023 https://doi.org/10.1007/s41666-023-00143-4

## 4. Proximal / acute codes
- Ramadan 2025 (JAMA Netw Open): same-admission ICD codes leak mortality labels (AUROC 0.97). https://doi.org/10.1001/jamanetworkopen.2025.50454 ; Davis 2023 https://doi.org/10.1093/jamia/ocad178
- EHRShot: no code-type exclusion; incident label + discharge-time prediction + prior-code exclusion. Beck 2016: acute sepsis is a legitimate outcome of chronic history. https://doi.org/10.1038/srep36624
- HCUP CCIR chronic/not-chronic flag for every ICD-10-CM code https://hcup-us.ahrq.gov/toolssoftware/chronic_icd10/CCIR-User-Guide-v2026-1.pdf ; POA indicators (EHRShot has none; analogue = inpatient share)
- Bridge nodes vs hospitalisation: indirect; inpatient-only networks have chronic central nodes (Gardinazzi 2026; Crisafulli 2026), so acute bridge survivors are most plausibly utilisation effects. Jones, Ma, McNally 2019 bridge centrality (DOI unverified) is the formal version of cluster-restricted betweenness. https://doi.org/10.1080/00273171.2019.1614898
- Practice = keep with ablation; exclusion reserved for same-admission outcomes.

Consensus tool notice (required by that server): Create or connect a free Consensus account to return more than 3 results per search in Claude Code.: https://consensus.app/sign-up/?utm_source=claude_code&auth=claude_code
