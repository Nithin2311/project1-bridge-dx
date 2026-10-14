"""Step 7: sensitivity runs.

  (a) ICD-9-era appendix graph (rows before era_start, bridged SNOMED -> ICD-10-CM category) on the main
      node set: Leiden at the main resolution, ARI vs main, top-20 betweenness overlap, edge Jaccard.
  (b) Follow-up strata: cohort tertiles by number of distinct categories, same comparisons.
  (c) Summary table joining (a), (b) with the step-4 full-graph and resolution-sweep results.
Rerunnable: `uv run python scripts/07_sensitivity.py` (after 05). ~5 min (concept_relationship scan ~1 min).
"""
from __future__ import annotations

import time

import networkx as nx
import polars as pl

from bridgedx import centrality as ce
from bridgedx import communities as cm
from bridgedx import config, data, edges as ed, graph as gr, sensitivity as se, subgroups as sg

cfg = config.load()
out = config.outputs_dir(cfg)
t0 = time.time()
k_top = int(cfg["subgroups"]["comparison"]["top_k"])
seed0 = cfg["communities"]["seeds"][0]

g_main = nx.read_graphml(out / "graph_main.graphml")
nodes = pl.read_csv(out / "nodes.csv").filter(pl.col("category").is_in(list(g_main.nodes)))
cats = nodes["category"].to_list()
comm = pl.read_csv(out / "communities.csv")
part_main = dict(zip(comm["category"], comm["leiden"]))
cent_main = pl.read_csv(out / "centrality.csv")
top_main = cent_main.sort("betweenness", descending=True)["category"].to_list()
pc = pl.read_parquet(out / "patient_category.parquet")
main_edges = set(map(frozenset, g_main.edges))


def compare(name: str, sub_pc: pl.DataFrame, members: list, note: str = "") -> dict:
    t1 = time.time()
    pairs = ed.weights(ed.fisher_filter(ed.pair_counts(sub_pc, categories=cats), cfg))
    pairs.write_csv(out / f"sensitivity_{name}_edges.csv")
    g = gr.build(nodes, pairs, weight=cfg["edges"]["weight"])
    gr.write_graphml(g, out / f"graph_{name}.graphml")
    p = cm.leiden(g, cfg, seed=seed0)
    bt = ce.weighted_betweenness(g)
    top = sorted(bt, key=lambda v: -bt[v])
    st = gr.basic_stats(g)
    e = set(map(frozenset, g.edges))
    row = {"variant": name, "n_patients": len(members), "n_edges": g.number_of_edges(), "density": st["density"],
           "avg_clustering": st["avg_clustering"], "n_isolates": st["n_isolates"],
           "n_communities": len(set(p.values())), "q": cm.modularity(g, p, cfg),
           "ari_vs_main": sg.compare_partitions(part_main, p),
           f"top{k_top}_bt_overlap_vs_main": sg.connector_overlap(top_main, top, k_top),
           "edge_jaccard_vs_main": len(e & main_edges) / len(e | main_edges) if e | main_edges else 0.0,
           "note": note}
    cm.partition_table(g, {"leiden": p}).write_csv(out / f"communities_{name}.csv")
    print(f"  {name:22s} n={len(members):,} edges={g.number_of_edges():,} dens={st['density']:.3f} comms={row['n_communities']} "
          f"Q={row['q']:.3f} ARI={row['ari_vs_main']:.3f} top{k_top}={row[f'top{k_top}_bt_overlap_vs_main']:.2f} "
          f"edgeJ={row['edge_jaccard_vs_main']:.2f}  ({time.time()-t1:.0f}s)")
    return row


rows = []
# (a) ICD-9 era
con = data.connect(cfg)
print("building SNOMED -> ICD-10-CM bridge (concept_relationship + concept_ancestor) ...", flush=True)
se.build_snomed_bridge(con, max_levels=3)
pc9, cov = se.icd9_era_patient_category(con, cfg)
pc9.write_parquet(out / "patient_category_icd9era.parquet")
n9 = cov["n_rows"]
print(f"ICD-9 era (< {cfg['cohort']['era_start']}): {n9:,} rows, {cov['n_patients']:,} patients; bridged direct "
      f"{cov['rows_direct']/n9:.1%}, via ancestor {cov['rows_ancestor']/n9:.1%}, unbridged {cov['rows_unbridged']/n9:.1%} "
      f"(concept_id 0: {cov['rows_unmapped_concept']/n9:.2%}); concepts with >1 category: {cov['concepts_ambiguous']:,} of {cov['concepts_total']:,}")
pats9 = pc9["person_id"].unique().to_list()
both = len(set(pats9) & set(pc["person_id"].unique().to_list()))
print(f"  ICD-9-era cohort (>= {cfg['cohort']['min_distinct_categories']} categories): {len(pats9):,} patients, {both:,} also in the main cohort; "
      f"{pc9.filter(pl.col('category').is_in(cats))['category'].n_unique()} of {len(cats)} main nodes present")
pl.DataFrame([cov]).write_csv(out / "sensitivity_icd9_bridge_coverage.csv")
rows.append(compare("icd9era", pc9, pats9, note="rows before era_start bridged via SNOMED; same node set"))

# (b) follow-up strata
print("\nfollow-up strata (tertiles of distinct-category count):")
for name, members in se.strata_by_n_categories(pc, 3).items():
    rows.append(compare(name, pc.filter(pl.col("person_id").is_in(members)), members, note="cohort tertile by n distinct categories"))

# (c) summary incl. step-4 variants
for tag, label in [("full", "full graph incl. R/Z, res main"), ("res1", "main graph, res 1.0"), ("res1.5", "main graph, res 1.5")]:
    f = out / f"communities_{tag}.csv"
    if f.exists():
        c2 = pl.read_csv(f)
        p2 = dict(zip(c2["category"], c2["leiden"]))
        s2 = pl.read_csv(out / f"communities_summary_{tag}.csv")
        rows.append({"variant": tag, "n_patients": pc["person_id"].n_unique(), "n_edges": None, "density": None,
                     "avg_clustering": None, "n_isolates": None, "n_communities": s2.height, "q": None,
                     "ari_vs_main": sg.compare_partitions(part_main, p2), f"top{k_top}_bt_overlap_vs_main": None,
                     "edge_jaccard_vs_main": None, "note": label})
summary = pl.DataFrame(rows)
summary.write_csv(out / "sensitivity_summary.csv")
print(f"\nwrote sensitivity_summary.csv, sensitivity_<variant>_edges.csv, graph_<variant>.graphml, communities_<variant>.csv, "
      f"patient_category_icd9era.parquet, sensitivity_icd9_bridge_coverage.csv  ({time.time()-t0:.0f}s)")
