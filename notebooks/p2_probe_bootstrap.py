"""P2-readiness probe B: bootstrap patients (resample cohort with replacement) and measure persistence of
edges, target-cluster membership and bridge ranking. Non-ML. Writes outputs/p2_probes/bootstrap_*.csv."""
from __future__ import annotations
import sys, time
import numpy as np, polars as pl, networkx as nx
from bridgedx import config, edges as ed, graph as gr, communities as cm, centrality as ce
cfg = config.load(); out = config.outputs_dir(cfg) / "p2_probes"
B = int(sys.argv[1]) if len(sys.argv) > 1 else 30
pc = pl.read_parquet(config.outputs_dir(cfg) / "patient_category.parquet")
g0 = nx.read_graphml(config.outputs_dir(cfg) / "graph_main.graphml")
nodes = pl.read_csv(config.outputs_dir(cfg) / "nodes.csv").filter(pl.col("category").is_in(list(g0.nodes)))
cats = nodes["category"].to_list()
comm = pl.read_csv(config.outputs_dir(cfg) / "communities.csv"); part0 = dict(zip(comm["category"], comm["leiden"]))
anchors = cfg["centrality"]["target_clusters"]
cl0 = ce.target_clusters(part0, anchors)
patients = pc["person_id"].unique().to_list()
rng = np.random.default_rng(0)
edge_count = {}; member_count = {k: {} for k in anchors}; bridge_rank = {}
t0 = time.time()
for b in range(B):
    samp = rng.choice(patients, size=len(patients), replace=True)
    # resampled patients get new ids so duplicates count as distinct patients
    pieces = []
    for i, p in enumerate(samp):
        pieces.append(pc.filter(pl.col("person_id") == p).with_columns(pl.lit(i).alias("person_id")))
    pcb = pl.concat(pieces)
    pairs = ed.weights(ed.fisher_filter(ed.pair_counts(pcb, categories=cats), cfg))
    g = gr.build(nodes, pairs, weight="phi")
    for u, v in g.edges:
        edge_count[frozenset((u, v))] = edge_count.get(frozenset((u, v)), 0) + 1
    p = cm.leiden(g, cfg, seed=0)
    cl = ce.target_clusters(p, anchors)
    for k in anchors:
        for v in cl[k]:
            member_count[k][v] = member_count[k].get(v, 0) + 1
    br = ce.cluster_bridges(g, cl0)  # fixed cluster sets from the main partition, no null (speed)
    for pair in br["pair"].unique():
        top = br.filter(pl.col("pair") == pair).sort("bridge_betweenness", descending=True).head(10)["category"].to_list()
        for r, v in enumerate(top):
            bridge_rank.setdefault((pair, v), []).append(r + 1)
    print(f"bootstrap {b+1}/{B}: {g.number_of_edges():,} edges, clusters {[len(cl[k]) for k in anchors]}  ({time.time()-t0:.0f}s)", flush=True)
m0 = set(map(frozenset, g0.edges))
pl.DataFrame({"u": [tuple(e)[0] for e in edge_count], "v": [tuple(e)[1] for e in edge_count],
              "persistence": [c / B for c in edge_count.values()],
              "in_main": [e in m0 for e in edge_count]}).write_csv(out / "bootstrap_edges.csv")
rows = [{"cluster": k, "category": v, "persistence": c / B, "in_main_cluster": v in cl0[k]} for k in anchors for v, c in member_count[k].items()]
pl.DataFrame(rows).write_csv(out / "bootstrap_cluster_membership.csv")
pl.DataFrame([{"pair": pr, "category": v, "n_in_top10": len(r), "share_in_top10": len(r) / B, "median_rank": float(np.median(r))}
              for (pr, v), r in bridge_rank.items()]).sort("pair", "share_in_top10", descending=[False, True]).write_csv(out / "bootstrap_bridges.csv")
print("done")
