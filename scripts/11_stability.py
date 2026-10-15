"""Step 11: patient-bootstrap stability of the main graph (non-ML).

Resample the main-graph patients with replacement n_bootstrap times; rebuild edges with the step-2 rule, Leiden
at the main resolution, target cores by anchor community, and cluster-restricted betweenness (no null, for
speed). Reports persistence of edges, target membership and bridge top-10 ranks.
Writes outputs/stability_edges.csv, stability_target_membership.csv, stability_bridges.csv, stability_summary.csv.
Rerunnable: `uv run python scripts/11_stability.py` (after 05). ~12 min for 30 resamples.
"""
from __future__ import annotations

import time

import networkx as nx
import numpy as np
import polars as pl

from bridgedx import centrality as ce, communities as cm, config, edges as ed, graph as gr, split

cfg = config.load()
out = config.outputs_dir(cfg)
B = int(cfg["stability"]["n_bootstrap"])
rng = np.random.default_rng(int(cfg["stability"]["seed"]))
t0 = time.time()

main_ids = split.train_ids(cfg)
pc = pl.read_parquet(out / "patient_category.parquet").filter(pl.col("person_id").is_in(main_ids))
g0 = nx.read_graphml(out / "graph_main.graphml")
nodes = pl.read_csv(out / "nodes.csv").filter(pl.col("category").is_in(list(g0.nodes)))
cats = nodes["category"].to_list()
anchors = cfg["centrality"]["target_clusters"]
td = pl.read_csv(out / "target_definitions.csv")
cores = {k: set(td.filter((pl.col("target") == k) & pl.col("in_core"))["category"].to_list()) for k in anchors}
br0 = pl.read_csv(out / "bridges.csv")
patients = np.array(sorted(pc["person_id"].unique().to_list()))
pc_by = {(k[0] if isinstance(k, tuple) else k): grp for k, grp in pc.group_by("person_id")}

edge_count, member_count, bridge_rank = {}, {k: {} for k in anchors}, {}
for b in range(B):
    samp = rng.choice(patients, size=len(patients), replace=True)
    pcb = pl.concat([pc_by[p].with_columns(pl.lit(i).alias("person_id")) for i, p in enumerate(samp)])
    pairs = ed.weights(ed.fisher_filter(ed.pair_counts(pcb, categories=cats), cfg))
    g = gr.build(nodes, pairs, weight=cfg["edges"]["weight"])
    for u, v in g.edges:
        e = frozenset((u, v)); edge_count[e] = edge_count.get(e, 0) + 1
    p = cm.leiden(g, cfg, seed=0)
    cl = ce.target_clusters(p, anchors)
    for k in anchors:
        for v in cl[k]:
            member_count[k][v] = member_count[k].get(v, 0) + 1
    br = ce.cluster_bridges(g, cores)
    for pair in br["pair"].unique():
        top = br.filter(pl.col("pair") == pair).sort("bridge_betweenness", descending=True).head(10)["category"].to_list()
        for r, v in enumerate(top):
            bridge_rank.setdefault((pair, v), []).append(r + 1)
    print(f"bootstrap {b+1}/{B}: {g.number_of_edges():,} edges, anchor communities {[len(cl[k]) for k in anchors]}  ({time.time()-t0:.0f}s)", flush=True)

m0 = set(map(frozenset, g0.edges))
ed_df = pl.DataFrame({"u": [tuple(e)[0] for e in edge_count], "v": [tuple(e)[1] for e in edge_count],
                      "persistence": [c / B for c in edge_count.values()], "in_main": [e in m0 for e in edge_count]})
ed_df.write_csv(out / "stability_edges.csv")
mem = pl.DataFrame([{"target": k, "category": v, "persistence": c / B, "in_core": v in cores[k]} for k in anchors for v, c in member_count[k].items()])
mem.write_csv(out / "stability_target_membership.csv")
brs = pl.DataFrame([{"pair": pr, "category": v, "share_in_top10": len(r) / B, "median_rank": float(np.median(r))} for (pr, v), r in bridge_rank.items()]).sort("pair", "share_in_top10", descending=[False, True])
brs.write_csv(out / "stability_bridges.csv")

mm = ed_df.filter("in_main")
rows = [{"quantity": "main edges persistence median", "value": mm["persistence"].median()},
        {"quantity": "main edges persistence >= 0.9 share", "value": (mm["persistence"] >= 0.9).mean()},
        {"quantity": "main edges persistence < 0.5 count", "value": (mm["persistence"] < 0.5).sum()}]
print(f"\nmain edges {mm.height:,}: persistence median {mm['persistence'].median():.2f}, >= 0.9 {(mm['persistence'] >= 0.9).mean():.1%}, < 0.5 {(mm['persistence'] < 0.5).sum():,}")
for k in anchors:
    s = mem.filter((pl.col("target") == k) & pl.col("in_core"))
    rows.append({"quantity": f"{k} core members persistence >= 0.8 share", "value": (s["persistence"] >= 0.8).mean() if s.height else None})
    print(f"  {k:18s} core {len(cores[k])}: in anchor community in >= 80% of resamples: {(s['persistence'] >= 0.8).sum()}; < 50%: {(s['persistence'] < 0.5).sum()}")
for pair in brs["pair"].unique().sort():
    surv = br0.filter((pl.col("pair") == pair) & pl.col("bridge_pass"))["category"].to_list()
    top = brs.filter(pl.col("pair") == pair).head(6)
    print(f"  {pair}: survivors {surv}; top-10 share: " + ", ".join(f"{r['category']} {r['share_in_top10']:.2f}" for r in top.iter_rows(named=True)))
    for v in surv:
        sh = brs.filter((pl.col("pair") == pair) & (pl.col("category") == v))
        rows.append({"quantity": f"bridge {pair} {v} top-10 share", "value": sh["share_in_top10"][0] if sh.height else 0.0})
pl.DataFrame(rows).write_csv(out / "stability_summary.csv")
print(f"wrote stability_*.csv  ({time.time()-t0:.0f}s)")
