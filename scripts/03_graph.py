"""Step 3: build + export graphs.

  graph_main.graphml         training patients, chapters in nodes.main_graph_excludes_chapters dropped -> every main analysis
  graph_allchapters.graphml  training patients, all nodes                                            -> chapter sensitivity
  graph_allpatients.graphml  all cohort patients, excluded chapters dropped                          -> split sensitivity
Deliverables graph_nodes.csv / graph_edges.csv describe the main graph. graph_stats.csv has all three.
Rerunnable: `uv run python scripts/03_graph.py` (after 01, 02).
"""
from __future__ import annotations

import time

import polars as pl

from bridgedx import config, graph

cfg = config.load()
out = config.outputs_dir(cfg)
t0 = time.time()

nodes = pl.read_csv(out / "nodes.csv")
excl = set(cfg["nodes"].get("main_graph_excludes_chapters", []))
edges_tr = pl.read_csv(out / "edges.csv")
edges_all = pl.read_csv(out / "edges_allpatients.csv")
w = cfg["edges"]["weight"]

g_allch = graph.build(nodes, edges_tr, weight=w)
g_main = g_allch.subgraph([v for v in g_allch.nodes if v[0] not in excl]).copy()
g_allp = graph.build(nodes, edges_all, weight=w)
g_allp = g_allp.subgraph([v for v in g_allp.nodes if v[0] not in excl]).copy()
print(f"main graph excludes chapters {sorted(excl)}: {g_allch.number_of_nodes() - g_main.number_of_nodes()} nodes dropped")

rows = []
for name, g in [("main", g_main), ("allchapters", g_allch), ("allpatients", g_allp)]:
    stats = graph.basic_stats(g)
    rows += [{"graph": name, "stat": k, "value": float(v)} for k, v in stats.items()]
    graph.write_graphml(g, out / f"graph_{name}.graphml")
    print(f"\n{name} graph: {stats['n_nodes']} nodes, {stats['n_edges']:,} edges, weight = {w}")
    for k in ["density", "degree_mean", "degree_median", "avg_clustering", "transitivity", "n_components", "n_isolates", "avg_shortest_path_gc", "diameter_gc"]:
        v = stats[k]
        print(f"  {k:28s} {v:,.4f}" if isinstance(v, float) else f"  {k:28s} {v:,}")
pl.DataFrame(rows).write_csv(out / "graph_stats.csv")
em, ea = set(map(frozenset, g_main.edges)), set(map(frozenset, g_allp.edges))
print(f"\nmain (train) vs all-patients graph: shared edges {len(em & ea):,}, Jaccard {len(em & ea)/len(em | ea):.2f}")

nt = graph.node_table(g_main)
et = graph.edge_table(g_main)
nt.write_csv(out / "graph_nodes.csv")
et.write_csv(out / "graph_edges.csv")
print("main graph, top 10 nodes by degree: " + ", ".join(f"{r['category']} ({r['degree']})" for r in nt.head(10).iter_rows(named=True)))
print("main graph, top 10 by weighted degree: " + ", ".join(f"{r['category']} ({r['weighted_degree']:.1f})" for r in nt.sort("weighted_degree", descending=True).head(10).iter_rows(named=True)))
iso = nt.filter(pl.col("degree") == 0)
if iso.height:
    print("isolates:", ", ".join(f"{r['category']} ({r['name'][:30]})" for r in iso.iter_rows(named=True)))
print(f"\nwrote graph_main/allchapters/allpatients.graphml, graph_nodes.csv, graph_edges.csv, graph_stats.csv  ({time.time()-t0:.0f}s)")
