"""Step 3: build + export graph.

Reads outputs/nodes.csv + outputs/edges.csv (kept edges only) and builds two phi-weighted graphs:
  graph_full.graphml  all 601 nodes                                          -> sensitivity
  graph_main.graphml  chapters in cfg nodes.main_graph_excludes_chapters dropped -> every main analysis
Deliverables graph_nodes.csv / graph_edges.csv describe the main graph. graph_stats.csv has both.
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
edges = pl.read_csv(out / "edges.csv")
g_full = graph.build(nodes, edges, weight=cfg["edges"]["weight"])
excl = set(cfg["nodes"].get("main_graph_excludes_chapters", []))
g_main = g_full.subgraph([v for v in g_full.nodes if v[0] not in excl]).copy()
print(f"main graph excludes chapters {sorted(excl)}: {g_full.number_of_nodes() - g_main.number_of_nodes()} nodes dropped")

rows = []
for name, g in [("main", g_main), ("full", g_full)]:
    stats = graph.basic_stats(g)
    rows += [{"graph": name, "stat": k, "value": float(v)} for k, v in stats.items()]
    graph.write_graphml(g, out / f"graph_{name}.graphml")
    print(f"\n{name} graph: {stats['n_nodes']} nodes, {stats['n_edges']:,} edges, weight = {cfg['edges']['weight']}")
    for k, v in stats.items():
        print(f"  {k:28s} {v:,.4f}" if isinstance(v, float) else f"  {k:28s} {v:,}")
pl.DataFrame(rows).write_csv(out / "graph_stats.csv")

nt = graph.node_table(g_main)
et = graph.edge_table(g_main)
nt.write_csv(out / "graph_nodes.csv")
et.write_csv(out / "graph_edges.csv")

print("\nmain graph, top 10 nodes by degree:")
for r in nt.head(10).iter_rows(named=True):
    print(f"  {r['category']}  {r['name'][:45]:45s} deg={r['degree']:>3}  wdeg={r['weighted_degree']:.1f}  prev={r['prevalence']:.2f}  clus={r['clustering']:.2f}")
print("main graph, top 10 nodes by weighted degree:")
for r in nt.sort("weighted_degree", descending=True).head(10).iter_rows(named=True):
    print(f"  {r['category']}  {r['name'][:45]:45s} deg={r['degree']:>3}  wdeg={r['weighted_degree']:.1f}  prev={r['prevalence']:.2f}")
iso = nt.filter(pl.col("degree") == 0)
if iso.height:
    print("isolates:", ", ".join(f"{r['category']} ({r['name'][:30]}, prev {r['prevalence']:.3f})" for r in iso.iter_rows(named=True)))
print(f"\nwrote graph_main.graphml, graph_full.graphml, graph_nodes.csv, graph_edges.csv, graph_stats.csv to {out}  ({time.time()-t0:.0f}s)")
