"""Step 3: build + export graph.

Reads outputs/nodes.csv + outputs/edges.csv (kept edges only), builds the phi-weighted NetworkX graph,
writes outputs/graph_main.graphml, outputs/graph_nodes.csv, outputs/graph_edges.csv, outputs/graph_stats.csv.
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
g = graph.build(nodes, edges, weight=cfg["edges"]["weight"])

stats = graph.basic_stats(g)
pl.DataFrame({"stat": list(stats), "value": [float(v) for v in stats.values()]}).write_csv(out / "graph_stats.csv")
graph.write_graphml(g, out / "graph_main.graphml")
nt = graph.node_table(g)
et = graph.edge_table(g)
nt.write_csv(out / "graph_nodes.csv")
et.write_csv(out / "graph_edges.csv")

print(f"graph: {stats['n_nodes']} nodes, {stats['n_edges']:,} edges, weight = {cfg['edges']['weight']}")
for k, v in stats.items():
    print(f"  {k:28s} {v:,.4f}" if isinstance(v, float) else f"  {k:28s} {v:,}")

print("\ntop 10 nodes by degree:")
for r in nt.head(10).iter_rows(named=True):
    print(f"  {r['category']}  {r['name'][:45]:45s} deg={r['degree']:>3}  wdeg={r['weighted_degree']:.1f}  "
          f"prev={r['prevalence']:.2f}  clus={r['clustering']:.2f}{'  [R/Z]' if r['flagged'] else ''}")
print("\ntop 10 nodes by weighted degree:")
for r in nt.sort("weighted_degree", descending=True).head(10).iter_rows(named=True):
    print(f"  {r['category']}  {r['name'][:45]:45s} deg={r['degree']:>3}  wdeg={r['weighted_degree']:.1f}  "
          f"prev={r['prevalence']:.2f}{'  [R/Z]' if r['flagged'] else ''}")
iso = nt.filter(pl.col("degree") == 0)
if iso.height:
    print("\nisolates:", ", ".join(f"{r['category']} ({r['name'][:30]}, prev {r['prevalence']:.3f})" for r in iso.iter_rows(named=True)))

h = graph.to_igraph(g)
print(f"\nigraph copy: {h.vcount()} vertices, {h.ecount()} edges")
print(f"wrote graph_main.graphml, graph_nodes.csv, graph_edges.csv, graph_stats.csv to {out}  ({time.time()-t0:.0f}s)")
