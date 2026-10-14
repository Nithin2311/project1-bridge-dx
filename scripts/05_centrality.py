"""Step 5: bridge measures on the main graph.

Weighted betweenness (distance 1/phi) with percentile against 100 rewired nulls, closeness, eigenvector,
Guimera-Amaral participation / within-module z / role, degree-vs-prevalence residual.
Writes outputs/centrality.csv (one row per node) and prints the connector table.
Rerunnable: `uv run python scripts/05_centrality.py` (after 04). ~5-10 min (100 null betweenness runs).
"""
from __future__ import annotations

import time

import networkx as nx
import polars as pl

from bridgedx import centrality as ce
from bridgedx import config

cfg = config.load()
out = config.outputs_dir(cfg)
t0 = time.time()

g = nx.read_graphml(out / "graph_main.graphml")
comm = pl.read_csv(out / "communities.csv")
partition = dict(zip(comm["category"], comm["leiden"]))
summary = pl.read_csv(out / "communities_summary.csv")
min_size = int(cfg["communities"].get("min_community_size_in_tables", 1))
big = set(summary.filter(pl.col("size") >= min_size)["community"].to_list())

print("betweenness + null ...", flush=True)
bt = ce.betweenness_null_percentile(g, cfg)
bt_df = pl.DataFrame([{"category": v, **d} for v, d in bt.items()])
oth = ce.other_centralities(g)
rl = ce.roles(g, partition)
pr = ce.prevalence_residual(g)
meta = pl.DataFrame([{"category": v, "name": d.get("name"), "chapter": d.get("chapter")} for v, d in g.nodes(data=True)])

df = meta.join(rl, on="category").join(bt_df, on="category").join(oth, on="category").join(pr, on="category")
df = df.sort("betweenness", descending=True)
df.write_csv(out / "centrality.csv")

n = df.height
print(f"\n{n} nodes. degree~prevalence R2 = {df['degree_r2'][0]:.3f}; weighted_degree~prevalence R2 = {df['weighted_degree_r2'][0]:.3f}")
print("role counts: " + ", ".join(f"{r}={c}" for r, c in df.group_by("role").len().sort("role").iter_rows()))

def row(r):
    return (f"  {r['category']:4s} {str(r['name'])[:34]:34s} c{r['community']:<2} {r['role']}  bt={r['betweenness']:.4f} "
            f"pct={r['bt_null_percentile']:.2f} z={r['bt_null_z']:5.1f}  P={r['participation']:.2f} wz={r['within_module_z']:5.2f} "
            f"deg={r['degree']:3d} prev={r['prevalence']:.2f} resid_z={r['degree_residual_z']:5.2f}")

print("\ntop 20 by weighted betweenness (pct = share of nulls below observed):")
for r in df.head(20).iter_rows(named=True):
    print(row(r))
print("\nconnector hubs / kinless (R6, R7, R4), sorted by participation:")
for r in df.filter(pl.col("role").is_in(["R4", "R6", "R7"])).sort("participation", descending=True).iter_rows(named=True):
    print(row(r))
print("\ntop 15 degree residual (more connected than prevalence predicts):")
for r in df.sort("degree_residual_z", descending=True).head(15).iter_rows(named=True):
    print(row(r))
print("\nbottom 5 degree residual (prevalence hubs: common but not well connected):")
for r in df.sort("degree_residual_z").head(5).iter_rows(named=True):
    print(row(r))
print(f"\nwrote {out/'centrality.csv'}  ({time.time()-t0:.0f}s)")
