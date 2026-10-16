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

# --- cluster-to-cluster bridges (the research question) ---
anchors = cfg["centrality"]["target_clusters"]
td = pl.read_csv(out / "target_definitions.csv")
clusters = {k: set(td.filter((pl.col("target") == k) & pl.col("in_core"))["category"].to_list()) for k in anchors}
names = dict(zip(meta["category"], meta["name"]))
print("\ntarget clusters = stable cores (target_definitions.csv, in_core):")
for k, members in clusters.items():
    print(f"  {k:18s} {len(members)} nodes: " + ", ".join(sorted(members)[:14]) + (" ..." if len(members) > 14 else ""))
print(f"cluster-restricted betweenness + {cfg['centrality'].get('bridge_null_n', 0)} rewired nulls ...", flush=True)
br = ce.cluster_bridges(g, clusters, cfg, seed=int(cfg["communities"]["seeds"][0]))
br = br.join(meta, on="category").join(df.select("category", "community", "degree", "prevalence", "bt_null_z"), on="category")
br.write_csv(out / "bridges.csv")
pct_min = float(cfg["centrality"].get("bridge_null_percentile_min", 0.95))
de = ce.direct_cluster_edges(g, clusters)
de.write_csv(out / "bridges_direct_edges.csv")
k = int(cfg["centrality"].get("bridge_top_k", 15))
for pair in br["pair"].unique().sort():
    sub = br.filter(pl.col("pair") == pair)
    a, b = pair.split("-")
    n_pass = int(sub["bridge_pass"].sum()) if "bridge_pass" in sub.columns else -1
    print(f"\n{pair}: direct edges between the clusters = {de.filter(pl.col('pair') == pair).height}; "
          f"{n_pass} nodes pass the null (pct >= {pct_min}); top {k} by restricted betweenness (* = pass):")
    for r in sub.head(k).iter_rows(named=True):
        mark = "*" if r.get("bridge_pass") else " "
        print(f" {mark}{r['category']:4s} {str(r['name'])[:36]:36s} c{r['community']:<2} bt_AB={r['bridge_betweenness']:.4f} "
              f"pct={r.get('bridge_null_percentile', float('nan')):.2f} share->{a[:5]}={r['share_to_a']:.2f} share->{b[:5]}={r['share_to_b']:.2f}  deg={r['degree']:3d} prev={r['prevalence']:.2f}")
    top_direct = de.filter(pl.col("pair") == pair).head(5)
    if top_direct.height:
        print("  strongest direct edges: " + "; ".join(f"{r['u']}-{r['v']} phi={r['phi']:.2f}" for r in top_direct.iter_rows(named=True)))
print(f"wrote {out/'bridges.csv'}, {out/'bridges_direct_edges.csv'}")

# --- directed precedence between target clusters (sequential analysis) ---
if cfg["centrality"].get("precedence", True):
    from bridgedx import split
    pc = pl.read_parquet(out / "patient_category.parquet").filter(pl.col("person_id").is_in(split.train_ids(cfg)))  # main-graph patients only
    prec = ce.cluster_precedence(pc, clusters)
    prec.write_csv(out / "cluster_precedence.csv")
    print(f"\nfirst-recording precedence between target clusters ({pc['person_id'].n_unique():,} main-graph patients entering both):")
    for r in prec.iter_rows(named=True):
        print(f"  {r['from']} -> {r['to']}: n_both={r['n_both']:,}  {r['from'][:6]} first {r['n_a_first']:,} ({r['share_a_first']:.0%}), "
              f"{r['to'][:6]} first {r['n_b_first']:,}, same day {r['n_same_day']:,}; median gap {r['median_gap_days_a_to_b']:.0f} / {r['median_gap_days_b_to_a']:.0f} d; sign-test p={r['p_sign_test']:.2g}")
    print(f"wrote {out/'cluster_precedence.csv'}")
