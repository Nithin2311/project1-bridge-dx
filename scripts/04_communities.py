"""Step 4: communities.

Leiden (+ Louvain) on the main graph at cfg resolution, modularity vs 100 degree-preserving rewired nulls,
seed stability, per-community description. Sensitivity: the full (R/Z-included) graph, and the main graph at
cfg communities.resolution_sensitivity. Rerunnable: `uv run python scripts/04_communities.py` (after 03).
Runtime ~10-15 min (100 null Leiden runs per variant; the full graph is the slow one).
Args: optional list of variants among  main full res<r>  (default: all).
"""
from __future__ import annotations

import copy
import sys
import time

import networkx as nx
import polars as pl
from sklearn.metrics import adjusted_rand_score

from bridgedx import communities as cm
from bridgedx import config

cfg = config.load()
out = config.outputs_dir(cfg)
seed0 = cfg["communities"]["seeds"][0]
res_main = float(cfg["communities"]["resolution"])
res_sens = [float(r) for r in cfg["communities"].get("resolution_sensitivity", [])]
default = ["main", "full"] + [f"res{r:g}" for r in res_sens]
variants = sys.argv[1:] or default


def run(tag: str, g: nx.Graph, c: dict, with_null: bool = True) -> dict:
    t0 = time.time()
    res = float(c["communities"]["resolution"])
    sfx = "" if tag == "main" else f"_{tag}"
    print(f"\n=== {tag}: {g.number_of_nodes()} nodes, {g.number_of_edges():,} edges, resolution {res:g} ===")
    lp = cm.leiden(g, c, seed=seed0)
    lv = cm.louvain(g, c, seed=seed0)
    nodes = list(g.nodes)
    ari = adjusted_rand_score([lp[v] for v in nodes], [lv[v] for v in nodes])
    print(f"leiden: {len(set(lp.values()))} communities, Q={cm.modularity(g, lp, c):.4f}   "
          f"louvain: {len(set(lv.values()))} communities, Q={cm.modularity(g, lv, c):.4f}   ARI(leiden, louvain)={ari:.3f}")
    stab = cm.seed_stability(g, c)
    print("seed stability: " + ", ".join(f"s{r['seed']}: k={r['n_communities']} Q={r['q']:.4f} ARI={r['ari_vs_seed0']:.3f}" for r in stab.iter_rows(named=True)))
    if with_null:
        null = cm.modularity_vs_null(g, lp, c)
        print(f"null ({null['n_null']} rewires): Q_null = {null['q_null_mean']:.4f} +/- {null['q_null_sd']:.4f} (max {null['q_null_max']:.4f}), "
              f"k_null = {null['n_communities_null_mean']:.1f};  Q_obs = {null['q_observed']:.4f}, z = {null['z']:.1f}, p = {null['p_empirical']:.3f}")
        pl.DataFrame({"q_null": null["q_null"]}).write_csv(out / f"modularity_null{sfx}.csv")
        with open(out / f"modularity{sfx}.txt", "w") as f:
            for k, v in null.items():
                if k != "q_null":
                    f.write(f"{k}\t{v}\n")
    desc = cm.describe(g, lp)
    min_size = int(c["communities"].get("min_community_size_in_tables", 1))
    print(f"{'c':>2} {'size':>4} {'purity':>6} {'dens':>5} {'clus':>5} {'int/ext':>7}  top chapter | chapters | top-5 by within-community weighted degree")
    for r in desc.filter(pl.col("size") >= min_size).iter_rows(named=True):
        print(f"{r['community']:>2} {r['size']:>4} {r['chapter_purity']:>6.2f} {r['density']:>5.2f} {r['avg_clustering']:>5.2f} "
              f"{r['internal_external_ratio']:>7.2f}  {r['top_chapter']} | {r['chapters']}\n      {r['top5']}")
    small = desc.filter(pl.col("size") < min_size)
    if small.height:
        print(f"  + {small.height} communities with < {min_size} nodes ({small['size'].sum()} nodes): "
              + ", ".join(small["top5_codes"].to_list()))
    cm.partition_table(g, {"leiden": lp, "louvain": lv}).write_csv(out / f"communities{sfx}.csv")
    desc.write_csv(out / f"communities_summary{sfx}.csv")
    stab.write_csv(out / f"leiden_seed_stability{sfx}.csv")
    print(f"({time.time()-t0:.0f}s)")
    return lp


g_main = nx.read_graphml(out / "graph_main.graphml")
g_full = nx.read_graphml(out / "graph_full.graphml")
parts = {}
for v in variants:
    if v == "main":
        parts[v] = run("main", g_main, cfg)
    elif v == "full":
        parts[v] = run("full", g_full, cfg)
    elif v.startswith("res"):
        c = copy.deepcopy(cfg)
        c["communities"]["resolution"] = float(v[3:])
        parts[v] = run(v, g_main, c, with_null=True)
if "main" in parts:
    base = parts["main"]
    for v, p in parts.items():
        if v == "main":
            continue
        common = [n for n in base if n in p]
        print(f"ARI(main, {v}) on {len(common)} shared nodes: {adjusted_rand_score([base[n] for n in common], [p[n] for n in common]):.3f}")
print(f"wrote communities*.csv, communities_summary*.csv, modularity_null*.csv, leiden_seed_stability*.csv to {out}")
