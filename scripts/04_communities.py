"""Step 4: communities.

Leiden (+ Louvain) on the main graph, modularity vs 100 degree-preserving rewired nulls, seed stability,
per-community description. Also runs the no-R/Z variant so the main graph can be chosen on evidence.
Rerunnable: `uv run python scripts/04_communities.py` (after 03). ~3-5 min (100 null Leiden runs per variant).
"""
from __future__ import annotations

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
variants = sys.argv[1:] or ["main", "noRZ"]


def run(variant: str, g: nx.Graph) -> dict:
    t0 = time.time()
    tag = "" if variant == "main" else f"_{variant}"
    print(f"\n=== {variant}: {g.number_of_nodes()} nodes, {g.number_of_edges():,} edges ===")
    lp = cm.leiden(g, cfg, seed=seed0)
    lv = cm.louvain(g, cfg, seed=seed0)
    nodes = list(g.nodes)
    ari = adjusted_rand_score([lp[v] for v in nodes], [lv[v] for v in nodes])
    print(f"leiden: {len(set(lp.values()))} communities, Q={cm.modularity(g, lp, cfg):.4f}   "
          f"louvain: {len(set(lv.values()))} communities, Q={cm.modularity(g, lv, cfg):.4f}   ARI(leiden, louvain)={ari:.3f}")

    stab = cm.seed_stability(g, cfg)
    print("seed stability: " + ", ".join(f"s{r['seed']}: k={r['n_communities']} Q={r['q']:.4f} ARI={r['ari_vs_seed0']:.3f}" for r in stab.iter_rows(named=True)))

    null = cm.modularity_vs_null(g, lp, cfg)
    print(f"null ({null['n_null']} rewires): Q_null = {null['q_null_mean']:.4f} +/- {null['q_null_sd']:.4f} (max {null['q_null_max']:.4f}), "
          f"k_null = {null['n_communities_null_mean']:.1f};  Q_obs = {null['q_observed']:.4f}, z = {null['z']:.1f}, p = {null['p_empirical']:.3f}")

    desc = cm.describe(g, lp)
    print(f"{'c':>2} {'size':>4} {'purity':>6} {'flagRZ':>6} {'dens':>5} {'clus':>5} {'int/ext':>7}  top chapter / chapters / top-5 by within-community weighted degree")
    for r in desc.iter_rows(named=True):
        print(f"{r['community']:>2} {r['size']:>4} {r['chapter_purity']:>6.2f} {r['share_flagged']:>6.2f} {r['density']:>5.2f} {r['avg_clustering']:>5.2f} "
              f"{r['internal_external_ratio']:>7.2f}  {r['top_chapter']} | {r['chapters']}\n      {r['top5']}")

    cm.partition_table(g, {"leiden": lp, "louvain": lv}).write_csv(out / f"communities{tag}.csv")
    desc.write_csv(out / f"communities_summary{tag}.csv")
    pl.DataFrame({"q_null": null["q_null"]}).write_csv(out / f"modularity_null{tag}.csv")
    stab.write_csv(out / f"leiden_seed_stability{tag}.csv")
    with open(out / f"modularity{tag}.txt", "w") as f:
        for k, v in null.items():
            if k != "q_null":
                f.write(f"{k}\t{v}\n")
    print(f"({time.time()-t0:.0f}s)")
    return {"leiden": lp, "desc": desc}


g_main = nx.read_graphml(out / "graph_main.graphml")
res = {}
if "main" in variants:
    res["main"] = run("main", g_main)
if "noRZ" in variants:
    keep = [v for v, d in g_main.nodes(data=True) if not d.get("flagged")]
    g_rz = g_main.subgraph(keep).copy()
    res["noRZ"] = run("noRZ", g_rz)
if len(res) == 2:
    common = list(res["noRZ"]["leiden"])
    ari = adjusted_rand_score([res["main"]["leiden"][v] for v in common], [res["noRZ"]["leiden"][v] for v in common])
    print(f"\nARI(main leiden, noRZ leiden) on the {len(common)} non-R/Z nodes: {ari:.3f}")
print(f"wrote communities*.csv, communities_summary*.csv, modularity_null*.csv, leiden_seed_stability*.csv to {out}")
