"""Step 4: communities.

Leiden (+ Louvain) on the main graph at cfg resolution, RB quality vs 100 degree-preserving rewired nulls,
standard Q, seed stability, per-community description, 100-seed co-assignment matrix and the stable core of
every target (cfg centrality.target_clusters, core_min_coassignment). Sensitivity variants: allchapters
(R/Z/O kept), allpatients (test patients included), and the main graph at cfg resolution_sensitivity.
Rerunnable: `uv run python scripts/04_communities.py` (after 03). Runtime ~20 min.
Args: optional list of variants among  main allchapters allpatients res<r>  (default: all).
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
default = ["main", "allchapters", "allpatients"] + [f"res{r:g}" for r in res_sens]
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
    print(f"leiden: {len(set(lp.values()))} communities, RB quality(gamma={res:g})={cm.modularity(g, lp, c):.4f}, standard Q={cm.modularity_standard(g, lp):.4f}   "
          f"louvain: {len(set(lv.values()))} communities, RB quality={cm.modularity(g, lv, c):.4f}   ARI(leiden, louvain)={ari:.3f}")
    with open(out / f"partition_quality{sfx}.txt", "w") as f:
        f.write(f"resolution\t{res}\nleiden_rb_quality\t{cm.modularity(g, lp, c)}\nleiden_q_standard\t{cm.modularity_standard(g, lp)}\n"
                f"louvain_rb_quality\t{cm.modularity(g, lv, c)}\nlouvain_q_standard\t{cm.modularity_standard(g, lv)}\nari_leiden_louvain\t{ari}\n"
                f"n_leiden\t{len(set(lp.values()))}\nn_louvain\t{len(set(lv.values()))}\n")
    stab = cm.seed_stability(g, c)
    print("seed stability: " + ", ".join(f"s{r['seed']}: k={r['n_communities']} Q={r['q']:.4f} ARI={r['ari_vs_seed0']:.3f}" for r in stab.iter_rows(named=True)))
    if with_null:
        null = cm.modularity_vs_null(g, lp, c)
        print(f"null ({null['n_null']} rewires, RB quality at gamma={res:g}): null = {null['q_null_mean']:.4f} +/- {null['q_null_sd']:.4f} (max {null['q_null_max']:.4f}), "
              f"k_null = {null['n_communities_null_mean']:.1f};  observed = {null['q_observed']:.4f}, z = {null['z']:.1f}, p = {null['p_empirical']:.3f}")
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
parts = {}
for v in variants:
    if v == "main":
        parts[v] = run("main", g_main, cfg)
    elif v in ("allchapters", "allpatients"):
        parts[v] = run(v, nx.read_graphml(out / f"graph_{v}.graphml"), cfg)
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
# --- consensus: 100-seed co-assignment on the main graph, stable core per target ---
if "main" in parts:
    t1 = time.time()
    n_seeds = int(cfg["centrality"].get("core_n_seeds", 100))
    co = cm.coassignment(g_main, cfg, n_seeds)
    co.write_csv(out / "coassignment.csv")
    min_share = float(cfg["centrality"].get("core_min_coassignment", 0.9))
    from bridgedx import centrality as ce
    anchors = cfg["centrality"]["target_clusters"]
    comm_sets = ce.target_clusters(parts["main"], anchors)
    chap_sets = ce.chapter_block_targets(list(g_main.nodes), cfg["centrality"].get("chapter_blocks", {}))
    names = {v: d.get("name") for v, d in g_main.nodes(data=True)}
    rows = []
    print(f"\nstable cores ({n_seeds} seeds at resolution {res_main:g}, co-assignment with anchors >= {min_share}):")
    for k, codes in anchors.items():
        core = cm.stable_core(co, codes, min_share)
        comm = comm_sets.get(k, set()); chap = chap_sets.get(k, set())
        for v in sorted(set(core) | comm | chap):
            rows.append({"target": k, "category": v, "name": names.get(v), "in_core": v in core, "in_community": v in comm,
                         "in_chapter_block": v in chap, "coassignment_with_anchors": float(co.filter(pl.col("category") == v).select(codes[0])[0, 0]) if codes[0] in co.columns and v in co["category"].to_list() else None})
        print(f"  {k:18s} core={len(core):3d} community(seed {seed0})={len(comm):3d} chapter_block={len(chap):3d} core&community={len(set(core) & comm)} core&chapter={len(set(core) & chap)}")
        print(f"     core: {' '.join(core[:30])}{' ...' if len(core) > 30 else ''}")
    pl.DataFrame(rows).write_csv(out / "target_definitions.csv")
    print(f"  ({time.time()-t1:.0f}s)")
print(f"wrote communities*.csv, communities_summary*.csv, modularity_null*.csv, leiden_seed_stability*.csv, coassignment.csv, target_definitions.csv to {out}")
