"""Step 6: subgroup graphs and comparison tests.

Assigns sex / age band per the yaml rule, builds female, male, under65, over65 graphs on the main node set
with the step-2 edge rule, runs Leiden (main resolution), weighted betweenness, cluster bridges (main target
cluster sets), then compares: ARI between partitions, top-20 connector overlap, edge-wise phi permutation test.
Rerunnable: `uv run python scripts/06_subgroups.py` (after 05). ~3-5 min.
"""
from __future__ import annotations

import time

import networkx as nx
import polars as pl

from bridgedx import centrality as ce
from bridgedx import communities as cm
from bridgedx import config, data, edges as ed, graph as gr, subgroups as sg

cfg = config.load()
out = config.outputs_dir(cfg)
t0 = time.time()
k_top = int(cfg["subgroups"]["comparison"]["top_k"])
seed0 = cfg["communities"]["seeds"][0]

pc = pl.read_parquet(out / "patient_category.parquet")
con = data.connect(cfg)
labels = sg.assign_subgroups(con, cfg, pc)
labels.write_csv(out / "subgroup_labels.csv")
print("subgroup sizes: " + ", ".join(f"{k}={len(v):,}" for k, v in sg.subgroup_members(labels, cfg).items())
      + f"; age at index median {labels['age_at_index'].median():.0f} (IQR {labels['age_at_index'].quantile(0.25):.0f}-{labels['age_at_index'].quantile(0.75):.0f})")

g_main = nx.read_graphml(out / "graph_main.graphml")
nodes_all = pl.read_csv(out / "nodes.csv")
nodes = nodes_all.filter(pl.col("category").is_in(list(g_main.nodes)))
cats = nodes["category"].to_list()
comm = pl.read_csv(out / "communities.csv")
part_main = dict(zip(comm["category"], comm["leiden"]))
clusters = ce.target_clusters(part_main, cfg["centrality"]["target_clusters"])
cent_main = pl.read_csv(out / "centrality.csv")
names = dict(zip(nodes["category"], nodes["name"]))

results = {"main": {"graph": g_main, "partition": part_main,
                    "top_bt": cent_main.sort("betweenness", descending=True)["category"].to_list(),
                    "stats": gr.basic_stats(g_main)}}
rows = []
for name, members in sg.subgroup_members(labels, cfg).items():
    t1 = time.time()
    sub_pc = pc.filter(pl.col("person_id").is_in(members))
    pairs = ed.weights(ed.fisher_filter(ed.pair_counts(sub_pc, categories=cats), cfg))
    pairs.write_csv(out / f"subgroup_{name}_edges.csv")
    g = gr.build(nodes, pairs, weight=cfg["edges"]["weight"])
    gr.write_graphml(g, out / f"graph_{name}.graphml")
    p = cm.leiden(g, cfg, seed=seed0)
    bt = ce.weighted_betweenness(g)
    top_bt = sorted(bt, key=lambda v: -bt[v])
    st = gr.basic_stats(g)
    br = ce.cluster_bridges(g, clusters).with_columns(pl.lit(name).alias("subgroup"))
    results[name] = {"graph": g, "partition": p, "top_bt": top_bt, "stats": st, "bridges": br, "betweenness": bt}
    rows.append({"subgroup": name, "n_patients": len(members), "n_edges": g.number_of_edges(), "density": st["density"],
                 "avg_clustering": st["avg_clustering"], "n_isolates": st["n_isolates"], "n_components": st["n_components"],
                 "n_communities": len(set(p.values())), "q": cm.modularity(g, p, cfg),
                 "ari_vs_main": sg.compare_partitions(part_main, p),
                 "top20_bt_overlap_vs_main": sg.connector_overlap(results["main"]["top_bt"], top_bt, k_top)})
    print(f"  {name:8s} n={len(members):,} edges={g.number_of_edges():,} density={st['density']:.3f} isolates={st['n_isolates']} "
          f"communities={len(set(p.values()))} Q={rows[-1]['q']:.3f} ARI_vs_main={rows[-1]['ari_vs_main']:.3f} "
          f"top{k_top}_bt_overlap_vs_main={rows[-1]['top20_bt_overlap_vs_main']:.2f}  ({time.time()-t1:.0f}s)")
summary = pl.DataFrame(rows)

# pairwise comparisons
pairs_cmp = [("female", "male", "sex"), ("under65", "over65", "age_band")]
cmp_rows = []
for a, b, col in pairs_cmp:
    ra, rb = results[a], results[b]
    ea, eb = set(map(frozenset, ra["graph"].edges)), set(map(frozenset, rb["graph"].edges))
    cmp_rows.append({"comparison": f"{a}-{b}", "ari": sg.compare_partitions(ra["partition"], rb["partition"]),
                     f"top{k_top}_bt_overlap": sg.connector_overlap(ra["top_bt"], rb["top_bt"], k_top),
                     "edges_only_a": len(ea - eb), "edges_only_b": len(eb - ea), "edges_both": len(ea & eb),
                     "jaccard_edges": len(ea & eb) / len(ea | eb)})
cmp = pl.DataFrame(cmp_rows)
summary.write_csv(out / "subgroup_comparison.csv")
cmp.write_csv(out / "subgroup_pairwise.csv")
print("\npairwise:")
for r in cmp.iter_rows(named=True):
    print(f"  {r['comparison']:16s} ARI={r['ari']:.3f}  top{k_top} betweenness overlap={r[f'top{k_top}_bt_overlap']:.2f}  "
          f"edges only {r['comparison'].split('-')[0]}={r['edges_only_a']:,}, only {r['comparison'].split('-')[1]}={r['edges_only_b']:,}, both={r['edges_both']:,} (Jaccard {r['jaccard_edges']:.2f})")

# top-k connectors per subgroup
print(f"\ntop {k_top} by weighted betweenness per subgroup:")
for name in ["main", "female", "male", "under65", "over65"]:
    print(f"  {name:8s} " + " ".join(results[name]["top_bt"][:k_top]))

# cluster bridges per subgroup
br_all = pl.concat([results[n]["bridges"] for n in results if n != "main"])
br_all.write_csv(out / "subgroup_bridges.csv")
for pair in br_all["pair"].unique().sort():
    print(f"\n{pair}: top 8 bridge nodes by restricted betweenness, per subgroup")
    lists = {}
    for name in ["female", "male", "under65", "over65"]:
        lists[name] = br_all.filter((pl.col("pair") == pair) & (pl.col("subgroup") == name)).sort("bridge_betweenness", descending=True)["category"].to_list()
        print(f"  {name:8s} " + ", ".join(f"{c} {names[c][:18]}" for c in lists[name][:8]))
    print(f"  overlap@{k_top}: female-male {sg.connector_overlap(lists['female'], lists['male'], k_top):.2f}, "
          f"under65-over65 {sg.connector_overlap(lists['under65'], lists['over65'], k_top):.2f}")

# edge-wise permutation test on main-graph edges
main_edges = pl.DataFrame({"u": [u for u, _ in g_main.edges], "v": [v for _, v in g_main.edges]})
alpha = float(cfg["edges"]["alpha"])
for a, b, col in pairs_cmp:
    t1 = time.time()
    tag = "sex" if col == "sex" else "age"
    pt = sg.edgewise_permutation(pc, labels, cfg, main_edges, col, a, b, cats, seed=seed0)
    pt = pt.with_columns(pl.col("u").replace_strict(names, default=None).alias("name_u"),
                         pl.col("v").replace_strict(names, default=None).alias("name_v")).sort("p_perm", "diff")
    pt.write_csv(out / f"subgroup_edge_permutation_{tag}.csv")
    n_sig = int((pt["p_fdr"] < alpha).sum())
    n_raw = int((pt["p_perm"] < alpha).sum())
    print(f"\nedge-wise permutation {a} vs {b} ({cfg['subgroups']['comparison']['n_permutations']} perms, {pt.height:,} edges): "
          f"{n_raw:,} edges p<{alpha} raw, {n_sig:,} at FDR<{alpha}  ({time.time()-t1:.0f}s)")
    sig = pt.filter(pl.col("p_fdr") < alpha)
    cl_of = {v: k for k, members in clusters.items() for v in members}
    sig = sig.with_columns(pl.col("u").replace_strict(cl_of, default=None).alias("cluster_u"),
                           pl.col("v").replace_strict(cl_of, default=None).alias("cluster_v"))
    sig.write_csv(out / f"subgroup_edge_permutation_{tag}_significant.csv")
    touch = sig.filter(pl.col("cluster_u").is_not_null() | pl.col("cluster_v").is_not_null())
    cross = touch.filter(pl.col("cluster_u").is_not_null() & pl.col("cluster_v").is_not_null() & (pl.col("cluster_u") != pl.col("cluster_v")))
    print(f"  of these, {touch.height} touch a target cluster and {cross.height} join two different target clusters")
    print(f"  direction: {int((sig['diff'] > 0).sum())} stronger in {a}, {int((sig['diff'] < 0).sum())} stronger in {b}; "
          f"touching a target cluster: {int((touch['diff'] > 0).sum())} stronger in {a}, {int((touch['diff'] < 0).sum())} stronger in {b}")
    for lab, df_ in [(f"stronger in {a}, all", sig.filter(pl.col("diff") > 0).sort("diff", descending=True).head(6)),
                     (f"stronger in {b}, all", sig.filter(pl.col("diff") < 0).sort("diff").head(6)),
                     (f"stronger in {a}, touching a target cluster", touch.filter(pl.col("diff") > 0).sort("diff", descending=True).head(10)),
                     (f"stronger in {b}, touching a target cluster", touch.filter(pl.col("diff") < 0).sort("diff").head(10))]:
        print(f"  {lab}:")
        for r in df_.iter_rows(named=True):
            tag_c = "/".join(x for x in [r.get("cluster_u"), r.get("cluster_v")] if x)
            print(f"    {r['u']}-{r['v']}  {r['name_u'][:24]} / {r['name_v'][:24]}  phi {r['phi_a']:.2f} vs {r['phi_b']:.2f}  diff={r['diff']:+.2f}  q={r['p_fdr']:.3f}  [{tag_c}]")

print(f"\nwrote subgroup_labels.csv, subgroup_<name>_edges.csv, graph_<name>.graphml, subgroup_comparison.csv, "
      f"subgroup_pairwise.csv, subgroup_bridges.csv, subgroup_edge_permutation_<sex|age>.csv  ({time.time()-t0:.0f}s)")
