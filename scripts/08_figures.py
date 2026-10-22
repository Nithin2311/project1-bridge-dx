"""Step 8: figures and tables for the report.

Reads outputs/* from steps 3-7 and writes outputs/fig_*.pdf (+png) and outputs/tables/*.tex.
Rerunnable: `uv run python scripts/08_figures.py` (after 07). ~1 min.
"""
from __future__ import annotations

import time

import networkx as nx
import numpy as np
import polars as pl

from bridgedx import config, viz

cfg = config.load()
out = config.outputs_dir(cfg)
tbl = out / "tables"
tbl.mkdir(exist_ok=True)
t0 = time.time()
k_top = int(cfg["subgroups"]["comparison"]["top_k"])
alpha = float(cfg["edges"]["alpha"])
min_size = int(cfg["communities"].get("min_community_size_in_tables", 5))

g = nx.read_graphml(out / "graph_main.graphml")
comm = pl.read_csv(out / "communities.csv")
partition = dict(zip(comm["category"], comm["leiden"]))
summary = pl.read_csv(out / "communities_summary.csv")
cent = pl.read_csv(out / "centrality.csv")
names = dict(zip(cent["category"], cent["name"]))
bridges = pl.read_csv(out / "bridges.csv")
sub_bridges = pl.read_csv(out / "subgroup_bridges.csv")
groups = cfg["subgroups"]["groups"]["sex"] + cfg["subgroups"]["groups"]["age"]["labels"]

# legend names: the four target communities carry the report's names and their anchor; the rest are described by
# top chapter and strongest member (describe before naming)
chapter_of = dict(zip(cent["category"], cent["chapter"]))  # chapter of the strongest member, not the community's plurality chapter
comm_names = {r["community"]: f"{chapter_of.get(r['top5_codes'].split()[0], r['top_chapter']).split('/')[0]}: {r['top5_codes'].split()[0]}"
              for r in summary.iter_rows(named=True)}
target_label = {"cardiometabolic": "Cardiometabolic", "cancer_metastatic": "Metastatic cancer", "cancer_hematologic": "Hematology", "mental_health": "Mental health"}
td_ = pl.read_csv(out / "target_definitions.csv")
for tgt, anchors in cfg["centrality"]["target_clusters"].items():
    members = td_.filter((pl.col("target") == tgt) & pl.col("in_core"))["category"].to_list()
    comms = [partition[m] for m in members if m in partition]
    if comms:
        c_mode = max(set(comms), key=comms.count)
        comm_names[c_mode] = f"{target_label.get(tgt, tgt)} ({'/'.join(anchors)})"
# short names shared by the report's bridge table and the stability bar chart
short_names = {"N17": "Acute kidney failure", "D63": "Chronic-disease anemia", "A41": "Sepsis", "J90": "Pleural effusion",
               "E43": "Severe malnutrition", "E87": "Fluid/electrolyte disorders", "F05": "Delirium", "E44": "Mild/moderate malnutrition",
               "D89": "Immune-mechanism disorders", "K59": "Functional intestinal disorders", "G89": "Pain", "K63": "Other intestinal diseases",
               "L29": "Pruritus"}

# color the target-cluster communities first (the research question), then the largest others
priority = []
for codes in cfg["centrality"]["target_clusters"].values():
    for code in codes:
        if code in partition and partition[code] not in priority:
            priority.append(partition[code])
viz.community_layout(g, partition, out / "fig_communities.pdf", top_k=int(cfg["figures"]["top_k"]),
                     comm_names=comm_names, seed=cfg["communities"]["seeds"][0], priority=priority)
viz.prevalence_residual_plot(cent, out / "fig_prevalence_residual.pdf")
q_null = pl.read_csv(out / "modularity_null.csv")["q_null"].to_numpy()
mod = dict(l.strip().split("\t") for l in open(out / "modularity.txt"))
viz.modularity_null_plot(q_null, float(mod["q_observed"]), out / "fig_modularity_null.pdf",
                         resolution=float(mod.get("resolution", cfg["communities"]["resolution"])),
                         q_standard=float(mod["q_standard_observed"]) if "q_standard_observed" in mod else None)
for pair in sub_bridges["pair"].unique().sort():
    viz.subgroup_bridges_plot(sub_bridges, out / f"fig_bridges_{pair}.pdf", pair, groups, names, top_n=8)  # 8 keeps it under a column
from bridgedx import subgroups as sg
ranges = cfg["nodes"].get("sex_specific_ranges", [])
panels = []
for tag, a, b, title in [("sex", "female", "male", "(a) recorded sex"), ("age", "under65", "over65", "(b) age at first eligible diagnosis")]:
    pt = pl.read_csv(out / f"subgroup_edge_permutation_{tag}.csv")
    pt = pt.with_columns((pl.col("u").map_elements(lambda c: sg.in_ranges(c, ranges), return_dtype=pl.Boolean)
                          | pl.col("v").map_elements(lambda c: sg.in_ranges(c, ranges), return_dtype=pl.Boolean)).alias("sex_specific"))
    panels.append((pt, a, b, title))
viz.edge_diff_panels(panels, out / "fig_edge_diff.pdf", alpha,
                     group_labels={"female": "female patients", "male": "male patients", "under65": "patients under 65", "over65": "patients 65 and over"})
for stale in ("fig_edge_diff_sex", "fig_edge_diff_age"):  # replaced by the two-panel fig_edge_diff
    for ext in (".pdf", ".png"):
        (out / f"{stale}{ext}").unlink(missing_ok=True)

# step 11 (patient bootstrap) is optional for this step; the figure is drawn only when its output exists
stab_path = out / "stability_bridges.csv"
if stab_path.exists():
    stab_df = pl.read_csv(stab_path)
    n_boot_ = int(cfg["stability"]["n_bootstrap"])
    viz.bridge_stability_plot(stab_df, bridges, out / "fig_bridge_stability.pdf", names, n_bootstrap=n_boot_)
    viz.bridge_stability_bars(stab_df, bridges, out / "fig_bridge_stability_bars.pdf", {**names, **short_names}, n_bootstrap=n_boot_)
    viz.compact_tables(bridges, stab_df, pl.read_csv(out / "subgroup_comparison.csv"), {**names, **short_names}, n_boot_, tbl)
else:
    print("stability_bridges.csv not found (run 11_stability.py); skipping fig_bridge_stability")

# tables
big = summary.filter(pl.col("size") >= min_size).sort("size", descending=True)
viz.latex_table(
    big.select(pl.col("community").alias("id"), "size", pl.col("top_chapter").alias("top chapter"),
               pl.col("chapter_purity").alias("purity"), "density", pl.col("avg_clustering").alias("clustering"),
               pl.col("internal_external_ratio").alias("int/ext"), pl.col("top5_codes").alias("top-5 members")),
    tbl / "communities.tex", "Leiden communities of the main graph (size $\\geq$ 5). Purity = share of the top ICD-10 chapter.",
    "tab:communities", {"purity": "{:.2f}", "density": "{:.2f}", "clustering": "{:.2f}", "int/ext": "{:.2f}"})

top_bt = cent.sort("betweenness", descending=True).head(15).select(
    "category", pl.col("name").str.slice(0, 40), pl.col("community").alias("comm"), "betweenness",
    pl.col("bt_null_percentile").alias("null pct"), pl.col("bt_null_z").alias("null z"), pl.col("participation").alias("P"),
    pl.col("within_module_z").alias("z_in"), "degree", "prevalence", pl.col("degree_residual_z").alias("resid z"))
viz.latex_table(top_bt, tbl / "betweenness.tex", "Top 15 categories by phi-weighted betweenness with rewired-null percentile, Guimer\\`a--Amaral participation $P$ and within-module $z$, and degree-vs-prevalence residual.",
                "tab:betweenness", {"betweenness": "{:.4f}", "null pct": "{:.2f}", "null z": "{:.1f}", "P": "{:.2f}", "z_in": "{:.2f}", "prevalence": "{:.2f}", "resid z": "{:.2f}"})

rows = []
pct_min = float(cfg["centrality"].get("bridge_null_percentile_min", 0.95))
n_boot = int(cfg["stability"]["n_bootstrap"])
stab_path = out / "stability_bridges.csv"
stab = {(r["pair"], r["category"]): r["share_in_top10"] for r in pl.read_csv(stab_path).iter_rows(named=True)} if stab_path.exists() else {}
for pair in bridges["pair"].unique().sort():
    s = bridges.filter(pl.col("pair") == pair)
    if "bridge_pass" in s.columns:
        s = s.filter(pl.col("bridge_pass"))
    for r in s.head(8).iter_rows(named=True):
        row = {"pair": pair.replace("_", " ").replace("-", " / "), "category": r["category"], "name": str(r["name"])[:30],
               "bt(A,B)": r["bridge_betweenness"], "null pct": r.get("bridge_null_percentile"), "null z": r.get("bridge_null_z")}
        if stab:  # bootstrap: resamples in which the category ranks in the pair's top-10
            k = stab.get((pair, r["category"]))
            row[f"top-10 in {n_boot} boots"] = "--" if k is None else f"{int(round(k * n_boot))}/{n_boot}"
        rows.append(row)
viz.latex_table(pl.DataFrame(rows), tbl / "bridges.tex",
                f"Bridge categories per cluster pair that beat the rewired null (percentile $\\geq$ {pct_min}): betweenness restricted to shortest paths between the two clusters, null percentile, null z"
                + (f", and the number of {n_boot} patient bootstraps in which the category stays in the pair's top-10." if stab else "."),
                "tab:bridges", {"bt(A,B)": "{:.4f}", "null pct": "{:.2f}", "null z": "{:.1f}"})
prec_path = out / "cluster_precedence.csv"
if prec_path.exists():
    prec = pl.read_csv(prec_path)
    viz.latex_table(prec.select(pl.col("from").str.replace_all("_", " ").alias("cluster A"), pl.col("to").str.replace_all("_", " ").alias("cluster B"),
                                pl.col("n_both").alias("patients in both"),
                                pl.col("n_a_first").alias("A first"), pl.col("n_b_first").alias("B first"), pl.col("n_same_day").alias("same day"),
                                pl.col("median_gap_days_a_to_b").alias("median days A to B"), pl.col("p_sign_test").alias("sign-test p")),
                    tbl / "precedence.tex", "Order of first recorded diagnosis between target clusters among patients who enter both (directed, sequential view).",
                    "tab:precedence", {"median days A to B": "{:.0f}", "sign-test p": "{:.2g}"})

sc = pl.read_csv(out / "subgroup_comparison.csv")
viz.latex_table(sc.select("subgroup", pl.col("n_patients").alias("patients"), pl.col("n_edges").alias("edges"),
                          pl.col("n_isolates").alias("isolates"), pl.col("avg_clustering").alias("clustering"),
                          pl.col("ari_vs_main").alias("ARI"), pl.col("ari_vs_main_non_isolates").alias("ARI (non-isol.)"),
                          pl.col(f"top{k_top}_bt_overlap_vs_main").alias(f"top-{k_top}")),
                tbl / "subgroups.tex", f"Subgroup graphs on the main node set: edges kept at the fixed thresholds, isolates, clustering, ARI against the pooled partition (all nodes; non-isolated nodes) and top-{k_top} betweenness overlap with the pooled graph.",
                "tab:subgroups", {"clustering": "{:.2f}", "ARI": "{:.2f}", "ARI (non-isol.)": "{:.2f}", f"top-{k_top}": "{:.2f}"})
pw = pl.read_csv(out / "subgroup_pairwise.csv")
viz.latex_table(pw.select("comparison", pl.col("ari").alias("ARI"), pl.col("ari_non_isolates").alias("ARI (non-isolates)"), pl.col(f"top{k_top}_bt_overlap").alias(f"top-{k_top} overlap"),
                          pl.col("edges_only_a").alias("edges only A"), pl.col("edges_only_b").alias("edges only B"), pl.col("edges_both").alias("shared"),
                          pl.col("jaccard_edges").alias("edge Jaccard")),
                tbl / "subgroup_pairwise.tex", "Pairwise subgroup comparison.", "tab:pairwise",
                {"ARI": "{:.2f}", "ARI (non-isolates)": "{:.2f}", f"top-{k_top} overlap": "{:.2f}", "edge Jaccard": "{:.2f}"})
sens = pl.read_csv(out / "sensitivity_summary.csv")
viz.latex_table(sens.select("variant", pl.col("n_patients").alias("patients"), pl.col("n_edges").alias("edges"), pl.col("n_isolates").alias("isolates"),
                            pl.col("n_communities").alias("communities"),
                            pl.col("ari_vs_main").alias("ARI vs main"), pl.col("ari_vs_main_non_isolates").alias("ARI (non-isol.)"),
                            pl.col(f"top{k_top}_bt_overlap_vs_main").alias(f"top-{k_top} overlap"), pl.col("edge_jaccard_vs_main").alias("edge Jaccard")),
                tbl / "sensitivity.tex", "Sensitivity runs compared with the main graph and partition.", "tab:sensitivity",
                {"ARI vs main": "{:.2f}", "ARI (non-isol.)": "{:.2f}", f"top-{k_top} overlap": "{:.2f}", "edge Jaccard": "{:.2f}"})
td = pl.read_csv(out / "target_definitions.csv")
trows = []
for k in cfg["centrality"]["target_clusters"]:
    t = td.filter(pl.col("target") == k)
    core = t.filter("in_core")["category"].to_list(); comm_ = t.filter("in_community")["category"].to_list(); chap = t.filter("in_chapter_block")["category"].to_list()
    trows.append({"target": k.replace("_", " "), "anchors": " ".join(cfg["centrality"]["target_clusters"][k]), "core": len(core), "community": len(comm_), "chapter block": len(chap),
                  "core and chapter": len(set(core) & set(chap)), "core members": " ".join(sorted(core))[:120]})
viz.latex_table(pl.DataFrame(trows), tbl / "targets.tex",
                f"Target definitions: stable core (co-assigned with the anchors in $\\geq$ {cfg['centrality']['core_min_coassignment']} of {cfg['centrality']['core_n_seeds']} Leiden runs), single-seed community, and a-priori ICD-10 chapter block.",
                "tab:targets")
stats = pl.read_csv(out / "graph_stats.csv").filter(pl.col("graph") == "main").select("stat", "value")
stats = stats.with_columns(pl.when(pl.col("value") == pl.col("value").round(0)).then(pl.col("value").cast(pl.Int64).cast(pl.Utf8))
                           .otherwise(pl.col("value").round(3).cast(pl.Utf8)).alias("value"))
viz.latex_table(stats, tbl / "graph_stats.tex", "Main graph statistics.", "tab:stats")

print("figures:", ", ".join(p.name for p in sorted(out.glob("fig_*.pdf"))))
print("tables: ", ", ".join(p.name for p in sorted(tbl.glob("*.tex"))))
print(f"({time.time()-t0:.0f}s)")
