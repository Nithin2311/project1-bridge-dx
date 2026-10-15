"""Step 7: sensitivity runs.

  (a) ICD-9-era appendix graph (rows before era_start, bridged SNOMED -> ICD-10-CM category) on the main
      node set: Leiden at the main resolution, ARI vs main, top-20 betweenness overlap, edge Jaccard.
  (b) Follow-up strata: cohort tertiles by number of distinct categories, same comparisons.
  (c) Summary table joining (a), (b) with the step-4 full-graph and resolution-sweep results.
Rerunnable: `uv run python scripts/07_sensitivity.py` (after 05). ~5 min (concept_relationship scan ~1 min).
"""
from __future__ import annotations

import time

import networkx as nx
import polars as pl

from bridgedx import centrality as ce
from bridgedx import communities as cm
from bridgedx import config, data, edges as ed, graph as gr, sensitivity as se, split, subgroups as sg

cfg = config.load()
out = config.outputs_dir(cfg)
t0 = time.time()
k_top = int(cfg["subgroups"]["comparison"]["top_k"])
seed0 = cfg["communities"]["seeds"][0]

g_main = nx.read_graphml(out / "graph_main.graphml")
nodes = pl.read_csv(out / "nodes.csv").filter(pl.col("category").is_in(list(g_main.nodes)))
cats = nodes["category"].to_list()
comm = pl.read_csv(out / "communities.csv")
part_main = dict(zip(comm["category"], comm["leiden"]))
cent_main = pl.read_csv(out / "centrality.csv")
top_main = cent_main.sort("betweenness", descending=True)["category"].to_list()
main_ids = split.train_ids(cfg)
pc = pl.read_parquet(out / "patient_category.parquet").filter(pl.col("person_id").is_in(main_ids))  # main-graph patients only
main_edges = set(map(frozenset, g_main.edges))
print(f"sensitivity runs on {len(main_ids):,} {'+'.join(split.main_splits(cfg))} patients")


def compare(name: str, sub_pc: pl.DataFrame, members: list, note: str = "") -> dict:
    """Build a graph on the main node set from sub_pc, Leiden at the main resolution, compare with main."""
    t1 = time.time()
    pairs = ed.weights(ed.fisher_filter(ed.pair_counts(sub_pc, categories=cats), cfg))
    pairs.write_csv(out / f"sensitivity_{name}_edges.csv")
    g = gr.build(nodes, pairs, weight=cfg["edges"]["weight"])
    gr.write_graphml(g, out / f"graph_{name}.graphml")
    p = cm.leiden(g, cfg, seed=seed0)
    bt = ce.weighted_betweenness(g)
    top = sorted(bt, key=lambda v: -bt[v])
    st = gr.basic_stats(g)
    e = set(map(frozenset, g.edges))
    non_iso = [v for v in g.nodes if g.degree(v) > 0]
    row = {"variant": name, "n_patients": len(members), "n_edges": g.number_of_edges(), "density": st["density"],
           "avg_clustering": st["avg_clustering"], "n_isolates": st["n_isolates"],
           "n_communities": len(set(p.values())), "q_standard": cm.modularity_standard(g, p),
           "ari_vs_main": sg.compare_partitions(part_main, p),
           "ari_vs_main_non_isolates": sg.compare_partitions(part_main, p, non_iso),
           f"top{k_top}_bt_overlap_vs_main": sg.connector_overlap(top_main, top, k_top),
           "edge_jaccard_vs_main": len(e & main_edges) / len(e | main_edges) if e | main_edges else 0.0,
           "note": note}
    cm.partition_table(g, {"leiden": p}).write_csv(out / f"communities_{name}.csv")
    print(f"  {name:22s} n={len(members):,} edges={g.number_of_edges():,} dens={st['density']:.3f} isol={st['n_isolates']} comms={row['n_communities']} "
          f"Q={row['q_standard']:.3f} ARI={row['ari_vs_main']:.3f} (non-iso {row['ari_vs_main_non_isolates']:.3f}) top{k_top}={row[f'top{k_top}_bt_overlap_vs_main']:.2f} "
          f"edgeJ={row['edge_jaccard_vs_main']:.2f}  ({time.time()-t1:.0f}s)")
    return row


rows = []
# (a) ICD-9 era
con = data.connect(cfg)
con.register("_pc", pc.to_arrow())
print("building SNOMED -> ICD-10-CM bridge (empirical era map, vocab + ancestor fallback) ...", flush=True)
excl = set(cfg["nodes"].get("main_graph_excludes_chapters", []))
allowed = [c for c in pl.read_csv(out / "nodes.csv")["category"].to_list() if c[0] not in excl]
bridge = se.build_snomed_bridge(con, cfg, allowed_fallback_categories=allowed)
bridge.write_csv(out / "sensitivity_snomed_bridge.csv")
pc9_all, cov = se.icd9_era_patient_category(con, cfg, min_categories=1)
pc9_all.write_parquet(out / "patient_category_preera_all.parquet")   # washout table for Project 2 labels: every patient, every bridged pre-era category
pc9 = pc9_all.filter(pl.col("person_id").is_in(main_ids))
keep9 = pc9.group_by("person_id").len().filter(pl.col("len") >= int(cfg["cohort"]["min_distinct_categories"]))["person_id"].to_list()
pc9 = pc9.filter(pl.col("person_id").is_in(keep9))
pc9.write_parquet(out / "patient_category_icd9era.parquet")
n9 = cov["n_rows"]
print(f"ICD-9 era (< {cfg['cohort']['era_start']}): {n9:,} rows, {cov['n_patients']:,} patients; hand map {cov['rows_manual']/n9:.1%}, bridged empirical "
      f"{cov['rows_empirical']/n9:.1%} (mean purity {cov['empirical_mean_purity']:.3f}, {cov['empirical_concepts']:,} concepts), "
      f"vocab {cov['rows_vocab']/n9:.1%}, ancestor {cov['rows_ancestor']/n9:.1%}, unbridged {cov['rows_unbridged']/n9:.1%} "
      f"(concept_id 0: {cov['rows_unmapped_concept']/n9:.2%})")
chk = con.execute("""
    select distinct co.condition_source_value as icd9, coalesce(m.category, b.category) as category, case when m.category is not null then 'manual' else b.source end as source
    from condition_occurrence co left join _manual m on m.sv = upper(trim(co.condition_source_value))
    left join snomed_bridge b on b.snomed_id = co.condition_concept_id
    where co.condition_source_value in ('300.00','401.9','250.00','202.80','585.9','272.4','311','496','263.9','600.00')
    order by 1""").pl()
print("  spot checks (ICD-9 -> category): " + "; ".join(f"{r['icd9']}->{r['category']} ({r['source'][:3]})" for r in chk.iter_rows(named=True)))
labels9 = pl.read_csv(out / "subgroup_labels.csv").select("person_id", "sex")
men_preg = pc9_all.join(labels9, on="person_id").filter((pl.col("sex") == "male") & pl.col("category").str.starts_with("O"))["person_id"].n_unique()
print(f"  men with pregnancy (O) codes in the bridged table (all patients): {men_preg}")
pats9 = pc9["person_id"].unique().to_list()
both = len(set(pats9) & set(pc["person_id"].unique().to_list()))
print(f"  ICD-9-era cohort (>= {cfg['cohort']['min_distinct_categories']} categories): {len(pats9):,} patients, {both:,} also in the main cohort; "
      f"{pc9.filter(pl.col('category').is_in(cats))['category'].n_unique()} of {len(cats)} main nodes present")
pl.DataFrame([cov]).write_csv(out / "sensitivity_icd9_bridge_coverage.csv")
rows.append(compare("icd9era", pc9, pats9, note="rows before era_start bridged via SNOMED; same node set"))

# (b) in-era temporal split
split = str(cfg["sensitivity"]["temporal_split_date"])
print(f"\nin-era temporal split at {split} (patients with rows in the window; same node set):")
pc_rows = con.execute(f"""
    select co.person_id, m.category, co.condition_start_DATE as dt
    from condition_occurrence co join cat_map m on m.sv = co.condition_source_value
    left join death d on d.person_id = co.person_id
    where co.condition_start_DATE >= date '{cfg["cohort"]["era_start"]}'
      and (d.death_DATE is null or co.condition_start_DATE <= d.death_DATE)
      and co.person_id in (select distinct person_id from _pc)
""").pl()
for name, cond in [(f"era_before_{split[:4]}", pl.col("dt") < pl.lit(split).str.to_date()),
                   (f"era_from_{split[:4]}", pl.col("dt") >= pl.lit(split).str.to_date())]:
    w = pc_rows.filter(cond).group_by("person_id", "category").agg(pl.col("dt").min().alias("first_date"))
    keep = w.group_by("person_id").len().filter(pl.col("len") >= int(cfg["cohort"]["min_distinct_categories"]))["person_id"].to_list()
    w = w.filter(pl.col("person_id").is_in(keep))
    rows.append(compare(name, w, keep, note=f"ICD-10-era rows {'before' if 'before' in name else 'on/after'} {split}, >= 5 categories in window"))
parts_t = {n: dict(zip(pl.read_csv(out / f"communities_{n}.csv")["category"], pl.read_csv(out / f"communities_{n}.csv")["leiden"]))
           for n in [f"era_before_{split[:4]}", f"era_from_{split[:4]}"]}
a, b = list(parts_t)
print(f"  ARI({a}, {b}) = {sg.compare_partitions(parts_t[a], parts_t[b]):.3f}")

# (c) follow-up strata
print("\nfollow-up strata (tertiles of distinct-category count; power-limited, see non-isolate ARI):")
for name, members in se.strata_by_n_categories(pc, 3).items():
    rows.append(compare(name, pc.filter(pl.col("person_id").is_in(members)), members, note="cohort tertile by n distinct categories"))

# (d) summary incl. step-4 variants
res_tags = [f"res{float(r):g}" for r in cfg["communities"].get("resolution_sensitivity", [])]
for tag, label in [("allchapters", "train+val patients, all chapters incl. R/Z/O"), ("allpatients", "all cohort patients incl. test split")] + [(t, f"main graph, res {t[3:]}") for t in res_tags]:
    f = out / f"communities_{tag}.csv"
    if f.exists():
        c2 = pl.read_csv(f)
        p2 = dict(zip(c2["category"], c2["leiden"]))
        s2 = pl.read_csv(out / f"communities_summary_{tag}.csv")
        rows.append({"variant": tag, "n_patients": pc["person_id"].n_unique(), "n_edges": None, "density": None,
                     "avg_clustering": None, "n_isolates": None, "n_communities": s2.height, "q_standard": None,
                     "ari_vs_main": sg.compare_partitions(part_main, p2), "ari_vs_main_non_isolates": None,
                     f"top{k_top}_bt_overlap_vs_main": None, "edge_jaccard_vs_main": None, "note": label})
summary = pl.DataFrame(rows)
summary.write_csv(out / "sensitivity_summary.csv")
print(f"\nwrote sensitivity_summary.csv, sensitivity_<variant>_edges.csv, graph_<variant>.graphml, communities_<variant>.csv, "
      f"patient_category_icd9era.parquet, sensitivity_icd9_bridge_coverage.csv  ({time.time()-t0:.0f}s)")
