"""Step 10: Project 2 input tables (non-ML).

  outputs/p2/node_features.csv          one row per main-graph node: structural, consensus, bridge, acuity, chronic,
                                        demographic and chapter descriptors, backbone degree
  outputs/p2/graph_backbone_topk.graphml top-k per-node phi backbone of the main graph
  outputs/p2/patient_index_subgraphs.csv one row per (patient, index date) in p2_labels: codes and induced edges per
                                        lookback window on the full and backbone graphs, 1-hop neighbour count
Rerunnable: `uv run python scripts/10_p2_tables.py` (after 09). ~3 min.
"""
from __future__ import annotations

import time

import networkx as nx
import numpy as np
import polars as pl

from bridgedx import cohort as _c, config, data, split, subgroups as sg, viz

cfg = config.load()
out = config.outputs_dir(cfg)
p2d = out / "p2"
p2d.mkdir(exist_ok=True)
pt = cfg["project2_tables"]
t0 = time.time()

g = nx.read_graphml(out / "graph_main.graphml")
nodes = list(g.nodes)
main_ids = split.train_ids(cfg)
con = data.connect(cfg)
_c._register_category_map(con)
con.register("_nodes", pl.DataFrame({"category": nodes}).to_arrow())
con.register("_main", pl.DataFrame({"person_id": main_ids}).to_arrow())

# ---- node features ----
cent = pl.read_csv(out / "centrality.csv")
td = pl.read_csv(out / "target_definitions.csv")
br = pl.read_csv(out / "bridges.csv")
comm = pl.read_csv(out / "communities.csv").select("category", pl.col("leiden").alias("community_seed0"), pl.col("louvain").alias("community_louvain"))
pr = nx.pagerank(g, weight="weight")
k = int(pt["backbone_top_k"])
bb = viz.sparsify_top_k(g, k)
nx.write_graphml(bb, p2d / "graph_backbone_topk.graphml")
feat = cent.select("category", "name", "chapter", "prevalence", "degree", "weighted_degree", "betweenness", "bt_null_percentile",
                   "closeness", "eigenvector", "participation", "within_module_z", "degree_residual_z")
feat = feat.join(comm, on="category", how="left").with_columns(
    pl.Series("pagerank", [pr[v] for v in feat["category"]]),
    pl.Series("backbone_degree", [bb.degree(v) for v in feat["category"]]),
    pl.col("category").str.slice(0, 1).alias("chapter_letter"),
    pl.col("category").map_elements(lambda c: _c.chapter_of(c), return_dtype=pl.Utf8).alias("chapter_name"))
for tname in cfg["centrality"]["target_clusters"]:
    for definition, col in [("core", "in_core"), ("community", "in_community"), ("chapter", "in_chapter_block")]:
        members = set(td.filter((pl.col("target") == tname) & pl.col(col))["category"].to_list())
        feat = feat.with_columns(pl.col("category").is_in(list(members)).alias(f"target_{tname}_{definition}"))
passing = br.filter(pl.col("bridge_pass")).group_by("category").agg(pl.col("pair").str.join("|").alias("bridge_pairs"))
feat = feat.join(passing, on="category", how="left").with_columns(pl.col("bridge_pairs").is_not_null().alias("bridge_survivor"))

# acuity: share of rows dated inside an inpatient stay of the same patient (main patients, era)
ipid = int(pt["inpatient_visit_concept_id"])
ip = con.execute(f"""
  with ipv as (select person_id, visit_start_DATE s, coalesce(visit_end_DATE, visit_start_DATE) e from visit_occurrence where visit_concept_id = {ipid}),
  rows as (select co.person_id, m.category, co.condition_start_DATE dt from condition_occurrence co join cat_map m on m.sv = co.condition_source_value
           where co.condition_start_DATE >= date '{cfg["cohort"]["era_start"]}' and m.category in (select category from _nodes)
             and co.person_id in (select person_id from _main)),
  flagged as (select r.category, exists(select 1 from ipv where ipv.person_id = r.person_id and r.dt between ipv.s and ipv.e) as inpt from rows r)
  select category, count(*) n_rows_main, avg(case when inpt then 1 else 0 end) inpatient_share from flagged group by 1""").pl()
feat = feat.join(ip, on="category", how="left").with_columns((pl.col("inpatient_share") >= float(pt["acute_share_threshold"])).alias("acute_node"))

# chronic: HCUP CCIR share of a 3-char category's codes flagged chronic (1), excluding 'no determination' (9)
ccir_path = config.REPO_ROOT / pt["ccir_file"]
if ccir_path.exists():
    import csv
    recs = []
    with open(ccir_path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.reader(fh):
            if len(row) < 3 or not row[0].strip("'\" ").isalnum() or row[0].strip("'\" ") in ("ICD-10-CM CODE",):
                continue
            code = row[0].strip("'\" ")
            try:
                chronic = int(row[-1].strip("'\" "))
            except ValueError:
                continue
            recs.append((code[:3], chronic))
    cc = pl.DataFrame(recs, schema=["category", "chronic"], orient="row")
    cc = cc.filter(pl.col("chronic") != 9).group_by("category").agg(pl.col("chronic").mean().alias("ccir_chronic_share"), pl.len().alias("ccir_n_codes"))
    print(f"CCIR: {len(recs):,} codes -> {cc.height} categories")
    feat = feat.join(cc, on="category", how="left").with_columns((pl.col("ccir_chronic_share") >= 0.5).alias("ccir_chronic"))
else:
    print(f"WARNING: CCIR file not found at {ccir_path}; chronic flag skipped")

# demographics: mean age at first occurrence and female share among main patients with the code
lab = pl.read_csv(out / "subgroup_labels.csv", try_parse_dates=True).select("person_id", "sex", "age_at_index", "index_date")
pc = pl.read_parquet(out / "patient_category.parquet").filter(pl.col("person_id").is_in(main_ids) & pl.col("category").is_in(nodes))
pc = pc.join(lab, on="person_id").with_columns(
    (pl.col("age_at_index") + (pl.col("first_date") - pl.col("index_date")).dt.total_days() / 365.25).alias("age_at_code"))
dem = pc.group_by("category").agg(pl.col("age_at_code").mean().alias("mean_age_first_occurrence"),
                                  (pl.col("sex") == "female").mean().alias("female_share"), pl.len().alias("n_patients_main"))
feat = feat.join(dem, on="category", how="left").sort("category")
feat.write_csv(p2d / "node_features.csv")
print(f"node_features.csv: {feat.height} nodes x {feat.width} columns; acute nodes {int(feat['acute_node'].sum())}; "
      f"chronic (CCIR) {int(feat['ccir_chronic'].fill_null(False).sum()) if 'ccir_chronic' in feat.columns else 'n/a'}; bridge survivors {int(feat['bridge_survivor'].sum())}; "
      f"backbone k={k}: {bb.number_of_edges():,} edges")

# ---- patient-index subgraph stats ----
lbl = pl.read_parquet(out / "p2_labels.parquet").select("person_id", "index_date", "split").unique().sort("person_id", "index_date")
era = con.execute(f"""
  select co.person_id, m.category, min(co.condition_start_DATE) as dt
  from condition_occurrence co join cat_map m on m.sv = co.condition_source_value
  left join death d on d.person_id = co.person_id
  where co.condition_start_DATE >= date '{cfg["cohort"]["era_start"]}' and (d.death_DATE is null or co.condition_start_DATE <= d.death_DATE)
    and m.category in (select category from _nodes)
  group by 1, 2""").pl()
by_pat = {(k[0] if isinstance(k, tuple) else k): (grp["category"].to_list(), np.array(grp["dt"].to_list())) for k, grp in era.group_by("person_id")}
# NB: uses FIRST occurrence per category in the era; a code first seen before the window counts as present (chronic history)
lookbacks = [int(x) for x in pt["lookback_days"]]
rows = []
for r in lbl.iter_rows(named=True):
    cats, dts = by_pat.get(r["person_id"], ([], np.array([])))
    row = {"person_id": r["person_id"], "index_date": r["index_date"], "split": r["split"]}
    for lb_ in lookbacks:
        if len(cats):
            lo = r["index_date"] - __import__("datetime").timedelta(days=lb_) if lb_ else None
            sel = [c for c, d in zip(cats, dts) if d <= r["index_date"] and (lo is None or d > lo)]
        else:
            sel = []
        sub = g.subgraph(sel); subb = bb.subgraph(sel)
        nb = set()
        for v in sel:
            nb.update(g.neighbors(v))
        tag = f"lb{lb_}" if lb_ else "lball"
        row.update({f"{tag}_n_codes": len(sel), f"{tag}_edges_full": sub.number_of_edges(), f"{tag}_edges_backbone": subb.number_of_edges(),
                    f"{tag}_isolates_full": sum(1 for v in sel if sub.degree(v) == 0), f"{tag}_hop1_neighbours": len(nb - set(sel))})
    rows.append(row)
sgs = pl.DataFrame(rows)
sgs.write_csv(p2d / "patient_index_subgraphs.csv")
for lb_ in lookbacks:
    tag = f"lb{lb_}" if lb_ else "lball"
    n = sgs[f"{tag}_n_codes"]; e = sgs[f"{tag}_edges_full"]
    print(f"  lookback {tag:6s}: codes median {n.median():.0f} (IQR {n.quantile(0.25):.0f}-{n.quantile(0.75):.0f}); edges median {e.median():.0f}; "
          f"0 edges {(e == 0).mean():.1%}; <3 codes {(n < 3).mean():.1%}; >=10 codes {(n >= 10).mean():.1%}")
print(f"patient_index_subgraphs.csv: {sgs.height:,} (patient, index) rows  ({time.time()-t0:.0f}s)")
