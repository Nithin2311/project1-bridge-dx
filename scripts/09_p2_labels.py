"""Step 9: Project 2 landmark label tables.

One row per (patient, index date, target, target definition, horizon) with eligibility, positive and censoring
flags (rules in config project2 and bridgedx.labels). Targets = the three definitions in target_definitions.csv
(stable core, single-seed community, chapter block). Writes outputs/p2_labels.parquet and outputs/p2_label_counts.csv.
Rerunnable: `uv run python scripts/09_p2_labels.py` (after 07).
"""
from __future__ import annotations

import time

import numpy as np
import polars as pl

from bridgedx import config, data, labels as lb

cfg = config.load()
out = config.outputs_dir(cfg)
t0 = time.time()
p2 = cfg["project2"]
min_pos = int(p2["min_positives"])

pc = pl.read_parquet(out / "patient_category.parquet")
pre_path = out / "patient_category_preera_all.parquet"
pre = pl.read_parquet(pre_path) if pre_path.exists() else pl.DataFrame({"person_id": [], "category": []}, schema={"person_id": pl.Int64, "category": pl.Utf8})
if not pre_path.exists():
    print("WARNING: patient_category_preera_all.parquet missing (run 07); pre-2015 history NOT used for washout")
sp = pl.read_csv(out / "patient_split.csv")
split = dict(zip(sp["person_id"], sp["split"]))
td = pl.read_csv(out / "target_definitions.csv")
targets = {}
for definition, col in [("core", "in_core"), ("community", "in_community"), ("chapter", "in_chapter_block")]:
    if definition in p2.get("target_definitions", ["core", "community", "chapter"]):
        targets[definition] = {k: set(td.filter((pl.col("target") == k) & pl.col(col))["category"].to_list()) for k in cfg["centrality"]["target_clusters"]}

con = data.connect(cfg)
con.register("_pc", pc.to_arrow())
patients = con.execute("""
    select p.person_id, p.first_era, r.last_dx, d.death_DATE as death
    from (select person_id, min(first_date) as first_era from _pc group by 1) p
    join (select co.person_id, max(co.condition_start_DATE) as last_dx from condition_occurrence co
          left join death d on d.person_id = co.person_id
          where d.death_DATE is null or co.condition_start_DATE <= d.death_DATE group by 1) r using (person_id)
    left join death d using (person_id)
""").pl()
enc = con.execute("""select co.person_id, co.condition_start_DATE as dt from condition_occurrence co left join death d on d.person_id = co.person_id
                     where co.person_id in (select person_id from _pc) and (d.death_DATE is null or co.condition_start_DATE <= d.death_DATE)""").pl()
encounters = {k: np.array(sorted(v)) for k, v in enc.group_by("person_id").agg(pl.col("dt")).rows()}

labels = lb.build_landmark_labels(cfg, pc, pre, patients, encounters, targets, split)
labels.write_parquet(out / "p2_labels.parquet")
summary = lb.summarize(labels, min_pos)
summary.write_csv(out / "p2_label_counts.csv")
grid = lb.landmark_grid(cfg, patients["last_dx"].max())
print(f"landmark grid: {len(grid)} index dates from {grid[0]} to {grid[-1]} every {p2['grid_step_days']} d; history >= {p2['min_history_days']} d; "
      f"encounter within {p2['require_encounter_days']} d; washout uses era + {pre['person_id'].n_unique():,} patients' bridged pre-era rows")
print(f"{labels.height:,} rows, {labels['person_id'].n_unique():,} patients\n")
for definition in targets:
    print(f"definition = {definition}")
    for h in [int(x) for x in p2["horizons_months"]]:
        sub = summary.filter((pl.col("definition") == definition) & (pl.col("horizon_months") == h))
        for k in cfg["centrality"]["target_clusters"]:
            s = sub.filter(pl.col("target") == k)
            tot = s.filter(pl.col("split") == "all").select("rows", "patients", "positives", "censored").row(0)
            by = ", ".join(f"{r['split']} {r['positives']}" for r in s.filter(pl.col("split") != "all").sort("split").iter_rows(named=True))
            flag = "ok " if tot[2] >= min_pos else "LOW"
            print(f"  {flag} {h:2d} mo {k:18s} rows={tot[0]:>6,} patients={tot[1]:>5,} positives={tot[2]:>4,} censored={tot[3]:>5,}  by split: {by}")
print(f"\nwrote p2_labels.parquet, p2_label_counts.csv  ({time.time()-t0:.0f}s)")
