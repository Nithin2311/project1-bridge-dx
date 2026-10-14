"""Step 9: community-entry label counts for the Project 2 paragraph.

For each candidate index rule x horizon x target (the three target clusters and every main community with
>= 5 nodes): eligible = cohort patients with >= 1 diagnosis on/before index and no target diagnosis on/before
index; positive = first target diagnosis within (index, index + horizon]; censored (excluded) = no positive
and death or record end inside the horizon; negative = the rest.
Writes outputs/p2_label_counts.csv. Rerunnable: `uv run python scripts/09_p2_labels.py` (after 05).
"""
from __future__ import annotations

import datetime as dt
import time

import polars as pl

from bridgedx import centrality as ce
from bridgedx import config, data

cfg = config.load()
out = config.outputs_dir(cfg)
t0 = time.time()
p2 = cfg["project2"]
min_pos = int(p2["min_positives"])
horizons = sorted({int(p2["horizon_months"]), 12})
min_size = int(cfg["communities"].get("min_community_size_in_tables", 5))

pc = pl.read_parquet(out / "patient_category.parquet")
comm = pl.read_csv(out / "communities.csv")
partition = dict(zip(comm["category"], comm["leiden"]))
summary = pl.read_csv(out / "communities_summary.csv")
targets = ce.target_clusters(partition, cfg["centrality"]["target_clusters"])
for r in summary.filter(pl.col("size") >= min_size).iter_rows(named=True):
    targets[f"c{r['community']} {r['top_chapter'].split('/')[0]} ({r['top5_codes'].split()[0]})"] = {
        v for v, c in partition.items() if c == r["community"]}

con = data.connect(cfg)
con.register("_pc", pc.to_arrow())
pat = con.execute("""
    select p.person_id, p.first_dx, r.last_dx, d.death_DATE as death
    from (select person_id, min(first_date) as first_dx from _pc group by 1) p
    join (select person_id, max(condition_start_DATE) as last_dx from condition_occurrence group by 1) r using (person_id)
    left join death d using (person_id)
""").pl()

rules = {
    "history_1y": pat.with_columns((pl.col("first_dx") + pl.duration(days=365)).alias("index")),
    "calendar_2019": pat.with_columns(pl.lit(dt.date(2019, 1, 1)).alias("index")).filter(pl.col("first_dx") <= pl.col("index")),
}

rows = []
for rule, base in rules.items():
    for name, members in targets.items():
        first_t = (pc.filter(pl.col("category").is_in(list(members)))
                   .group_by("person_id").agg(pl.col("first_date").min().alias("first_target")))
        df = base.join(first_t, on="person_id", how="left")
        df = df.filter(pl.col("first_target").is_null() | (pl.col("first_target") > pl.col("index")))  # no prior target dx
        for h in horizons:
            end = pl.col("index") + pl.duration(days=int(round(h * 30.44)))
            d = df.with_columns(
                (pl.col("first_target").is_not_null() & (pl.col("first_target") <= end)).alias("positive"),
                ((pl.col("death").is_not_null() & (pl.col("death") <= end)) | (pl.col("last_dx") < end)).alias("ends_inside"),
            ).with_columns((~pl.col("positive") & pl.col("ends_inside")).alias("censored"))
            n_pos = int(d["positive"].sum())
            n_cen = int(d["censored"].sum())
            n_el = d.height
            rows.append({"rule": rule, "horizon_months": h, "target": name, "n_target_nodes": len(members),
                         "eligible": n_el, "positives": n_pos, "censored": n_cen, "negatives": n_el - n_pos - n_cen,
                         "positive_rate": n_pos / (n_el - n_cen) if n_el - n_cen else 0.0,
                         "enough": n_pos >= min_pos})
res = pl.DataFrame(rows)
res.write_csv(out / "p2_label_counts.csv")

print(f"cohort {pat.height:,}; record end = last diagnosis date; min positives {min_pos}\n")
for rule in rules:
    print(f"rule {rule}: {rules[rule].height:,} patients with an index date")
    for h in horizons:
        sub = res.filter((pl.col("rule") == rule) & (pl.col("horizon_months") == h)).sort("positives", descending=True)
        print(f"  horizon {h} months: {int(sub['enough'].sum())} of {sub.height} targets reach {min_pos} positives")
        for r in sub.iter_rows(named=True):
            flag = "ok " if r["enough"] else "LOW"
            print(f"    {flag} {r['target'][:38]:38s} eligible={r['eligible']:>5,} pos={r['positives']:>4,} "
                  f"cens={r['censored']:>4,} neg={r['negatives']:>5,} rate={r['positive_rate']:.2f}")
print(f"\nwrote {out/'p2_label_counts.csv'}  ({time.time()-t0:.0f}s)")
