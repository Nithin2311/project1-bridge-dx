"""Step 1: cohort, patient split, node list.

Reads condition_occurrence / death / concept through DuckDB, applies the rules in config/option_a.yaml,
writes outputs/patient_category.parquet (all cohort patients), outputs/patient_split.csv (train/val/test),
outputs/nodes.csv (categories with prevalence >= min_prevalence among TRAINING patients), outputs/cohort_funnel.csv.
Rerunnable from the raw CSVs: `uv run python scripts/01_cohort.py`.
"""
from __future__ import annotations

import time

import polars as pl

from bridgedx import cohort, config, data, split

PROPOSAL = {"cohort": 4131, "nodes": 476}  # numbers from "Project 1 Ideas proposal.docx" (Oct 8)

cfg = config.load()
out = config.outputs_dir(cfg)
con = data.connect(cfg)
t0 = time.time()

funnel = cohort.cohort_funnel(con, cfg)
funnel.write_csv(out / "cohort_funnel.csv")
print("cohort funnel")
for step, pats, rows in funnel.iter_rows():
    print(f"  {step:48s} patients={pats:>6,}" + (f"  rows={rows:>10,}" if rows is not None else ""))

pc = cohort.build_patient_category(con, cfg)
pc.write_parquet(out / "patient_category.parquet")
n_pat = pc["person_id"].n_unique()
per_pat = pc.group_by("person_id").len()["len"]
print(f"\ncohort: {n_pat:,} patients with >= {cfg['cohort']['min_distinct_categories']} distinct categories "
      f"(era >= {cfg['cohort']['era_start']}); {pc.height:,} (patient, category) pairs; "
      f"{pc['category'].n_unique():,} distinct categories")
print(f"  categories per patient: median {per_pat.median():.0f}, IQR {per_pat.quantile(0.25):.0f}-{per_pat.quantile(0.75):.0f}, max {per_pat.max()}")

sp = split.assign_split(pc["person_id"].unique().to_list(), cfg)
sp.write_csv(out / "patient_split.csv")
counts = sp.group_by("split").len().sort("split")
print(f"\nsplit (source {sp['split_source'][0]}): " + ", ".join(f"{r['split']}={r['len']:,}" for r in counts.iter_rows(named=True))
      + (f"; {int((sp['split_source'] == 'missing').sum())} cohort patients missing from the file -> own split" if "missing" in sp["split_source"].to_list() else ""))
main_split = "+".join(split.main_splits(cfg))
pc_train = pc.filter(pl.col("person_id").is_in(sp.filter(pl.col("split").is_in(split.main_splits(cfg)))["person_id"].to_list()))

nodes = cohort.build_nodes(pc_train, con, cfg)
nodes.write_csv(out / "nodes.csv")
n_flag = int(nodes["flagged"].sum())
print(f"\nnodes (prevalence >= {cfg['nodes']['min_prevalence']:.0%} of {pc_train['person_id'].n_unique():,} {main_split} patients): {nodes.height} categories; "
      f"{n_flag} flagged ({'/'.join(cfg['nodes']['flag_chapters'])} chapters); "
      f"{int(nodes['name'].eq(nodes['category']).sum())} without a concept name; "
      f"{nodes.filter(~pl.col('category').str.slice(0, 1).is_in(cfg['nodes']['main_graph_excludes_chapters'])).height} outside excluded chapters {cfg['nodes']['main_graph_excludes_chapters']}")
print("  by chapter:")
for ch, n in nodes.group_by("chapter").len().sort("len", descending=True).iter_rows():
    print(f"    {ch:36s} {n:>4}")
print("  top 10 by prevalence: " + ", ".join(f"{r['category']} {r['prevalence']:.2f}" for r in nodes.head(10).iter_rows(named=True)))

print(f"\nvs proposal: cohort {n_pat:,} (proposal {PROPOSAL['cohort']:,}); nodes {nodes.height} on the training split (proposal {PROPOSAL['nodes']} on all patients)")
print(f"wrote patient_category.parquet, patient_split.csv, nodes.csv, cohort_funnel.csv to {out}  ({time.time()-t0:.0f}s)")
