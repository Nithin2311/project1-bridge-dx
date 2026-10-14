"""Step 1: cohort + node list.

Reads condition_occurrence / death / concept through DuckDB, applies the rules in config/option_a.yaml,
writes outputs/patient_category.parquet, outputs/nodes.csv, outputs/cohort_funnel.csv.
Rerunnable from the raw CSVs: `uv run python scripts/01_cohort.py`.
"""
from __future__ import annotations

import time

import polars as pl

from bridgedx import cohort, config, data

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

nodes = cohort.build_nodes(pc, con, cfg)
nodes.write_csv(out / "nodes.csv")
n_flag = int(nodes["flagged"].sum())
print(f"\nnodes: {nodes.height} categories with prevalence >= {cfg['nodes']['min_prevalence']:.0%}; "
      f"{n_flag} flagged ({'/'.join(cfg['nodes']['flag_chapters'])} chapters); "
      f"{int(nodes['name'].eq(nodes['category']).sum())} without a concept name")
print("  by chapter:")
for ch, n in nodes.group_by("chapter").len().sort("len", descending=True).iter_rows():
    print(f"    {ch:36s} {n:>4}")
print("  top 15 by prevalence:")
with pl.Config(tbl_rows=15, fmt_str_lengths=60, tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True):
    print(nodes.head(15))

print(f"\nvs proposal: cohort {n_pat:,} (proposal {PROPOSAL['cohort']:,}, diff {n_pat - PROPOSAL['cohort']:+,}); "
      f"nodes {nodes.height} (proposal {PROPOSAL['nodes']}, diff {nodes.height - PROPOSAL['nodes']:+})")
print(f"wrote {out/'patient_category.parquet'}, {out/'nodes.csv'}, {out/'cohort_funnel.csv'}  ({time.time()-t0:.0f}s)")
