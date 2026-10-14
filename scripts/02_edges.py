"""Step 2: edge list.

Reads outputs/patient_category.parquet + outputs/nodes.csv, tests every node pair, writes outputs/edges.csv
(all pairs, with `kept` flag) and prints the proposal's "raw count vs phi" comparison.
Rerunnable: `uv run python scripts/02_edges.py` (after 01).
"""
from __future__ import annotations

import time

import polars as pl

from bridgedx import config, edges

cfg = config.load()
out = config.outputs_dir(cfg)
e = cfg["edges"]
t0 = time.time()

pc = pl.read_parquet(out / "patient_category.parquet")
nodes = pl.read_csv(out / "nodes.csv")
cats = nodes["category"].to_list()
names = dict(zip(nodes["category"], nodes["name"]))

pairs = edges.pair_counts(pc, categories=cats)
print(f"{len(cats)} nodes, {pc['person_id'].n_unique():,} patients -> {pairs.height:,} pairs")
pairs = edges.fisher_filter(pairs, cfg)
pairs = edges.weights(pairs)
pairs.write_csv(out / "edges.csv")

kept = pairs.filter("kept")
print(f"\nfilter: p_bonf < {e['alpha']} (m = {pairs.height:,}), OR >= {e['min_odds_ratio']}, n_both >= {e['min_shared_patients']}")
print(f"  pairs with n_both >= {e['min_shared_patients']}:            {pairs.filter(pl.col('n_both') >= e['min_shared_patients']).height:>8,}")
print(f"  ... and Bonferroni-significant:        {pairs.filter((pl.col('n_both') >= e['min_shared_patients']) & (pl.col('p_bonf') < e['alpha'])).height:>8,}")
print(f"  ... and OR >= {e['min_odds_ratio']} (kept edges):        {kept.height:>8,}")
neg = pairs.filter((pl.col('p_bonf') < e['alpha']) & (pl.col('phi') < 0)).height
print(f"  significant negative associations (dropped by OR floor): {neg:,}")
print(f"  kept phi: min {kept['phi'].min():.3f}, median {kept['phi'].median():.3f}, max {kept['phi'].max():.3f}; "
      f"kept OR: median {kept['odds_ratio'].median():.2f}")
deg = pl.concat([kept.select(pl.col("u").alias("c")), kept.select(pl.col("v").alias("c"))]).group_by("c").len()
print(f"  nodes with >= 1 kept edge: {deg.height} of {len(cats)}; degree median {deg['len'].median():.0f}, max {deg['len'].max()}")


def show(df: pl.DataFrame, title: str) -> None:
    print(f"\n{title}")
    for i, r in enumerate(df.iter_rows(named=True), 1):
        print(f"  {i:>2}. {r['u']}-{r['v']}  {names[r['u']][:38]} / {names[r['v']][:38]}  "
              f"n={r['n_both']:,}  OR={r['odds_ratio']:.1f}  phi={r['phi']:.3f}  J={r['jaccard']:.2f}")


cols = ["u", "v", "n_both", "odds_ratio", "phi", "jaccard"]
show(kept.sort("n_both", descending=True).head(10).select(cols), "top 10 kept edges by raw shared-patient count")
show(kept.sort("phi", descending=True).head(10).select(cols), "top 10 kept edges by phi")

watch = [("O99", "Z37"), ("T86", "Z94"), ("I12", "N18"), ("A41", "R65")]
ranked = kept.sort("phi", descending=True).with_row_index("rank_phi", offset=1)
print("\nproposal's named phi pairs:")
for u, v in watch:
    r = ranked.filter(((pl.col("u") == u) & (pl.col("v") == v)) | ((pl.col("u") == v) & (pl.col("v") == u)))
    print(f"  {u}-{v}: " + (f"rank {r['rank_phi'][0]} by phi, n={r['n_both'][0]}, phi={r['phi'][0]:.3f}" if r.height else "not a kept edge / not in node set"))

print(f"\nwrote {out/'edges.csv'} ({pairs.height:,} rows, {kept.height:,} kept)  ({time.time()-t0:.0f}s)")
