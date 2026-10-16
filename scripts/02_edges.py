"""Step 2: edge list.

Main edges (outputs/edges.csv): every node pair tested on the main-graph patients only (cfg split.main_graph_patients = train + val).
Sensitivity edges (outputs/edges_allpatients.csv): the same test on all cohort patients.
Rerunnable: `uv run python scripts/02_edges.py` (after 01).
"""
from __future__ import annotations

import time

import polars as pl

from bridgedx import config, edges, split

cfg = config.load()
out = config.outputs_dir(cfg)
e = cfg["edges"]
t0 = time.time()

pc_all = pl.read_parquet(out / "patient_category.parquet")
nodes = pl.read_csv(out / "nodes.csv")
cats = nodes["category"].to_list()
names = dict(zip(nodes["category"], nodes["name"]))
tr = split.train_ids(cfg)


def run(pc: pl.DataFrame, label: str, fname: str) -> pl.DataFrame:
    pairs = edges.pair_counts(pc, categories=cats)
    print(f"\n[{label}] {len(cats)} nodes, {pc['person_id'].n_unique():,} patients -> {pairs.height:,} pairs")
    pairs = edges.weights(edges.fisher_filter(pairs, cfg))
    pairs.write_csv(out / fname)
    kept = pairs.filter("kept")
    print(f"  filter: p_bonf < {e['alpha']} (m = {pairs.height:,}), OR >= {e['min_odds_ratio']}, n_both >= {e['min_shared_patients']}")
    print(f"  pairs with n_both >= {e['min_shared_patients']}: {pairs.filter(pl.col('n_both') >= e['min_shared_patients']).height:,}; "
          f"Bonferroni-significant: {pairs.filter((pl.col('n_both') >= e['min_shared_patients']) & (pl.col('p_bonf') < e['alpha'])).height:,}; "
          f"kept (OR >= {e['min_odds_ratio']}): {kept.height:,}")
    print(f"  kept phi: min {kept['phi'].min():.3f}, median {kept['phi'].median():.3f}, max {kept['phi'].max():.3f}")
    return pairs


main = run(pc_all.filter(pl.col("person_id").is_in(tr)), f"main = {'+'.join(split.main_splits(cfg))} split(s)", "edges.csv")
allp = run(pc_all, "sensitivity = all cohort patients", "edges_allpatients.csv")
km, ka = set(map(frozenset, zip(*[main.filter("kept")[c] for c in ("u", "v")]))), set(map(frozenset, zip(*[allp.filter("kept")[c] for c in ("u", "v")])))
print(f"\nkept-edge overlap train vs all patients: both {len(km & ka):,}, train only {len(km - ka):,}, all-patients only {len(ka - km):,}, Jaccard {len(km & ka)/len(km | ka):.2f}")


def show(df: pl.DataFrame, title: str) -> None:
    print(f"\n{title}")
    for i, r in enumerate(df.iter_rows(named=True), 1):
        print(f"  {i:>2}. {r['u']}-{r['v']}  {names[r['u']][:34]} / {names[r['v']][:34]}  n={r['n_both']:,}  OR={r['odds_ratio']:.1f}  phi={r['phi']:.3f}")


kept = main.filter("kept")
show(kept.sort("n_both", descending=True).head(10).select("u", "v", "n_both", "odds_ratio", "phi"), "main: top 10 kept edges by raw shared-patient count")
show(kept.sort("phi", descending=True).head(10).select("u", "v", "n_both", "odds_ratio", "phi"), "main: top 10 kept edges by phi")
print(f"\nwrote edges.csv ({main.height:,} rows, {kept.height:,} kept) and edges_allpatients.csv  ({time.time()-t0:.0f}s)")
