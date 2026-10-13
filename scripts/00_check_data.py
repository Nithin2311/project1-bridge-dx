"""Step 0: verify the data folder, register views, print row counts. Safe to run any time."""
from __future__ import annotations

import sys
import time

from bridgedx import config, data

cfg = config.load()
d = config.data_dir(cfg)
print(f"data dir: {d}")
if not d.exists():
    sys.exit(f"missing. set {cfg['paths']['data_env']} to the folder with the 13 CSVs")

avail = data.available_tables(cfg)
missing = [t for t in data.TABLES if t not in avail]
for t, p in avail.items():
    print(f"  {t:22s} {p.stat().st_size/1e6:10.1f} MB")
if missing:
    print("  missing:", ", ".join(missing))

con = data.connect(cfg)
for t in ["person", "condition_occurrence", "death", "observation_period", "visit_occurrence"]:
    if t in avail:
        t0 = time.time()
        n = con.execute(f"select count(*) from {t}").fetchone()[0]
        print(f"  rows {t:22s} {n:>12,}  ({time.time()-t0:.1f}s)")
print(f"cache: {config.cache_path(cfg)}")
