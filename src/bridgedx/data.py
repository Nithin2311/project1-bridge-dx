"""DuckDB access to the EHRShot CSVs. Views are registered lazily; nothing is copied."""
from __future__ import annotations

from pathlib import Path

import duckdb

from . import config

TABLES = [
    "person", "visit_occurrence", "condition_occurrence", "drug_exposure", "procedure_occurrence",
    "death", "observation_period", "concept", "concept_ancestor", "concept_relationship",
    "concept_class", "domain", "relationship", "vocabulary",
]


def connect(cfg: dict | None = None, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open the cache DB and make every CSV available as a view named after its table."""
    cfg = cfg or config.load()
    data = config.data_dir(cfg)
    con = duckdb.connect(str(config.cache_path(cfg)), read_only=read_only)
    existing = {r[0] for r in con.execute("select view_name from duckdb_views() where not internal").fetchall()}
    for t in TABLES:
        csv = data / f"{t}.csv"
        if csv.exists() and t not in existing and not read_only:
            con.execute(f"create view {t} as select * from read_csv_auto('{csv}', header=true, sample_size=-1)")
    return con


def available_tables(cfg: dict | None = None) -> dict[str, Path]:
    data = config.data_dir(cfg)
    return {t: data / f"{t}.csv" for t in TABLES if (data / f"{t}.csv").exists()}
