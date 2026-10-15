"""Patient split: train / val / test by person_id, fixed before any graph quantity is computed.

Source (config split.source): 'file' = EHRSHOT_ASSETS/splits/person_id_map.csv from Redivis (columns split,
omop_person_id); 'own' = deterministic hash-free split: person_ids sorted, shuffled with numpy seed, cut by
fractions; 'auto' = file if it exists else own. Patients in the cohort but not in the file are assigned 'own'
style and flagged.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from . import config


def assign_split(person_ids: list, cfg: dict) -> pl.DataFrame:
    sp = cfg["split"]
    ids = sorted(int(p) for p in person_ids)
    source = sp.get("source", "auto")
    path = config.REPO_ROOT / sp["file"] if sp.get("file") else None
    use_file = source == "file" or (source == "auto" and path is not None and path.exists())
    if use_file:
        f = pl.read_csv(path)
        col = "omop_person_id" if "omop_person_id" in f.columns else "person_id"
        m = dict(zip(f[col].cast(pl.Int64).to_list(), f["split"].to_list()))
        rows = [{"person_id": p, "split": m.get(p), "split_source": "file" if p in m else "missing"} for p in ids]
        df = pl.DataFrame(rows)
        missing = df.filter(pl.col("split").is_null())["person_id"].to_list()
        if missing:
            own = _own(missing, sp)
            df = df.join(own.rename({"split": "split_own"}), on="person_id", how="left").with_columns(
                pl.coalesce(["split", "split_own"]).alias("split")).drop("split_own")
        return df.sort("person_id")
    df = _own(ids, sp).with_columns(pl.lit("own").alias("split_source"))
    return df.sort("person_id")


def _own(ids: list, sp: dict) -> pl.DataFrame:
    rng = np.random.default_rng(int(sp.get("seed", 0)))
    arr = np.array(sorted(ids))
    perm = rng.permutation(len(arr))
    fr = sp.get("fractions", [0.7, 0.15, 0.15])
    n_tr = int(round(fr[0] * len(arr)))
    n_va = int(round(fr[1] * len(arr)))
    lab = np.empty(len(arr), dtype=object)
    lab[perm[:n_tr]] = "train"
    lab[perm[n_tr:n_tr + n_va]] = "val"
    lab[perm[n_tr + n_va:]] = "test"
    return pl.DataFrame({"person_id": arr.tolist(), "split": lab.tolist()})


def main_splits(cfg: dict) -> list[str]:
    m = cfg["split"].get("main_graph_patients", "train")
    return [m] if isinstance(m, str) else list(m)


def train_ids(cfg: dict | None = None) -> list:
    """person_ids of the splits that build the main graph (cfg split.main_graph_patients)."""
    cfg = cfg or config.load()
    df = pl.read_csv(config.outputs_dir(cfg) / "patient_split.csv")
    return df.filter(pl.col("split").is_in(main_splits(cfg)))["person_id"].to_list()
