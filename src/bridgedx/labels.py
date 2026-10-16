"""Project 2 landmark labels (non-ML): one row per (patient, index date, target, definition, horizon).

Rules (config project2):
  index dates: calendar grid from grid_start every grid_step_days while index + max horizon <= data end
  eligible row:  first ICD-10-era diagnosis <= index - min_history_days
                 >= 1 diagnosis row (any) in (index - require_encounter_days, index]
                 no target code on/before index in the ICD-10 era, and none in the bridged pre-era table
  positive:      first target code in (index, index + horizon]
  censored:      no positive and (death <= index + horizon or last diagnosis < index + horizon)
  no row is emitted once the patient's record has ended (last pre-death diagnosis < index) or after death.
  a patient contributes rows until the first positive or censoring for that target.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl


def landmark_grid(cfg: dict, data_end: dt.date) -> list[dt.date]:
    p2 = cfg["project2"]
    start = dt.date.fromisoformat(str(p2["grid_start"]))
    step = int(p2["grid_step_days"])
    hmax = max(int(h) for h in p2["horizons_months"])
    end = data_end - dt.timedelta(days=int(round(hmax * 30.44)))
    grid, t = [], start
    while t <= end:
        grid.append(t)
        t += dt.timedelta(days=step)
    return grid


def build_landmark_labels(cfg: dict, pc: pl.DataFrame, pre: pl.DataFrame, patients: pl.DataFrame,
                          encounters: dict, targets: dict[str, dict[str, set]], split: dict) -> pl.DataFrame:
    """pc: era (person_id, category, first_date); pre: bridged pre-era (person_id, category);
    patients: person_id, first_era, last_dx, death; encounters: person_id -> sorted np.array of dates;
    targets: definition -> {target_name -> set(categories)}; split: person_id -> split label."""
    p2 = cfg["project2"]
    data_end = patients["last_dx"].max()
    grid = landmark_grid(cfg, data_end)
    min_hist = dt.timedelta(days=int(p2["min_history_days"]))
    enc_win = dt.timedelta(days=int(p2["require_encounter_days"]))
    horizons = [int(h) for h in p2["horizons_months"]]
    rows = []
    for definition, tsets in targets.items():
        for tname, members in tsets.items():
            m = list(members)
            ft = dict(pc.filter(pl.col("category").is_in(m)).group_by("person_id").agg(pl.col("first_date").min().alias("ft")).rows())
            prior = set(pre.filter(pl.col("category").is_in(m))["person_id"].to_list())
            for r in patients.iter_rows(named=True):
                pid = r["person_id"]
                if pid in prior:
                    continue
                f = ft.get(pid)
                enc = encounters.get(pid)
                for h in horizons:
                    H = dt.timedelta(days=int(round(h * 30.44)))
                    for t in grid:
                        if r["first_era"] > t - min_hist:
                            continue
                        if r["last_dx"] < t or (r["death"] is not None and r["death"] < t):
                            break  # record ended before this index date (third review M4)
                        if enc is None or not ((enc > t - enc_win) & (enc <= t)).any():
                            continue
                        if f is not None and f <= t:
                            break
                        end = t + H
                        positive = f is not None and f <= end
                        ends = (r["death"] is not None and r["death"] <= end) or r["last_dx"] < end
                        censored = (not positive) and ends
                        rows.append({"person_id": pid, "split": split.get(pid), "index_date": t, "target": tname,
                                     "definition": definition, "horizon_months": h, "positive": positive,
                                     "censored": censored, "days_to_event": (f - t).days if positive else None,
                                     "history_days": (t - r["first_era"]).days,
                                     "followup_days": (min(r["last_dx"], r["death"]) if r["death"] else r["last_dx"]) .__sub__(t).days})
                        if positive or censored:
                            break
    return pl.DataFrame(rows)


def summarize(labels: pl.DataFrame, min_pos: int) -> pl.DataFrame:
    """Counts per (definition, target, horizon, split) plus an 'all' split row; `enough` = positives >= min_pos
    on the 'all' row (per-split rows carry their own count)."""
    agg = [pl.len().alias("rows"), pl.col("person_id").n_unique().alias("patients"),
           pl.col("positive").sum().alias("positives"), pl.col("censored").sum().alias("censored")]
    per = labels.group_by("definition", "target", "horizon_months", "split").agg(*agg)
    tot = labels.group_by("definition", "target", "horizon_months").agg(*agg).with_columns(pl.lit("all").alias("split"))
    return (pl.concat([per, tot.select(per.columns)])
            .with_columns((pl.col("positives") / (pl.col("rows") - pl.col("censored"))).alias("positive_rate"),
                          (pl.col("positives") >= min_pos).alias("enough"))
            .sort("definition", "target", "horizon_months", "split"))
