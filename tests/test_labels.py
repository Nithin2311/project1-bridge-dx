import datetime as dt

import numpy as np
import polars as pl

from bridgedx import labels as lb

CFG = {"project2": {"grid_start": "2018-01-01", "grid_step_days": 182, "min_history_days": 365,
                    "require_encounter_days": 365, "horizons_months": [12], "min_positives": 1}}


def test_landmark_labels_rules():
    pc = pl.DataFrame({"person_id": [1, 1, 2, 2, 3, 3],
                       "category": ["A", "T", "A", "T", "A", "B"],
                       "first_date": [dt.date(2016, 1, 1), dt.date(2018, 9, 1), dt.date(2016, 1, 1), dt.date(2017, 1, 1),
                                      dt.date(2016, 1, 1), dt.date(2019, 1, 1)]})
    pre = pl.DataFrame({"person_id": [3], "category": ["T"]})  # patient 3 had T before the era -> excluded entirely
    patients = pl.DataFrame({"person_id": [1, 2, 3], "first_era": [dt.date(2016, 1, 1)] * 3,
                             "last_dx": [dt.date(2021, 1, 1)] * 3, "death": [None, None, None]}).with_columns(pl.col("death").cast(pl.Date))
    enc = {p: np.array([dt.date(2016, 1, 1) + dt.timedelta(days=90 * i) for i in range(25)]) for p in [1, 2, 3]}
    targets = {"core": {"T": {"T"}}}
    df = lb.build_landmark_labels(CFG, pc, pre, patients, enc, targets, {1: "train", 2: "val", 3: "test"})
    # patient 1: index 2018-01-01 -> T on 2018-09-01 within 12 months -> positive, then stop
    p1 = df.filter(pl.col("person_id") == 1)
    assert p1.height == 1 and p1["positive"][0] is True and p1["days_to_event"][0] == 243
    # patient 2: T before the first index -> prior, no rows
    assert df.filter(pl.col("person_id") == 2).height == 0
    assert df.filter(pl.col("person_id") == 3).height == 0
    s = lb.summarize(df, 1)
    assert s["positives"].sum() == 1 and s["split"].to_list() == ["train"]


def test_landmark_grid_respects_horizon():
    g = lb.landmark_grid(CFG, dt.date(2020, 1, 1))
    assert g[0] == dt.date(2018, 1, 1) and all(d + dt.timedelta(days=365) <= dt.date(2020, 1, 1) for d in g)
