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
    assert s.filter(pl.col("split") == "all")["positives"][0] == 1 and sorted(s["split"].to_list()) == ["all", "train"]


def test_landmark_grid_respects_horizon():
    g = lb.landmark_grid(CFG, dt.date(2020, 1, 1))
    assert g[0] == dt.date(2018, 1, 1) and all(d + dt.timedelta(days=365) <= dt.date(2020, 1, 1) for d in g)


def test_landmark_labels_censoring_and_death():
    import datetime as dt
    import numpy as np
    pc = pl.DataFrame({"person_id": [1, 2, 3], "category": ["A", "A", "A"], "first_date": [dt.date(2016, 1, 1)] * 3})
    pre = pl.DataFrame({"person_id": [], "category": []}, schema={"person_id": pl.Int64, "category": pl.Utf8})
    # 1: record ends 2018-06-01 (< index + 12 mo) -> censored row at 2018-01-01, then stop
    # 2: dies 2017-06-01 -> no row at all (dead before the first index)
    # 3: full follow-up, never gets T -> negative rows at every grid date
    patients = pl.DataFrame({"person_id": [1, 2, 3], "first_era": [dt.date(2016, 1, 1)] * 3,
                             "last_dx": [dt.date(2018, 6, 1), dt.date(2017, 5, 1), dt.date(2021, 1, 1)],
                             "death": [None, dt.date(2017, 6, 1), None]}).with_columns(pl.col("death").cast(pl.Date))
    enc = {p: np.array([dt.date(2016, 1, 1) + dt.timedelta(days=60 * i) for i in range(40)]) for p in [1, 2, 3]}
    df = lb.build_landmark_labels(CFG, pc, pre, patients, enc, {"core": {"T": {"T"}}}, {1: "train", 2: "train", 3: "test"})
    p1 = df.filter(pl.col("person_id") == 1)
    assert p1.height == 1 and p1["censored"][0] is True and p1["positive"][0] is False
    assert df.filter(pl.col("person_id") == 2).height == 0
    p3 = df.filter(pl.col("person_id") == 3)
    assert p3.height == len(lb.landmark_grid(CFG, dt.date(2021, 1, 1))) and not p3["positive"].any() and not p3["censored"].any()
    assert (df["followup_days"] >= 0).all()
    s = lb.summarize(df, 1)
    assert "all" in s["split"].to_list()
