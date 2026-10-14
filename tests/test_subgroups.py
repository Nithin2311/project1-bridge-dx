import datetime as dt

import duckdb
import numpy as np
import polars as pl

from bridgedx import subgroups as sg

CFG = {
    "subgroups": {
        "groups": {"sex": ["female", "male"], "age": {"cut": 65, "labels": ["under65", "over65"]}},
        "age_rule": {"birth_field": "birth_DATETIME", "sex_concept_ids": {"female": 8532, "male": 8507}},
        "comparison": {"n_permutations": 200, "top_k": 2},
    }
}


def test_assign_subgroups_age_rule():
    con = duckdb.connect()
    con.execute("create table person(person_id bigint, gender_concept_id bigint, birth_DATETIME timestamp)")
    con.execute("insert into person values (1, 8532, '1950-06-15'), (2, 8507, '1951-06-16'), (3, 8532, '1990-01-01')")
    pc = pl.DataFrame({"person_id": [1, 1, 2, 3], "category": ["A", "B", "A", "B"],
                       "first_date": [dt.date(2016, 1, 1), dt.date(2015, 6, 15), dt.date(2016, 6, 15), dt.date(2020, 1, 1)]})
    df = sg.assign_subgroups(con, CFG, pc).sort("person_id")
    assert df["sex"].to_list() == ["female", "male", "female"]
    assert df["age_at_index"].to_list() == [65, 64, 30]  # 1: exactly 65 on birthday; 2: day before 65th birthday
    assert df["age_band"].to_list() == ["over65", "under65", "under65"]
    assert df["index_date"][0] == dt.date(2015, 6, 15)


def test_overlap_and_ari():
    assert sg.connector_overlap(["a", "b", "c"], ["b", "a", "z"], k=2) == 1.0
    assert sg.connector_overlap(["a", "b"], ["c", "d"], k=2) == 0.0
    assert sg.compare_partitions({"a": 0, "b": 0, "c": 1}, {"a": 5, "b": 5, "c": 9, "zz": 1}) == 1.0


def test_edgewise_permutation_detects_group_specific_edge():
    rng = np.random.default_rng(0)
    rows, labs = [], []
    for p in range(400):
        grp = "female" if p < 200 else "male"
        labs.append((p, grp))
        has_a = rng.random() < 0.5
        has_b = (has_a if grp == "female" else rng.random() < 0.5)  # A-B linked only in females
        has_c = rng.random() < 0.5
        for c, h in [("A", has_a), ("B", has_b), ("C", has_c)]:
            if h:
                rows.append((p, c))
    pc = pl.DataFrame(rows, schema=["person_id", "category"], orient="row")
    labels = pl.DataFrame(labs, schema=["person_id", "sex"], orient="row")
    edges = pl.DataFrame({"u": ["A", "A"], "v": ["B", "C"]})
    res = sg.edgewise_permutation(pc, labels, CFG, edges, "sex", "female", "male", ["A", "B", "C"], seed=1)
    ab = res.filter(pl.col("u") == "A").filter(pl.col("v") == "B").row(0, named=True)
    ac = res.filter(pl.col("v") == "C").row(0, named=True)
    assert ab["phi_a"] > 0.95 and ab["p_perm"] < 0.01 and ab["diff"] > 0.8
    assert ac["p_perm"] > 0.05
    assert res["n_a"][0] == 200 and res["n_b"][0] == 200
