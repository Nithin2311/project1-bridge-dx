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


def test_bh_fdr_matches_statsmodels_style_reference():
    import numpy as np
    # reuse the permutation machinery on random data and check the BH transform against a direct implementation
    rng = np.random.default_rng(3)
    rows, labs = [], []
    for p in range(120):
        labs.append((p, "female" if p % 2 else "male"))
        for c in "ABCDE":
            if rng.random() < 0.4:
                rows.append((p, c))
    pc = pl.DataFrame(rows, schema=["person_id", "category"], orient="row")
    labels = pl.DataFrame(labs, schema=["person_id", "sex"], orient="row")
    edges = pl.DataFrame({"u": ["A", "A", "A", "B", "B", "C"], "v": ["B", "C", "D", "C", "E", "E"]})
    res = sg.edgewise_permutation(pc, labels, CFG, edges, "sex", "female", "male", list("ABCDE"), seed=0)
    p = res["p_perm"].to_numpy()
    m = len(p)
    order = np.argsort(p)
    q_ref = np.empty(m)
    prev = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        prev = min(prev, p[i] * m / rank)
        q_ref[i] = prev
    assert np.allclose(res["p_fdr"].to_numpy(), q_ref)


def test_in_ranges():
    r = [["N40", "N51"], ["O00", "O9A"], ["C50", "C58"]]
    assert sg.in_ranges("N40", r) and sg.in_ranges("N45", r) and sg.in_ranges("O99", r) and sg.in_ranges("O9A", r) and sg.in_ranges("C50", r)
    assert not sg.in_ranges("N39", r) and not sg.in_ranges("N52", r) and not sg.in_ranges("I10", r) and not sg.in_ranges("C59", r)


def test_age_rule_feb29_and_unknown_gender():
    import datetime as dt
    import duckdb
    con = duckdb.connect()
    con.execute("create table person(person_id bigint, gender_concept_id bigint, birth_DATETIME timestamp)")
    con.execute("insert into person values (1, 8532, '1952-02-29'), (2, 0, '1960-01-01')")
    pc = pl.DataFrame({"person_id": [1, 2], "category": ["A", "A"], "first_date": [dt.date(2017, 2, 28), dt.date(2017, 1, 1)]})
    df = sg.assign_subgroups(con, CFG, pc).sort("person_id")
    assert df["age_at_index"].to_list() == [64, 57]  # Feb 28 2017 is the day before the leap-birthday's Mar 1 anniversary
    assert df["sex"].to_list() == ["female", None]    # unknown gender concept -> null sex, patient kept for age bands
