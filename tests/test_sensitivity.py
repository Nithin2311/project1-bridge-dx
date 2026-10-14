import polars as pl

from bridgedx import sensitivity as se


def test_strata_by_n_categories():
    pc = pl.DataFrame({"person_id": [1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 5, 5, 5, 5, 5, 6],
                       "category": list("ABABCABCDAABCDEA")})
    strata = se.strata_by_n_categories(pc, 3)
    assert len(strata) == 3
    names = list(strata)
    assert names[0].startswith("low_ncat_1-") and names[-1].startswith("high_ncat_")
    assert sorted(sum(strata.values(), [])) == [1, 2, 3, 4, 5, 6]
    assert strata[names[-1]] == [3, 5]  # the two patients with most categories


def test_snomed_bridge_prefers_empirical_single_category():
    import duckdb
    con = duckdb.connect()
    con.execute("create table condition_occurrence(person_id bigint, condition_concept_id bigint, condition_source_value varchar, condition_start_DATE date)")
    rows = [(i, 100, "F41.9", "2016-01-01") for i in range(6)] + [(9, 100, "F32.9", "2016-01-01")]  # concept 100: F41 x6, F32 x1
    rows += [(i, 200, "I10", "2017-01-01") for i in range(3)]                                           # concept 200: only 3 era rows (< min 5)
    rows += [(1, 300, "401.9", "2010-01-01")]                                                            # concept 300: pre-era only
    con.executemany("insert into condition_occurrence values (?, ?, ?, ?)", rows)
    con.execute("create table concept(concept_id bigint, concept_code varchar, vocabulary_id varchar)")
    con.execute("insert into concept values (1, 'I10', 'ICD10CM'), (2, 'E11.9', 'ICD10CM'), (3, 'E08.0', 'ICD10CM')")
    con.execute("create table concept_relationship(concept_id_1 bigint, concept_id_2 bigint, relationship_id varchar)")
    con.execute("insert into concept_relationship values (1, 300, 'Maps to'), (2, 400, 'Maps to'), (3, 400, 'Maps to')")  # 400 is two-category
    con.execute("create table concept_ancestor(ancestor_concept_id bigint, descendant_concept_id bigint, min_levels_of_separation int)")
    con.execute("insert into concept_ancestor values (300, 500, 1)")  # 500 -> ancestor 300 (I10)
    con.execute("insert into condition_occurrence values (2, 400, '250.00', date '2011-01-01'), (3, 500, '401.1', date '2011-01-01')")
    b = se.build_snomed_bridge(con, {"cohort": {"era_start": "2015-10-01"}, "sensitivity": {"icd9_min_era_rows_per_concept": 5}})
    m = {r["snomed_id"]: (r["category"], r["source"]) for r in b.iter_rows(named=True)}
    assert m[100] == ("F41", "empirical")           # majority, not keep-all
    assert 200 not in m  # too few era rows and no vocab map -> unbridged
    assert m[300] == ("I10", "vocab")
    assert 400 not in m                              # two vocab categories -> excluded, not duplicated
    assert m[500] == ("I10", "ancestor")
    assert b.filter(pl.col("snomed_id") == 100)["purity"][0] == 6 / 7
