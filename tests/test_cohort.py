"""Unit tests on synthetic rows; no real data."""
import datetime as dt

import duckdb
import polars as pl

from bridgedx import cohort

CFG = {
    "cohort": {"era_start": "2015-10-01", "min_distinct_categories": 2, "drop_events_after_death": True},
    "nodes": {"min_prevalence": 0.5, "flag_chapters": ["R", "Z"]},
}


def test_icd10_3char():
    assert cohort.icd10_3char("I10") == "I10"
    assert cohort.icd10_3char("I10.9") == "I10"
    assert cohort.icd10_3char("E11.65") == "E11"
    assert cohort.icd10_3char("D3A.8") == "D3A"
    assert cohort.icd10_3char("M1A.9XX0") == "M1A"
    assert cohort.icd10_3char("NULL-C50.911") == "C50"
    assert cohort.icd10_3char(" e78.5 ") == "E78"
    assert cohort.icd10_3char("V97.33XD") == "V97"  # ICD-10 external cause (7th char)
    assert cohort.icd10_3char("401.9") is None  # ICD-9
    assert cohort.icd10_3char("E812.0") is None  # ICD-9 E-code
    assert cohort.icd10_3char("V12.3") is None  # ICD-9 V-code
    assert cohort.icd10_3char("19004") is None
    assert cohort.icd10_3char("") is None
    assert cohort.icd10_3char(None) is None


def test_chapter_of():
    assert cohort.chapter_of("I10") == "circulatory"
    assert cohort.chapter_of("D3A") == "neoplasms"
    assert cohort.chapter_of("D64") == "blood/immune"
    assert cohort.chapter_of("H60") == "ear"
    assert cohort.chapter_of("T86") == "injury/poisoning"
    assert cohort.chapter_of("Z94") == "factors influencing health status"


def _synthetic_con():
    con = duckdb.connect()
    rows = [
        # person 1: I10, E11 (dup), R94 -> 3 categories, kept
        (1, "I10", dt.date(2016, 1, 1)), (1, "E11.9", dt.date(2016, 2, 1)), (1, "E11.65", dt.date(2016, 3, 1)),
        (1, "R94.31", dt.date(2016, 4, 1)),
        # person 2: I10 in era, 401.9 ICD-9 in era (ignored), E11 before era (ignored) -> 1 category, dropped
        (2, "I10", dt.date(2017, 1, 1)), (2, "401.9", dt.date(2017, 1, 1)), (2, "E11.9", dt.date(2014, 1, 1)),
        # person 3: I10, R94; Z94 dated after death -> 2 categories, kept
        (3, "I10", dt.date(2018, 1, 1)), (3, "R94.31", dt.date(2018, 1, 1)), (3, "Z94.0", dt.date(2019, 6, 1)),
    ]
    con.execute("create table condition_occurrence(person_id bigint, condition_source_value varchar, condition_start_DATE date)")
    con.executemany("insert into condition_occurrence values (?, ?, ?)", rows)
    con.execute("create table death(person_id bigint, death_DATE date)")
    con.execute("insert into death values (3, date '2019-01-01')")
    con.execute("create table person(person_id bigint)")
    con.execute("insert into person values (1), (2), (3)")
    con.execute("create table concept(concept_code varchar, concept_name varchar, vocabulary_id varchar)")
    con.execute("insert into concept values ('I10', 'Essential (primary) hypertension', 'ICD10CM'), "
                "('E11', 'Type 2 diabetes mellitus', 'ICD10CM'), ('R94', 'Abnormal results of function studies', 'ICD10CM')")
    return con


def test_build_patient_category_and_nodes():
    con = _synthetic_con()
    pc = cohort.build_patient_category(con, CFG)
    assert set(pc["person_id"].unique()) == {1, 3}
    assert pc.filter(pl.col("person_id") == 1)["category"].to_list() == ["E11", "I10", "R94"]
    assert pc.filter(pl.col("person_id") == 3)["category"].to_list() == ["I10", "R94"]  # Z94 after death dropped
    assert pc.filter((pl.col("person_id") == 1) & (pl.col("category") == "E11"))["first_date"][0] == dt.date(2016, 2, 1)

    nodes = cohort.build_nodes(pc, con, CFG)
    # cohort = 2 patients; I10 and R94 in both (prev 1.0), E11 in one (0.5) -> all kept at min_prevalence 0.5
    assert nodes["category"].to_list()[:2] == ["I10", "R94"]
    assert nodes.filter(pl.col("category") == "R94")["flagged"][0] is True
    assert nodes.filter(pl.col("category") == "I10")["flagged"][0] is False
    assert nodes.filter(pl.col("category") == "I10")["name"][0] == "Essential (primary) hypertension"
    assert nodes.filter(pl.col("category") == "E11")["prevalence"][0] == 0.5

    funnel = cohort.cohort_funnel(con, CFG)
    assert funnel.filter(pl.col("step") == "rows dated after death (dropped)")["rows_"][0] == 1
