"""Sensitivity inputs: ICD-9-era patient x category table bridged through SNOMED, and follow-up strata.

Step 7 of the Option A pipeline. See docs/OPTION-A-SPEC.md.
  * ICD-9 era = rows before cohort.era_start. condition_concept_id is already the SNOMED standard concept
    (ETL output). SNOMED -> ICD-10-CM category via concept_relationship 'Maps to' (ICD10CM -> SNOMED, read
    backwards). Concepts with no direct map fall back to the nearest SNOMED ancestor (concept_ancestor,
    min_levels_of_separation <= max_levels) that has one. Concepts mapping to several categories contribute
    all of them (ambiguity rate reported).
  * Strata by number of distinct categories: tertiles of the cohort (unequal follow-up check).
"""
from __future__ import annotations

import polars as pl


def build_snomed_bridge(con, max_levels: int = 3) -> pl.DataFrame:
    """SNOMED concept_id -> ICD-10-CM 3-char category (direct, or via nearest mapped ancestor).
    Registers temp table `snomed_bridge` (snomed_id, category, levels) and returns it."""
    con.execute("""
        create or replace temp table _direct as
        select distinct r.concept_id_2 as snomed_id, substr(c.concept_code, 1, 3) as category
        from concept_relationship r
        join concept c on c.concept_id = r.concept_id_1
        where r.relationship_id = 'Maps to' and c.vocabulary_id = 'ICD10CM'
          and (r.invalid_reason is null or r.invalid_reason = '')
    """)
    con.execute("""
        create or replace temp table _need as
        select distinct condition_concept_id as snomed_id from condition_occurrence
        where condition_concept_id <> 0 and condition_concept_id not in (select snomed_id from _direct)
    """)
    con.execute(f"""
        create or replace temp table _via_ancestor as
        with cand as (
            select a.descendant_concept_id as snomed_id, d.category, a.min_levels_of_separation as levels
            from _need n
            join concept_ancestor a on a.descendant_concept_id = n.snomed_id
            join _direct d on d.snomed_id = a.ancestor_concept_id
            where a.min_levels_of_separation between 1 and {int(max_levels)}
        ),
        best as (select snomed_id, min(levels) as levels from cand group by 1)
        select distinct c.snomed_id, c.category, c.levels from cand c join best b using (snomed_id, levels)
    """)
    con.execute("""
        create or replace temp table snomed_bridge as
        select snomed_id, category, 0 as levels from _direct
        union all
        select snomed_id, category, levels from _via_ancestor
    """)
    return con.execute("select * from snomed_bridge").pl()


def icd9_era_patient_category(con, cfg: dict) -> tuple[pl.DataFrame, dict]:
    """Distinct (person_id, category) from rows before era_start, bridged through SNOMED; post-death rows
    dropped; patients with >= min_distinct_categories kept. Returns (table, coverage stats)."""
    c = cfg["cohort"]
    death_clause = (
        "and (d.death_DATE is null or co.condition_start_DATE <= d.death_DATE)"
        if c.get("drop_events_after_death", True) else ""
    )
    cov = con.execute(f"""
        with pre as (
            select co.person_id, co.condition_concept_id as cid, co.condition_start_DATE as dt
            from condition_occurrence co left join death d on d.person_id = co.person_id
            where co.condition_start_DATE < date '{c["era_start"]}' {death_clause}
        )
        select count(*) as n_rows, count(distinct person_id) as n_patients,
               count(*) filter (where cid in (select snomed_id from snomed_bridge where levels = 0)) as rows_direct,
               count(*) filter (where cid in (select snomed_id from snomed_bridge where levels > 0)) as rows_ancestor,
               count(*) filter (where cid not in (select snomed_id from snomed_bridge)) as rows_unbridged,
               count(*) filter (where cid = 0) as rows_unmapped_concept
        from pre
    """).pl().row(0, named=True)
    amb = con.execute("""
        select count(*) filter (where n > 1) as ambiguous, count(*) as total
        from (select snomed_id, count(distinct category) as n from snomed_bridge group by 1)
    """).pl().row(0, named=True)
    cov.update({"concepts_ambiguous": amb["ambiguous"], "concepts_total": amb["total"]})
    pc = con.execute(f"""
        with era as (
            select co.person_id, b.category, co.condition_start_DATE as dt
            from condition_occurrence co
            join snomed_bridge b on b.snomed_id = co.condition_concept_id
            left join death d on d.person_id = co.person_id
            where co.condition_start_DATE < date '{c["era_start"]}' {death_clause}
        ),
        pc as (select person_id, category, min(dt) as first_date from era group by 1, 2),
        keep as (select person_id from pc group by 1 having count(*) >= {int(c["min_distinct_categories"])})
        select pc.person_id, pc.category, pc.first_date from pc join keep using (person_id)
        order by person_id, category
    """).pl()
    return pc, cov


def strata_by_n_categories(pc: pl.DataFrame, n_strata: int = 3) -> dict[str, list]:
    """Cohort split into quantile strata by number of distinct categories; name -> person_ids."""
    counts = pc.group_by("person_id").len().sort("len", "person_id")
    n = counts.height
    out = {}
    labels = ["low", "mid", "high"] if n_strata == 3 else [f"q{i+1}" for i in range(n_strata)]
    for i, lab in enumerate(labels[:n_strata]):
        lo, hi = i * n // n_strata, (i + 1) * n // n_strata
        part = counts[lo:hi]
        out[f"{lab}_ncat_{part['len'].min()}-{part['len'].max()}"] = part["person_id"].to_list()
    return out
