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


def build_snomed_bridge(con, cfg: dict | None = None, min_era_rows: int = 5, max_levels: int = 3,
                        allowed_fallback_categories: list[str] | None = None) -> pl.DataFrame:
    """SNOMED concept_id -> ONE ICD-10-CM 3-char category. Registers temp table `snomed_bridge`
    (snomed_id, category, source, purity) and returns it. `allowed_fallback_categories` restricts the vocab and
    ancestor sources (default: every category; step 7 passes the node list minus excluded chapters).
      source = 'empirical': majority category among this hospital's own ICD-10-era rows carrying that
               condition_concept_id (>= min_era_rows rows); purity = majority share
      source = 'vocab':     fallback for concepts unseen in the era: vocabulary 'Maps to' (ICD10CM -> SNOMED read
               backwards), accepted only when it yields exactly one category
      source = 'ancestor':  last resort, nearest SNOMED ancestor (<= max_levels) with a single vocab category
    The earlier pure-vocabulary bridge gave men pregnancy codes and 12% multi-category rows (review 2026-10-14)."""
    if cfg is not None:
        min_era_rows = int(cfg.get("sensitivity", {}).get("icd9_min_era_rows_per_concept", min_era_rows))
    era_start = (cfg or {}).get("cohort", {}).get("era_start", "2015-10-01")
    from . import cohort as _cohort
    _cohort._register_category_map(con)
    con.execute(f"""
        create or replace temp table _empirical as
        with rows as (
            select co.condition_concept_id as snomed_id, m.category
            from condition_occurrence co join cat_map m on m.sv = co.condition_source_value
            where co.condition_start_DATE >= date '{era_start}' and co.condition_concept_id <> 0
        ),
        cnt as (select snomed_id, category, count(*) as n from rows group by 1, 2),
        tot as (select snomed_id, sum(n) as n_total from cnt group by 1),
        best as (select snomed_id, category, n, row_number() over (partition by snomed_id order by n desc, category) as rk from cnt)
        select b.snomed_id, b.category, 'empirical' as source, b.n * 1.0 / t.n_total as purity
        from best b join tot t using (snomed_id)
        where b.rk = 1 and t.n_total >= {int(min_era_rows)}
    """)
    allow = ""
    if allowed_fallback_categories is not None:
        con.register("_allowed_df", pl.DataFrame({"category": list(allowed_fallback_categories)}).to_arrow())
        con.execute("create or replace temp table _allowed as select * from _allowed_df")
        allow = "and substr(c.concept_code, 1, 3) in (select category from _allowed)"
    con.execute(f"""
        create or replace temp table _vocab as
        select snomed_id, min(category) as category, 'vocab' as source, 1.0 as purity
        from (
            select distinct r.concept_id_2 as snomed_id, substr(c.concept_code, 1, 3) as category
            from concept_relationship r join concept c on c.concept_id = r.concept_id_1
            where r.relationship_id = 'Maps to' and c.vocabulary_id = 'ICD10CM' {allow}
        ) group by snomed_id having count(*) = 1
    """)
    con.execute("""
        create or replace temp table _need as
        select distinct condition_concept_id as snomed_id from condition_occurrence
        where condition_concept_id <> 0
          and condition_concept_id not in (select snomed_id from _empirical)
          and condition_concept_id not in (select snomed_id from _vocab)
    """)
    con.execute(f"""
        create or replace temp table _ancestor as
        with cand as (
            select a.descendant_concept_id as snomed_id, v.category, a.min_levels_of_separation as levels
            from _need n join concept_ancestor a on a.descendant_concept_id = n.snomed_id
            join _vocab v on v.snomed_id = a.ancestor_concept_id
            where a.min_levels_of_separation between 1 and {int(max_levels)}
        ),
        best as (select snomed_id, min(levels) as levels from cand group by 1),
        one as (select c.snomed_id, min(c.category) as category, count(distinct c.category) as k
                from cand c join best b using (snomed_id, levels) group by 1)
        select snomed_id, category, 'ancestor' as source, 1.0 as purity from one where k = 1
    """)
    con.execute("""
        create or replace temp table snomed_bridge as
        select * from _empirical
        union all select * from _vocab where snomed_id not in (select snomed_id from _empirical)
        union all select * from _ancestor
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
    manual = (cfg.get("sensitivity", {}) or {}).get("icd9_manual_map", {}) or {}
    con.register("_manual_df", pl.DataFrame({"sv": [str(k).upper() for k in manual], "category": [str(v) for v in manual.values()]},
                                            schema={"sv": pl.Utf8, "category": pl.Utf8}).to_arrow())
    con.execute("create or replace temp table _manual as select * from _manual_df")
    cov = con.execute(f"""
        with pre as (
            select co.person_id, co.condition_concept_id as cid, co.condition_source_value as sv, co.condition_start_DATE as dt
            from condition_occurrence co left join death d on d.person_id = co.person_id
            where co.condition_start_DATE < date '{c["era_start"]}' {death_clause}
        )
        select count(*) as n_rows, count(distinct person_id) as n_patients,
               count(*) filter (where upper(trim(sv)) in (select sv from _manual)) as rows_manual,
               count(*) filter (where cid in (select snomed_id from snomed_bridge where source = 'empirical')) as rows_empirical,
               count(*) filter (where cid in (select snomed_id from snomed_bridge where source = 'vocab')) as rows_vocab,
               count(*) filter (where cid in (select snomed_id from snomed_bridge where source = 'ancestor')) as rows_ancestor,
               count(*) filter (where cid not in (select snomed_id from snomed_bridge) and upper(trim(sv)) not in (select sv from _manual)) as rows_unbridged,
               count(*) filter (where cid = 0) as rows_unmapped_concept
        from pre
    """).pl().row(0, named=True)
    pur = con.execute("select avg(purity) as mean_purity, count(*) as concepts from snomed_bridge where source = 'empirical'").pl().row(0, named=True)
    cov.update({"empirical_concepts": pur["concepts"], "empirical_mean_purity": pur["mean_purity"]})
    pc = con.execute(f"""
        with era as (
            select co.person_id, coalesce(m.category, b.category) as category, co.condition_start_DATE as dt
            from condition_occurrence co
            left join _manual m on m.sv = upper(trim(co.condition_source_value))
            left join snomed_bridge b on b.snomed_id = co.condition_concept_id
            left join death d on d.person_id = co.person_id
            where co.condition_start_DATE < date '{c["era_start"]}' {death_clause}
              and coalesce(m.category, b.category) is not null
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
