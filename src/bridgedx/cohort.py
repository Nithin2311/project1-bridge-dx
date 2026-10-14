"""Cohort and node list.

Step 1 of the Option A pipeline. See docs/OPTION-A-SPEC.md.
Inputs:  condition_occurrence, person, death, concept (via bridgedx.data)
Outputs: outputs/patient_category.parquet (distinct person_id x category), outputs/nodes.csv

Rules (all parameters come from config/option_a.yaml):
  * era:        condition_start_DATE >= cohort.era_start AND the source code parses as ICD-10
  * dedupe:     distinct (person_id, category); the earliest start date in the era is kept as first_date
  * post-death: rows dated strictly after death_DATE are dropped (same-day rows kept)
  * cohort:     patients with >= cohort.min_distinct_categories distinct categories
  * nodes:      categories with prevalence >= nodes.min_prevalence of the cohort
  * names:      ICD10CM 3-char concept in concept.csv (fallback: ICD10, then the code itself)
  * flagged:    first letter in nodes.flag_chapters (R = symptoms, Z = status/admin)
"""
from __future__ import annotations

import re

import polars as pl

# ICD-10-CM category: letter, digit, then digit OR letter (D3A, C7A, C7B, Z3A, M1A are real categories).
_ICD10_CAT = re.compile(r"^([A-Z][0-9][0-9A-Z])(?:\.|$|[0-9A-Z])")
# ICD-9 look-alikes that must not pass: E800-E999 external-cause codes, and V-codes in ICD-9 form
# (V## optionally .d or .dd). ICD-10 V-codes carry a 7th-character letter or >=3 chars after the dot.
_ICD9_E = re.compile(r"^E[89][0-9]{2}")
_ICD9_V = re.compile(r"^V[0-9]{2}(\.[0-9]{0,2})?$")

# ICD-10-CM chapters, by first letter and 2-digit number of the category.
_CHAPTERS = [
    ("A", 0, "B", 99, "infectious"),
    ("C", 0, "D", 49, "neoplasms"),
    ("D", 50, "D", 89, "blood/immune"),
    ("E", 0, "E", 89, "endocrine/metabolic"),
    ("F", 1, "F", 99, "mental/behavioral"),
    ("G", 0, "G", 99, "nervous"),
    ("H", 0, "H", 59, "eye"),
    ("H", 60, "H", 95, "ear"),
    ("I", 0, "I", 99, "circulatory"),
    ("J", 0, "J", 99, "respiratory"),
    ("K", 0, "K", 95, "digestive"),
    ("L", 0, "L", 99, "skin"),
    ("M", 0, "M", 99, "musculoskeletal"),
    ("N", 0, "N", 99, "genitourinary"),
    ("O", 0, "O", 99, "pregnancy"),
    ("P", 0, "P", 96, "perinatal"),
    ("Q", 0, "Q", 99, "congenital"),
    ("R", 0, "R", 99, "symptoms/abnormal findings"),
    ("S", 0, "T", 88, "injury/poisoning"),
    ("U", 0, "U", 85, "special purpose"),
    ("V", 0, "Y", 99, "external causes"),
    ("Z", 0, "Z", 99, "factors influencing health status"),
]


def icd10_3char(source_value: str | None) -> str | None:
    """'I10.9' -> 'I10'; 'E11.65' -> 'E11'; 'D3A.8' -> 'D3A'. Return None for non-ICD-10 strings
    ('401.9', 'E812.0', 'V12.3', '19004', '')."""
    if source_value is None:
        return None
    code = str(source_value).strip().upper()
    code = re.sub(r"^NULL-", "", code)  # prefix left by the shc_v_cancer_staging source system
    if not code or _ICD9_E.match(code) or _ICD9_V.match(code):
        return None
    m = _ICD10_CAT.match(code)
    return m.group(1) if m else None


def chapter_of(category: str) -> str:
    """ICD-10-CM chapter label for a 3-char category."""
    letter = category[0]
    num = int(category[1]) * 10 + (int(category[2]) if category[2].isdigit() else 0)
    for lo_l, lo_n, hi_l, hi_n, name in _CHAPTERS:
        if (letter, num) >= (lo_l, lo_n) and (letter, num) <= (hi_l, hi_n):
            return name
    return "unknown"


def _register_category_map(con) -> None:
    """Map every distinct condition_source_value to its 3-char category once (18k strings),
    so the 2.5M-row table is never processed in Python."""
    vals = con.execute(
        "select distinct condition_source_value as sv from condition_occurrence where condition_source_value is not null"
    ).pl()
    mapped = vals.with_columns(
        pl.col("sv").map_elements(icd10_3char, return_dtype=pl.Utf8).alias("category")
    ).drop_nulls("category")
    con.register("_cat_map_df", mapped.to_arrow())
    con.execute("create or replace temp table cat_map as select * from _cat_map_df")


def build_patient_category(con, cfg: dict) -> pl.DataFrame:
    """Distinct (person_id, category) in the ICD-10 era, post-death rows dropped,
    patients with >= cfg['cohort']['min_distinct_categories'] kept.
    Columns: person_id, category, first_date (earliest era start date of that category)."""
    c = cfg["cohort"]
    _register_category_map(con)
    death_clause = (
        "and (d.death_DATE is null or co.condition_start_DATE <= d.death_DATE)"
        if c.get("drop_events_after_death", True) else ""
    )
    sql = f"""
    with era as (
        select co.person_id, m.category, co.condition_start_DATE as dt
        from condition_occurrence co
        join cat_map m on m.sv = co.condition_source_value
        left join death d on d.person_id = co.person_id
        where co.condition_start_DATE >= date '{c["era_start"]}'
        {death_clause}
    ),
    pc as (
        select person_id, category, min(dt) as first_date
        from era group by 1, 2
    ),
    keep as (
        select person_id from pc group by 1
        having count(*) >= {int(c["min_distinct_categories"])}
    )
    select pc.person_id, pc.category, pc.first_date
    from pc join keep using (person_id)
    order by person_id, category
    """
    return con.execute(sql).pl()


def cohort_funnel(con, cfg: dict) -> pl.DataFrame:
    """Patient counts at each inclusion step, for the Data Preparation section."""
    c = cfg["cohort"]
    _register_category_map(con)
    sql = f"""
    with base as (
        select co.person_id, co.condition_source_value as sv, co.condition_start_DATE as dt, m.category,
               d.death_DATE
        from condition_occurrence co
        left join cat_map m on m.sv = co.condition_source_value
        left join death d on d.person_id = co.person_id
    )
    select 'patients in person'                                   as step, (select count(*) from person) as patients, null::bigint as rows_
    union all select 'patients with >=1 condition row',           count(distinct person_id), count(*) from base
    union all select 'rows on/after era_start',                   count(distinct person_id), count(*) from base where dt >= date '{c["era_start"]}'
    union all select 'rows on/after era_start with ICD-10 code', count(distinct person_id), count(*) from base where dt >= date '{c["era_start"]}' and category is not null
    union all select 'rows dated after death (dropped)',          count(distinct person_id), count(*) from base where dt >= date '{c["era_start"]}' and category is not null and dt > death_DATE
    union all select 'distinct (patient, category) pairs',        count(distinct person_id), count(*) from (
                  select distinct person_id, category from base
                  where dt >= date '{c["era_start"]}' and category is not null and (death_DATE is null or dt <= death_DATE))
    """
    return con.execute(sql).pl()


def build_nodes(pc: pl.DataFrame, con, cfg: dict) -> pl.DataFrame:
    """Categories with prevalence >= cfg['nodes']['min_prevalence'];
    columns: category, name (from concept), chapter, n_patients, prevalence, flagged (R/Z)."""
    n_cohort = pc["person_id"].n_unique()
    min_prev = float(cfg["nodes"]["min_prevalence"])
    flag = set(cfg["nodes"]["flag_chapters"])

    counts = (
        pc.group_by("category").agg(pl.col("person_id").n_unique().alias("n_patients"))
        .with_columns((pl.col("n_patients") / n_cohort).alias("prevalence"))
        .filter(pl.col("prevalence") >= min_prev)
    )

    # Names: ICD10CM 3-char concept, fallback ICD10 (WHO), fallback the code itself.
    con.register("_node_codes", counts.select("category").to_arrow())
    names = con.execute(
        """
        select n.category,
               coalesce(max(c.concept_name) filter (where c.vocabulary_id = 'ICD10CM'),
                        max(c.concept_name) filter (where c.vocabulary_id = 'ICD10')) as name
        from _node_codes n
        left join concept c on c.concept_code = n.category and c.vocabulary_id in ('ICD10CM', 'ICD10')
        group by n.category
        """
    ).pl()

    nodes = (
        counts.join(names, on="category", how="left")
        .with_columns(
            pl.col("name").fill_null(pl.col("category")),
            pl.col("category").map_elements(chapter_of, return_dtype=pl.Utf8).alias("chapter"),
            pl.col("category").str.slice(0, 1).is_in(list(flag)).alias("flagged"),
        )
        .select("category", "name", "chapter", "n_patients", "prevalence", "flagged")
        .sort("n_patients", "category", descending=[True, False])
    )
    return nodes
