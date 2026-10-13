"""Cohort and node list.

Step 1 of the Option A pipeline. See docs/OPTION-A-SPEC.md.
Inputs:  condition_occurrence, person, death, concept (via bridgedx.data)
Outputs: outputs/patient_category.parquet (distinct person_id x category), outputs/nodes.csv
Status:  STUB — not implemented. Signatures are the contract; fill in, keep names.
"""
from __future__ import annotations

import polars as pl


def icd10_3char(source_value: str) -> str | None:
    """'I10.9' -> 'I10'; 'E11.65' -> 'E11'. Return None for non-ICD-10 strings."""
    raise NotImplementedError


def build_patient_category(con, cfg: dict) -> pl.DataFrame:
    """Distinct (person_id, category) in the ICD-10 era, post-death rows dropped,
    patients with >= cfg['cohort']['min_distinct_categories'] kept."""
    raise NotImplementedError


def build_nodes(pc: pl.DataFrame, con, cfg: dict) -> pl.DataFrame:
    """Categories with prevalence >= cfg['nodes']['min_prevalence'];
    columns: category, name (from concept), chapter, n_patients, prevalence, flagged (R/Z)."""
    raise NotImplementedError

