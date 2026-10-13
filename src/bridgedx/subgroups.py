"""Subgroup graphs (sex, age) on the same node set, and the comparison tests.

Step 6 of the Option A pipeline. See docs/OPTION-A-SPEC.md.
Inputs:  patient_category, person, nodes
Outputs: outputs/subgroup_<name>_edges.csv, outputs/subgroup_comparison.csv
Status:  STUB — not implemented. Signatures are the contract; fill in, keep names.
"""
from __future__ import annotations

def assign_subgroups(con, cfg: dict) -> "pl.DataFrame":
    """person_id -> sex, age_band. Age at first ICD-10-era diagnosis (write the rule down once)."""
    raise NotImplementedError


def compare_partitions(p_a: dict, p_b: dict) -> float:
    """Adjusted Rand index."""
    raise NotImplementedError


def connector_overlap(c_a: list, c_b: list, k: int = 20) -> float:
    raise NotImplementedError


def edgewise_permutation(pc, labels, cfg: dict) -> "pl.DataFrame":
    """Edge-wise phi difference with cfg['subgroups']['comparison']['n_permutations'] label permutations."""
    raise NotImplementedError

