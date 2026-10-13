"""Edge list with statistical filtering and phi / Jaccard weights.

Step 2 of the Option A pipeline. See docs/OPTION-A-SPEC.md.
Inputs:  outputs/patient_category.parquet, outputs/nodes.csv
Outputs: outputs/edges.csv (u, v, n_shared, odds_ratio, p_fisher, p_bonf, phi, jaccard, kept)
Status:  STUB — not implemented. Signatures are the contract; fill in, keep names.
"""
from __future__ import annotations

import polars as pl


def pair_counts(pc: pl.DataFrame) -> pl.DataFrame:
    """For every category pair: n_both, n_u_only, n_v_only, n_neither (2x2 table counts)."""
    raise NotImplementedError


def fisher_filter(pairs: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """Add p_fisher, p_bonf, odds_ratio; kept = p_bonf < alpha and OR >= min_odds_ratio and n_both >= min_shared."""
    raise NotImplementedError


def weights(pairs: pl.DataFrame) -> pl.DataFrame:
    """phi coefficient and Jaccard from the 2x2 counts."""
    raise NotImplementedError

