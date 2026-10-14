"""Edge list with statistical filtering and phi / Jaccard weights.

Step 2 of the Option A pipeline. See docs/OPTION-A-SPEC.md.
Inputs:  outputs/patient_category.parquet, outputs/nodes.csv
Outputs: outputs/edges.csv (u, v, n_both, n_u, n_v, n_u_only, n_v_only, n_neither,
                            odds_ratio, p_fisher, p_bonf, phi, jaccard, kept)

Rules (parameters from config/option_a.yaml, section `edges`):
  * one 2x2 table per unordered category pair, built from distinct (patient, category) indicators
  * two-sided Fisher exact test on every pair among the node set
  * Bonferroni over m = C(n_nodes, 2), the number of pairs tested
  * kept = p_bonf < alpha  and  odds_ratio >= min_odds_ratio  and  n_both >= min_shared_patients
  * phi = (ad - bc) / sqrt((a+b)(c+d)(a+c)(b+d)); Jaccard = a / (a+b+c)
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import fisher_exact
from tqdm import tqdm


def pair_counts(pc: pl.DataFrame, categories: list[str] | None = None) -> pl.DataFrame:
    """For every category pair: n_both, n_u_only, n_v_only, n_neither (2x2 table counts).
    `categories` restricts (and orders) the columns; default = every category in pc."""
    cats = list(categories) if categories is not None else sorted(pc["category"].unique().to_list())
    cat_idx = {c: i for i, c in enumerate(cats)}
    sub = pc.filter(pl.col("category").is_in(cats))
    pats = pc["person_id"].unique().sort()  # N = every cohort patient, not only those with a node category
    pat_idx = {p: i for i, p in enumerate(pats.to_list())}
    n_pat = len(pats)

    m = np.zeros((n_pat, len(cats)), dtype=np.int32)
    rows = sub["person_id"].to_numpy()
    cols = sub["category"].to_numpy()
    m[[pat_idx[p] for p in rows], [cat_idx[c] for c in cols]] = 1

    both = m.T @ m                     # n_both for every ordered pair; diagonal = n_u
    n_u = np.diag(both).copy()
    iu, iv = np.triu_indices(len(cats), k=1)
    a = both[iu, iv]
    nu, nv = n_u[iu], n_u[iv]
    return pl.DataFrame({
        "u": [cats[i] for i in iu],
        "v": [cats[i] for i in iv],
        "n_both": a,
        "n_u": nu,
        "n_v": nv,
        "n_u_only": nu - a,
        "n_v_only": nv - a,
        "n_neither": n_pat - nu - nv + a,
    })


def fisher_filter(pairs: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """Add p_fisher, p_bonf, odds_ratio; kept = p_bonf < alpha and OR >= min_odds_ratio and n_both >= min_shared."""
    e = cfg["edges"]
    a = pairs["n_both"].to_numpy().astype(np.int64)
    b = pairs["n_u_only"].to_numpy().astype(np.int64)
    c = pairs["n_v_only"].to_numpy().astype(np.int64)
    d = pairs["n_neither"].to_numpy().astype(np.int64)

    with np.errstate(divide="ignore", invalid="ignore"):
        odds = np.where(b * c > 0, (a * d) / (b * c), np.inf)
        odds = np.where(a * d == 0, 0.0, odds)

    p = np.empty(len(a))
    for i in tqdm(range(len(a)), desc="fisher", unit="pair", mininterval=2.0):
        p[i] = fisher_exact([[a[i], b[i]], [c[i], d[i]]], alternative="two-sided")[1]
    m_tests = len(a)
    p_bonf = np.minimum(p * m_tests, 1.0)

    kept = (
        (p_bonf < float(e["alpha"]))
        & (odds >= float(e["min_odds_ratio"]))
        & (a >= int(e["min_shared_patients"]))
    )
    return pairs.with_columns(
        pl.Series("odds_ratio", odds),
        pl.Series("p_fisher", p),
        pl.Series("p_bonf", p_bonf),
        pl.lit(m_tests).alias("n_tests"),
        pl.Series("kept", kept),
    )


def weights(pairs: pl.DataFrame) -> pl.DataFrame:
    """phi coefficient and Jaccard from the 2x2 counts."""
    a = pairs["n_both"].cast(pl.Float64)
    b = pairs["n_u_only"].cast(pl.Float64)
    c = pairs["n_v_only"].cast(pl.Float64)
    d = pairs["n_neither"].cast(pl.Float64)
    denom = ((a + b) * (c + d) * (a + c) * (b + d)).sqrt()
    phi = ((a * d - b * c) / denom).fill_nan(0.0)
    jac = a / (a + b + c)
    return pairs.with_columns(phi.alias("phi"), jac.alias("jaccard"))
