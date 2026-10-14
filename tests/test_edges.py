"""Edge statistics on a tiny synthetic patient-category table."""
import math

import polars as pl
from scipy.stats import fisher_exact

from bridgedx import edges

CFG = {"edges": {"alpha": 0.05, "min_odds_ratio": 1.2, "min_shared_patients": 2}}


def _pc():
    # 6 patients. A and B co-occur in 3; C is in everyone with A; D alone.
    rows = [
        (1, "A"), (1, "B"), (1, "C"),
        (2, "A"), (2, "B"), (2, "C"),
        (3, "A"), (3, "B"), (3, "C"),
        (4, "A"), (4, "C"),
        (5, "D"),
        (6, "D"), (6, "B"),
    ]
    return pl.DataFrame(rows, schema=["person_id", "category"], orient="row")


def test_pair_counts_2x2():
    p = edges.pair_counts(_pc())
    ab = p.filter((pl.col("u") == "A") & (pl.col("v") == "B")).row(0, named=True)
    assert (ab["n_both"], ab["n_u_only"], ab["n_v_only"], ab["n_neither"]) == (3, 1, 1, 1)
    assert ab["n_u"] == 4 and ab["n_v"] == 4
    assert p.height == 6  # C(4,2)
    # restricting categories drops pairs and keeps counts over the same patients
    p2 = edges.pair_counts(_pc(), categories=["A", "C"])
    assert p2.height == 1 and p2["n_both"][0] == 4 and p2["n_neither"][0] == 2


def test_fisher_and_weights():
    p = edges.weights(edges.fisher_filter(edges.pair_counts(_pc()), CFG))
    ab = p.filter((pl.col("u") == "A") & (pl.col("v") == "B")).row(0, named=True)
    assert math.isclose(ab["p_fisher"], fisher_exact([[3, 1], [1, 1]])[1])
    assert math.isclose(ab["p_bonf"], min(1.0, ab["p_fisher"] * 6))
    assert math.isclose(ab["odds_ratio"], 3.0)
    assert math.isclose(ab["phi"], (3 * 1 - 1 * 1) / math.sqrt(4 * 2 * 4 * 2))
    assert math.isclose(ab["jaccard"], 3 / 5)
    assert ab["kept"] is False  # not significant at n=6
    ac = p.filter((pl.col("u") == "A") & (pl.col("v") == "C")).row(0, named=True)
    assert ac["odds_ratio"] == math.inf and math.isclose(ac["phi"], 1.0) and ac["jaccard"] == 1.0
    ad = p.filter((pl.col("u") == "A") & (pl.col("v") == "D")).row(0, named=True)
    assert ad["odds_ratio"] == 0.0 and ad["phi"] < 0 and ad["kept"] is False
