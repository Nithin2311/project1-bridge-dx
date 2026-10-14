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
