import math

import networkx as nx
import polars as pl

from bridgedx import centrality as ce

CFG = {"communities": {"seeds": [0], "null_model": {"n": 4}}}


def _barbell():
    # two 5-cliques joined through a single bridge node x
    g = nx.Graph()
    for side in "ab":
        for i in range(5):
            for j in range(i + 1, 5):
                g.add_edge(f"{side}{i}", f"{side}{j}", weight=0.5, distance=2.0)
        g.add_edge("x", f"{side}0", weight=0.5, distance=2.0)
    for v in g.nodes:
        g.nodes[v].update(prevalence=0.1 if v != "x" else 0.3, name=v, chapter=v[0])
    part = {v: (0 if v[0] == "a" else 1 if v[0] == "b" else 0) for v in g.nodes}
    return g, part


def test_betweenness_and_null():
    g, _ = _barbell()
    bt = ce.weighted_betweenness(g)
    assert bt["x"] == max(bt.values())
    r = ce.betweenness_null_percentile(g, CFG, observed=bt)
    assert set(r) == set(g.nodes) and 0 <= r["x"]["bt_null_percentile"] <= 1
    assert r["x"]["betweenness"] == bt["x"]


def test_roles():
    g, part = _barbell()
    df = ce.roles(g, part)
    x = df.filter(pl.col("category") == "x").row(0, named=True)
    assert math.isclose(x["participation"], 0.5)  # one edge into each community
    assert x["n_communities_touched"] == 2
    a1 = df.filter(pl.col("category") == "a1").row(0, named=True)
    assert a1["participation"] == 0.0 and a1["role"] in ("R1",)
    assert set(df["role_name"].drop_nulls()) <= set(ce.ROLE_NAMES.values())


def test_prevalence_residual():
    g, _ = _barbell()
    df = ce.prevalence_residual(g)
    assert df.height == g.number_of_nodes()
    assert abs(df["degree_residual"].sum()) < 1e-9  # OLS residuals sum to zero
    assert "degree_r2" in df.columns and 0 <= df["degree_r2"][0] <= 1


def test_target_clusters_and_bridges():
    g, part = _barbell()
    cl = ce.target_clusters(part, {"A": ["a1"], "B": ["b1"]})
    assert cl["A"] == {f"a{i}" for i in range(5)} | {"x"} and cl["B"] == {f"b{i}" for i in range(5)}
    cl2 = {"A": {f"a{i}" for i in range(5)}, "B": {f"b{i}" for i in range(5)}}
    br = ce.cluster_bridges(g, cl2)
    assert br.height == 1 and br["category"][0] == "x" and br["bridge_betweenness"][0] > 0
    assert br["share_to_a"][0] == 0.5 and br["min_share"][0] == 0.5
    assert ce.direct_cluster_edges(g, cl2).height == 0
    brn = ce.cluster_bridges(g, cl2, n_null=5)
    assert {"bridge_null_percentile", "bridge_pass"} <= set(brn.columns) and 0 <= brn["bridge_null_percentile"][0] <= 1
    # missing anchor code -> empty cluster, not a crash
    assert ce.target_clusters(part, {"A": ["zzz"]})["A"] == set()


def test_cluster_bridges_prefers_two_hop_over_absent_direct():
    # A-cluster and B-cluster joined only via x (no direct edge): x must carry all restricted betweenness
    g, _ = _barbell()
    cl = {"A": {f"a{i}" for i in range(5)}, "B": {f"b{i}" for i in range(5)}}
    br = ce.cluster_bridges(g, cl)
    assert br["bridge_betweenness"][0] > 0
    g.add_edge("a0", "b0", weight=0.9, distance=1 / 0.9)  # strong direct edge now beats the x route
    br2 = ce.cluster_bridges(g, cl)
    assert br2["bridge_betweenness"][0] < br["bridge_betweenness"][0]


def test_cluster_precedence():
    import datetime as dt
    pc = pl.DataFrame({"person_id": [1, 1, 2, 2, 3, 3, 4],
                       "category": ["A1", "B1", "A1", "B1", "A1", "B1", "A1"],
                       "first_date": [dt.date(2016, 1, 1), dt.date(2016, 6, 1), dt.date(2017, 1, 1), dt.date(2016, 1, 1),
                                      dt.date(2018, 1, 1), dt.date(2018, 1, 1), dt.date(2016, 1, 1)]})
    pr = ce.cluster_precedence(pc, {"A": {"A1"}, "B": {"B1"}})
    r = pr.row(0, named=True)
    assert (r["n_both"], r["n_a_first"], r["n_b_first"], r["n_same_day"]) == (3, 1, 1, 1)
    assert r["median_gap_days_a_to_b"] == 152.0 and 0 < r["p_sign_test"] <= 1


def test_chapter_block_targets():
    t = ce.chapter_block_targets(["I10", "I25", "E11", "F32", "C78", "K21"], {"cm": [["I10", "I15"], ["E08", "E13"]], "mh": [["F20", "F48"]]})
    assert t == {"cm": {"I10", "E11"}, "mh": {"F32"}}
