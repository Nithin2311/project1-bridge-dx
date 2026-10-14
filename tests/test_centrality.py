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
