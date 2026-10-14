import math

import polars as pl

from bridgedx import graph


def _inputs():
    nodes = pl.DataFrame({
        "category": ["A", "B", "C", "D"], "name": ["a", "b", "c", "d"],
        "chapter": ["x", "x", "y", "y"], "n_patients": [4, 4, 4, 1],
        "prevalence": [0.4, 0.4, 0.4, 0.1], "flagged": [False, False, False, True],
    })
    edges = pl.DataFrame({
        "u": ["A", "A", "B", "A"], "v": ["B", "C", "C", "D"],
        "n_both": [3, 4, 3, 1], "odds_ratio": [3.0, 9.0, 3.0, 0.5], "p_bonf": [0.01] * 4,
        "phi": [0.5, 0.25, 0.5, -0.1], "jaccard": [0.6, 1.0, 0.6, 0.1],
        "kept": [True, True, True, False],
    })
    return nodes, edges


def test_build_and_stats():
    nodes, edges = _inputs()
    g = graph.build(nodes, edges)
    assert g.number_of_nodes() == 4 and g.number_of_edges() == 3  # D isolate, unkept edge dropped
    assert g.nodes["D"]["flagged"] is True and g.nodes["A"]["name"] == "a"
    assert g["A"]["C"]["weight"] == 0.25 and g["A"]["C"]["distance"] == 4.0
    s = graph.basic_stats(g)
    assert s["n_components"] == 2 and s["giant_component_nodes"] == 3 and s["n_isolates"] == 1
    assert math.isclose(s["density"], 3 / 6)
    assert math.isclose(s["transitivity"], 1.0)
    nt = graph.node_table(g)
    assert nt.filter(pl.col("category") == "A").row(0, named=True)["weighted_degree"] == 0.75
    et = graph.edge_table(g)
    assert et.height == 3 and et["weight"][0] == 0.5
    h = graph.to_igraph(g)
    assert h.vcount() == 4 and h.ecount() == 3 and set(h.vs["name"]) == {"A", "B", "C", "D"}
    assert math.isclose(sum(h.es["weight"]), 1.25)
