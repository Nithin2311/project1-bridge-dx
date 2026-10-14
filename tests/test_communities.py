import networkx as nx

from bridgedx import communities as cm
from bridgedx import null_models

CFG = {"communities": {"resolution": 1.0, "seeds": [0, 1], "null_model": {"n": 3}}}


def _two_cliques():
    g = nx.Graph()
    for i in range(6):
        for j in range(i + 1, 6):
            g.add_edge(f"a{i}", f"a{j}", weight=0.5)
            g.add_edge(f"b{i}", f"b{j}", weight=0.5)
    g.add_edge("a0", "b0", weight=0.1)
    for v in g.nodes:
        g.nodes[v].update(name=v, chapter=v[0], flagged=(v[0] == "b"), prevalence=0.1)
    return g


def test_leiden_finds_two_cliques():
    g = _two_cliques()
    p = cm.leiden(g, CFG, seed=0)
    assert len(set(p.values())) == 2
    assert len({p[f"a{i}"] for i in range(6)}) == 1 and p["a0"] != p["b0"]
    assert cm.modularity(g, p, CFG) > 0.4
    q = cm.louvain(g, CFG, seed=0)
    assert len(set(q.values())) == 2
    d = cm.describe(g, p)
    assert d.height == 2 and d["chapter_purity"].min() == 1.0 and d["size"].to_list() == [6, 6]
    assert d["internal_edges"].to_list() == [15, 15] and d["external_edge_ends"].to_list() == [1, 1]
    assert sorted(d["share_flagged"].to_list()) == [0.0, 1.0]


def test_rewired_preserves_degrees_and_weights():
    g = _two_cliques()
    deg = dict(g.degree())
    w = sorted(d["weight"] for _, _, d in g.edges(data=True))
    hs = list(null_models.rewired(g, 3, seed=0))
    assert len(hs) == 3
    for h in hs:
        assert dict(h.degree()) == deg
        assert sorted(d["weight"] for _, _, d in h.edges(data=True)) == w
        assert set(h.nodes) == set(g.nodes)
    assert any(set(h.edges) != set(g.edges) for h in hs)


def test_modularity_vs_null_and_stability():
    g = _two_cliques()
    p = cm.leiden(g, CFG, seed=0)
    r = cm.modularity_vs_null(g, p, CFG)
    assert r["n_null"] == 3 and len(r["q_null"]) == 3 and r["q_observed"] >= r["q_null_max"]
    s = cm.seed_stability(g, CFG)
    assert s.height == 2 and s["ari_vs_seed0"].to_list() == [1.0, 1.0]
