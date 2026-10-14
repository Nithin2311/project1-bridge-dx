"""Degree-preserving rewiring used by communities and centrality.

Step 4/5 of the Option A pipeline. See docs/OPTION-A-SPEC.md.
Inputs:  graph
Outputs: generator of rewired graphs
"""
from __future__ import annotations

import random

import networkx as nx


def rewired(g: nx.Graph, n: int, seed: int = 0, swaps_per_edge: int = 10):
    """Yield n degree-preserving rewired copies (nx.double_edge_swap, weights shuffled).
    Each copy: m*swaps_per_edge successful swaps, then the multiset of edge weights is reassigned at random,
    so degree sequence and weight distribution are kept while topology and weight placement are randomized."""
    m = g.number_of_edges()
    weights = [d["weight"] for _, _, d in g.edges(data=True)]
    for i in range(n):
        rng = random.Random(seed + i)
        h = nx.Graph()
        h.add_nodes_from(g.nodes(data=True))
        h.add_edges_from(g.edges())
        if m > 1:
            nx.double_edge_swap(h, nswap=m * swaps_per_edge, max_tries=m * swaps_per_edge * 20, seed=seed + i)
        w = weights[:]
        rng.shuffle(w)
        for (u, v), wt in zip(h.edges(), w):
            h[u][v]["weight"] = wt
            h[u][v]["distance"] = 1.0 / wt
        h.graph.update(g.graph)
        yield h
