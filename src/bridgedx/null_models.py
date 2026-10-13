"""Degree-preserving rewiring used by communities and centrality.

Step 4/5 of the Option A pipeline. See docs/OPTION-A-SPEC.md.
Inputs:  graph
Outputs: generator of rewired graphs
Status:  STUB — not implemented. Signatures are the contract; fill in, keep names.
"""
from __future__ import annotations

import networkx as nx


def rewired(g: nx.Graph, n: int, seed: int = 0):
    """Yield n degree-preserving rewired copies (nx.double_edge_swap, weights shuffled)."""
    raise NotImplementedError

