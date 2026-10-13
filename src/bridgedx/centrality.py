"""Bridge measures: betweenness with null percentile, Guimera-Amaral roles, prevalence residual.

Step 5 of the Option A pipeline. See docs/OPTION-A-SPEC.md.
Inputs:  graph, partition
Outputs: outputs/centrality.csv
Status:  STUB — not implemented. Signatures are the contract; fill in, keep names.
"""
from __future__ import annotations

import networkx as nx


def weighted_betweenness(g: nx.Graph) -> dict:
    """distance = 1/phi."""
    raise NotImplementedError


def betweenness_null_percentile(g: nx.Graph, cfg: dict) -> dict:
    """Percentile of each node's observed betweenness against rewired nulls."""
    raise NotImplementedError


def roles(g: nx.Graph, partition: dict) -> "pl.DataFrame":
    """Participation coefficient and within-module degree z (Guimera & Amaral 2005)."""
    raise NotImplementedError


def prevalence_residual(g: nx.Graph) -> "pl.DataFrame":
    """Residual of degree ~ prevalence: which nodes are more connected than frequency predicts."""
    raise NotImplementedError

