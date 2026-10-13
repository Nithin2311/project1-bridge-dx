"""Community detection with null-model check and description.

Step 4 of the Option A pipeline. See docs/OPTION-A-SPEC.md.
Inputs:  graph
Outputs: outputs/communities.csv, outputs/modularity_null.csv
Status:  STUB — not implemented. Signatures are the contract; fill in, keep names.
"""
from __future__ import annotations

import networkx as nx


def leiden(g: nx.Graph, cfg: dict, seed: int = 0) -> dict:
    """node -> community id. Also run Louvain for the report."""
    raise NotImplementedError


def modularity_vs_null(g: nx.Graph, partition: dict, cfg: dict) -> dict:
    """Observed modularity vs cfg['communities']['null_model']['n'] degree-preserving rewires."""
    raise NotImplementedError


def describe(g: nx.Graph, partition: dict) -> "pl.DataFrame":
    """Per community: size, chapter purity, top-5 nodes by within-community degree, density, clustering,
    internal/external edge ratio. Describe before naming."""
    raise NotImplementedError

