"""Build NetworkX / igraph graphs from the edge list; export node and edge CSVs.

Step 3 of the Option A pipeline. See docs/OPTION-A-SPEC.md.
Inputs:  outputs/nodes.csv, outputs/edges.csv
Outputs: outputs/graph_main.graphml, outputs/graph_<subgroup>.graphml
Status:  STUB — not implemented. Signatures are the contract; fill in, keep names.
"""
from __future__ import annotations

import networkx as nx
import polars as pl


def build(nodes: pl.DataFrame, edges: pl.DataFrame, weight: str = "phi") -> nx.Graph:
    """Undirected weighted graph. Node attrs: name, chapter, prevalence, flagged. Edge attr: weight, distance=1/weight."""
    raise NotImplementedError


def to_igraph(g: nx.Graph):
    """igraph copy for Leiden."""
    raise NotImplementedError

