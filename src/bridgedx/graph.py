"""Build NetworkX / igraph graphs from the edge list; export node and edge CSVs.

Step 3 of the Option A pipeline. See docs/OPTION-A-SPEC.md.
Inputs:  outputs/nodes.csv, outputs/edges.csv (all tested pairs with a `kept` flag)
Outputs: outputs/graph_main.graphml, outputs/graph_nodes.csv, outputs/graph_edges.csv
         (graph_<subgroup>.* written by step 6 with the same functions)
"""
from __future__ import annotations

import igraph as ig
import networkx as nx
import polars as pl

NODE_ATTRS = ["name", "chapter", "n_patients", "prevalence", "flagged"]
EDGE_ATTRS = ["n_both", "odds_ratio", "p_bonf", "phi", "jaccard"]


def build(nodes: pl.DataFrame, edges: pl.DataFrame, weight: str = "phi") -> nx.Graph:
    """Undirected weighted graph. Node attrs: name, chapter, prevalence, flagged. Edge attr: weight, distance=1/weight.
    Every node in `nodes` is added (isolates included); only edges with kept == True (if the column exists)
    whose endpoints are both nodes are used."""
    g = nx.Graph()
    for r in nodes.iter_rows(named=True):
        g.add_node(r["category"], **{k: r[k] for k in NODE_ATTRS if k in r})
    e = edges.filter("kept") if "kept" in edges.columns else edges
    e = e.filter(pl.col("u").is_in(g.nodes) & pl.col("v").is_in(g.nodes) & (pl.col(weight) > 0))
    for r in e.iter_rows(named=True):
        attrs = {k: r[k] for k in EDGE_ATTRS if k in r}
        attrs["weight"] = float(r[weight])
        attrs["distance"] = 1.0 / float(r[weight])
        g.add_edge(r["u"], r["v"], **attrs)
    g.graph["weight"] = weight
    return g


def to_igraph(g: nx.Graph) -> ig.Graph:
    """igraph copy for Leiden. Vertex attr 'name' = category; edge attr 'weight' copied."""
    names = list(g.nodes)
    idx = {n: i for i, n in enumerate(names)}
    h = ig.Graph(n=len(names), edges=[(idx[u], idx[v]) for u, v in g.edges], directed=False)
    h.vs["name"] = names
    h.es["weight"] = [d["weight"] for _, _, d in g.edges(data=True)]
    return h


def basic_stats(g: nx.Graph) -> dict:
    """Density, clustering, components; the numbers for the Network Construction section."""
    n, m = g.number_of_nodes(), g.number_of_edges()
    comps = sorted(nx.connected_components(g), key=len, reverse=True)
    gc = g.subgraph(comps[0]) if comps else g
    deg = dict(g.degree())
    wdeg = dict(g.degree(weight="weight"))
    stats = {
        "n_nodes": n,
        "n_edges": m,
        "density": nx.density(g),
        "degree_mean": sum(deg.values()) / n if n else 0.0,
        "degree_median": float(sorted(deg.values())[n // 2]) if n else 0.0,
        "degree_max": max(deg.values()) if n else 0,
        "weighted_degree_mean": sum(wdeg.values()) / n if n else 0.0,
        "avg_clustering": nx.average_clustering(g),
        "avg_clustering_weighted": nx.average_clustering(g, weight="weight"),
        "transitivity": nx.transitivity(g),
        "n_components": len(comps),
        "giant_component_nodes": gc.number_of_nodes(),
        "n_isolates": sum(1 for _, d in deg.items() if d == 0),
        "degree_assortativity": nx.degree_assortativity_coefficient(g) if m else float("nan"),
        "avg_shortest_path_gc": nx.average_shortest_path_length(gc) if gc.number_of_nodes() > 1 else 0.0,
        "diameter_gc": nx.diameter(gc) if gc.number_of_nodes() > 1 else 0,
    }
    return stats


def node_table(g: nx.Graph) -> pl.DataFrame:
    """Node list with graph-derived columns (degree, weighted degree, clustering) for export."""
    clus = nx.clustering(g)
    wclus = nx.clustering(g, weight="weight")
    rows = []
    for v, d in g.nodes(data=True):
        rows.append({"category": v, **{k: d.get(k) for k in NODE_ATTRS},
                     "degree": g.degree(v), "weighted_degree": g.degree(v, weight="weight"),
                     "clustering": clus[v], "clustering_weighted": wclus[v]})
    return pl.DataFrame(rows).sort("degree", "category", descending=[True, False])


def edge_table(g: nx.Graph) -> pl.DataFrame:
    rows = [{"u": u, "v": v, "weight": d["weight"], "distance": d["distance"],
             **{k: d.get(k) for k in EDGE_ATTRS}} for u, v, d in g.edges(data=True)]
    return pl.DataFrame(rows).sort("weight", descending=True)


def write_graphml(g: nx.Graph, path) -> None:
    nx.write_graphml(g, path)
