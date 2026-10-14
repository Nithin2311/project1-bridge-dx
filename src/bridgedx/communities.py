"""Community detection with null-model check and description.

Step 4 of the Option A pipeline. See docs/OPTION-A-SPEC.md.
Inputs:  graph
Outputs: outputs/communities.csv, outputs/communities_summary.csv, outputs/modularity_null.csv
"""
from __future__ import annotations

from collections import Counter

import leidenalg as la
import networkx as nx
import numpy as np
import polars as pl
from sklearn.metrics import adjusted_rand_score

from . import graph as graphmod
from . import null_models


def leiden(g: nx.Graph, cfg: dict, seed: int = 0) -> dict:
    """node -> community id (0 = largest). Weighted modularity (RB configuration model) at
    cfg['communities']['resolution']. Also run Louvain for the report (see `louvain`)."""
    h = graphmod.to_igraph(g)
    part = la.find_partition(
        h, la.RBConfigurationVertexPartition, weights="weight",
        resolution_parameter=float(cfg["communities"].get("resolution", 1.0)), seed=seed, n_iterations=-1,
    )
    return _relabel({h.vs[i]["name"]: c for i, c in enumerate(part.membership)})


def louvain(g: nx.Graph, cfg: dict, seed: int = 0) -> dict:
    comms = nx.community.louvain_communities(
        g, weight="weight", resolution=float(cfg["communities"].get("resolution", 1.0)), seed=seed)
    return _relabel({v: i for i, c in enumerate(comms) for v in c})


def _relabel(p: dict) -> dict:
    """Renumber communities by decreasing size (ties by smallest member name) so ids are stable to read."""
    sizes = Counter(p.values())
    members = {c: min(v for v, cc in p.items() if cc == c) for c in sizes}
    order = sorted(sizes, key=lambda c: (-sizes[c], members[c]))
    new = {c: i for i, c in enumerate(order)}
    return {v: new[c] for v, c in p.items()}


def modularity(g: nx.Graph, partition: dict, cfg: dict | None = None, resolution: float | None = None) -> float:
    """Reichardt-Bornholdt quality at `resolution` (default: cfg communities.resolution; 1.0 = standard
    modularity Q). Leiden optimises this quantity and the rewired null is scored on it."""
    comms = {}
    for v, c in partition.items():
        comms.setdefault(c, set()).add(v)
    if resolution is None:
        resolution = float(cfg["communities"].get("resolution", 1.0)) if cfg else 1.0
    return nx.community.modularity(g, list(comms.values()), weight="weight", resolution=resolution)


def modularity_standard(g: nx.Graph, partition: dict) -> float:
    """Standard Newman-Girvan modularity (gamma = 1) of any partition, for reporting."""
    return modularity(g, partition, resolution=1.0)


def modularity_vs_null(g: nx.Graph, partition: dict, cfg: dict) -> dict:
    """Observed modularity vs cfg['communities']['null_model']['n'] degree-preserving rewires.
    Each null graph gets its own Leiden run, so the null is 'best modularity Leiden can find on a random
    graph with this degree sequence and these weights'."""
    n = int(cfg["communities"]["null_model"]["n"])
    q_obs = modularity(g, partition, cfg)
    q_null, k_null = [], []
    for h in null_models.rewired(g, n, seed=cfg["communities"]["seeds"][0]):
        p = leiden(h, cfg, seed=0)
        q_null.append(modularity(h, p, cfg))
        k_null.append(len(set(p.values())))
    q_null = np.array(q_null)
    return {
        "resolution": float(cfg["communities"].get("resolution", 1.0)),
        "q_observed": q_obs,
        "q_standard_observed": modularity_standard(g, partition),
        "n_communities_observed": len(set(partition.values())),
        "n_null": n,
        "q_null_mean": float(q_null.mean()),
        "q_null_sd": float(q_null.std(ddof=1)),
        "q_null_max": float(q_null.max()),
        "z": float((q_obs - q_null.mean()) / q_null.std(ddof=1)),
        "p_empirical": float((np.sum(q_null >= q_obs) + 1) / (n + 1)),
        "n_communities_null_mean": float(np.mean(k_null)),
        "q_null": q_null.tolist(),
    }


def seed_stability(g: nx.Graph, cfg: dict) -> pl.DataFrame:
    """Leiden across cfg['communities']['seeds']: modularity, #communities, ARI vs seed 0."""
    seeds = cfg["communities"]["seeds"]
    parts = {s: leiden(g, cfg, seed=s) for s in seeds}
    base = parts[seeds[0]]
    nodes = list(g.nodes)
    rows = [{"seed": s, "q": modularity(g, parts[s], cfg), "q_standard": modularity_standard(g, parts[s]),
             "n_communities": len(set(parts[s].values())),
             "ari_vs_seed0": adjusted_rand_score([base[v] for v in nodes], [parts[s][v] for v in nodes])}
            for s in seeds]
    return pl.DataFrame(rows)


def describe(g: nx.Graph, partition: dict) -> pl.DataFrame:
    """Per community: size, chapter purity, top-5 nodes by within-community degree, density, clustering,
    internal/external edge ratio. Describe before naming."""
    comms = {}
    for v, c in partition.items():
        comms.setdefault(c, []).append(v)
    rows = []
    for c, members in sorted(comms.items()):
        sub = g.subgraph(members)
        mset = set(members)
        internal = sub.number_of_edges()
        external = sum(1 for v in members for u in g.neighbors(v) if u not in mset)
        internal_w = sum(d["weight"] for _, _, d in sub.edges(data=True))
        external_w = sum(g[v][u]["weight"] for v in members for u in g.neighbors(v) if u not in mset)
        chapters = Counter(g.nodes[v].get("chapter") for v in members)
        top_chapter, top_n = chapters.most_common(1)[0]
        wdeg = dict(sub.degree(weight="weight"))
        top5 = sorted(members, key=lambda v: -wdeg[v])[:5]
        rows.append({
            "community": c,
            "size": len(members),
            "top_chapter": top_chapter,
            "chapter_purity": top_n / len(members),
            "chapters": "; ".join(f"{k}:{n}" for k, n in chapters.most_common(3)),
            "share_flagged": sum(bool(g.nodes[v].get("flagged")) for v in members) / len(members),
            "top5": "; ".join(f"{v} {str(g.nodes[v].get('name'))[:30]}" for v in top5),
            "top5_codes": " ".join(top5),
            "density": nx.density(sub) if len(members) > 1 else 0.0,
            "avg_clustering": nx.average_clustering(sub) if len(members) > 2 else 0.0,
            "internal_edges": internal,
            "external_edge_ends": external,
            "internal_external_ratio": internal / external if external else float("inf"),
            "internal_external_weight_ratio": internal_w / external_w if external_w else float("inf"),
            "mean_prevalence": float(np.mean([g.nodes[v].get("prevalence", 0.0) for v in members])),
        })
    return pl.DataFrame(rows)


def partition_table(g: nx.Graph, parts: dict[str, dict]) -> pl.DataFrame:
    """node -> one column per named partition, plus name/chapter/flagged."""
    rows = [{"category": v, "name": g.nodes[v].get("name"), "chapter": g.nodes[v].get("chapter"),
             "flagged": g.nodes[v].get("flagged"), **{k: p[v] for k, p in parts.items()}} for v in g.nodes]
    return pl.DataFrame(rows).sort(list(parts)[0], "category")
