"""Bridge measures: betweenness with null percentile, Guimera-Amaral roles, prevalence residual.

Step 5 of the Option A pipeline. See docs/OPTION-A-SPEC.md.
Inputs:  graph, partition
Outputs: outputs/centrality.csv

Definitions
  * weighted betweenness: shortest paths on distance = 1/phi, normalized (networkx default)
  * null percentile: share of degree-preserving rewired graphs (weights shuffled) whose betweenness for the
    node is below the observed value; z = (obs - null mean) / null sd
  * participation coefficient P_i = 1 - sum_s (k_is / k_i)^2 over communities s (Guimera & Amaral 2005)
  * within-module degree z_i = (k_i^in - mean_s) / sd_s, computed within node i's community
  * roles (Guimera & Amaral 2005): non-hubs (z < 2.5): R1 ultra-peripheral P<0.05, R2 peripheral P<0.62,
    R3 non-hub connector P<0.80, R4 non-hub kinless P>=0.80; hubs (z >= 2.5): R5 provincial P<0.30,
    R6 connector hub P<0.75, R7 kinless hub P>=0.75
  * prevalence residual: OLS degree ~ prevalence; residual > 0 = more connected than frequency predicts
"""
from __future__ import annotations

from collections import defaultdict

import networkx as nx
import numpy as np
import polars as pl

from . import null_models

ROLE_NAMES = {
    "R1": "ultra-peripheral", "R2": "peripheral", "R3": "non-hub connector", "R4": "non-hub kinless",
    "R5": "provincial hub", "R6": "connector hub", "R7": "kinless hub",
}


def weighted_betweenness(g: nx.Graph) -> dict:
    """distance = 1/phi."""
    return nx.betweenness_centrality(g, weight="distance", normalized=True)


def other_centralities(g: nx.Graph) -> pl.DataFrame:
    """Closeness (distance-weighted) and eigenvector (phi-weighted) for the required-technique table."""
    clo = nx.closeness_centrality(g, distance="distance")
    try:
        eig = nx.eigenvector_centrality(g, weight="weight", max_iter=2000, tol=1e-8)
    except nx.PowerIterationFailedConvergence:
        eig = nx.eigenvector_centrality_numpy(g, weight="weight")
    return pl.DataFrame({"category": list(g.nodes), "closeness": [clo[v] for v in g.nodes],
                         "eigenvector": [eig[v] for v in g.nodes]})


def betweenness_null_percentile(g: nx.Graph, cfg: dict, observed: dict | None = None) -> dict:
    """Percentile of each node's observed betweenness against rewired nulls.
    Returns {node: {"betweenness": obs, "null_mean", "null_sd", "percentile", "z"}}."""
    n = int(cfg["communities"]["null_model"]["n"])
    seed = int(cfg["communities"]["seeds"][0])
    obs = observed or weighted_betweenness(g)
    nodes = list(g.nodes)
    null = np.empty((n, len(nodes)))
    for i, h in enumerate(null_models.rewired(g, n, seed=seed + 1000)):
        b = weighted_betweenness(h)
        null[i] = [b[v] for v in nodes]
    out = {}
    for j, v in enumerate(nodes):
        col = null[:, j]
        sd = col.std(ddof=1) if n > 1 else 0.0
        out[v] = {
            "betweenness": obs[v],
            "bt_null_mean": float(col.mean()),
            "bt_null_sd": float(sd),
            "bt_null_percentile": float((col < obs[v]).mean()),
            "bt_null_z": float((obs[v] - col.mean()) / sd) if sd > 0 else 0.0,
        }
    return out


def _role(z: float, p: float) -> str:
    if z < 2.5:
        return "R1" if p < 0.05 else "R2" if p < 0.62 else "R3" if p < 0.80 else "R4"
    return "R5" if p < 0.30 else "R6" if p < 0.75 else "R7"


def roles(g: nx.Graph, partition: dict) -> pl.DataFrame:
    """Participation coefficient and within-module degree z (Guimera & Amaral 2005).
    Unweighted versions drive the role label; phi-weighted versions are reported alongside."""
    rows = []
    kin = {}
    kin_w = {}
    for v in g.nodes:
        c = partition[v]
        per = defaultdict(float)
        per_w = defaultdict(float)
        for u in g.neighbors(v):
            per[partition[u]] += 1
            per_w[partition[u]] += g[v][u]["weight"]
        k = sum(per.values())
        kw = sum(per_w.values())
        kin[v] = per.get(c, 0.0)
        kin_w[v] = per_w.get(c, 0.0)
        p = 1 - sum((x / k) ** 2 for x in per.values()) if k else 0.0
        pw = 1 - sum((x / kw) ** 2 for x in per_w.values()) if kw else 0.0
        rows.append({"category": v, "community": c, "degree": int(k), "weighted_degree": kw,
                     "k_in": kin[v], "k_in_w": kin_w[v], "participation": p, "participation_w": pw,
                     "n_communities_touched": sum(1 for x in per.values() if x > 0)})
    df = pl.DataFrame(rows)
    # within-module z: standardize k_in inside each community (population sd; sd 0 -> z 0)
    stats = df.group_by("community").agg(
        pl.col("k_in").mean().alias("m"), pl.col("k_in").std(ddof=0).alias("s"),
        pl.col("k_in_w").mean().alias("mw"), pl.col("k_in_w").std(ddof=0).alias("sw"))
    df = df.join(stats, on="community").with_columns(
        pl.when(pl.col("s") > 0).then((pl.col("k_in") - pl.col("m")) / pl.col("s")).otherwise(0.0).alias("within_module_z"),
        pl.when(pl.col("sw") > 0).then((pl.col("k_in_w") - pl.col("mw")) / pl.col("sw")).otherwise(0.0).alias("within_module_z_w"),
    ).drop("m", "s", "mw", "sw")
    df = df.with_columns(
        pl.struct("within_module_z", "participation").map_elements(
            lambda r: _role(r["within_module_z"], r["participation"]), return_dtype=pl.Utf8).alias("role"))
    return df.with_columns(pl.col("role").replace_strict(ROLE_NAMES, default=None).alias("role_name"))


def prevalence_residual(g: nx.Graph) -> pl.DataFrame:
    """Residual of degree ~ prevalence: which nodes are more connected than frequency predicts.
    OLS on degree and on weighted degree; residual_z = residual / residual sd."""
    nodes = list(g.nodes)
    prev = np.array([float(g.nodes[v].get("prevalence", 0.0)) for v in nodes])
    out = {"category": nodes, "prevalence": prev}
    for col, vals in [("degree", np.array([g.degree(v) for v in nodes], dtype=float)),
                      ("weighted_degree", np.array([g.degree(v, weight="weight") for v in nodes]))]:
        slope, intercept = np.polyfit(prev, vals, 1)
        fitted = intercept + slope * prev
        resid = vals - fitted
        ss_res, ss_tot = float((resid ** 2).sum()), float(((vals - vals.mean()) ** 2).sum())
        out[f"{col}_fitted"] = fitted
        out[f"{col}_residual"] = resid
        out[f"{col}_residual_z"] = resid / resid.std(ddof=1)
        out[f"{col}_r2"] = np.full(len(nodes), 1 - ss_res / ss_tot if ss_tot else 0.0)
    return pl.DataFrame(out)
