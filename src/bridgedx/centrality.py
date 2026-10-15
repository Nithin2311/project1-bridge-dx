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
            # z is meaningless when the null mass sits at ~0 (tiny observed values give z in the hundreds)
            "bt_null_z": float((obs[v] - col.mean()) / sd) if sd > 0 and col.mean() >= 1e-4 else float("nan"),
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


def target_clusters(partition: dict, anchors: dict[str, list[str]]) -> dict[str, set]:
    """cluster name -> set of nodes = union of communities containing the anchor codes."""
    comms = defaultdict(set)
    for v, c in partition.items():
        comms[c].add(v)
    out = {}
    for name, codes in anchors.items():
        members = set()
        for code in codes:
            if code in partition:
                members |= comms[partition[code]]
        out[name] = members
    return out


def _restricted_betweenness(g: nx.Graph, clusters: dict[str, set]) -> dict[str, dict]:
    names = list(clusters)
    out = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            out[f"{a}-{b}"] = nx.betweenness_centrality_subset(
                g, sources=list(clusters[a]), targets=list(clusters[b]), weight="distance", normalized=True)
    return out


def cluster_bridges(g: nx.Graph, clusters: dict[str, set], cfg: dict | None = None, n_null: int | None = None,
                    seed: int = 0) -> pl.DataFrame:
    """For every pair of target clusters: betweenness restricted to shortest paths (distance 1/phi) from
    cluster A nodes to cluster B nodes, for nodes outside both clusters; each node's phi-weight share into A
    and into B; and, if n_null > 0 (default cfg centrality.bridge_null_n), the percentile and z of the
    restricted betweenness against degree-preserving rewired graphs (weights shuffled). One row per (pair, node)."""
    if n_null is None:
        n_null = int(cfg["centrality"].get("bridge_null_n", 0)) if cfg else 0
    pct_min = float(cfg["centrality"].get("bridge_null_percentile_min", 0.95)) if cfg else 0.95
    min_z = float(cfg["centrality"].get("bridge_min_z", 2.0)) if cfg else 2.0
    min_share = float(cfg["centrality"].get("bridge_min_share_of_max", 0.10)) if cfg else 0.10
    names = list(clusters)
    wdeg = dict(g.degree(weight="weight"))
    obs = _restricted_betweenness(g, clusters)
    nodes = list(g.nodes)
    null = {pair: np.empty((n_null, len(nodes))) for pair in obs} if n_null else {}
    if n_null:
        for i, h in enumerate(null_models.rewired(g, n_null, seed=seed + 2000)):
            r = _restricted_betweenness(h, clusters)
            for pair in obs:
                null[pair][i] = [r[pair][v] for v in nodes]
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            A, B = clusters[a], clusters[b]
            pair = f"{a}-{b}"
            pair_max = max((obs[pair][v] for v in nodes if v not in A and v not in B), default=0.0)
            for idx, v in enumerate(nodes):
                if v in A or v in B:
                    continue
                w_a = sum(g[v][u]["weight"] for u in g.neighbors(v) if u in A)
                w_b = sum(g[v][u]["weight"] for u in g.neighbors(v) if u in B)
                row = {"pair": pair, "category": v, "bridge_betweenness": obs[pair][v],
                       "w_to_a": w_a, "w_to_b": w_b,
                       "share_to_a": w_a / wdeg[v] if wdeg[v] else 0.0,
                       "share_to_b": w_b / wdeg[v] if wdeg[v] else 0.0,
                       "min_share": min(w_a, w_b) / wdeg[v] if wdeg[v] else 0.0}
                if n_null:
                    col = null[pair][:, idx]
                    sd = col.std(ddof=1) if n_null > 1 else 0.0
                    pct = float((col < obs[pair][v]).mean())
                    z = float((obs[pair][v] - col.mean()) / sd) if sd > 0 and col.mean() >= 1e-5 else float("nan")
                    o = obs[pair][v]
                    magnitude_ok = (z == z and z >= min_z) or (pair_max > 0 and o >= min_share * pair_max)
                    row.update({"bridge_null_mean": float(col.mean()), "bridge_null_percentile": pct, "bridge_null_z": z,
                                "bridge_share_of_pair_max": o / pair_max if pair_max > 0 else 0.0,
                                "bridge_pass": bool(pct >= pct_min and o > 0 and magnitude_ok)})
                rows.append(row)
    df = pl.DataFrame(rows)
    return df.sort("pair", "bridge_betweenness", descending=[False, True])


def cluster_precedence(pc: pl.DataFrame, clusters: dict[str, set]) -> pl.DataFrame:
    """Directed first-date precedence between target clusters (sequential analysis). For each ordered pair
    (A, B) among patients who enter both: n_A_first, n_B_first, n_same_day, median gap days, two-sided
    binomial sign test on A-first vs B-first."""
    from scipy.stats import binomtest
    entry = {}
    for name, members in clusters.items():
        e = (pc.filter(pl.col("category").is_in(list(members)))
             .group_by("person_id").agg(pl.col("first_date").min().alias(name)))
        entry[name] = e
    names = list(clusters)
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            d = entry[a].join(entry[b], on="person_id", how="inner")
            gap = (d[b] - d[a]).dt.total_days()
            n_a, n_b, n_same = int((gap > 0).sum()), int((gap < 0).sum()), int((gap == 0).sum())
            p = binomtest(n_a, n_a + n_b, 0.5).pvalue if n_a + n_b else 1.0
            rows.append({"from": a, "to": b, "n_both": d.height, "n_a_first": n_a, "n_b_first": n_b, "n_same_day": n_same,
                         "share_a_first": n_a / (n_a + n_b) if n_a + n_b else float("nan"),
                         "median_gap_days_a_to_b": float(gap.filter(gap > 0).median()) if n_a else float("nan"),
                         "median_gap_days_b_to_a": float((-gap).filter(gap < 0).median()) if n_b else float("nan"),
                         "p_sign_test": float(p)})
    return pl.DataFrame(rows)


def direct_cluster_edges(g: nx.Graph, clusters: dict[str, set]) -> pl.DataFrame:
    """Edges that join two target clusters directly, by phi."""
    names = list(clusters)
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            for u in clusters[a]:
                for v in g.neighbors(u):
                    if v in clusters[b]:
                        rows.append({"pair": f"{a}-{b}", "u": u, "v": v, "phi": g[u][v]["weight"], "n_both": g[u][v].get("n_both")})
    return pl.DataFrame(rows).sort("pair", "phi", descending=[False, True]) if rows else pl.DataFrame(
        schema={"pair": pl.Utf8, "u": pl.Utf8, "v": pl.Utf8, "phi": pl.Float64, "n_both": pl.Int64})
