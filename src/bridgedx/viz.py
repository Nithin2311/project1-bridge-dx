"""Figures: community layout, bridge table, residual plot, subgroup comparison.

Step 8 of the Option A pipeline. See docs/OPTION-A-SPEC.md.
Inputs:  outputs/*
Outputs: outputs/fig_*.pdf (+ .png previews), outputs/tables/*.tex

Style: categorical hues assigned in fixed order (validated palette, 8 slots), extra communities fold into
gray "other"; thin marks; recessive grid; direct labels; one axis per plot.
"""
from __future__ import annotations

from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OTHER = "#b8b7b2"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "pdf.fonttype": 42, "savefig.dpi": 200,
})


def _save(fig, path):
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(str(path).replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)


def sparsify_top_k(g: nx.Graph, k: int) -> nx.Graph:
    """Keep each node's k strongest edges (union over nodes); layout only."""
    keep = set()
    for v in g.nodes:
        nb = sorted(g[v].items(), key=lambda kv: -kv[1]["weight"])[:k]
        keep.update(frozenset((v, u)) for u, _ in nb)
    h = nx.Graph()
    h.add_nodes_from(g.nodes(data=True))
    for e in keep:
        u, v = tuple(e)
        h.add_edge(u, v, **g[u][v])
    return h


def community_layout(g, partition, path, top_k: int = 5, n_colored: int = 8, labels_per_comm: int = 2,
                     comm_names: dict | None = None, seed: int = 0, priority: list | None = None):
    """Sparsified layout (top-k phi edges per node, unweighted spring so tight modules do not fly off),
    nodes colored by community: `priority` communities first (the target clusters), then largest, up to
    n_colored in fixed slot order; rest gray. Size ~ prevalence; top nodes per colored community labeled."""
    h = sparsify_top_k(g, top_k)
    h.remove_nodes_from([v for v in list(h.nodes) if h.degree(v) == 0])  # isolates: listed in the text, not drawn
    sizes = Counter(partition.values())
    order = [c for c in (priority or []) if c in sizes]
    order += [c for c, _ in sizes.most_common() if c not in order]
    slot = {c: PALETTE[i] for i, c in enumerate(order[:n_colored])}
    pos = nx.spring_layout(h, weight=None, seed=seed, k=1.6 / np.sqrt(h.number_of_nodes()), iterations=300)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.set_axis_off()
    ax.grid(False)
    for u, v, d in h.edges(data=True):
        same = partition[u] == partition[v]
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color=slot.get(partition[u], OTHER) if same else "#d2d1cc",
                lw=0.35 + 1.2 * d["weight"], alpha=0.45 if same else 0.35, zorder=1, solid_capstyle="round")
    prev = np.array([float(g.nodes[v].get("prevalence", 0.01)) for v in h.nodes])
    cols = [slot.get(partition[v], OTHER) for v in h.nodes]
    xy = np.array([pos[v] for v in h.nodes])
    ax.scatter(xy[:, 0], xy[:, 1], s=12 + 260 * prev, c=cols, edgecolors="white", linewidths=0.6, zorder=3)
    wdeg = dict(g.degree(weight="weight"))
    for c in order[:n_colored]:
        members = [v for v in g.nodes if partition[v] == c]
        inner = sorted(members, key=lambda v: -sum(g[v][u]["weight"] for u in g.neighbors(v) if partition[u] == c))
        for j, v in enumerate(inner[:labels_per_comm]):
            ax.annotate(v, pos[v], fontsize=6.5, color=INK, ha="center", va="center", zorder=4,
                        xytext=(0, 7 if j % 2 == 0 else -7), textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8))
    handles = []
    for i, c in enumerate(order[:n_colored]):
        name = (comm_names or {}).get(c, f"community {c}")
        handles.append(plt.Line2D([], [], marker="o", ls="", color=PALETTE[i], markersize=6, label=f"{name} ({sizes[c]})"))
    handles.append(plt.Line2D([], [], marker="o", ls="", color=OTHER, markersize=6,
                              label=f"other ({sum(sizes[c] for c in order[n_colored:])})"))
    ax.legend(handles=handles, loc="lower left", fontsize=6.5, frameon=False, ncol=2, bbox_to_anchor=(0.0, -0.08))
    ax.set_title(f"Diagnosis co-occurrence graph, {g.number_of_nodes()} ICD-10 categories; layout on each node's top-{top_k} phi edges",
                 fontsize=8, color=INK2, loc="left")
    _save(fig, path)


def prevalence_residual_plot(df: pl.DataFrame, path, n_label: int = 8):
    """Degree vs prevalence with OLS line; labeled extremes of the residual."""
    x = df["prevalence"].to_numpy()
    y = df["degree"].to_numpy()
    fit = df["degree_fitted"].to_numpy()
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.scatter(x, y, s=9, color=PALETTE[0], alpha=0.55, edgecolors="none")
    o = np.argsort(x)
    ax.plot(x[o], fit[o], color=INK2, lw=1.2)
    rz = df["degree_residual_z"].to_numpy()
    idx = list(np.argsort(rz)[-n_label:]) + list(np.argsort(rz)[:4])
    for i in idx:
        ax.annotate(df["category"][int(i)], (x[i], y[i]), fontsize=6, color=INK, xytext=(3, 2), textcoords="offset points")
    r2 = df["degree_r2"][0]
    ax.set_xlabel("prevalence in cohort")
    ax.set_ylabel("degree (kept edges)")
    ax.set_title(f"Degree vs prevalence, OLS R² = {r2:.2f}", fontsize=8, color=INK2, loc="left")
    _save(fig, path)


def modularity_null_plot(q_null: np.ndarray, q_obs: float, path, resolution: float = 1.0, q_standard: float | None = None):
    """Null distribution of the RB quality at the working resolution (what Leiden optimises), observed value
    marked; the null is drawn with its own x-range inset so the narrow distribution is visible."""
    fig, ax = plt.subplots(figsize=(3.6, 2.3))
    lo, hi = float(q_null.min()), float(q_null.max())
    ax.hist(q_null, bins=20, color=PALETTE[0], alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.axvline(q_obs, color=PALETTE[1], lw=1.5)
    ax.set_xlim(min(lo, q_obs) - 0.01, max(hi, q_obs) + 0.01)
    ins = ax.inset_axes([0.42, 0.45, 0.3, 0.45])
    ins.hist(q_null, bins=15, color=PALETTE[0], alpha=0.85, edgecolor="white", linewidth=0.4)
    ins.set_title("null, zoomed", fontsize=6, color=INK2)
    ins.tick_params(labelsize=5)
    ins.grid(False)
    ax.annotate(f"observed {q_obs:.3f}\n(z = {(q_obs - q_null.mean()) / q_null.std(ddof=1):.0f})", (q_obs, ax.get_ylim()[1] * 0.85),
                fontsize=7, color=INK, ha="right", xytext=(-4, 0), textcoords="offset points")
    ax.annotate(f"null {q_null.mean():.3f} ± {q_null.std(ddof=1):.3f}", ((lo + hi) / 2, ax.get_ylim()[1] * 0.85), fontsize=7, color=INK2, ha="left", xytext=(6, 0), textcoords="offset points")
    label = f"RB quality at γ = {resolution:g}" if resolution != 1.0 else "modularity Q"
    ax.set_xlabel(label + (f"   (standard Q of the partition = {q_standard:.3f})" if q_standard is not None and resolution != 1.0 else ""))
    ax.set_ylabel("rewired graphs")
    ax.set_title(f"Leiden partition quality vs {len(q_null)} degree-preserving rewires", fontsize=8, color=INK2, loc="left")
    _save(fig, path)


def subgroup_bridges_plot(df: pl.DataFrame, path, pair: str, groups: list[str], names: dict, top_n: int = 12):
    """Dot plot: rows = bridge nodes (union of each subgroup's top-n by restricted betweenness), columns =
    subgroups, dot size/position = rank within the subgroup. Single hue, direct labels."""
    sub = df.filter(pl.col("pair") == pair)
    ranks = {}
    for gname in groups:
        s = sub.filter(pl.col("subgroup") == gname).sort("bridge_betweenness", descending=True)
        ranks[gname] = {c: i + 1 for i, c in enumerate(s["category"].to_list())}
    rows = sorted({c for gname in groups for c, r in ranks[gname].items() if r <= top_n},
                  key=lambda c: sum(min(ranks[gname].get(c, 99), 30) for gname in groups))
    fig, ax = plt.subplots(figsize=(3.6, 0.22 * len(rows) + 0.9))
    for j, gname in enumerate(groups):
        for i, c in enumerate(rows):
            r = ranks[gname].get(c)
            if r is None or r > 30:
                ax.scatter(j, i, s=6, color=GRID, zorder=2)
            else:
                ax.scatter(j, i, s=max(6, 110 - 3.4 * r), color=PALETTE[0] if r <= top_n else OTHER, alpha=0.85, zorder=3, edgecolors="white", linewidths=0.5)
                if r <= 5:
                    ax.annotate(str(r), (j, i), fontsize=5.5, color="white", ha="center", va="center", zorder=4)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{c}  {names.get(c, '')[:28]}" for c in rows], fontsize=6.5)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=7)
    ax.invert_yaxis()
    ax.grid(False)
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_title(f"Bridges {pair.replace('_', ' ')}: rank by restricted betweenness", fontsize=8, color=INK2, loc="left")
    _save(fig, path)


def edge_diff_plot(pt: pl.DataFrame, path, a: str, b: str, alpha: float = 0.05):
    """phi in group a vs b for every main-graph edge; FDR-significant edges highlighted."""
    sig = pt["p_fdr"] < alpha
    flag = pt["sex_specific"] if "sex_specific" in pt.columns else pl.Series([False] * pt.height)
    fig, ax = plt.subplots(figsize=(3.3, 3.1))
    ax.plot([-0.1, 1], [-0.1, 1], color=GRID, lw=1)
    ax.scatter(pt["phi_b"].filter(~sig), pt["phi_a"].filter(~sig), s=5, color=OTHER, alpha=0.5, edgecolors="none")
    ax.scatter(pt["phi_b"].filter(sig & flag), pt["phi_a"].filter(sig & flag), s=9, color=INK2, alpha=0.6, edgecolors="none",
               label=f"significant, reproductive/sex-organ code ({int((sig & flag).sum())})")
    ax.scatter(pt["phi_b"].filter(sig & ~flag), pt["phi_a"].filter(sig & ~flag), s=9, color=PALETTE[1], alpha=0.9, edgecolors="none",
               label=f"significant, other ({int((sig & ~flag).sum())})")
    lab = pt.filter(sig & ~flag).sort(pl.col("diff").abs(), descending=True).head(5)
    for i, r in enumerate(lab.iter_rows(named=True)):
        ax.annotate(f"{r['u']}-{r['v']}", (r["phi_b"], r["phi_a"]), fontsize=5.5, color=INK,
                    xytext=(4, 6 if i % 2 == 0 else -8), textcoords="offset points")
    ax.legend(fontsize=5.5, frameon=False, loc="lower right")
    ax.set_xlabel(f"phi in {b}")
    ax.set_ylabel(f"phi in {a}")
    ax.set_title(f"Edge weights {a} vs {b}; {int(sig.sum())} of {pt.height:,} edges differ at FDR {alpha}", fontsize=8, color=INK2, loc="left")
    _save(fig, path)


def latex_table(df: pl.DataFrame, path, caption: str, label: str, col_fmt: dict | None = None, escape: bool = True):
    """Minimal booktabs table writer; numeric formats via col_fmt {col: '{:.2f}'}."""
    col_fmt = col_fmt or {}
    def esc(s):
        return str(s).replace("&", "\\&").replace("%", "\\%").replace("_", "\\_") if escape else str(s)
    lines = ["\\begin{table}[t]", "\\centering", "\\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\begin{tabular}{" + "l" * df.width + "}", "\\toprule",
             " & ".join(esc(c) for c in df.columns) + " \\\\", "\\midrule"]
    for row in df.iter_rows(named=True):
        cells = []
        for c, v in row.items():
            if v is None or (isinstance(v, float) and (v != v or v in (float("inf"), float("-inf")))):
                cells.append("--" if v is None or v != v else ("$\\infty$" if v > 0 else "$-\\infty$"))
            else:
                cells.append(esc(col_fmt[c].format(v)) if c in col_fmt else esc(v))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))
