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
    placed = []  # stagger labels of points that sit on top of each other
    for i in sorted(idx, key=lambda j: (x[j], y[j])):
        near = sum(1 for (px, py) in placed if abs(px - x[i]) < 0.02 and abs(py - y[i]) < 14)
        ax.annotate(df["category"][int(i)], (x[i], y[i]), fontsize=6, color=INK, xytext=(3, 2 - 8 * near), textcoords="offset points")
        placed.append((x[i], y[i]))
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
    ins = ax.inset_axes([0.30, 0.40, 0.34, 0.45])
    ins.hist(q_null, bins=15, color=PALETTE[0], alpha=0.85, edgecolor="white", linewidth=0.4)
    ins.set_title(f"null, zoomed: {q_null.mean():.3f} ± {q_null.std(ddof=1):.3f}", fontsize=6, color=INK2)
    ins.tick_params(labelsize=5)
    ins.grid(False)
    ax.annotate(f"observed {q_obs:.3f}\n(z = {(q_obs - q_null.mean()) / q_null.std(ddof=1):.0f})", (q_obs, ax.get_ylim()[1] * 0.30),
                fontsize=7, color=INK, ha="right", xytext=(-4, 0), textcoords="offset points")
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


def _edge_diff_ax(ax, pt: pl.DataFrame, a: str, b: str, alpha: float, title: str, group_labels: dict):
    """phi in group a vs b for every main-graph edge on one axis; FDR-significant edges highlighted,
    the five largest non-reproductive differences labelled. Returns the three scatter handles."""
    sig = pt["p_fdr"] < alpha
    flag = pt["sex_specific"] if "sex_specific" in pt.columns else pl.Series([False] * pt.height)
    ax.plot([-0.1, 1], [-0.1, 1], color=GRID, lw=1, zorder=0)
    h0 = ax.scatter(pt["phi_b"].filter(~sig), pt["phi_a"].filter(~sig), s=4, color=OTHER, alpha=0.5, edgecolors="none",
                    label="not significant", rasterized=True)
    h1 = ax.scatter(pt["phi_b"].filter(sig & flag), pt["phi_a"].filter(sig & flag), s=11, color=INK2, alpha=0.7, edgecolors="none",
                    label="significant, reproductive / sex-organ code")
    h2 = ax.scatter(pt["phi_b"].filter(sig & ~flag), pt["phi_a"].filter(sig & ~flag), s=11, color=PALETTE[1], alpha=0.9, edgecolors="none",
                    label="significant, other")
    lab = pt.filter(sig & ~flag).sort(pl.col("diff").abs(), descending=True).head(5)
    for i, r in enumerate(lab.iter_rows(named=True)):
        ax.annotate(f"{r['u']}-{r['v']}", (r["phi_b"], r["phi_a"]), fontsize=5.5, color=INK,
                    xytext=(4, 6 if i % 2 == 0 else -8), textcoords="offset points")
    ax.set_xlim(-0.12, 1.0)
    ax.set_ylim(-0.12, 1.0)
    ax.set_xlabel(f"phi in {group_labels.get(b, b)}")
    ax.set_ylabel(f"phi in {group_labels.get(a, a)}")
    ax.set_title(f"{title}: {group_labels.get(a, a)} vs {group_labels.get(b, b)}", fontsize=7.5, color=INK2, loc="left")
    ax.text(0.03, 0.97, f"{int(sig.sum())} of {pt.height:,} edges differ at FDR {alpha}\n"
            f"{int((sig & flag).sum())} reproductive / sex-organ, {int((sig & ~flag).sum())} other",
            transform=ax.transAxes, fontsize=6.5, color=INK2, va="top", ha="left")
    return h0, h1, h2


def edge_diff_panels(panels: list, path, alpha: float = 0.05, group_labels: dict | None = None):
    """One full-width figure, one panel per subgroup split: panels = [(pt, a, b, title), ...].
    Shared legend below; per-panel counts in the panel titles."""
    group_labels = group_labels or {}
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels) + 0.3, 3.0))
    axes = np.atleast_1d(axes)
    handles = None
    for ax, (pt, a, b, title) in zip(axes, panels):
        handles = _edge_diff_ax(ax, pt, a, b, alpha, title, group_labels)
    fig.legend(handles=list(handles), loc="lower center", ncol=3, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.06, 1, 1], w_pad=2.0)
    _save(fig, path)


def bridge_stability_plot(st: pl.DataFrame, bridges: pl.DataFrame, path, names: dict, n_bootstrap: int, top_n: int = 8):
    """Patient-bootstrap stability of bridge rank, one panel per cluster pair: the top_n categories by the
    share of resamples in which they rank in the pair's top-10 by restricted betweenness. Filled bars = the
    category also passes the rewired null in the main graph (bridge_pass); hollow = it does not."""
    pairs = sorted(st["pair"].unique().to_list())
    ncol = 3
    nrow = int(np.ceil(len(pairs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.1, 1.75 * nrow + 0.5), squeeze=False)
    for ax in axes.flat[len(pairs):]:
        ax.set_axis_off()
    for ax, pair in zip(axes.flat, pairs):
        s = st.filter(pl.col("pair") == pair).sort(["share_in_top10", "median_rank"], descending=[True, False]).head(top_n)
        passed = set(bridges.filter((pl.col("pair") == pair) & pl.col("bridge_pass"))["category"].to_list())
        cats = s["category"].to_list()
        vals = s["share_in_top10"].to_numpy()
        ys = np.arange(len(cats))
        for y, c, v in zip(ys, cats, vals):
            ax.barh(y, v, height=0.62, color=PALETTE[0] if c in passed else "white",
                    edgecolor=PALETTE[0], linewidth=0.8, zorder=3)
            ax.annotate(f"{int(np.rint(v * n_bootstrap))}/{n_bootstrap}", (v, y), fontsize=5.5, color=INK2,
                        xytext=(2, 0), textcoords="offset points", va="center")
        ax.set_yticks(ys)
        ax.set_yticklabels([f"{c}  {names.get(c, '')[:20]}" for c in cats], fontsize=5.8)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.22)
        ax.set_xticks([0, 0.5, 1.0])
        ax.tick_params(axis="x", labelsize=6)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="y", visible=False)
        ax.spines["left"].set_visible(False)
        ax.set_title(pair.replace("_", " ").replace("-", " vs "), fontsize=7.5, color=INK2, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=PALETTE[0], ec=PALETTE[0], lw=0.8, label="passes the rewired null in the main graph"),
               plt.Rectangle((0, 0), 1, 1, fc="white", ec=PALETTE[0], lw=0.8, label="does not pass the rewired null")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Share of {n_bootstrap} patient resamples in which the category ranks in the pair's top-10 bridges",
                 fontsize=8, color=INK2, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0.05, 1, 0.95], w_pad=1.6, h_pad=1.2)
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
