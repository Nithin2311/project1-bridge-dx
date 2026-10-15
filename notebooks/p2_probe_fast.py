"""P2-readiness probes (non-ML): A consensus clustering, C inpatient share per node, D patient-subgraph sizes,
E alternative label designs, F temporal split feasibility, H disparity-filter backbone. Writes outputs/p2_probes/."""
from __future__ import annotations
import datetime as dt, time
import numpy as np, polars as pl, networkx as nx
from collections import Counter
from scipy.stats import binomtest
from sklearn.metrics import adjusted_rand_score
from bridgedx import config, data, communities as cm, centrality as ce
cfg = config.load(); O = config.outputs_dir(cfg); out = O / "p2_probes"; out.mkdir(exist_ok=True)
g = nx.read_graphml(O / "graph_main.graphml"); nodes = list(g.nodes)
pc = pl.read_parquet(O / "patient_category.parquet")
comm = pl.read_csv(O / "communities.csv"); part0 = dict(zip(comm["category"], comm["leiden"]))
anchors = cfg["centrality"]["target_clusters"]; cl0 = ce.target_clusters(part0, anchors)
names = dict(zip(pl.read_csv(O / "nodes.csv")["category"], pl.read_csv(O / "nodes.csv")["name"]))
t0 = time.time()

# ---------- A. consensus clustering across seeds (and resolutions) ----------
print("A. consensus clustering")
idx = {v: i for i, v in enumerate(nodes)}
def coassign(resolutions, seeds):
    M = np.zeros((len(nodes), len(nodes))); n = 0
    for r in resolutions:
        c = {"communities": dict(cfg["communities"], resolution=r)}
        for s in seeds:
            p = cm.leiden(g, c, seed=s)
            lab = np.array([p[v] for v in nodes])
            M += (lab[:, None] == lab[None, :]); n += 1
    return M / n
rows = []
for label, res, seeds in [("res1.5_30seeds", [1.5], range(30)), ("res1-1.5-2_10seeds", [1.0, 1.5, 2.0], range(10))]:
    M = coassign(res, seeds)
    for k, codes in anchors.items():
        a = [idx[c] for c in codes if c in idx]
        co = M[a].mean(axis=0)  # mean co-assignment with the anchor codes
        core = [nodes[i] for i in np.argsort(-co) if co[i] >= 0.9]
        soft = [nodes[i] for i in np.argsort(-co) if 0.5 <= co[i] < 0.9]
        main_members = cl0[k]
        rows.append({"setting": label, "cluster": k, "core_ge0.9": len(core), "soft_0.5-0.9": len(soft),
                     "main_cluster_size": len(main_members), "main_in_core": sum(v in core for v in main_members),
                     "core_codes": " ".join(core), "soft_codes": " ".join(soft)})
        print(f"  {label:20s} {k:16s} core(>=0.9)={len(core):3d} soft(0.5-0.9)={len(soft):3d} main={len(main_members)} main_in_core={sum(v in core for v in main_members)}")
        print(f"      core: {' '.join(core[:40])}{' ...' if len(core) > 40 else ''}")
    # also: pairwise ARI distribution among seeds at 1.5
pl.DataFrame(rows).write_csv(out / "consensus_clusters.csv")
# full co-assignment for 1.5 saved for P2 (node x node)
M15 = coassign([1.5], range(30))
pl.DataFrame({"category": nodes, **{nodes[j]: M15[:, j] for j in range(len(nodes))}}).write_csv(out / "coassignment_res1.5_30seeds.csv")
ari = []
parts = [cm.leiden(g, cfg, seed=s) for s in range(10)]
for i in range(10):
    for j in range(i + 1, 10):
        ari.append(adjusted_rand_score([parts[i][v] for v in nodes], [parts[j][v] for v in nodes]))
print(f"  pairwise ARI among 10 seeds at 1.5: median {np.median(ari):.3f}, min {min(ari):.3f}, max {max(ari):.3f}  ({time.time()-t0:.0f}s)")

# ---------- C. inpatient / ED share per node ----------
print("C. visit-type share per node")
con = data.connect(cfg)
con.register("_nodes", pl.DataFrame({"category": nodes}).to_arrow())
from bridgedx import cohort as _c; _c._register_category_map(con)
vt = con.execute(f"""
  with rows as (
    select m.category, v.visit_concept_id as vc
    from condition_occurrence co join cat_map m on m.sv = co.condition_source_value
    left join visit_occurrence v on v.visit_occurrence_id = co.visit_occurrence_id
    where co.condition_start_DATE >= date '{cfg["cohort"]["era_start"]}' and m.category in (select category from _nodes))
  select category, count(*) n_rows,
    avg(case when vc = 9201 then 1 else 0 end) share_inpatient,
    avg(case when vc = 9203 then 1 else 0 end) share_ed,
    avg(case when vc in (9201, 9203) then 1 else 0 end) share_acute,
    avg(case when vc is null then 1 else 0 end) share_novisit
  from rows group by 1""").pl()
print("  visit_concept_id distribution:", con.execute("select visit_concept_id, count(*) from visit_occurrence group by 1 order by 2 desc limit 6").fetchall())
br = pl.read_csv(O / "bridges.csv")
passing = set(br.filter(pl.col("bridge_pass"))["category"].to_list())
vt = vt.with_columns(pl.col("category").is_in(list(passing)).alias("is_bridge"))
vt.write_csv(out / "node_visit_type_share.csv")
for flag, sub in [("bridges (pass null)", vt.filter("is_bridge")), ("all other nodes", vt.filter(~pl.col("is_bridge")))]:
    print(f"  {flag:20s} n={sub.height:3d} median share_acute={sub['share_acute'].median():.2f}  share_inpatient={sub['share_inpatient'].median():.2f}  share_ed={sub['share_ed'].median():.2f}")
top = vt.sort("share_acute", descending=True).head(12)
print("  most acute nodes: " + ", ".join(f"{r['category']} {r['share_acute']:.2f}" for r in top.iter_rows(named=True)))
print(f"  nodes with share_acute >= 0.5: {vt.filter(pl.col('share_acute') >= 0.5).height}  ({time.time()-t0:.0f}s)")

# ---------- D. patient subgraph sizes for P2 ----------
print("D. patient subgraphs (index = first era dx + 365 d, lookback 365 d, main node set)")
first = pc.group_by("person_id").agg(pl.col("first_date").min().alias("first_dx"))
rowsq = con.execute(f"""
  select co.person_id, m.category, co.condition_start_DATE dt
  from condition_occurrence co join cat_map m on m.sv = co.condition_source_value
  left join death d on d.person_id = co.person_id
  where co.condition_start_DATE >= date '{cfg["cohort"]["era_start"]}' and (d.death_DATE is null or co.condition_start_DATE <= d.death_DATE)
    and m.category in (select category from _nodes)""").pl().join(first, on="person_id")
win = rowsq.filter((pl.col("dt") <= pl.col("first_dx") + pl.duration(days=365))).select("person_id", "category").unique()
sizes = win.group_by("person_id").agg(pl.col("category").alias("codes"))
n_nodes_l, n_edges_l = [], []
for r in sizes.iter_rows(named=True):
    sub = g.subgraph(r["codes"]); n_nodes_l.append(sub.number_of_nodes()); n_edges_l.append(sub.number_of_edges())
nn, ne = np.array(n_nodes_l), np.array(n_edges_l)
print(f"  patients {len(nn):,}; nodes/subgraph median {np.median(nn):.0f} (IQR {np.percentile(nn,25):.0f}-{np.percentile(nn,75):.0f}); "
      f"edges median {np.median(ne):.0f}; share with 0 edges {np.mean(ne==0):.1%}, <3 nodes {np.mean(nn<3):.1%}, >=5 nodes {np.mean(nn>=5):.1%}")
pl.DataFrame({"person_id": sizes["person_id"], "n_nodes": nn, "n_edges": ne}).write_csv(out / "patient_subgraph_sizes.csv")

# ---------- E. alternative label designs ----------
print("E. label designs (12-month horizon; prior history excluded incl. bridged pre-era)")
pre = pl.read_parquet(O / "patient_category_icd9era.parquet")
pat = con.execute("""select co.person_id, max(co.condition_start_DATE) last_dx from condition_occurrence co left join death d on d.person_id=co.person_id
   where d.death_DATE is null or co.condition_start_DATE <= d.death_DATE group by 1""").pl()
death = con.execute("select person_id, death_DATE as death from death").pl()
base = first.join(pat, on="person_id").join(death, on="person_id", how="left")
H = 365
def count_labels(targets: dict, index_rule: str):
    res = {}
    for k, members in targets.items():
        ft = pc.filter(pl.col("category").is_in(list(members))).group_by("person_id").agg(pl.col("first_date").min().alias("ft"))
        prior = set(pre.filter(pl.col("category").is_in(list(members)))["person_id"].to_list())
        df = base.join(ft, on="person_id", how="left").filter(~pl.col("person_id").is_in(list(prior)))
        if index_rule == "single":
            df = df.with_columns((pl.col("first_dx") + pl.duration(days=365)).alias("index"))
            df = df.filter(pl.col("ft").is_null() | (pl.col("ft") > pl.col("index")))
            end = pl.col("index") + pl.duration(days=H)
            d = df.with_columns((pl.col("ft").is_not_null() & (pl.col("ft") <= end)).alias("pos"),
                                ((pl.col("death").is_not_null() & (pl.col("death") <= end)) | (pl.col("last_dx") < end)).alias("ends"))
            d = d.with_columns((~pl.col("pos") & pl.col("ends")).alias("cens"))
            res[k] = (d.height, int(d["pos"].sum()), int(d["cens"].sum()))
        else:  # landmarks every 182 d from first_dx+365 while index + H <= last_dx (uncensored by construction) or positive
            tot = pos = 0
            for r in df.iter_rows(named=True):
                t = r["first_dx"] + dt.timedelta(days=365)
                while True:
                    if r["ft"] is not None and r["ft"] <= t: break
                    end = t + dt.timedelta(days=H)
                    is_pos = r["ft"] is not None and r["ft"] <= end
                    ends = (r["death"] is not None and r["death"] <= end) or r["last_dx"] < end
                    if is_pos or not ends:
                        tot += 1; pos += int(is_pos)
                    if is_pos or ends: break
                    t += dt.timedelta(days=182)
            res[k] = (tot, pos, 0)
    return res
targets = dict(cl0)
targets["F_chapter"] = {v for v in nodes if v.startswith("F")}
targets["any_new_target_cluster"] = None
for rule in ["single", "landmark"]:
    for k, members in targets.items():
        if members is None: continue
        n, p, c = count_labels({k: members}, rule)[k]
        print(f"  {rule:9s} {k:24s} eligible rows={n:>6,} positives={p:>5,} censored={c:>5,} rate={(p/(n-c) if n-c else 0):.2f}")
# any-new-cluster: patient positive if enters ANY of the three clusters it was not in
ft_all = {k: pc.filter(pl.col("category").is_in(list(m))).group_by("person_id").agg(pl.col("first_date").min().alias("ft")) for k, m in cl0.items()}
df = base.with_columns((pl.col("first_dx") + pl.duration(days=365)).alias("index"))
pos = 0; elig = 0
for r in df.iter_rows(named=True):
    pid = r["person_id"]; idxd = r["index"]; end = idxd + dt.timedelta(days=H)
    cand = []
    for k in cl0:
        f = ft_all[k].filter(pl.col("person_id") == pid)
        f = f["ft"][0] if f.height else None
        prior = pid in set(pre.filter(pl.col("category").is_in(list(cl0[k])))["person_id"].to_list()) if False else False
        if f is None or f > idxd: cand.append(f)
    if not cand: continue
    ends = (r["death"] is not None and r["death"] <= end) or r["last_dx"] < end
    is_pos = any(f is not None and f <= end for f in cand)
    if is_pos or not ends:
        elig += 1; pos += int(is_pos)
print(f"  single    any_new_target_cluster  eligible={elig:,} positives={pos:,} (pre-era history not excluded here)  ({time.time()-t0:.0f}s)")

# ---------- F. temporal split feasibility ----------
print("F. temporal split (index = first era dx + 365 d)")
df = base.with_columns((pl.col("first_dx") + pl.duration(days=365)).alias("index"))
yr = df["index"].dt.year()
print("  index year counts:", dict(sorted(Counter(yr.to_list()).items())))
for k, members in cl0.items():
    ft = pc.filter(pl.col("category").is_in(list(members))).group_by("person_id").agg(pl.col("first_date").min().alias("ft"))
    prior = set(pre.filter(pl.col("category").is_in(list(members)))["person_id"].to_list())
    d = df.join(ft, on="person_id", how="left").filter(~pl.col("person_id").is_in(list(prior)))
    d = d.filter(pl.col("ft").is_null() | (pl.col("ft") > pl.col("index")))
    end = pl.col("index") + pl.duration(days=H)
    d = d.with_columns((pl.col("ft").is_not_null() & (pl.col("ft") <= end)).alias("pos"),
                       ((pl.col("death").is_not_null() & (pl.col("death") <= end)) | (pl.col("last_dx") < end)).alias("ends"))
    d = d.filter(pl.col("pos") | ~pl.col("ends"))
    early = d.filter(pl.col("index") < dt.date(2019, 1, 1)); late = d.filter(pl.col("index") >= dt.date(2019, 1, 1))
    print(f"  {k:16s} index<2019: n={early.height:,} pos={int(early['pos'].sum())}   index>=2019: n={late.height:,} pos={int(late['pos'].sum())}")

# ---------- H. disparity-filter backbone ----------
print("H. disparity filter backbone (Serrano 2009) on the main graph")
def disparity(g, alpha):
    keep = set()
    for v in g.nodes:
        k = g.degree(v)
        if k < 2: continue
        s = sum(d["weight"] for _, _, d in g.edges(v, data=True))
        for _, u, d in g.edges(v, data=True):
            pij = d["weight"] / s
            a = (1 - pij) ** (k - 1)
            if a < alpha: keep.add(frozenset((v, u)))
    h = nx.Graph(); h.add_nodes_from(g.nodes(data=True))
    for e in keep:
        u, v = tuple(e); h.add_edge(u, v, **g[u][v])
    return h
rowsH = []
for alpha in [0.05, 0.01]:
    h = disparity(g, alpha)
    p = cm.leiden(h, cfg, seed=0)
    clh = ce.target_clusters(p, anchors)
    iso = sum(1 for v in h.nodes if h.degree(v) == 0)
    ari = adjusted_rand_score([part0[v] for v in nodes], [p[v] for v in nodes])
    comps = nx.number_connected_components(h)
    print(f"  alpha={alpha}: edges={h.number_of_edges():,} (density {nx.density(h):.3f}) isolates={iso} components={comps} "
          f"Leiden k={len(set(p.values()))} ARI_vs_main={ari:.3f} target sizes={[len(clh[k]) for k in anchors]} "
          f"overlap with main targets={[len(clh[k] & cl0[k]) for k in anchors]}")
    nx.write_graphml(h, out / f"graph_backbone_alpha{alpha}.graphml")
    rowsH.append({"alpha": alpha, "edges": h.number_of_edges(), "density": nx.density(h), "isolates": iso, "ari_vs_main": ari})
pl.DataFrame(rowsH).write_csv(out / "backbone_summary.csv")
print(f"done ({time.time()-t0:.0f}s)")
