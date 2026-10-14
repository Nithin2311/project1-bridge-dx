"""Subgroup graphs (sex, age) on the same node set, and the comparison tests.

Step 6 of the Option A pipeline. See docs/OPTION-A-SPEC.md.
Inputs:  patient_category, person, nodes
Outputs: outputs/subgroup_<name>_edges.csv, outputs/graph_<name>.graphml, outputs/subgroup_comparison.csv,
         outputs/subgroup_edge_permutation_<sex|age>.csv

Rules (config subgroups.*):
  * index_date = earliest first_date in patient_category for the patient
  * age_at_index = floor(years from person.birth_DATETIME to index_date); over65 if >= cut
  * sex from person.gender_concept_id via subgroups.age_rule.sex_concept_ids
  * each subgroup graph: same node set as the main graph; edges re-tested inside the subgroup with the
    step-2 rule (Fisher, Bonferroni over the same m, OR, min shared patients); weight = phi in the subgroup
  * edge-wise permutation: for every main-graph edge, phi_A - phi_B observed vs n_permutations shuffles of
    the group label across patients; two-sided empirical p, Benjamini-Hochberg FDR
"""
from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.metrics import adjusted_rand_score


def assign_subgroups(con, cfg: dict, pc: pl.DataFrame | None = None) -> pl.DataFrame:
    """person_id -> sex, age_band. Age at first ICD-10-era diagnosis (rule written in config/option_a.yaml)."""
    sg = cfg["subgroups"]
    cut = int(sg["groups"]["age"]["cut"])
    under, over = sg["groups"]["age"]["labels"]
    ids = sg["age_rule"]["sex_concept_ids"]
    birth = sg["age_rule"].get("birth_field", "birth_DATETIME")
    if pc is None:
        from . import config as _cfg
        pc = pl.read_parquet(_cfg.outputs_dir(cfg) / "patient_category.parquet")
    idx = pc.group_by("person_id").agg(pl.col("first_date").min().alias("index_date"))
    con.register("_idx", idx.to_arrow())
    df = con.execute(f"""
        select i.person_id, i.index_date, p.gender_concept_id, cast(p.{birth} as date) as birth_date,
               date_diff('year', cast(p.{birth} as date), i.index_date)
                 - case when (month(i.index_date), day(i.index_date)) < (month(cast(p.{birth} as date)), day(cast(p.{birth} as date)))
                        then 1 else 0 end as age_at_index
        from _idx i join person p using (person_id)
    """).pl()
    sex_map = {int(v): k for k, v in ids.items()}
    return df.with_columns(
        pl.col("gender_concept_id").replace_strict(sex_map, default=None).alias("sex"),
        pl.when(pl.col("age_at_index") >= cut).then(pl.lit(over)).otherwise(pl.lit(under)).alias("age_band"),
    ).select("person_id", "sex", "age_at_index", "age_band", "index_date")


def subgroup_members(labels: pl.DataFrame, cfg: dict) -> dict[str, list]:
    """subgroup name -> list of person_id, for every label in config subgroups.groups."""
    sg = cfg["subgroups"]["groups"]
    out = {}
    for s in sg["sex"]:
        out[s] = labels.filter(pl.col("sex") == s)["person_id"].to_list()
    for a in sg["age"]["labels"]:
        out[a] = labels.filter(pl.col("age_band") == a)["person_id"].to_list()
    return out


def compare_partitions(p_a: dict, p_b: dict) -> float:
    """Adjusted Rand index over the shared nodes."""
    common = [v for v in p_a if v in p_b]
    return float(adjusted_rand_score([p_a[v] for v in common], [p_b[v] for v in common]))


def connector_overlap(c_a: list, c_b: list, k: int = 20) -> float:
    """|top-k(a) & top-k(b)| / k."""
    return len(set(c_a[:k]) & set(c_b[:k])) / k


def _binary_matrix(pc: pl.DataFrame, categories: list[str], patients: list) -> np.ndarray:
    cat_idx = {c: i for i, c in enumerate(categories)}
    pat_idx = {p: i for i, p in enumerate(patients)}
    sub = pc.filter(pl.col("category").is_in(categories) & pl.col("person_id").is_in(patients))
    m = np.zeros((len(patients), len(categories)), dtype=np.float64)
    m[[pat_idx[p] for p in sub["person_id"].to_list()], [cat_idx[c] for c in sub["category"].to_list()]] = 1.0
    return m


def _phi_for_edges(m: np.ndarray, iu: np.ndarray, iv: np.ndarray) -> np.ndarray:
    """phi for the listed column pairs from a 0/1 patient x category matrix (rows = one group)."""
    n = m.shape[0]
    both = m.T @ m
    k = np.diag(both)
    a = both[iu, iv]
    ku, kv = k[iu], k[iv]
    num = n * a - ku * kv
    den = np.sqrt(ku * (n - ku) * kv * (n - kv))
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = np.where(den > 0, num / den, 0.0)
    return phi


def edgewise_permutation(pc: pl.DataFrame, labels: pl.DataFrame, cfg: dict, edges: pl.DataFrame,
                         group_col: str, group_a: str, group_b: str, categories: list[str],
                         seed: int = 0) -> pl.DataFrame:
    """Edge-wise phi difference (group_a - group_b) for the given edges (columns u, v) with
    cfg['subgroups']['comparison']['n_permutations'] label permutations across patients.
    Returns u, v, phi_a, phi_b, diff, p_perm (two-sided), p_fdr, n_a, n_b."""
    n_perm = int(cfg["subgroups"]["comparison"]["n_permutations"])
    lab = labels.filter(pl.col(group_col).is_in([group_a, group_b]))
    patients = lab["person_id"].to_list()
    is_a = (lab[group_col] == group_a).to_numpy()
    m = _binary_matrix(pc, categories, patients)
    cat_idx = {c: i for i, c in enumerate(categories)}
    iu = np.array([cat_idx[u] for u in edges["u"]])
    iv = np.array([cat_idx[v] for v in edges["v"]])

    phi_a = _phi_for_edges(m[is_a], iu, iv)
    phi_b = _phi_for_edges(m[~is_a], iu, iv)
    obs = phi_a - phi_b
    rng = np.random.default_rng(seed)
    count = np.zeros(len(obs))
    for _ in range(n_perm):
        perm = rng.permutation(is_a)
        d = _phi_for_edges(m[perm], iu, iv) - _phi_for_edges(m[~perm], iu, iv)
        count += np.abs(d) >= np.abs(obs)
    p = (count + 1) / (n_perm + 1)
    # Benjamini-Hochberg
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    p_fdr = np.empty_like(p)
    p_fdr[order] = np.minimum(q, 1.0)
    return pl.DataFrame({
        "u": edges["u"], "v": edges["v"], "phi_a": phi_a, "phi_b": phi_b, "diff": obs,
        "p_perm": p, "p_fdr": p_fdr,
    }).with_columns(pl.lit(group_a).alias("group_a"), pl.lit(group_b).alias("group_b"),
                    pl.lit(int(is_a.sum())).alias("n_a"), pl.lit(int((~is_a).sum())).alias("n_b"))
