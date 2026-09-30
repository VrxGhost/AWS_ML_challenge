"""Pairwise features for (Source 1 record, candidate record).

Missing information is encoded as NaN (never as 0 = "disagree"); gradient
boosting handles NaN natively. No feature uses the country label, so the
model transfers to a country it never saw in training.

v2 (with normalization.py):
  - FIX: name_ndigit -> name_n_digits (new column name), key bits follow blocking.KEY_TYPES.
  - NEW: state_eq, house_sfx_eq, name_key_exact, had_alias, addr_empty_a/b, legal_conflict.
  - kept from before: fork/spawn-safe worker pool, name_was_indic.
"""
from __future__ import annotations

import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

from blocking import KEY_TYPES

try:
    from rapidfuzz import fuzz as _fuzz
    from rapidfuzz import process as _process
    from rapidfuzz.distance import JaroWinkler as _JW
    HAVE_RAPIDFUZZ = True
except Exception:  # pragma: no cover
    HAVE_RAPIDFUZZ = False

LANDMARK = {"near", "opposite", "behind", "beside", "next", "adjacent"}


# --------------------------------------------------------------------------
# String similarity back-ends (rapidfuzz fast path, pure-python fallback)
# --------------------------------------------------------------------------
def _py_ratio(a, b):
    if not a and not b:
        return 100.0
    return 100.0 * SequenceMatcher(None, a, b).ratio()


def _py_tsort(a, b):
    return _py_ratio(" ".join(sorted(a.split())), " ".join(sorted(b.split())))


def _py_tset(a, b):
    sa, sb = set(a.split()), set(b.split())
    inter = " ".join(sorted(sa & sb))
    da = " ".join(sorted(sa - sb))
    db = " ".join(sorted(sb - sa))
    c1 = (inter + " " + da).strip()
    c2 = (inter + " " + db).strip()
    if inter and (not da or not db):
        return 100.0
    return max(_py_ratio(inter, c1), _py_ratio(inter, c2), _py_ratio(c1, c2))


def _py_partial(a, b):
    if not a or not b:
        return 0.0
    s, l = (a, b) if len(a) <= len(b) else (b, a)
    best = 0.0
    for k in range(0, len(l) - len(s) + 1, max(1, len(s) // 4)):
        best = max(best, _py_ratio(s, l[k:k + len(s)]))
    return best


def _py_jw(a, b):
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if not la or not lb:
        return 0.0
    md = max(la, lb) // 2 - 1
    ma, mb = [False] * la, [False] * lb
    m = 0
    for i in range(la):
        for j in range(max(0, i - md), min(lb, i + md + 1)):
            if not mb[j] and a[i] == b[j]:
                ma[i] = mb[j] = True
                m += 1
                break
    if not m:
        return 0.0
    t, k = 0, 0
    for i in range(la):
        if ma[i]:
            while not mb[k]:
                k += 1
            t += a[i] != b[k]
            k += 1
    jaro = (m / la + m / lb + (m - t / 2) / m) / 3
    p = 0
    for x, y in zip(a[:4], b[:4]):
        if x != y:
            break
        p += 1
    return jaro + p * 0.1 * (1 - jaro)


def pair_sim(a: list, b: list, kind: str) -> np.ndarray:
    if HAVE_RAPIDFUZZ:
        scorer = {"ratio": _fuzz.ratio, "tset": _fuzz.token_set_ratio, "tsort": _fuzz.token_sort_ratio,
                  "partial": _fuzz.partial_ratio, "jw": _JW.normalized_similarity}[kind]
        return np.asarray(_process.cpdist(a, b, scorer=scorer, workers=-1), dtype=np.float32)
    fn = {"ratio": _py_ratio, "tset": _py_tset, "tsort": _py_tsort, "partial": _py_partial,
          "jw": _py_jw}[kind]
    return np.fromiter((fn(x, y) for x, y in zip(a, b)), dtype=np.float32, count=len(a))


# --------------------------------------------------------------------------
# Token-set features (pure python loop, parallelised by chunks)
# --------------------------------------------------------------------------
_G = {}


def _init_worker(ta, tb, stats_df, n, kind):   # used when fork is unavailable (Windows / macOS)
    _G.update(ta=ta, tb=tb, df=stats_df, n=n, kind=kind)


def _set_feats(lo: int, hi: int):
    ta, tb, stats_df, n = _G["ta"], _G["tb"], _G["df"], _G["n"]
    out = np.full((hi - lo, 6), np.nan, dtype=np.float32)
    for r in range(lo, hi):
        a, b = set(ta[r]), set(tb[r])
        if not a or not b:
            continue
        idf = {t: math.log((n + 1) / (stats_df.get(t, 0) + 1)) for t in a | b}
        inter = a & b
        wi = sum(idf[t] for t in inter)
        wa = sum(idf[t] for t in a)
        wb = sum(idf[t] for t in b)
        wu = wa + wb - wi
        out[r - lo, 0] = wi / wu if wu > 0 else np.nan                       # weighted jaccard
        out[r - lo, 1] = wi / wa if wa > 0 else np.nan                       # coverage of S1 side
        out[r - lo, 2] = wi / wb if wb > 0 else np.nan                       # coverage of cand side
        out[r - lo, 3] = max((idf[t] for t in inter), default=0.0)           # rarest shared token
        out[r - lo, 4] = max((idf[t] for t in a ^ b), default=0.0)           # rarest unshared token
        out[r - lo, 5] = len(inter) / len(a | b)                             # plain jaccard
    return out


def set_features(ta, tb, stats_df, n, kind, n_jobs):
    # Only the counts of words that occur in these pairs are needed.  Sending the full table
    # (millions of words) to every worker on Windows copied it n_jobs times -> out of memory.
    need = set()
    for x in ta:
        need.update(x)
    for x in tb:
        need.update(x)
    stats_df = {t: stats_df.get(t, 0) for t in need}
    _G.update(ta=ta, tb=tb, df=stats_df, n=n, kind=kind)
    N = len(ta)
    if n_jobs <= 1 or N < 200_000:
        return _set_feats(0, N)
    step = math.ceil(N / (n_jobs * 4))
    bounds = [(s, min(s + step, N)) for s in range(0, N, step)]
    if "fork" in mp.get_all_start_methods():      # Linux: children inherit _G for free
        ex = ProcessPoolExecutor(n_jobs, mp_context=mp.get_context("fork"))
    else:                                         # Windows / macOS spawn
        ex = ProcessPoolExecutor(n_jobs, initializer=_init_worker, initargs=(ta, tb, stats_df, n, kind))
    with ex:
        parts = list(ex.map(_set_feats, *zip(*bounds)))
    return np.vstack(parts)


def _eq3(a: pd.Series, b: pd.Series) -> np.ndarray:
    """1 agree / 0 explicit disagreement / NaN if missing on either side."""
    a = a.values
    b = b.values
    res = (a == b).astype(np.float32)
    res[(a == "") | (b == "")] = np.nan
    return res


def _digits(s: pd.Series) -> pd.Series:
    return s.str.replace(r"\D", "", regex=True)


# --------------------------------------------------------------------------
# Main entry
# --------------------------------------------------------------------------
def build_features(cand: pd.DataFrame, s1: pd.DataFrame, tgts: dict, stats, freq, n_jobs: int = 1,
                   log=print) -> pd.DataFrame:
    """cand: output of blocking.generate_candidates. tgts: {2: repr_S2, 3: repr_S3}."""
    F = pd.DataFrame(index=cand.index)
    A = s1.iloc[cand["i"].values].reset_index(drop=True)
    pieces, order = [], []
    for src, rep in tgts.items():
        pos = np.where(cand["src"].values == src)[0]
        pieces.append(rep.iloc[cand["j"].values[pos]].reset_index(drop=True))
        order.append(pos)
    B = pd.concat(pieces, ignore_index=True)
    B.index = np.concatenate(order)
    B = B.sort_index()
    A.index = cand.index
    B.index = cand.index

    log("    string similarities")
    an, bn = A["name_core"].tolist(), B["name_core"].tolist()
    aa, ba = A["addr_norm"].tolist(), B["addr_norm"].tolist()
    F["cos_name"] = cand["cos_name"].values
    F["cos_addr"] = cand["cos_addr"].values
    F["name_ratio"] = pair_sim(an, bn, "ratio")
    F["name_tset"] = pair_sim(an, bn, "tset")
    F["name_tsort"] = pair_sim(an, bn, "tsort")
    F["name_partial"] = pair_sim(an, bn, "partial")
    F["name_jw"] = pair_sim(A["name_compact"].tolist(), B["name_compact"].tolist(), "jw")
    F["addr_ratio"] = pair_sim(aa, ba, "ratio")
    F["addr_tset"] = pair_sim(aa, ba, "tset")
    F["addr_tsort"] = pair_sim(aa, ba, "tsort")
    ast_, bst = A["addr_street"].tolist(), B["addr_street"].tolist()
    st = pair_sim(ast_, bst, "ratio")
    st[(A["addr_street"].values == "") | (B["addr_street"].values == "")] = np.nan
    F["street_ratio"] = st

    log("    exact / structural")
    F["name_core_exact"] = (A["name_core"].values == B["name_core"].values).astype(np.float32)
    F["name_norm_exact"] = (A["name_norm"].values == B["name_norm"].values).astype(np.float32)
    F["name_key_exact"] = (A["name_key"].values == B["name_key"].values).astype(np.float32)   # NEW
    F["addr_exact"] = (A["addr_norm"].values == B["addr_norm"].values).astype(np.float32)
    ia, ib = A["name_initials"].values, B["name_initials"].values
    F["initials_eq"] = ((ia == ib) & (pd.Series(ia).str.len().values >= 2)).astype(np.float32)
    ca, cb = A["name_compact"].values, B["name_compact"].values
    F["acronym"] = ((ia == cb) | (ib == ca)).astype(np.float32)
    ta = A["name_tokens"].tolist()
    tb = B["name_tokens"].tolist()
    F["first_tok_eq"] = np.array([(x[0] == y[0]) if x and y else np.nan for x, y in zip(ta, tb)], dtype=np.float32)
    F["last_tok_eq"] = np.array([(x[-1] == y[-1]) if x and y else np.nan for x, y in zip(ta, tb)], dtype=np.float32)
    F["legal_eq"] = _eq3(A["name_legal"], B["name_legal"])
    # NEW: legal forms that cannot both be true (llc vs private limited); NaN if either is missing
    la_, lb_ = A["name_legal"].tolist(), B["name_legal"].tolist()
    F["legal_conflict"] = np.array([np.nan if not x or not y else float(not (set(x.split()) & set(y.split())))
                                    for x, y in zip(la_, lb_)], dtype=np.float32)
    F["house_eq"] = _eq3(A["addr_house"], B["addr_house"])
    F["house_digits_eq"] = _eq3(_digits(A["addr_house"]), _digits(B["addr_house"]))
    F["house_sfx_eq"] = _eq3(A["addr_house_sfx"], B["addr_house_sfx"])          # NEW 12 vs 12 bis
    F["postal_eq"] = _eq3(A["addr_postal"], B["addr_postal"])
    F["state_eq"] = _eq3(A["addr_state"], B["addr_state"])                     # NEW
    na_, nb_ = A["addr_nums"].tolist(), B["addr_nums"].tolist()
    nj = np.array([len(set(x) & set(y)) / len(set(x) | set(y)) if x and y else np.nan
                   for x, y in zip(na_, nb_)], dtype=np.float32)
    F["nums_jacc"] = nj
    F["nums_conflict"] = np.where(np.isnan(nj), np.nan, (nj == 0).astype(np.float32))
    ca_, cb_ = A["addr_comps"].tolist(), B["addr_comps"].tolist()
    F["comp_jacc"] = np.array([len(set(x) & set(y)) / len(set(x) | set(y)) if x and y else np.nan
                               for x, y in zip(ca_, cb_)], dtype=np.float32)
    F["landmark_a"] = np.array([bool(LANDMARK & set(x)) for x in A["addr_tokens"]], dtype=np.float32)
    F["landmark_b"] = np.array([bool(LANDMARK & set(x)) for x in B["addr_tokens"]], dtype=np.float32)
    F["name_was_indic"] = B["name_was_indic"].values.astype(np.float32)
    F["had_alias"] = np.maximum(A["name_had_alias"].values, B["name_had_alias"].values).astype(np.float32)  # NEW
    F["addr_empty_a"] = A["addr_empty"].values.astype(np.float32)             # NEW: nothing to compare
    F["addr_empty_b"] = B["addr_empty"].values.astype(np.float32)

    log("    rarity-weighted token overlap")
    nf = set_features(ta, tb, stats.name_df, stats.n, "name", n_jobs)
    for k, c in enumerate(["name_wjacc", "name_cov_a", "name_cov_b", "name_max_shared_idf",
                           "name_max_unshared_idf", "name_jacc"]):
        F[c] = nf[:, k]
    af = set_features(A["addr_tokens"].tolist(), B["addr_tokens"].tolist(), stats.addr_df, stats.n,
                      "addr", n_jobs)
    for k, c in enumerate(["addr_wjacc", "addr_cov_a", "addr_cov_b", "addr_max_shared_idf",
                           "addr_max_unshared_idf", "addr_jacc"]):
        F[c] = af[:, k]

    # rarity on a 0-1 scale, whatever the number of records
    scale = np.float32(np.log(stats.n + 1))
    for c in ("name_max_shared_idf", "name_max_unshared_idf", "addr_max_shared_idf", "addr_max_unshared_idf"):
        F[c] = F[c] / scale

    log("    lengths / frequencies / contradictions")
    F["name_len_diff"] = np.abs(A["name_len"].values - B["name_len"].values)
    F["name_len_ratio"] = (np.minimum(A["name_len"].values, B["name_len"].values) /
                           np.maximum(1, np.maximum(A["name_len"].values, B["name_len"].values)))
    F["name_ntok_diff"] = np.abs(A["name_ntok"].values - B["name_ntok"].values)
    F["name_ndigit_diff"] = np.abs(A["name_n_digits"].values - B["name_n_digits"].values)
    F["addr_len_ratio"] = (np.minimum(A["addr_len"].values, B["addr_len"].values) /
                           np.maximum(1, np.maximum(A["addr_len"].values, B["addr_len"].values)))
    F["addr_ntok_diff"] = np.abs(A["addr_ntok"].values - B["addr_ntok"].values)
    # how common is this name / address (per million records) -> generic names are weak evidence
    F["name_freq"] = np.log1p(A["name_core"].map(freq["name"]).fillna(1).values * 1e6 / freq["n"])
    F["addr_freq"] = np.log1p(A["addr_norm"].map(freq["addr"]).fillna(1).values * 1e6 / freq["n"])
    F["name_unique"] = (A["name_core"].map(freq["name"]).fillna(1).values <= 1).astype(np.float32)
    name_hi = (F["name_tset"] >= 90)
    addr_lo = (F["addr_tset"] < 50)
    F["name_hi_addr_lo"] = (name_hi & addr_lo).astype(np.float32)
    F["addr_hi_name_lo"] = ((F["addr_tset"] >= 90) & (F["name_tset"] < 50)).astype(np.float32)
    conflicts = np.zeros(len(F), dtype=np.float32)
    agrees = np.zeros(len(F), dtype=np.float32)
    for c in ["house_eq", "postal_eq", "legal_eq", "state_eq"]:
        v = F[c].values
        conflicts += (v == 0)
        agrees += (v == 1)
    conflicts += (F["nums_conflict"].values == 1)
    F["n_conflicts"] = conflicts
    F["n_agrees"] = agrees
    F["name_hi_house_conflict"] = (name_hi & (F["house_eq"] == 0)).astype(np.float32)

    # provenance
    F["src"] = cand["src"].values.astype(np.float32)
    F["n_keys"] = cand["n_keys"].values
    for b, k in enumerate(KEY_TYPES):
        F[f"key_{k}"] = ((cand["key_mask"].values.astype(np.int64) >> b) & 1).astype(np.float32)
    F["cheap"] = cand["cheap"].values
    F["rank_cheap"] = cand["rank_cheap"].values
    return F


def frequency_tables(s1: pd.DataFrame) -> dict:
    return {"name": s1["name_core"].value_counts(), "addr": s1["addr_norm"].value_counts(), "n": len(s1)}


# --------------------------------------------------------------------------
# Context features (computed from a first-stage probability p1)
# --------------------------------------------------------------------------
def context_features(cand: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    d = pd.DataFrame({"i": cand["i"].values, "j": cand["j"].values, "src": cand["src"].values, "p": p})
    C = pd.DataFrame(index=cand.index)
    g = d.groupby(["i", "src"])["p"]
    best = g.transform("max").values
    C["ctx_rank_es"] = g.rank(ascending=False, method="first").values
    C["ctx_gap_es"] = best - d["p"].values
    C["ctx_n_es"] = g.transform("size").values
    # margin over the runner-up within (entity, source)
    second = _second_best(d, ["i", "src"])
    C["ctx_margin_es"] = np.where(C["ctx_rank_es"].values == 1, d["p"].values - second, d["p"].values - best)
    ge = d.groupby("i")["p"]
    C["ctx_sum_e"] = ge.transform("sum").values
    C["ctx_n_hi_e"] = d.assign(h=(d["p"] > 0.5)).groupby("i")["h"].transform("sum").values
    # reverse view: among all Source 1 entities that picked this record
    gr = d.groupby(["src", "j"])["p"]
    rbest = gr.transform("max").values
    C["ctx_rank_rev"] = gr.rank(ascending=False, method="first").values
    C["ctx_gap_rev"] = rbest - d["p"].values
    C["ctx_n_rev"] = gr.transform("size").values
    rsecond = _second_best(d, ["src", "j"])
    C["ctx_margin_rev"] = np.where(C["ctx_rank_rev"].values == 1, d["p"].values - rsecond, d["p"].values - rbest)
    # support from the other source for the same entity
    mx = d.groupby(["i", "src"])["p"].max().unstack()
    other = np.full(len(d), np.nan, dtype=np.float32)
    for s in mx.columns:
        others = [c for c in mx.columns if c != s]
        if not others:
            continue
        om = mx[others].max(axis=1)
        sel = d["src"].values == s
        other[sel] = om.reindex(d.loc[sel, "i"].values).values
    C["ctx_other_src_max"] = other
    return C


def _second_best(d: pd.DataFrame, keys) -> np.ndarray:
    o = d.sort_values(keys + ["p"], ascending=[True] * len(keys) + [False])
    o["_r"] = o.groupby(keys).cumcount()
    sec = o[o["_r"] == 1].set_index(keys)["p"]
    idx = pd.MultiIndex.from_frame(d[keys])
    v = sec.reindex(idx).values
    return np.where(np.isnan(v), 0.0, v)
