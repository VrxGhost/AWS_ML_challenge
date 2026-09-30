"""Candidate generation.

Stage A - multi-pass key blocking. Every record emits a handful of keys built
from its RAREST name / address tokens (IDF-ranked, so generic words such as
"group", "limited", "road" never form a block). Oversized blocks are dropped.
Stage B - a cheap TF-IDF (character n-gram) cosine on name and address ranks
the union of key hits and keeps the top-K per (Source 1 entity, source).

The Stage-B output is exactly the set the matching model scores, i.e. what
goes into output/candidate_pairs.tsv.

v2 (with normalization.py):
  - NEW key "nk": the order-free name key (sorted core words minus filler words), so
    "Solutions Advisory Akash" and "Akash Advisory Solutions Services" meet.
  - NEW key "pn": postcode + rarest name word (strong, cheap, works for any country
    that writes a postcode, France included).
  - KEY_TYPES is the single source of truth; features.py reads its length.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

KEY_TYPES = ["n1", "nn", "nc", "nf", "a2", "hs", "na", "ia", "np", "nk", "pn"]
KT = {k: i for i, k in enumerate(KEY_TYPES)}


# --------------------------------------------------------------------------
# Token statistics (document frequency -> IDF)
# --------------------------------------------------------------------------
class TokenStats:
    def __init__(self, reprs: List[pd.DataFrame]):
        self.name_df: Counter = Counter()
        self.addr_df: Counter = Counter()
        n = 0
        for r in reprs:
            n += len(r)
            for toks in r["name_tokens"]:
                self.name_df.update(set(toks))
            for toks in r["addr_tokens"]:
                self.addr_df.update(set(toks))
        self.n = max(n, 1)

    def name_idf(self, t: str) -> float:
        return math.log((self.n + 1) / (self.name_df.get(t, 0) + 1))

    def addr_idf(self, t: str) -> float:
        return math.log((self.n + 1) / (self.addr_df.get(t, 0) + 1))


def block_limits(cfg, n_records: int):
    """(max target count per key, max pairs per key).  The limits grow with
    the data, relative to the training universe: a key shared by 300 records in a 2M-record
    training set is as specific as one shared by 1,800 in a 12M-record test set."""
    scale = 1.0
    if getattr(cfg, "ref_records", 0):
        scale = min(max(1.0, n_records / cfg.ref_records), 4.0)     # capped at 4x
    return cfg.max_block_target * scale, cfg.max_block_pairs * scale


def _skeleton(t: str) -> str:
    if not t:
        return ""
    out = [t[0]]
    for ch in t[1:]:
        if ch in "aeiouhwy" or ch.isdigit():
            continue
        if ch != out[-1]:
            out.append(ch)
    return "".join(out[:5])


def record_keys(rep: pd.DataFrame, stats: TokenStats, use_country: bool = True) -> pd.DataFrame:
    """One row per (record position, key, key type)."""
    rows_pos, rows_key, rows_kt = [], [], []
    ndf, adf = stats.name_df, stats.addr_df
    for pos, (ntoks, alpha, house, compact, initials, country, nkey, postal) in enumerate(zip(
            rep["name_tokens"], rep["addr_alpha"], rep["addr_house"], rep["name_compact"],
            rep["name_initials"], rep["country_norm"], rep["name_key"], rep["addr_postal"])):
        c = country if use_country else ""
        nt = sorted({t for t in ntoks if len(t) > 1}, key=lambda t: (ndf.get(t, 0), t))
        at = sorted(set(alpha), key=lambda t: (adf.get(t, 0), t))
        keys = []
        if nt:
            keys.append(("n1", nt[0]))
            keys.append(("nf", _skeleton(nt[0]) + ("|" + nt[1][0] if len(nt) > 1 else "")))
        if len(nt) > 1:
            keys.append(("nn", "|".join(sorted(nt[:2]))))
        if compact:
            keys.append(("nc", compact))
        if nkey and " " in nkey:                      # single-word keys are already n1
            keys.append(("nk", nkey))
        if postal and nt:
            keys.append(("pn", postal + "|" + nt[0]))
        if len(at) > 1:
            keys.append(("a2", "|".join(sorted(at[:2]))))
        if house and at:
            keys.append(("hs", house + "|" + at[0]))
        if nt and at:
            keys.append(("na", nt[0] + "|" + at[0]))
        if len(initials) >= 2 and at:
            keys.append(("ia", initials + "|" + at[0]))
        if len(compact) >= 5 and at:
            keys.append(("np", compact[:5] + "|" + at[0]))
        for kt, k in keys:
            rows_pos.append(pos)
            rows_key.append(f"{kt}|{c}|{k}")
            rows_kt.append(KT[kt])
    kdf = pd.DataFrame({"pos": np.asarray(rows_pos, dtype=np.int64),
                        "key": pd.util.hash_pandas_object(pd.Series(rows_key, dtype=object),
                                                          index=False).values,
                        "kt": np.asarray(rows_kt, dtype=np.int8)})
    return kdf.drop_duplicates(["pos", "key"])


# --------------------------------------------------------------------------
# Cheap scorer
# --------------------------------------------------------------------------
class CheapScorer:
    """Character n-gram TF-IDF for names and addresses (fit once per corpus)."""

    def __init__(self, max_fit: int = 2_000_000, seed: int = 0):
        self.name_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2,
                                        dtype=np.float32, sublinear_tf=True)
        self.addr_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2,
                                        dtype=np.float32, sublinear_tf=True)
        self.max_fit = max_fit
        self.seed = seed

    def fit(self, reprs: List[pd.DataFrame]):
        names = pd.concat([r["name_core"] for r in reprs], ignore_index=True)
        addrs = pd.concat([r["addr_norm"] for r in reprs], ignore_index=True)
        if len(names) > self.max_fit:
            names = names.sample(self.max_fit, random_state=self.seed)
            addrs = addrs.sample(self.max_fit, random_state=self.seed)
        self.name_vec.fit(names.values)
        self.addr_vec.fit(addrs.values)
        return self

    def transform(self, rep: pd.DataFrame):
        return (self.name_vec.transform(rep["name_core"].values).tocsr(),
                self.addr_vec.transform(rep["addr_norm"].values).tocsr())


def rowwise_cos(A, B, i: np.ndarray, j: np.ndarray, chunk: int = 1_000_000) -> np.ndarray:
    out = np.empty(len(i), dtype=np.float32)
    for s in range(0, len(i), chunk):
        e = s + chunk
        out[s:e] = np.asarray(A[i[s:e]].multiply(B[j[s:e]]).sum(axis=1)).ravel()
    return out


# --------------------------------------------------------------------------
# Candidate generation
# --------------------------------------------------------------------------
def generate_candidates(s1: pd.DataFrame, targets: Dict[str, pd.DataFrame], stats: TokenStats,
                        scorer: CheapScorer, cfg, log=print) -> pd.DataFrame:
    """Return columns: i (row in s1), j (row in target), src (2/3), key_mask,
    n_keys, cos_name, cos_addr, cheap, rank_cheap."""
    k1 = record_keys(s1, stats, cfg.use_country_in_keys)
    X1n, X1a = scorer.transform(s1)
    k1_count = k1.groupby("key").size()
    out = []
    for src_name, tgt in targets.items():
        src = 2 if src_name.endswith("2") else 3
        kt = record_keys(tgt, stats, cfg.use_country_in_keys)
        kt_count = kt.groupby("key").size()
        both = pd.concat([k1_count.rename("a"), kt_count.rename("b")], axis=1, join="inner")
        lim_t, lim_p = block_limits(cfg, stats.n)
        ok = both[(both["b"] <= lim_t) & (both["a"] * both["b"] <= lim_p)].index
        kt_ok = kt[kt["key"].isin(ok)][["key", "pos", "kt"]].rename(columns={"pos": "j"})
        k1_ok = k1[k1["key"].isin(ok)][["key", "pos", "kt"]].rename(columns={"pos": "i"})
        log(f"  [{src_name}] usable keys {len(ok):,} / {len(both):,} shared")
        Xtn, Xta = scorer.transform(tgt)
        n1 = len(s1)
        for s in range(0, n1, cfg.block_chunk):
            part = k1_ok[(k1_ok["i"] >= s) & (k1_ok["i"] < s + cfg.block_chunk)]
            m = part.merge(kt_ok[["key", "j"]], on="key", how="inner")
            if m.empty:
                continue
            m = m[["i", "j", "kt"]].drop_duplicates()
            m["bit"] = np.left_shift(np.int64(1), m["kt"].values.astype(np.int64))
            g = m.groupby(["i", "j"], sort=False)["bit"].agg(["sum", "size"]).reset_index()
            g.columns = ["i", "j", "key_mask", "n_keys"]
            ii, jj = g["i"].values, g["j"].values
            g["cos_name"] = rowwise_cos(X1n, Xtn, ii, jj)
            g["cos_addr"] = rowwise_cos(X1a, Xta, ii, jj)
            g["cheap"] = 0.55 * g["cos_name"] + 0.45 * g["cos_addr"]
            grp = g.groupby("i", sort=False)
            r_joint = grp["cheap"].rank(ascending=False, method="first")
            r_name = grp["cos_name"].rank(ascending=False, method="first")
            r_addr = grp["cos_addr"].rank(ascending=False, method="first")
            keep = (r_joint <= cfg.topk_joint) | (r_name <= cfg.topk_name) | (r_addr <= cfg.topk_addr)
            keep &= g["cheap"] >= cfg.min_cheap
            g = g[keep].copy()
            g["src"] = np.int8(src)
            out.append(g)
            log(f"  [{src_name}] s1 rows {s:,}-{min(s + cfg.block_chunk, n1):,}: "
                f"raw pairs {len(r_joint):,} -> kept {len(g):,}")
    cand = pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["i", "j", "key_mask", "n_keys", "cos_name", "cos_addr", "cheap", "src"])
    cand["rank_cheap"] = cand.groupby(["i", "src"])["cheap"].rank(ascending=False, method="first")
    return cand
