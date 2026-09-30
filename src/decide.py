"""Entity-level decisions and the official metric.

Per Source 1 entity the metric is F0.5 = 1.25*TP / (1.25*TP + 0.25*FN + FP),
and with k predicted ids the denominator simplifies to (k + 0.25*G), where G
is the number of true matches. Given calibrated probabilities we choose, per
entity, the k (0..n, top-k by probability) that maximises expected F0.5:
    E[F | k>=1] ~= 1.25 * sum_{top k} p / (k + 0.25 * E[G])
    E[F | k=0]   = P(no true match) = prod(1 - p) * (1 - miss)
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Set

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Metric
# --------------------------------------------------------------------------
def f05(pred: Set[str], true: Set[str]) -> float:
    if not true and not pred:
        return 1.0
    if not true or not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)


def macro_f05(pred: Dict[str, Set[str]], truth: Dict[str, Set[str]], ids: Iterable[str]) -> float:
    ids = list(ids)
    return float(np.mean([f05(pred.get(e, set()), truth.get(e, set())) for e in ids])) if ids else float("nan")


# --------------------------------------------------------------------------
# Decisions
# --------------------------------------------------------------------------
def decide_threshold(d: pd.DataFrame, t: float) -> pd.Series:
    """d has columns i, p. Returns boolean mask of kept rows."""
    return d["p"] >= t


def decide_expected_f(d: pd.DataFrame, miss: float = 0.0, w0: float = 1.0, min_p: float = 0.05) -> pd.Series:
    """Choose k per entity maximising expected F0.5. w0 scales the value of
    predicting nothing (tuned on validation), miss = expected number of true
    matches lost by blocking per entity."""
    o = d[["i", "p"]].copy()
    o["p"] = o["p"].clip(1e-6, 1 - 1e-6)
    o = o.sort_values(["i", "p"], ascending=[True, False])
    g = o.groupby("i", sort=False)
    k = g.cumcount().values + 1
    cum = g["p"].cumsum().values
    G = g["p"].transform("sum").values + miss
    ef = 1.25 * cum / (k + 0.25 * G)
    ef[o["p"].values < min_p] = -1.0
    o["ef"] = ef
    o["k"] = k
    best_idx = o.groupby("i", sort=False)["ef"].idxmax()
    best = o.loc[best_idx, ["i", "ef", "k"]].set_index("i")
    log_none = np.log1p(-o["p"].values)
    p_none = np.exp(pd.Series(log_none, index=o.index).groupby(o["i"].values).sum()) * (1 - min(miss, 0.99))
    best["e0"] = w0 * p_none.reindex(best.index).values
    best["kstar"] = np.where(best["e0"] >= best["ef"], 0, best["k"])
    kstar = best["kstar"].reindex(o["i"].values).values
    keep = pd.Series(o["k"].values <= kstar, index=o.index)
    return keep.reindex(d.index)


def enforce_single_owner(d: pd.DataFrame, keep: pd.Series) -> pd.Series:
    """If each Source 2/3 record belongs to at most one Source 1 entity, keep
    only the highest-probability owner among the kept rows."""
    k = d[keep.values][["i", "j", "src", "p"]]
    if k.empty:
        return keep
    top = k.sort_values("p", ascending=False).drop_duplicates(["src", "j"])
    out = pd.Series(False, index=d.index)
    out.loc[top.index] = True
    return out


def to_prediction(d: pd.DataFrame, keep: pd.Series, s1_ids: np.ndarray, cand_ids: Dict[int, np.ndarray]) -> Dict[str, Set[str]]:
    k = d[keep.values]
    pred: Dict[str, Set[str]] = {}
    for src, ids in cand_ids.items():
        sub = k[k["src"] == src]
        for i, j in zip(sub["i"].values, sub["j"].values):
            pred.setdefault(s1_ids[i], set()).add(ids[j])
    return pred


def tune_decision(d: pd.DataFrame, truth: Dict[str, Set[str]], s1_ids, cand_ids, eval_ids, single_owner: bool,
                  log=print) -> dict:
    """Grid-search the decision rule on a validation slice; returns best params."""
    results: List[tuple] = []
    for t in [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
        keep = decide_threshold(d, t)
        for so in ([False, True] if single_owner else [False]):
            kk = enforce_single_owner(d, keep) if so else keep
            s = macro_f05(to_prediction(d, kk, s1_ids, cand_ids), truth, eval_ids)
            results.append((s, {"mode": "threshold", "t": t, "single_owner": so}))
    for miss in [0.0, 0.05, 0.15]:
        for w0 in [0.8, 1.0, 1.2, 1.5]:
            for min_p in [0.05, 0.2]:
                keep = decide_expected_f(d, miss, w0, min_p)
                for so in ([False, True] if single_owner else [False]):
                    kk = enforce_single_owner(d, keep) if so else keep
                    s = macro_f05(to_prediction(d, kk, s1_ids, cand_ids), truth, eval_ids)
                    results.append((s, {"mode": "expected_f", "miss": miss, "w0": w0, "min_p": min_p,
                                        "single_owner": so}))
    results.sort(key=lambda x: -x[0])
    best_thr = max((r for r in results if r[1]["mode"] == "threshold"), key=lambda r: r[0])
    log(f"    best global threshold : {best_thr[0]:.4f}  {best_thr[1]}")
    log(f"    best overall          : {results[0][0]:.4f}  {results[0][1]}")
    return results[0][1]


def apply_decision(d: pd.DataFrame, params: dict) -> pd.Series:
    if params["mode"] == "threshold":
        keep = decide_threshold(d, params["t"])
    else:
        keep = decide_expected_f(d, params["miss"], params["w0"], params["min_p"])
    if params.get("single_owner"):
        keep = enforce_single_owner(d, keep)
    return keep
