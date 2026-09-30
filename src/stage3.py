"""stage3.py - a small third model trained on the DEV set, where every business competes.

    python src/stage3.py --work-dir work_dense --dev-scores work_dense/dev_scores.pkl \
        --dev-truth dataset_dev_dense/dev_ground_truth.tsv --dev-s1 dataset_dev_dense/test/test_source1.tsv \
        --test-scores work_dense/test_scores.pkl --test-s1 dataset/test/test_source1.tsv --out-work work_s3

Why: dense training (train_dense.py) has to leave out the "reverse" context features ("which other
Source 1 businesses also picked this record, and is this one the best?"), because there only the
sampled businesses have shortlists.  The dev set is scored like the test: every dev business has a
shortlist, so those features are real there, and the dev set has answers.  So:

  1. dev pairs + their calibrated probability p (from predict_lowmem --save-scores)
  2. context features computed with full competition, country by country (records only compete
     inside a country, the blocking keys contain it)
  3. dev businesses split by id hash:  50% fit  |  25% tune  |  25% check (never used before the end)
  4. LightGBM on [p, logit p, context]; decision rule tuned on the tune part, for the new score AND,
     for a fair comparison, for the old p
  5. both rules scored on the check part.  Stage 3 is used ONLY if it beats the old score there.
  6. if it does: the same features on the test pairs -> <out-work>/test_scores_s3.pkl and
     <out-work>/model.pkl (a copy of model.pkl with the new rule).  Then:
        python src/predict_lowmem.py --data-dir dataset --work-dir <out-work> --out-dir output_s3 \
               --from-scores <out-work>/test_scores_s3.pkl

Memory: works one country at a time; the test part needs the most (about 6-8 GB for ~70M pairs).
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import pickle
import sys
import time
import zlib

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import context_features  # noqa: E402
from tune_on_dev import Fast, read_truth  # noqa: E402

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def s1_countries(path, ids1):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3,
                     usecols=["entity_id", "country"])
    m = dict(zip(df["entity_id"], df["country"].str.strip().str.lower()))
    return np.array([m.get(e, "?") for e in ids1], dtype=object)


def ctx_block(d: pd.DataFrame) -> pd.DataFrame:
    """[p, logit p, context features with full competition] for one country's pairs."""
    p = d["p"].to_numpy(np.float32)
    cand = d[["i", "j", "src"]].reset_index(drop=True)
    C = context_features(cand, p).astype(np.float32)
    q = np.clip(p, 1e-6, 1 - 1e-6)
    C.insert(0, "p", p)
    C.insert(1, "logit_p", np.log(q / (1 - q)).astype(np.float32))
    return C


def features(d: pd.DataFrame, cn_of_row: np.ndarray, cols=None):
    """Features for all pairs, country by country; returns float32 matrix in d's row order."""
    out = None
    for c in pd.unique(cn_of_row):
        sel = np.flatnonzero(cn_of_row == c)
        C = ctx_block(d.iloc[sel])
        if cols is None:
            cols = list(C.columns)
        if out is None:
            out = np.full((len(d), len(cols)), np.nan, dtype=np.float32)
        out[sel] = C[cols].to_numpy(np.float32)
        log(f"    context features country={c}: {len(sel):,} pairs")
        del C
        gc.collect()
    return out, cols


def rule_grid():
    g = []
    for so in (False, True):
        for t in [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98]:
            g.append({"mode": "threshold", "t": t, "single_owner": so})
        for miss in [0.0, 0.05, 0.15, 0.3]:
            for w0 in [0.8, 1.0, 1.2, 1.5, 2.0, 3.0]:
                for min_p in [0.05, 0.2, 0.4, 0.6]:
                    g.append({"mode": "expected_f", "miss": miss, "w0": w0, "min_p": min_p, "single_owner": so})
    return g


def best_rule(F: Fast):
    res = sorted(((F.score(F.keep_for(r))[0], r) for r in rule_grid()), key=lambda x: -x[0])
    return res[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", default="work_dense", help="folder with model.pkl")
    ap.add_argument("--dev-scores", default="work_dense/dev_scores.pkl")
    ap.add_argument("--dev-truth", default="dataset_dev_dense/dev_ground_truth.tsv")
    ap.add_argument("--dev-s1", default="dataset_dev_dense/test/test_source1.tsv")
    ap.add_argument("--test-scores", default="work_dense/test_scores.pkl")
    ap.add_argument("--test-s1", default="dataset/test/test_source1.tsv")
    ap.add_argument("--out-work", default="work_s3")
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--max-fit-rows", type=int, default=6_000_000)
    ap.add_argument("--min-gain", type=float, default=0.0005, help="needed gain on the check part")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    with open(os.path.join(a.work_dir, "model.pkl"), "rb") as f:
        B = pickle.load(f)

    # ---------------- dev
    log(f"loading dev scores {a.dev_scores}")
    S = pd.read_pickle(a.dev_scores)
    d = S["d"].reset_index(drop=True)
    ids = {1: np.asarray(S["ids1"], dtype=object), 2: np.asarray(S["ids2"], dtype=object),
           3: np.asarray(S["ids3"], dtype=object)}
    truth = read_truth(a.dev_truth)
    log(f"dev: {len(d):,} pairs, {len(ids[1]):,} Source 1 businesses, {len(truth):,} with answers")
    cn = s1_countries(a.dev_s1, ids[1])
    X, cols = features(d, cn[d["i"].to_numpy()])

    # labels as integer keys (no big string columns)
    pos = {s: pd.Series(np.arange(len(ids[s])), index=ids[s]) for s in (2, 3)}
    pos1 = pd.Series(np.arange(len(ids[1])), index=ids[1])
    tk = []
    for e, ts in truth.items():
        if e not in pos1.index:
            continue
        ie = int(pos1[e])
        for t in ts:
            s = 2 if t.startswith("S2-") else 3
            j = pos[s].get(t)
            if j is not None:
                tk.append((ie * 4 + s) * (1 << 24) + int(j))
    key = (d["i"].to_numpy(np.int64) * 4 + d["src"].to_numpy(np.int64)) * (1 << 24) + d["j"].to_numpy(np.int64)
    y = np.isin(key, np.array(tk, dtype=np.int64))
    log(f"dev true pairs in the shortlists: {int(y.sum()):,}")

    part_ent = np.array([zlib.crc32(e.encode()) % 4 for e in ids[1]], dtype=np.int8)
    part = part_ent[d["i"].to_numpy()]
    fit_m, tune_m, chk_m = part <= 1, part == 2, part == 3
    rng = np.random.RandomState(a.seed)
    fit_idx = np.flatnonzero(fit_m)
    if len(fit_idx) > a.max_fit_rows:
        fit_idx = np.sort(rng.choice(fit_idx, a.max_fit_rows, replace=False))
    tune_idx = np.flatnonzero(tune_m)
    es_idx = tune_idx if len(tune_idx) <= 2_000_000 else np.sort(rng.choice(tune_idx, 2_000_000, replace=False))
    log(f"fit on {len(fit_idx):,} pairs (pos {int(y[fit_idx].sum()):,}), early stop on {len(es_idx):,}")
    try:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(n_estimators=1500, learning_rate=0.08, num_leaves=63, min_child_samples=100,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.9, reg_lambda=1.0,
                                   random_state=a.seed, n_jobs=a.n_jobs, verbose=-1)
        model.fit(X[fit_idx], y[fit_idx], eval_set=[(X[es_idx], y[es_idx])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        log(f"trees: {model.best_iteration_}")
        imp = pd.Series(model.booster_.feature_importance("gain"), index=cols).sort_values(ascending=False)
        log("top stage-3 features: " + ", ".join(f"{k} {v / imp.sum():.0%}" for k, v in imp.head(6).items()))
    except ImportError:                                   # same fallback as model.py
        from sklearn.ensemble import HistGradientBoostingClassifier
        log("lightgbm not installed: using sklearn HistGradientBoosting (slower)")
        model = HistGradientBoostingClassifier(learning_rate=0.08, max_iter=600, max_leaf_nodes=63,
                                               min_samples_leaf=100, early_stopping=True,
                                               validation_fraction=0.1, n_iter_no_change=30,
                                               random_state=a.seed)
        model.fit(X[fit_idx], y[fit_idx])

    p3 = np.empty(len(d), dtype=np.float32)
    for m in (tune_m, chk_m):
        idx = np.flatnonzero(m)
        p3[idx] = model.predict_proba(X[idx])[:, 1]
    p0 = d["p"].to_numpy(np.float32)
    ent_part = {e: part_ent[pos1[e]] for e in truth if e in pos1.index}

    def evaluate(mask, pcol, which):
        sub = d.loc[mask, ["i", "j", "src"]].assign(p=pcol[mask])
        tr = {e: v for e, v in truth.items() if ent_part.get(e) == which}
        return Fast(sub, ids, tr)

    log("tuning the decision rule on the tune part (old score vs stage 3)")
    f_old_tune, r_old = best_rule(evaluate(tune_m, p0, 2))
    f_new_tune, r_new = best_rule(evaluate(tune_m, p3, 2))
    Fo, Fn = evaluate(chk_m, p0, 3), evaluate(chk_m, p3, 3)
    so, sn = Fo.score(Fo.keep_for(r_old)), Fn.score(Fn.keep_for(r_new))
    print("\n  CHECK PART (never used for fitting or tuning)      F0.5    prec    recall  empty")
    print(f"   old score  + its best rule  {str(r_old):<60} {so[0]:.4f}  {so[1]:.3f}  {so[2]:.3f}  {so[3]:.1%}")
    print(f"   stage 3    + its best rule  {str(r_new):<60} {sn[0]:.4f}  {sn[1]:.3f}  {sn[2]:.3f}  {sn[3]:.1%}")
    gain = sn[0] - so[0]
    print(f"\n  STAGE 3 GAIN on the check part: {gain:+.4f}\n")
    os.makedirs(a.out_work, exist_ok=True)
    report = {"check_old": so[0], "check_stage3": sn[0], "gain": gain, "rule_old": r_old, "rule_stage3": r_new}
    with open(os.path.join(a.out_work, "stage3_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    if gain < a.min_gain:
        log(f"stage 3 is not better by at least {a.min_gain} -> NOT used. Keep the current submission.")
        return
    del X, d, S, y, key
    gc.collect()

    # ---------------- test
    log(f"stage 3 helps -> scoring the test pairs {a.test_scores}")
    T = pd.read_pickle(a.test_scores)
    dt = T["d"].reset_index(drop=True)
    cnt = s1_countries(a.test_s1, np.asarray(T["ids1"], dtype=object))[dt["i"].to_numpy()]
    pt = np.empty(len(dt), dtype=np.float32)
    for c in pd.unique(cnt):
        sel = np.flatnonzero(cnt == c)
        C = ctx_block(dt.iloc[sel])
        pt[sel] = model.predict_proba(C[cols].to_numpy(np.float32))[:, 1]
        log(f"    test country={c}: {len(sel):,} pairs scored")
        del C
        gc.collect()
    out_scores = os.path.join(a.out_work, "test_scores_s3.pkl")
    pd.to_pickle({"d": dt[["i", "j", "src"]].assign(p=pt), "ids1": T["ids1"], "ids2": T["ids2"],
                  "ids3": T["ids3"]}, out_scores)
    B2 = copy.copy(B)
    B2["params"] = r_new
    B2["stage3"] = {"model": model, "cols": cols, "report": report}
    with open(os.path.join(a.out_work, "model.pkl"), "wb") as f:
        pickle.dump(B2, f)
    log(f"saved {out_scores} and {a.out_work}/model.pkl (rule {r_new})")
    log(f"next: python src/predict_lowmem.py --data-dir dataset --work-dir {a.out_work} "
        f"--out-dir output_s3 --from-scores {out_scores}")


if __name__ == "__main__":
    main()
