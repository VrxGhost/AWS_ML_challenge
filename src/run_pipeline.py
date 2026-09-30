"""End-to-end pipeline (v2: one normaliser).

    python src/run_pipeline.py train   --data-dir dataset --work-dir work
    python src/run_pipeline.py predict --data-dir dataset --work-dir work --out-dir output
    python src/run_pipeline.py all     ...   (train then predict)

What changed in v2
  - normalization.py replaces normalize.py + indic_latin.py.  One Normalizer object holds
    everything learned from training (Indian-script dictionary, abbreviation maps, protected
    look-alike words).  It is fitted ONCE on ALL training true matches (before sampling),
    saved to work/norm_state.json AND inside model.pkl, and predict() reuses it unchanged.
    (Old flow: normalise -> mine abbreviations -> normalise everything a second time.)
  - Normalised frames are cached in work/cache/ (keyed by the normaliser state + sample),
    so a second train/predict run after changing only blocking/model settings skips this step.
  - predict() prints the country labels found in each test source (France check).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blocking import CheapScorer, TokenStats, generate_candidates  # noqa: E402
from config import Config  # noqa: E402
from decide import apply_decision, macro_f05, to_prediction, tune_decision  # noqa: E402
from features import build_features, context_features, frequency_tables  # noqa: E402
from io_utils import read_sources, read_truth, self_validate, write_id_lists  # noqa: E402
from model import GBM, HAVE_LGB  # noqa: E402
from normalization import Normalizer, repr_frame  # noqa: E402

T0 = time.time()


def log(msg: str):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------
def fit_normalizer(src: dict, truth: dict, cfg: Config) -> Normalizer:
    """Learn translit + abbreviations + protected words from ALL training true matches."""
    owner = {b: a for a, bs in truth.items() for b in bs}
    s1_rows = zip(src["s1"]["entity_id"], src["s1"]["business_name"],
                  src["s1"]["business_address"], src["s1"]["country"])
    tg_rows = []
    for k in ("s2", "s3"):
        d = src[k]
        m = d["entity_id"].isin(owner).values
        tg_rows += list(zip(d["entity_id"].values[m], d["business_name"].values[m],
                            d["business_address"].values[m], d["country"].values[m]))
    return Normalizer().fit(s1_rows, tg_rows, owner, abbrev_min_count=cfg.mine_abbrev_min_count, log=log)


def get_normalizer(src, truth, cfg: Config, refit: bool) -> Normalizer:
    path = cfg.norm_state
    if path and os.path.exists(path) and not refit:
        log(f"normaliser: loading {path} (use --refit-norm to learn it again)")
        return Normalizer.load(path)
    log("normaliser: fitting on all training true matches")
    norm = fit_normalizer(src, truth, cfg)
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        norm.save(path)
        log(f"normaliser: saved {path}")
    return norm


def _state_hash(norm: Normalizer) -> str:
    return hashlib.md5(json.dumps(norm.state(), sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:10]


def normalise_all(src: dict, norm: Normalizer, cfg: Config, tag: str) -> dict:
    """{s1,s2,s3} raw frames -> representation frames (cached on disk)."""
    reps = {}
    h = _state_hash(norm)
    cdir = os.path.join(cfg.work_dir, "cache")
    for k, df in src.items():
        ids = df["entity_id"].values
        sig = hashlib.md5((ids[0] + ids[-1] + str(len(ids))).encode()).hexdigest()[:8] if len(ids) else "0"
        path = os.path.join(cdir, f"{tag}_{k}_{h}_{sig}.pkl")
        if cfg.use_cache and os.path.exists(path):
            reps[k] = pd.read_pickle(path)
            log(f"  {k}: {len(reps[k]):,} rows from cache")
            continue
        t = time.time()
        reps[k] = repr_frame(df, norm, cfg.n_jobs, cache_mode=cfg.cache_mode)
        log(f"  {k}: normalised {len(df):,} rows ({len(df) / max(1e-9, time.time() - t):,.0f}/s)")
        if cfg.use_cache:
            os.makedirs(cdir, exist_ok=True)
            reps[k].to_pickle(path)
    return reps


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def consistent_sample(src: dict, truth: dict, n: int, seed: int):
    """Sample n S1 entities and keep a self-consistent universe."""
    s1 = src["s1"]
    if n is None or n >= len(s1):
        return src, truth
    rng = np.random.RandomState(seed)
    keep_s1 = set(s1["entity_id"].values[rng.choice(len(s1), n, replace=False)])
    frac = n / len(s1)
    matched_all = {x for v in truth.values() for x in v}
    matched_keep = {x for e in keep_s1 for x in truth.get(e, ())}
    out = {"s1": s1[s1["entity_id"].isin(keep_s1)].reset_index(drop=True)}
    for k in ("s2", "s3"):
        d = src[k]
        orphan = ~d["entity_id"].isin(matched_all)
        take_orphan = orphan & (rng.rand(len(d)) < frac)
        out[k] = d[d["entity_id"].isin(matched_keep) | take_orphan].reset_index(drop=True)
    return out, {e: truth.get(e, set()) for e in keep_s1}


def label_pairs(cand, s1_ids, tgt_ids, truth) -> np.ndarray:
    pos = {(a, b) for a, bs in truth.items() for b in bs}
    y = np.zeros(len(cand), dtype=np.int8)
    for src, ids in tgt_ids.items():
        m = np.where(cand["src"].values == src)[0]
        a = s1_ids[cand["i"].values[m]]
        b = ids[cand["j"].values[m]]
        y[m] = [(x, z) in pos for x, z in zip(a, b)]
    return y


def candidates_and_features(reps, cfg):
    stats = TokenStats(list(reps.values()))
    if not cfg.ref_records:                            # training: remember the universe size
        cfg.ref_records = stats.n
    scorer = CheapScorer(seed=cfg.seed).fit(list(reps.values()))
    log("blocking")
    cand = generate_candidates(reps["s1"], {"s2": reps["s2"], "s3": reps["s3"]}, stats, scorer, cfg, log)
    cand = cand.reset_index(drop=True)
    log(f"candidates: {len(cand):,} pairs, {cand['i'].nunique():,} S1 entities with >=1 candidate, "
        f"avg {len(cand) / max(1, len(reps['s1'])):.1f} per S1 entity")
    log("features")
    F = build_features(cand, reps["s1"], {2: reps["s2"], 3: reps["s3"]}, stats,
                       frequency_tables(reps["s1"]), cfg.n_jobs, log)
    return cand, F


def ids_of(reps):
    return (reps["s1"]["entity_id"].values,
            {2: reps["s2"]["entity_id"].values, 3: reps["s3"]["entity_id"].values})


def fit_two_stage(F, y, groups, cfg):
    """Stage 1 on pair features; out-of-fold stage-1 scores -> context features."""
    rows = np.arange(len(F))
    if cfg.neg_keep < 1.0:
        rng = np.random.RandomState(cfg.seed)
        rows = rows[(y == 1) | (rng.rand(len(y)) < cfg.neg_keep)]
    m1 = GBM(cfg.seed, cfg.n_jobs).fit(F.iloc[rows], y[rows])
    if not cfg.use_context:
        return m1, None, None
    oof = np.zeros(len(F), dtype=np.float32)
    for tr, va in GroupKFold(cfg.n_folds).split(F, y, groups):
        tr = np.intersect1d(tr, rows)
        oof[va] = GBM(cfg.seed, cfg.n_jobs).fit(F.iloc[tr], y[tr]).predict(F.iloc[va])
    return m1, oof, rows


def score(m1, m2, F, cand, p1=None):
    if p1 is None:
        p1 = m1.predict(F)
    if m2 is None:
        return p1, p1
    X2 = pd.concat([F, context_features(cand, p1)], axis=1)
    return p1, m2.predict(X2)


# --------------------------------------------------------------------------
# TRAIN
# --------------------------------------------------------------------------
def train(cfg: Config, refit_norm: bool = False):
    os.makedirs(cfg.work_dir, exist_ok=True)
    tdir = os.path.join(cfg.data_dir, "train")
    truth_all = read_truth(os.path.join(tdir, "train_ground_truth.tsv"))

    # one-owner check: does any S2/S3 record belong to more than one S1 entity?
    owners = pd.Series([x for v in truth_all.values() for x in v])
    multi = int(owners.duplicated().sum())
    single_owner = multi == 0
    n_single = sum(1 for v in truth_all.values() if not v)
    log(f"ground truth: {len(truth_all):,} S1 entities, singletons {n_single / max(1, len(truth_all)):.1%}, "
        f"records with >1 owner: {multi:,} -> single-owner rule {'ON' if single_owner else 'OFF'}")

    src = read_sources(tdir, "train")
    log(f"loaded train: S1 {len(src['s1']):,}  S2 {len(src['s2']):,}  S3 {len(src['s3']):,}")
    norm = get_normalizer(src, truth_all, cfg, refit_norm)           # learned from ALL train
    src, truth = consistent_sample(src, truth_all, cfg.train_sample_s1, cfg.seed)
    del truth_all
    log(f"sample: S1 {len(src['s1']):,}  S2 {len(src['s2']):,}  S3 {len(src['s3']):,}")
    reps = normalise_all(src, norm, cfg, f"train{cfg.train_sample_s1 or 'all'}s{cfg.seed}")
    del src

    cand, F = candidates_and_features(reps, cfg)
    s1_ids, tgt_ids = ids_of(reps)
    y = label_pairs(cand, s1_ids, tgt_ids, truth)
    n_pos_total = sum(len(v) for v in truth.values())
    log(f"BLOCKING RECALL: {y.sum():,} / {n_pos_total:,} = {y.sum() / max(1, n_pos_total):.4f}  "
        f"(pairs/entity {len(cand) / len(s1_ids):.1f})")
    ub_pred = to_prediction(cand.assign(p=1.0), pd.Series(y == 1, index=cand.index), s1_ids, tgt_ids)
    log(f"upper-bound macro F0.5 with a perfect matcher on these candidates: "
        f"{macro_f05(ub_pred, truth, s1_ids):.4f}")

    # entity-level split: train / holdout(A: calibrate+tune, B: report)
    rng = np.random.RandomState(cfg.seed)
    ent = rng.permutation(len(s1_ids))
    n_hold = int(len(ent) * cfg.holdout_frac)
    hold_A, hold_B = set(ent[: n_hold // 2]), set(ent[n_hold // 2: n_hold])
    is_A = cand["i"].isin(hold_A).values
    is_B = cand["i"].isin(hold_B).values
    is_tr = ~(is_A | is_B)
    tr_idx = np.where(is_tr)[0]
    log(f"training pairs {is_tr.sum():,} (pos {y[is_tr].sum():,}); holdout pairs {(~is_tr).sum():,}")

    log("stage 1 (+ out-of-fold scores)")
    m1, oof, _ = fit_two_stage(F.iloc[tr_idx].reset_index(drop=True), y[tr_idx],
                               cand["i"].values[tr_idx], cfg)
    p1 = m1.predict(F)
    m2 = None
    if cfg.use_context:
        p1_ctx = p1.copy()
        p1_ctx[tr_idx] = oof           # train rows use honest out-of-fold scores
        X2 = pd.concat([F, context_features(cand, p1_ctx)], axis=1)
        log("stage 2 (pair + context features)")
        m2 = GBM(cfg.seed, cfg.n_jobs).fit(X2.iloc[tr_idx], y[tr_idx])
        p = m2.predict(X2)
    else:
        p = p1

    # calibrate on A, tune decision on A, report on B
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p[is_A], y[is_A])
    pc = iso.predict(p).astype(np.float32)
    d = cand[["i", "j", "src"]].assign(p=pc)
    ids_A = s1_ids[sorted(hold_A)]
    ids_B = s1_ids[sorted(hold_B)]
    log("decision tuning on holdout A")
    params = tune_decision(d[is_A], truth, s1_ids, tgt_ids, ids_A, single_owner, log)
    scores = {}
    for name, mask, ids in [("A", is_A, ids_A), ("B", is_B, ids_B)]:
        keep = apply_decision(d[mask], params)
        base = d[mask]["p"] >= 0.5
        sc = macro_f05(to_prediction(d[mask], keep, s1_ids, tgt_ids), truth, ids)
        sb = macro_f05(to_prediction(d[mask], base, s1_ids, tgt_ids), truth, ids)
        scores[name] = sc
        log(f"HOLDOUT {name}: macro F0.5 = {sc:.4f}   (plain p>=0.5: {sb:.4f})")
    keep_B = apply_decision(d[is_B], params)
    pred_B = to_prediction(d[is_B], keep_B, s1_ids, tgt_ids)
    sing = [e for e in ids_B if not truth.get(e)]
    if sing:
        log(f"  singletons in B: {len(sing):,}, correctly left empty: "
            f"{np.mean([not pred_B.get(e) for e in sing]):.4f}")
    # score by country on B (France is not in train, but US vs India tells you where to work)
    cn = reps["s1"]["country_norm"].values
    for c in sorted(set(cn[sorted(hold_B)])):
        ids_c = [s1_ids[i] for i in sorted(hold_B) if cn[i] == c]
        log(f"  HOLDOUT B country={c}: macro F0.5 = {macro_f05(pred_B, truth, ids_c):.4f} ({len(ids_c):,} entities)")

    if HAVE_LGB:
        imp = (m2 or m1).importance().head(25)
        log("top features:\n" + imp.to_string())

    bundle = {"cfg": cfg.to_dict(), "norm_state": norm.state(), "m1": m1, "m2": m2, "iso": iso,
              "params": params, "single_owner": single_owner}
    with open(os.path.join(cfg.work_dir, "model.pkl"), "wb") as f:
        pickle.dump(bundle, f)
    with open(os.path.join(cfg.work_dir, "train_report.json"), "w") as f:
        json.dump({"decision": params, "blocking_recall": float(y.sum() / max(1, n_pos_total)),
                   "pairs_per_entity": float(len(cand) / len(s1_ids)),
                   "holdout_f05": scores}, f, indent=2)
    log("saved model bundle")


# --------------------------------------------------------------------------
# PREDICT
# --------------------------------------------------------------------------
def predict(cfg: Config):
    with open(os.path.join(cfg.work_dir, "model.pkl"), "rb") as f:
        B = pickle.load(f)
    norm = Normalizer.from_state(B["norm_state"])             # exactly what training used
    cfg.ref_records = B["cfg"].get("ref_records", 0)          # training universe size
    tdir = os.path.join(cfg.data_dir, "test")
    src = read_sources(tdir, "test")
    log(f"loaded test: S1 {len(src['s1']):,}  S2 {len(src['s2']):,}  S3 {len(src['s3']):,}")
    for k, df in src.items():                                  # France check: spelled the same way?
        log(f"  {k} country labels: {df['country'].value_counts().head(8).to_dict()}")
    reps = normalise_all(src, norm, cfg, "test")
    del src
    cand, F = candidates_and_features(reps, cfg)
    s1_ids, tgt_ids = ids_of(reps)
    log("scoring")
    _, p = score(B["m1"], B["m2"], F, cand)
    pc = B["iso"].predict(p).astype(np.float32)
    d = cand[["i", "j", "src"]].assign(p=pc)
    keep = apply_decision(d, B["params"])
    pred = to_prediction(d, keep, s1_ids, tgt_ids)
    allc = to_prediction(d, pd.Series(True, index=d.index), s1_ids, tgt_ids)

    os.makedirs(cfg.out_dir, exist_ok=True)
    mp_ = os.path.join(cfg.out_dir, "matching_results.tsv")
    cp = os.path.join(cfg.out_dir, "candidate_pairs.tsv")
    write_id_lists(mp_, "matched_entity_ids", s1_ids, pred)
    write_id_lists(cp, "candidate_entity_ids", s1_ids, allc)
    n_empty = sum(1 for e in s1_ids if not pred.get(e))
    log(f"wrote {mp_} ({len(s1_ids):,} rows, {n_empty / len(s1_ids):.1%} empty) and {cp}")
    by_country = reps["s1"].assign(empty=[not pred.get(e) for e in s1_ids]).groupby("country_norm")["empty"].mean()
    log("share predicted as no-match, by country (compare with train; France far off = look at it):\n"
        + by_country.to_string())
    issues = self_validate(mp_, cp, s1_ids, tgt_ids[2], tgt_ids[3])
    log("self-validation: " + ("PASS" if not issues else "ISSUES\n  " + "\n  ".join(issues[:20])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["train", "predict", "all"])
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--sample-s1", type=int, default=Config.train_sample_s1,
                    help="train on N sampled Source 1 entities (0 = all)")
    ap.add_argument("--n-jobs", type=int, default=Config.n_jobs)
    ap.add_argument("--no-context", action="store_true")
    ap.add_argument("--refit-norm", action="store_true",
                    help="learn the normaliser again even if work/norm_state.json exists")
    ap.add_argument("--no-cache", action="store_true", help="do not read/write work/cache/")
    ap.add_argument("--low-mem", action="store_true",
                    help="small normaliser caches per worker (use with a smaller --sample-s1 on a laptop)")
    a = ap.parse_args()
    cfg = Config(data_dir=a.data_dir, work_dir=a.work_dir, out_dir=a.out_dir,
                 train_sample_s1=(a.sample_s1 or None), n_jobs=a.n_jobs, use_context=not a.no_context,
                 norm_state=os.path.join(a.work_dir, "norm_state.json"), use_cache=not a.no_cache,
                 cache_mode="low" if a.low_mem else "full")
    log(f"LightGBM available: {HAVE_LGB}")
    if a.stage in ("train", "all"):
        train(cfg, a.refit_norm)
    if a.stage in ("predict", "all"):
        predict(cfg)


if __name__ == "__main__":
    main()
