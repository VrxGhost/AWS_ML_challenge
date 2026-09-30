"""train_dense.py - train the model at TEST density, in bounded memory .

    python src/train_dense.py --data-dir dataset --work-dir work_dense --entities 200000 --n-jobs 4

Why: run_pipeline.py trains on a thin sample (50k Source 1 entities with only their own matches
and a few other records).  The test set is dense: every business has many look-alikes.  A model
that never saw those look-alikes makes false matches.  Here the model learns from pairs built
exactly like the test pairs:
  - word rarity and name frequencies are counted over ALL training records (2.2M Source 1 +
    all Source 2/3, about the size of the test set, so these numbers also match the test);
  - each sampled Source 1 entity gets its shortlist against ALL Source 2/3 records, so its
    look-alikes are among the candidates, as in the test.
Only --entities Source 1 entities get candidates, features and labels (to keep time and RAM
small); the rest of the data is there as competitors.

Then, like run_pipeline.py train: split the sampled entities 80% fit / 10% A / 10% B, stage-1 model
with out-of-fold scores, stage-2 model with context features, calibration on A, decision rule
tuned on A, macro F0.5 reported on B.  Holdout B is dense, so its score is comparable with the
dev score of tune_on_dev.py (the leaderboard predictor), not with the old 0.98 holdout.

Writes <work-dir>/model.pkl (same format as run_pipeline.py), so the usual commands use it:
    python src/predict_lowmem.py --data-dir dataset --work-dir work_dense --out-dir output_dense --save-scores work_dense/test_scores.pkl

Needs the cleaner: <work-dir>/norm_state.json (copy it from your work folder), or it is fitted
from the training files (a few minutes).
Disk: about 2 GB per 100k entities of temporary files in <work-dir>/dense_tmp (deleted at the end).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import shutil
import sys
import time
import zlib

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import normalization as N  # noqa: E402
from config import Config  # noqa: E402
from decide import apply_decision, macro_f05, to_prediction, tune_decision  # noqa: E402
from features import build_features, context_features  # noqa: E402
from model import GBM, HAVE_LGB  # noqa: E402
from predict_lowmem import _load, _mem, blocking_country, pass1  # noqa: E402
from run_pipeline import fit_two_stage  # noqa: E402

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def read_truth(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for ln in f:
            p = ln.rstrip("\r\n").split("\t")
            out[p[0].strip()] = {x.strip() for x in (p[1].split(",") if len(p) > 1 else []) if x.strip()}
    return out


def features_country(cand, s1c, tparts, stats, freq, cfg, fdir, tag):
    """Features for every candidate pair of one country, one Source 2/3 slice at a time.
    Returns [(pair rows, path of the saved feature table)]."""
    out = []
    for src, files in tparts.items():
        for n, (fp, _) in enumerate(files):
            d = pd.read_pickle(fp)
            sel = np.where((cand["src"].values == src) & np.isin(cand["j"].values, d["gidx"].values))[0]
            if not len(sel):
                continue
            cg = cand.iloc[sel].reset_index(drop=True)
            local = pd.Series(np.arange(len(d)), index=d["gidx"].values)
            F = build_features(cg.assign(j=local.reindex(cg["j"].values).values), s1c, {src: d}, stats, freq,
                               cfg.n_jobs, log=lambda *_: None)
            path = os.path.join(fdir, f"{tag}_s{src}_{n:04d}.pkl")
            F.to_pickle(path)
            out.append((cg, path))
            del d, F
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--work-dir", default="work_dense")
    ap.add_argument("--entities", type=int, default=200_000, help="Source 1 entities that get labelled pairs")
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=200_000)
    ap.add_argument("--block-chunk", type=int, default=100_000)
    ap.add_argument("--cache", default="low", choices=["low", "full"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-context", action="store_true")
    ap.add_argument("--keep-temp", action="store_true")
    a = ap.parse_args()

    cfg = Config(data_dir=a.data_dir, work_dir=a.work_dir, n_jobs=a.n_jobs, seed=a.seed,
                 train_sample_s1=None, use_context=not a.no_context,
                 norm_state=os.path.join(a.work_dir, "norm_state.json"), cache_mode=a.cache)
    cfg.block_chunk = a.block_chunk
    os.makedirs(a.work_dir, exist_ok=True)
    tmp = os.path.join(a.work_dir, "dense_tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(os.path.join(tmp, "features"))

    # ---- cleaner
    if os.path.exists(cfg.norm_state):
        norm = N.Normalizer.load(cfg.norm_state)
        log(f"cleaner: loaded {cfg.norm_state}")
    else:
        log("cleaner: no norm_state.json in the work folder, fitting it from the training files")
        norm = N.fit_from_files(a.data_dir, 1, log=log)
        norm.save(cfg.norm_state)

    # ---- PASS 1 over ALL training files: slices on disk + global counts (same as predict)
    tdir = os.path.join(a.data_dir, "train")
    truth = read_truth(os.path.join(tdir, "train_ground_truth.tsv"))
    owners = pd.Series([x for v in truth.values() for x in v])
    single_owner = not owners.duplicated().any()
    stats, freq, scorer, ids, parts = pass1(a.data_dir, tmp, norm.state(), a, split="train")
    gc.collect()
    log(f"PASS 1 done, RAM {_mem()}")
    cfg.ref_records = stats.n                                   # universe size for size-independent features

    # ---- sampled entities (by a stable hash of the id, so re-runs pick the same ones)
    s1_ids = ids[1]
    frac = min(1.0, a.entities / max(1, len(s1_ids)))
    focus = {e for e in s1_ids if (zlib.crc32(e.encode()) % 1_000_000) < frac * 1_000_000}
    log(f"{len(focus):,} of {len(s1_ids):,} Source 1 entities get labelled pairs "
        f"(all {len(ids[2]) + len(ids[3]):,} Source 2/3 records are candidates)")

    # ---- PASS 2: dense shortlist + features for the sampled entities, one country at a time
    countries = sorted(parts[1]) if cfg.use_country_in_keys else [None]
    pieces = []
    for c in countries:
        if c is None:
            s1_files = [x for v in parts[1].values() for x in v]
            tparts = {k: [x for v in parts[k].values() for x in v] for k in (2, 3)}
        else:
            s1_files = parts[1][c]
            tparts = {k: parts[k].get(c, []) for k in (2, 3)}
        s1c = _load(s1_files)
        s1c = s1c[s1c["entity_id"].isin(focus)].reset_index(drop=True)
        if s1c.empty:
            continue
        log(f"country={c}: {len(s1c):,} sampled S1 vs S2 {sum(n for _, n in tparts[2]):,} + "
            f"S3 {sum(n for _, n in tparts[3]):,}")
        cand = blocking_country(s1c, tparts, stats, scorer, cfg)
        if cand is None:
            continue
        log(f"    candidates {len(cand):,} ({len(cand) / len(s1c):.1f} per entity), RAM {_mem()}")
        for cg, path in features_country(cand, s1c, tparts, stats, freq, cfg,
                                         os.path.join(tmp, "features"), str(c)):
            cg["i"] = s1c["gidx"].values[cg["i"].values]
            pieces.append((cg, path))
        del s1c, cand
        gc.collect()

    cand = pd.concat([p for p, _ in pieces], ignore_index=True)
    F = pd.concat([pd.read_pickle(f) for _, f in pieces], ignore_index=True)
    shutil.rmtree(os.path.join(tmp, "features"), ignore_errors=True)
    tgt_ids = {2: ids[2], 3: ids[3]}
    a_ids = s1_ids[cand["i"].values]
    b_ids = np.empty(len(cand), dtype=object)
    for s in (2, 3):
        m = cand["src"].values == s
        b_ids[m] = tgt_ids[s][cand["j"].values[m]]
    pairs = {f"{e}|{t}" for e in focus for t in truth.get(e, ())}
    y = (pd.Series(a_ids) + "|" + pd.Series(b_ids)).isin(pairs).to_numpy().astype(np.int8)
    n_pos_total = sum(len(truth.get(e, ())) for e in focus)
    log(f"DENSE pairs {len(cand):,} ({len(cand) / max(1, len(focus)):.1f} per entity), "
        f"BLOCKING RECALL {y.sum() / max(1, n_pos_total):.4f}, RAM {_mem()}")

    # ---- train exactly like run_pipeline.train, on the dense pairs
    focus_rows = np.unique(cand["i"].values)
    rng = np.random.RandomState(a.seed)
    ent = rng.permutation(np.array(sorted(focus_rows)))
    n_hold = int(len(ent) * cfg.holdout_frac)
    hold_A, hold_B = set(ent[: n_hold // 2]), set(ent[n_hold // 2: n_hold])
    is_A = cand["i"].isin(hold_A).values
    is_B = cand["i"].isin(hold_B).values
    tr_idx = np.where(~(is_A | is_B))[0]
    log(f"training pairs {len(tr_idx):,} (pos {y[tr_idx].sum():,}); holdout pairs {len(cand) - len(tr_idx):,}")
    log("stage 1 (+ out-of-fold scores)")
    m1, oof, _ = fit_two_stage(F.iloc[tr_idx].reset_index(drop=True), y[tr_idx], cand["i"].values[tr_idx], cfg)
    p1 = m1.predict(F)
    m2 = None
    if cfg.use_context:
        p1_ctx = p1.copy()
        p1_ctx[tr_idx] = oof
        C = context_features(cand, p1_ctx)
        # The REVERSE view ("how many / which Source 1 entities also picked this record") is left
        # out here.  In this training only the sampled entities (a few % of Source 1) have
        # shortlists, so a record almost never has a competitor (ctx_n_rev ~ 1).  In the test every
        # Source 1 entity competes, so these columns look completely different there and the model
        # would misread them.  The per-entity context columns are fine: each sampled entity has its
        # full test-like shortlist.  GBM.predict only uses the columns it was trained on, so predict
        # needs no change.
        C = C.drop(columns=[c for c in C.columns if c.endswith("_rev")])
        X2 = pd.concat([F, C], axis=1)
        log(f"stage 2 (pair + {C.shape[1]} context features; reverse-view columns left out)")
        m2 = GBM(cfg.seed, cfg.n_jobs).fit(X2.iloc[tr_idx], y[tr_idx])
        p = m2.predict(X2)
        del X2
    else:
        p = p1
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p[is_A], y[is_A])
    d = cand[["i", "j", "src"]].assign(p=iso.predict(p).astype(np.float32))
    truth_f = {e: truth.get(e, set()) for e in focus}
    ids_A = [s1_ids[i] for i in sorted(hold_A)]
    ids_B = [s1_ids[i] for i in sorted(hold_B)]
    log("decision tuning on dense holdout A")
    params = tune_decision(d[is_A], truth_f, s1_ids, tgt_ids, ids_A, single_owner, log)
    scores = {}
    for name, mask, idl in [("A", is_A, ids_A), ("B", is_B, ids_B)]:
        keep = apply_decision(d[mask], params)
        scores[name] = macro_f05(to_prediction(d[mask], keep, s1_ids, tgt_ids), truth_f, idl)
        log(f"DENSE HOLDOUT {name}: macro F0.5 = {scores[name]:.4f}")
    pred_B = to_prediction(d[is_B], apply_decision(d[is_B], params), s1_ids, tgt_ids)
    cn = {}
    for c, files in parts[1].items():
        for fp, _ in files:
            for e in pd.read_pickle(fp)["entity_id"].values:
                if e in focus:
                    cn[e] = c
    for c in sorted(set(cn.values())):
        idc = [e for e in ids_B if cn.get(e) == c]
        log(f"   DENSE HOLDOUT B country={c}: macro F0.5 = {macro_f05(pred_B, truth_f, idc):.4f} ({len(idc):,})")
    if HAVE_LGB:
        log("top features:\n" + (m2 or m1).importance().head(20).to_string())

    bundle = {"cfg": cfg.to_dict(), "norm_state": norm.state(), "m1": m1, "m2": m2, "iso": iso,
              "params": params, "single_owner": single_owner, "dense": True,
              "focus_frac": frac}          # make_devset.py uses this to leave the training entities out
    with open(os.path.join(a.work_dir, "model.pkl"), "wb") as f:
        pickle.dump(bundle, f)
    with open(os.path.join(a.work_dir, "train_report.json"), "w") as f:
        json.dump({"dense": True, "entities": len(focus), "focus_frac": frac, "pairs": int(len(cand)),
                   "blocking_recall": float(y.sum() / max(1, n_pos_total)), "decision": params,
                   "dense_holdout_f05": scores}, f, indent=2)
    log(f"saved {os.path.join(a.work_dir, 'model.pkl')}")
    if not a.keep_temp:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
