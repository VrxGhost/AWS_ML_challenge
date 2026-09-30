"""make_devset.py - build a DEV set from the training data that looks like the real test set.

    python src/make_devset.py --data-dir dataset --work-dir work --out dataset_dev

Why: the holdout score printed by training (0.98+) is measured in a small, "thin" sample, while
the test set is dense (1.7M Source 1 records, ~10M Source 2/3 records).  In a dense set every
record has many more look-alikes, so the model makes more false matches.  This dev set has test
density AND known answers, so its score predicts the leaderboard and can be used to tune the
decision rule (tune_on_dev.py).

What it writes (same layout as dataset/test, so every predict script reads it unchanged):
    <out>/test/test_source1.tsv      Source 1 entities NOT used to train work/model.pkl
    <out>/test/test_source2.tsv      Source 2 records, minus those owned by training entities
    <out>/test/test_source3.tsv      Source 3 records, minus those owned by training entities
    <out>/dev_ground_truth.tsv       the answers for the dev Source 1 entities

The training entities are found by repeating exactly the sampling that made model.pkl, so no dev
entity was seen in training:
  - run_pipeline.py model: the same random sample (size and seed are stored in model.pkl);
  - train_dense.py model:  the same id-hash rule (the share is stored in model.pkl as focus_frac).
    A dense model.pkl from before focus_frac was saved needs --dense-entities N (the --entities
    you gave train_dense.py).
--frac 0.5 keeps a random half of the dev Source 1 entities (faster) but still keeps ALL
Source 2/3 records as competitors, so the density stays test-like.

Streams the files line by line: memory stays small (ids + the ground truth only).
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
import zlib

import numpy as np

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def first_field(line: str) -> str:
    return line.split("\t", 1)[0].strip()


def read_truth(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for ln in f:
            p = ln.rstrip("\r\n").split("\t")
            out[p[0].strip()] = [x.strip() for x in (p[1].split(",") if len(p) > 1 else []) if x.strip()]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--work-dir", default="work", help="folder with the model.pkl whose training sample to exclude")
    ap.add_argument("--out", default="dataset_dev")
    ap.add_argument("--frac", type=float, default=1.0, help="share of dev Source 1 entities to keep (1 = all)")
    ap.add_argument("--dense-entities", type=int, default=0,
                    help="only for an older train_dense model.pkl without focus_frac: the --entities it used")
    a = ap.parse_args()

    tdir = os.path.join(a.data_dir, "train")
    with open(os.path.join(a.work_dir, "model.pkl"), "rb") as f:
        B = pickle.load(f)
    cfg = B["cfg"]
    n_sample, seed = cfg.get("train_sample_s1"), cfg.get("seed", 42)

    # 1) Source 1 ids in file order -> repeat run_pipeline.consistent_sample's choice
    s1_path = os.path.join(tdir, "train_source1.tsv")
    with open(s1_path, encoding="utf-8") as f:
        next(f)
        s1_ids = [first_field(ln) for ln in f if ln.strip()]
    if B.get("dense"):                                  # train_dense.py: same id-hash rule it used
        frac = B.get("focus_frac")
        if frac is None:
            if not a.dense_entities:
                sys.exit("this model.pkl comes from train_dense.py but does not store focus_frac.\n"
                         "Run again with --dense-entities N (the --entities value you trained with).")
            frac = min(1.0, a.dense_entities / max(1, len(s1_ids)))
        if frac >= 1.0:
            sys.exit("the dense model used ALL Source 1 entities, so no unseen entities are left for a dev set.")
        used = {e for e in s1_ids if (zlib.crc32(e.encode()) % 1_000_000) < frac * 1_000_000}
        log(f"dense model: training entities re-found with the id-hash rule (share {frac:.4f})")
    else:                                               # run_pipeline.py: repeat consistent_sample
        if not n_sample or n_sample >= len(s1_ids):
            sys.exit("model.pkl was trained on ALL Source 1 entities, so no unseen entities are left for a dev set.\n"
                     "Train with --sample-s1 (for example 150000) and run this again.")
        rng = np.random.RandomState(seed)
        used = set(np.asarray(s1_ids, dtype=object)[rng.choice(len(s1_ids), n_sample, replace=False)])
    keep = lambda e: e not in used and (a.frac >= 1 or zlib.crc32(e.encode()) % 1000 < a.frac * 1000)
    dev = {e for e in s1_ids if keep(e)}
    log(f"Source 1: {len(s1_ids):,} entities, {len(used):,} used in training -> {len(dev):,} dev entities")

    # 2) records owned by training entities are left out (their owner is not in the dev set)
    truth = read_truth(os.path.join(tdir, "train_ground_truth.tsv"))
    owned_by_train = {t for e in used for t in truth.get(e, ())}

    out_t = os.path.join(a.out, "test")
    os.makedirs(out_t, exist_ok=True)
    kept_tgt = set()
    for k in (1, 2, 3):
        src = os.path.join(tdir, f"train_source{k}.tsv")
        dst = os.path.join(out_t, f"test_source{k}.tsv")
        n_in = n_out = 0
        with open(src, encoding="utf-8", newline="") as fi, open(dst, "w", encoding="utf-8", newline="") as fo:
            fo.write(next(fi))                                   # header
            for ln in fi:
                if not ln.strip():
                    continue
                n_in += 1
                e = first_field(ln)
                ok = (e in dev) if k == 1 else (e not in owned_by_train)
                if ok:
                    fo.write(ln if ln.endswith("\n") else ln + "\n")
                    n_out += 1
                    if k != 1:
                        kept_tgt.add(e)
        log(f"source {k}: {n_in:,} -> {n_out:,} records -> {dst}")

    # 3) answers for the dev entities
    gt = os.path.join(a.out, "dev_ground_truth.tsv")
    n_match = n_single = 0
    with open(gt, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for e in s1_ids:
            if e in dev:
                m = [t for t in truth.get(e, ()) if t in kept_tgt]
                n_match += len(m)
                n_single += not m
                f.write(f"{e}\t{','.join(m)}\n")
    log(f"wrote {gt}: {len(dev):,} entities, {n_match:,} true matches, "
        f"{n_single / max(1, len(dev)):.1%} with no match")
    log(f"next: python src/predict_lowmem.py --data-dir {a.out} --work-dir {a.work_dir} "
        f"--out-dir {a.out}/output --save-scores {a.work_dir}/dev_scores.pkl")


if __name__ == "__main__":
    main()
