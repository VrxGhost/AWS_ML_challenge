"""tune_on_dev.py - pick the decision rule on the dense DEV set, then (optionally) save it into model.pkl.

    python src/tune_on_dev.py --scores work/dev_scores.pkl --truth dataset_dev/dev_ground_truth.tsv --work-dir work
    python src/tune_on_dev.py ... --write          # also store the best rule in work/model.pkl (backup kept)

Inputs
  --scores  pairs + calibrated probabilities saved by
            predict_lowmem.py --data-dir dataset_dev ... --save-scores work/dev_scores.pkl
  --truth   dataset_dev/dev_ground_truth.tsv from make_devset.py

What it prints
  - the dev macro F0.5 of the rule currently in model.pkl (your best leaderboard predictor)
  - the best rules found, per family (global threshold / expected-F, each with and without one-owner)
  - how the best rule changes precision, recall and the share of empty predictions
  - per-country dev score (if --by-country)

The search uses fast vectorised copies of decide.py's rules; the winner is re-checked with the
real decide.apply_decision + macro_f05 before anything is written, so predict gets exactly the
score shown here.  Afterwards re-apply the rule to the TEST pairs without recomputing anything:
    python src/predict_lowmem.py --work-dir work --from-scores work/test_scores.pkl --out-dir output
"""
from __future__ import annotations

import argparse
import os
import pickle
import shutil
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decide import apply_decision, macro_f05, to_prediction  # noqa: E402

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def read_truth(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for ln in f:
            p = ln.rstrip("\r\n").split("\t")
            out[p[0].strip()] = {x.strip() for x in (p[1].split(",") if len(p) > 1 else []) if x.strip()}
    return out


class Fast:
    """Everything that does not depend on the rule is computed once here."""

    def __init__(self, d, ids, truth):
        self.n_ent = len(ids[1])
        self.i = d["i"].to_numpy(np.int64)
        self.p = np.clip(d["p"].to_numpy(np.float64), 1e-6, 1 - 1e-6)
        self.p32 = d["p"].to_numpy(np.float32)     # thresholds compared exactly like decide.py (float32)
        src = d["src"].to_numpy(np.int64)
        j = d["j"].to_numpy(np.int64)
        s1 = pd.Series(ids[1][self.i])
        tgt = np.empty(len(d), dtype=object)
        for s in (2, 3):
            m = src == s
            tgt[m] = ids[s][j[m]]
        pairs = {f"{a}|{b}" for a, bs in truth.items() for b in bs}
        self.y = (s1 + "|" + pd.Series(tgt)).isin(pairs).to_numpy()
        # G = number of true matches per entity (including ones blocking missed)
        g = pd.Series({e: len(v) for e, v in truth.items()})
        self.G = g.reindex(ids[1]).fillna(0).to_numpy(np.float64)
        self.in_truth = pd.Series(ids[1]).isin(truth.keys()).to_numpy()
        # order by entity, probability descending -> expected-F rule
        self.oe = np.lexsort((-self.p, self.i))
        ie = self.i[self.oe]
        self.starts = np.flatnonzero(np.r_[True, ie[1:] != ie[:-1]])
        self.gent = ie[self.starts]
        self.rank = np.arange(len(ie)) - np.repeat(self.starts, np.diff(np.r_[self.starts, len(ie)])) + 1
        pe = self.p[self.oe]
        grp = np.repeat(np.arange(len(self.starts)), np.diff(np.r_[self.starts, len(ie)]))
        self.grp = grp
        c = np.cumsum(pe)
        base = np.r_[0.0, c[self.starts[1:] - 1]]
        self.cum = c - base[grp]
        self.gsum = np.add.reduceat(pe, self.starts)
        self.lognone = np.add.reduceat(np.log1p(-pe), self.starts)
        self.pe = pe
        # order by target, probability descending -> one-owner rule
        key = src * (j.max() + 1 if len(j) else 1) + j
        self.ot = np.lexsort((-self.p, key))
        kt = key[self.ot]
        self.tgrp = np.cumsum(np.r_[True, kt[1:] != kt[:-1]]) - 1

    def expected_f(self, miss, w0, min_p):
        ef = 1.25 * self.cum / (self.rank + 0.25 * (self.gsum[self.grp] + miss))
        ef[self.pe < min_p] = -1.0
        best = np.maximum.reduceat(ef, self.starts)
        # first position of the max within each group
        is_best = ef == best[self.grp]
        first = np.full(len(self.starts), -1)
        pos = np.flatnonzero(is_best)
        g = self.grp[pos]
        u, idx = np.unique(g, return_index=True)
        first[u] = self.rank[pos[idx]]
        e0 = w0 * np.exp(self.lognone) * (1 - min(miss, 0.99))
        kstar = np.where(e0 >= best, 0, first)
        keep_sorted = self.rank <= kstar[self.grp]
        keep = np.zeros(len(self.p), dtype=bool)
        keep[self.oe] = keep_sorted
        return keep

    def single_owner(self, keep):
        k = keep[self.ot]
        pos = np.flatnonzero(k)
        _, first = np.unique(self.tgrp[pos], return_index=True)
        out = np.zeros(len(self.p), dtype=bool)
        out[self.ot[pos[first]]] = True
        return out

    def score(self, keep):
        tp = np.bincount(self.i[keep], weights=self.y[keep].astype(float), minlength=self.n_ent)
        pn = np.bincount(self.i[keep], minlength=self.n_ent).astype(float)
        den = pn + 0.25 * self.G
        f = np.where(den > 0, 1.25 * tp / np.where(den > 0, den, 1), 1.0)
        f = f[self.in_truth]
        prec = tp.sum() / max(1, pn.sum())
        rec = tp.sum() / max(1, self.G[self.in_truth].sum())
        empty = (pn[self.in_truth] == 0).mean()
        return float(f.mean()), prec, rec, empty

    def keep_for(self, prm):
        keep = self.p32 >= np.float32(prm["t"]) if prm["mode"] == "threshold" else self.expected_f(prm["miss"], prm["w0"], prm["min_p"])
        return self.single_owner(keep) if prm.get("single_owner") else keep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", default="work/dev_scores.pkl")
    ap.add_argument("--truth", default="dataset_dev/dev_ground_truth.tsv")
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--write", action="store_true", help="save the best rule into model.pkl (a backup is kept)")
    ap.add_argument("--by-country", metavar="DEV_S1_TSV",
                    help="dataset_dev/test/test_source1.tsv, to print the dev score per country")
    a = ap.parse_args()

    mpath = os.path.join(a.work_dir, "model.pkl")
    with open(mpath, "rb") as f:
        B = pickle.load(f)
    S = pd.read_pickle(a.scores)
    d, ids = S["d"].reset_index(drop=True), {1: S["ids1"], 2: S["ids2"], 3: S["ids3"]}
    truth = read_truth(a.truth)
    log(f"{len(d):,} scored pairs, {len(ids[1]):,} dev entities, "
        f"{sum(len(v) for v in truth.values()):,} true matches")
    F = Fast(d, ids, truth)
    log(f"true matches reaching the shortlist (blocking recall): {F.y.sum() / max(1, F.G.sum()):.4f}")

    cur = B["params"]
    fc, pc, rc, ec = F.score(F.keep_for(cur))
    log(f"CURRENT rule {cur}")
    log(f"   dev macro F0.5 = {fc:.4f}   precision {pc:.3f}  recall {rc:.3f}  empty {ec:.1%}  "
        f"(true empty share {np.mean([not v for v in truth.values()]):.1%})")

    grid = []
    for so in (False, True):
        for t in [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98]:
            grid.append({"mode": "threshold", "t": t, "single_owner": so})
        for miss in [0.0, 0.05, 0.15, 0.3]:
            for w0 in [0.8, 1.0, 1.2, 1.5, 2.0, 3.0]:
                for min_p in [0.05, 0.2, 0.4, 0.6]:
                    grid.append({"mode": "expected_f", "miss": miss, "w0": w0, "min_p": min_p, "single_owner": so})
    log(f"searching {len(grid)} rules")
    res = []
    for prm in grid:
        res.append((F.score(F.keep_for(prm)), prm))
    res.sort(key=lambda r: -r[0][0])
    print("\n  best rules (dev macro F0.5 | precision | recall | empty share):")
    shown = set()
    for (f, pr, rc_, em), prm in res:
        fam = (prm["mode"], prm["single_owner"])
        if fam in shown:
            continue
        shown.add(fam)
        print(f"   {f:.4f} | {pr:.3f} | {rc_:.3f} | {em:6.1%}   {prm}")
    (fb, pb, rb, eb), best = res[0]
    print(f"\n  BEST {fb:.4f} vs CURRENT {fc:.4f}  (gain {fb - fc:+.4f})\n")

    # re-check the winner with the real decide.py code (exactly what predict will run)
    keep = apply_decision(d, best)
    pred = to_prediction(d, keep, ids[1], {2: ids[2], 3: ids[3]})
    check = macro_f05(pred, truth, list(truth.keys()))
    same = abs(check - fb) < 1e-4                  # float rounding at the threshold can move the 7th decimal
    log(f"check with decide.py: {check:.4f}" + ("" if same else "  (!! differs from fast path)"))

    if a.by_country:
        c = pd.read_csv(a.by_country, sep="\t", dtype=str, keep_default_na=False, quoting=3,
                        usecols=["entity_id", "country"]).set_index("entity_id")["country"]
        for cn, grp in c.groupby(c):
            e = [x for x in grp.index if x in truth]
            log(f"   country {cn}: dev macro F0.5 = {macro_f05(pred, truth, e):.4f} ({len(e):,} entities)")

    if a.write:
        if not same or check <= fc:
            log("not writing: the best rule is not better than the current one (or the check differed)")
            return
        bak = mpath + time.strftime(".bak_%Y%m%d_%H%M%S")
        shutil.copy2(mpath, bak)
        B["params"] = best
        B.setdefault("history", []).append({"tuned_on": a.scores, "dev_f05": fb, "previous": cur})
        with open(mpath, "wb") as f:
            pickle.dump(B, f)
        log(f"saved the new rule into {mpath} (old model kept as {bak})")
        log("apply it to the test pairs:  python src/predict_lowmem.py --work-dir "
            f"{a.work_dir} --from-scores {a.work_dir}/test_scores.pkl --out-dir output")


if __name__ == "__main__":
    main()
