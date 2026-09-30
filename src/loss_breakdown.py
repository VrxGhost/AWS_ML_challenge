"""loss_breakdown.py - WHERE is the F0.5 being lost?  Run on the DEV set (it has answers).

    python src/loss_breakdown.py --work-dir work_dense --scores work_dense/dev_scores.pkl \
        --truth dataset_dev_dense/dev_ground_truth.tsv --dev-dir dataset_dev_dense --examples 300

Every dev Source 1 entity that does not score 1.0 is put in ONE bucket:
  A  true singleton, but we predicted a match            -> false merge, costs the full 1.0
  B  has matches, none reached the shortlist             -> blocking problem
  C  has matches in the shortlist, we predicted nothing  -> decision too strict
  D  we predicted at least one WRONG record              -> false merge (model / rule)
  E  all our picks right, but we missed some             -> recall (blocking or rule)
The "points lost" column adds up to (1 - dev F0.5), so the biggest bucket is where to work.
--examples N writes N sample errors with names/addresses to <work-dir>/errors_sample.tsv.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tune_on_dev import Fast, read_truth  # noqa: E402

BUCKETS = {
    "A": "singleton, we predicted a match (false merge)",
    "B": "has matches, none in shortlist (blocking)",
    "C": "matches in shortlist, we predicted none (too strict)",
    "D": "predicted a wrong record (false merge)",
    "E": "picks right, missed some (recall)",
}


def load_src(path, wanted):
    """Only the rows whose id is in `wanted` (the dev S2/S3 files have ~5M rows each: loading them
    whole would need several GB of RAM)."""
    out = {}
    for ch in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3, chunksize=500_000):
        ch = ch[ch["entity_id"].isin(wanted)]
        out.update(zip(ch["entity_id"], zip(ch["business_name"], ch["business_address"], ch["country"])))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", default="work_dense")
    ap.add_argument("--scores", default="work_dense/dev_scores.pkl")
    ap.add_argument("--truth", default="dataset_dev_dense/dev_ground_truth.tsv")
    ap.add_argument("--dev-dir", default="dataset_dev_dense", help="folder with test/test_source1..3.tsv")
    ap.add_argument("--examples", type=int, default=0)
    a = ap.parse_args()

    with open(os.path.join(a.work_dir, "model.pkl"), "rb") as f:
        rule = pickle.load(f)["params"]
    S = pd.read_pickle(a.scores)
    d = S["d"].reset_index(drop=True)
    ids = {k: np.asarray(S[f"ids{k}"], dtype=object) for k in (1, 2, 3)}
    truth = read_truth(a.truth)
    F = Fast(d, ids, truth)
    keep = F.keep_for(rule)
    print(f"rule in model.pkl: {rule}")

    n = F.n_ent
    tp = np.bincount(F.i[keep], weights=F.y[keep].astype(float), minlength=n)
    pn = np.bincount(F.i[keep], minlength=n).astype(float)
    reach = np.bincount(F.i, weights=F.y.astype(float), minlength=n)   # true matches in shortlist
    G = F.G
    den = pn + 0.25 * G
    f = np.where(den > 0, 1.25 * tp / np.where(den > 0, den, 1), 1.0)

    cat = np.full(n, "", dtype=object)
    cat[(G > 0) & (pn > 0) & (pn == tp) & (tp < G)] = "E"
    cat[(G > 0) & (pn > tp)] = "D"
    cat[(G > 0) & (pn == 0) & (reach > 0)] = "C"
    cat[(G > 0) & (pn == 0) & (reach == 0)] = "B"
    cat[(G == 0) & (pn > 0)] = "A"

    m = F.in_truth
    N = int(m.sum())
    cn = np.full(n, "all", dtype=object)
    if os.path.exists(os.path.join(a.dev_dir, "test", "test_source1.tsv")):
        c = pd.read_csv(os.path.join(a.dev_dir, "test", "test_source1.tsv"), sep="\t", dtype=str,
                        keep_default_na=False, quoting=3, usecols=["entity_id", "country"])
        cmap = dict(zip(c["entity_id"], c["country"].str.strip()))
        cn = np.array([cmap.get(e, "?") for e in ids[1]], dtype=object)

    for country in ["ALL"] + sorted(set(cn[m])):
        sel = m if country == "ALL" else m & (cn == country)
        Nc = int(sel.sum())
        print(f"\n=== {country}: {Nc:,} entities, dev F0.5 = {f[sel].mean():.4f}, "
              f"singletons {np.mean(G[sel] == 0):.1%}, blocking recall "
              f"{reach[sel].sum() / max(1, G[sel].sum()):.4f}")
        print(f"  {'bucket':<55} {'entities':>9} {'points lost':>12}")
        for b, txt in BUCKETS.items():
            s = sel & (cat == b)
            print(f"  {b} {txt:<53} {int(s.sum()):>9,} {(1 - f[s]).sum() / max(1, Nc):>12.4f}")
    lost_block = (G[m] - reach[m]).sum()
    print(f"\ntrue matches never in the shortlist: {int(lost_block):,} of {int(G[m].sum()):,}")

    if a.examples:
        tgt = np.empty(len(d), dtype=object)
        src = d["src"].to_numpy()
        j = d["j"].to_numpy()
        for s in (2, 3):
            tgt[src == s] = ids[s][j[src == s]]
        rng = np.random.RandomState(0)
        bad = np.flatnonzero(m & (cat != ""))
        pick = set(rng.choice(bad, min(a.examples, len(bad)), replace=False))
        rows = []
        rowsel = np.flatnonzero(np.isin(F.i, list(pick)))
        wanted = {ids[1][e] for e in pick} | set(tgt[rowsel]) | {t for e in pick for t in truth.get(ids[1][e], ())}
        srcs = {k: load_src(os.path.join(a.dev_dir, "test", f"test_source{k}.tsv"), wanted) for k in (1, 2, 3)}
        pmap = {(F.i[r], tgt[r]): (float(F.p[r]), bool(keep[r])) for r in rowsel}
        for e in sorted(pick):
            eid = ids[1][e]
            n1, a1, c1 = srcs[1].get(eid, ("", "", ""))
            predicted = {t for (ie, t), (_, k) in pmap.items() if ie == e and k}
            for t in predicted | truth.get(eid, set()):
                kind = ("OK" if t in truth.get(eid, set()) else "WRONG_PICK") if t in predicted else "MISSED"
                if kind == "OK":
                    continue
                n2, a2, _ = srcs[2 if t.startswith("S2-") else 3].get(t, ("", "", ""))
                p = pmap.get((e, t), (None, False))[0]
                rows.append([cat[e], c1, eid, n1, a1, kind, t, n2, a2, "" if p is None else f"{p:.3f}"])
        out = os.path.join(a.work_dir, "errors_sample.tsv")
        pd.DataFrame(rows, columns=["bucket", "country", "s1_id", "s1_name", "s1_address", "error",
                                    "other_id", "other_name", "other_address", "p"]).to_csv(out, sep="\t", index=False)
        print(f"wrote {len(rows)} error rows -> {out}")


if __name__ == "__main__":
    main()