"""check_blocking.py - test ONLY the blocking step, in minutes, without training a model.

Runs load -> normalise -> blocking on a sample of the TRAINING data (where the
answers are known) and tells you how good the shortlist is and why matches were lost.

    python src/check_blocking.py --sample-s1 100000 --n-jobs 8
    python src/check_blocking.py --sample-s1 100000 --topk-joint 25 --max-block-target 500

Prints
  1. RECALL + COST    share of true matches that made the shortlist, pairs per entity
  2. RECALL BY SLICE  Source 2 vs 3, each country, Indian-script names vs Latin names
  3. KEY USEFULNESS   how many true matches each key type found, and how many ONLY that key found
  4. WHY MISSED       every lost true match gets one reason:
       no shared key        the two records share no tag at all      -> needs a new key type
       only oversized keys  they shared a tag, but its group was too big and was dropped
                                                                     -> raise max_block_target / max_block_pairs
       below min_cheap      they met, but the quick similarity was under min_cheap
                                                                     -> lower min_cheap
       ranked below top-K   they met, but other candidates ranked higher
                                                                     -> raise topk_joint / topk_name / topk_addr
Writes work/blocking_misses.tsv with examples of each reason.

v2: uses normalization.py (the same Normalizer as run_pipeline.py; it reuses work/norm_state.json
if you already ran `python src/run_normalization.py fit`).
When a setting works better here, copy its value into config.py so train and predict use it.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blocking import KEY_TYPES, CheapScorer, TokenStats, generate_candidates, record_keys, rowwise_cos  # noqa: E402
from config import Config  # noqa: E402
from io_utils import read_sources, read_truth  # noqa: E402
from normalization import LearnedDict  # noqa: E402
from run_pipeline import consistent_sample, get_normalizer, normalise_all  # noqa: E402

T0 = time.time()

KEY_HELP = {
    "n1": "rarest name word",
    "nn": "two rarest name words",
    "nc": "whole cleaned name",
    "nf": "typo-tolerant sound of rarest name word",
    "a2": "two rarest address words",
    "hs": "house number + rarest address word",
    "na": "rarest name word + rarest address word",
    "ia": "initials + rarest address word",
    "np": "first 5 letters of name + rarest address word",
    "nk": "order-free name key (sorted words, fillers dropped)",
    "pn": "postcode + rarest name word",
}
ADVICE = {
    "no shared key": "add a new key type in blocking.record_keys(), or fix cleaning so the words agree",
    "only oversized keys": "raise max_block_target / max_block_pairs (costs more pairs)",
    "below min_cheap": "lower min_cheap (e.g. 0.10)",
    "ranked below top-K": "raise topk_joint / topk_name / topk_addr",
}


def log(msg: str):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- diagnosis
def true_pairs(reps, truth):
    """All true matches in the sample as row positions (i in S1, src 2/3, j in that source)."""
    s1_pos = {e: i for i, e in enumerate(reps["s1"]["entity_id"].values)}
    tpos = {2: {e: j for j, e in enumerate(reps["s2"]["entity_id"].values)},
            3: {e: j for j, e in enumerate(reps["s3"]["entity_id"].values)}}
    rows, lost = [], 0
    for a, bs in truth.items():
        i = s1_pos.get(a)
        if i is None:
            continue
        for b in bs:
            if b in tpos[2]:
                rows.append((i, 2, tpos[2][b]))
            elif b in tpos[3]:
                rows.append((i, 3, tpos[3][b]))
            else:
                lost += 1
    return pd.DataFrame(rows, columns=["i", "src", "j"]).astype(np.int64), lost


def explain_misses(miss, reps, stats, scorer, cfg, found_cand):
    """Give every missed true pair a reason (see module docstring)."""
    k1 = record_keys(reps["s1"], stats, cfg.use_country_in_keys)
    k1_count = k1.groupby("key").size()
    X1n, X1a = scorer.transform(reps["s1"])
    parts = []
    for src in (2, 3):
        m = miss[miss["src"] == src].copy()
        if m.empty:
            continue
        tgt = reps[f"s{src}"]
        kt = record_keys(tgt, stats, cfg.use_country_in_keys)
        kt_count = kt.groupby("key").size()
        both = pd.concat([k1_count.rename("a"), kt_count.rename("b")], axis=1, join="inner")
        ok = set(both[(both["b"] <= cfg.max_block_target) &
                      (both["a"] * both["b"] <= cfg.max_block_pairs)].index)
        shared = (m[["i", "j"]]
                  .merge(k1.rename(columns={"pos": "i"}), on="i")
                  .merge(kt.rename(columns={"pos": "j"})[["j", "key"]], on=["j", "key"]))
        shared["ok"] = shared["key"].isin(ok)
        shared["kname"] = [KEY_TYPES[x] for x in shared["kt"].values]
        agg = shared.groupby(["i", "j"]).agg(any_ok=("ok", "any"),
                                             shared_keys=("kname", lambda s: ",".join(sorted(set(s)))))
        m = m.merge(agg.reset_index(), on=["i", "j"], how="left")
        Xtn, Xta = scorer.transform(tgt)
        ii, jj = m["i"].values, m["j"].values
        m["cos_name"] = rowwise_cos(X1n, Xtn, ii, jj)
        m["cos_addr"] = rowwise_cos(X1a, Xta, ii, jj)
        m["cheap"] = 0.55 * m["cos_name"] + 0.45 * m["cos_addr"]
        reason = np.where(m["shared_keys"].isna(), "no shared key",
                 np.where(~m["any_ok"].fillna(False).astype(bool), "only oversized keys",
                 np.where(m["cheap"] < cfg.min_cheap, "below min_cheap", "ranked below top-K")))
        m["reason"] = reason
        kept = found_cand.loc[found_cand["src"] == src, ["i", "cheap"]].rename(columns={"cheap": "kc"})
        mm = m[["i", "j", "cheap"]].merge(kept, on="i")
        nb = mm[mm["kc"] >= mm["cheap"]].groupby(["i", "j"]).size().rename("n_better")
        m = m.merge(nb.reset_index(), on=["i", "j"], how="left")
        m["n_better"] = m["n_better"].fillna(0).astype(int)
        parts.append(m)
    return pd.concat(parts, ignore_index=True) if parts else miss.assign(reason=[])


def pct(x):
    return f"{100 * x:6.2f}%"


def main():
    ap = argparse.ArgumentParser(description="Measure and diagnose the blocking step on training data.")
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--sample-s1", type=int, default=100_000, help="S1 entities to test on (0 = all)")
    ap.add_argument("--n-jobs", type=int, default=Config.n_jobs)
    ap.add_argument("--no-translit", action="store_true", help="turn OFF the Indian-script dictionary (to compare)")
    ap.add_argument("--no-country", action="store_true", help="do not put the country inside the keys")
    ap.add_argument("--n-examples", type=int, default=300, help="examples per miss reason in the TSV")
    for k in ("max_block_target", "max_block_pairs", "topk_joint", "topk_name", "topk_addr", "block_chunk"):
        ap.add_argument("--" + k.replace("_", "-"), type=int)
    ap.add_argument("--min-cheap", type=float)
    a = ap.parse_args()

    cfg = Config(data_dir=a.data_dir, work_dir=a.work_dir, train_sample_s1=(a.sample_s1 or None), n_jobs=a.n_jobs,
                 norm_state=os.path.join(a.work_dir, "norm_state.json"))
    for k in ("max_block_target", "max_block_pairs", "topk_joint", "topk_name", "topk_addr", "block_chunk", "min_cheap"):
        v = getattr(a, k)
        if v is not None:
            setattr(cfg, k, v)
    if a.no_country:
        cfg.use_country_in_keys = False
    os.makedirs(a.work_dir, exist_ok=True)
    knobs = {k: getattr(cfg, k) for k in ("max_block_target", "max_block_pairs", "topk_joint", "topk_name",
                                          "topk_addr", "min_cheap", "use_country_in_keys")}
    log(f"settings: {knobs}")

    # ---- load + normalise (same code as run_pipeline.train) ----
    tdir = os.path.join(a.data_dir, "train")
    truth_all = read_truth(os.path.join(tdir, "train_ground_truth.tsv"))
    src = read_sources(tdir, "train")
    log(f"loaded: S1 {len(src['s1']):,}  S2 {len(src['s2']):,}  S3 {len(src['s3']):,}")
    norm = get_normalizer(src, truth_all, cfg, refit=False)
    if a.no_translit:
        norm.translit = LearnedDict()
        norm._reset_caches()
    src, truth = consistent_sample(src, truth_all, cfg.train_sample_s1, cfg.seed)
    del truth_all
    log(f"sample: S1 {len(src['s1']):,}  S2 {len(src['s2']):,}  S3 {len(src['s3']):,}")
    reps = normalise_all(src, norm, cfg, f"train{cfg.train_sample_s1 or 'all'}s{cfg.seed}")

    # ---- blocking (exactly what the pipeline does) ----
    t = time.time()
    stats = TokenStats(list(reps.values()))
    scorer = CheapScorer(seed=cfg.seed).fit(list(reps.values()))
    cand = generate_candidates(reps["s1"], {"s2": reps["s2"], "s3": reps["s3"]}, stats, scorer, cfg, log)
    cand = cand.reset_index(drop=True)
    for c in ("src", "i", "j"):
        cand[c] = cand[c].astype(np.int64)
    t_block = time.time() - t

    tp, lost = true_pairs(reps, truth)
    if lost:
        log(f"!! {lost:,} ground-truth ids are not in the source files - run fix_source_files.py first")
    tp = tp.merge(cand[["i", "src", "j", "key_mask"]], on=["i", "src", "j"], how="left")
    tp["found"] = tp["key_mask"].notna()
    n1 = len(reps["s1"])
    recall = tp["found"].mean() if len(tp) else float("nan")

    print("\n=== 1. RECALL + COST ===")
    print(f"  true matches in sample   {len(tp):,}")
    print(f"  found by blocking        {int(tp['found'].sum()):,}")
    print(f"  BLOCKING RECALL          {recall:.4f}   (target: above 0.95)")
    print(f"  candidate pairs          {len(cand):,}  = {len(cand) / max(1, n1):.1f} per S1 entity")
    print(f"  S1 with no candidates    {pct(1 - cand['i'].nunique() / max(1, n1))}")
    print(f"  blocking time            {t_block:.0f}s for {n1:,} S1 entities")
    ent = tp.groupby("i")["found"].all()
    print(f"  entities with all matches found  {pct(ent.mean() if len(ent) else float('nan'))}")

    print("\n=== 2. RECALL BY SLICE ===")
    tp["source"] = np.where(tp["src"] == 2, "S2", "S3")
    tp["country"] = reps["s1"]["country_norm"].values[tp["i"].values]
    ind = np.zeros(len(tp), dtype=np.int8)
    for s in (2, 3):
        m = tp["src"].values == s
        ind[m] = reps[f"s{s}"]["name_was_indic"].values[tp["j"].values[m]]
    tp["name_script"] = np.where(ind == 1, "indian-script", "latin")
    for col in ("source", "country", "name_script"):
        g = tp.groupby(col)["found"].agg(["mean", "size"])
        for k, r in g.iterrows():
            print(f"  {col:<12} {str(k):<16} recall {r['mean']:.4f}   ({int(r['size']):,} true matches)")

    print("\n=== 3. KEY USEFULNESS (on true matches that were found) ===")
    fm = tp.loc[tp["found"], "key_mask"].astype(np.int64).values
    print(f"  {'key':<4} {'found':>8} {'only this':>10}  what it is")
    for b, k in enumerate(KEY_TYPES):
        has = int(((fm >> b) & 1).sum())
        only = int((fm == (1 << b)).sum())
        print(f"  {k:<4} {pct(has / max(1, len(fm)))} {only:>10,}  {KEY_HELP[k]}")
    print("  ('only this' = matches no other key would have found. Near 0 means the key adds cost, not recall.)")

    print("\n=== 4. WHY TRUE MATCHES WERE MISSED ===")
    miss = tp.loc[~tp["found"], ["i", "src", "j"]]
    if miss.empty:
        print("  nothing missed")
        return
    ex = explain_misses(miss, reps, stats, scorer, cfg, cand)
    cnt = ex["reason"].value_counts()
    for r, c in cnt.items():
        print(f"  {r:<20} {c:>8,}  = {pct(c / len(tp))} of all true matches  -> {ADVICE[r]}")
    rk = ex[ex["reason"] == "ranked below top-K"]
    if len(rk):
        q = np.percentile(rk["n_better"], [50, 90])
        print(f"  (ranked below top-K: median {q[0]:.0f}, 90th pct {q[1]:.0f} kept candidates scored higher)")
    oc = ex[ex["reason"] == "only oversized keys"]
    if len(oc):
        print(f"  (oversized: most common shared key types {oc['shared_keys'].value_counts().head(3).to_dict()})")

    rows = []
    for r in cnt.index:
        rows.append(ex[ex["reason"] == r].sample(min(a.n_examples, int(cnt[r])), random_state=cfg.seed))
    ex = pd.concat(rows, ignore_index=True)
    s1 = reps["s1"]
    out = pd.DataFrame({
        "reason": ex["reason"].values,
        "s1_id": s1["entity_id"].values[ex["i"].values],
        "s1_name": s1["business_name"].values[ex["i"].values],
        "s1_address": s1["business_address"].values[ex["i"].values],
        "s1_name_core": s1["name_core"].values[ex["i"].values],
        "cos_name": ex["cos_name"].round(3).values,
        "cos_addr": ex["cos_addr"].round(3).values,
        "shared_keys": ex["shared_keys"].fillna("").values,
        "n_better": ex["n_better"].values,
    })
    for col in ("id", "name", "address", "name_core"):
        vals = []
        for s, j in zip(ex["src"].values, ex["j"].values):
            t_ = reps[f"s{s}"]
            vals.append(t_[{"id": "entity_id", "name": "business_name", "address": "business_address",
                            "name_core": "name_core"}[col]].values[j])
        out["tgt_" + col] = vals
    path = os.path.join(a.work_dir, "blocking_misses.tsv")
    out.to_csv(path, sep="\t", index=False)
    print(f"\nExamples of each miss reason written to {path}")
    print("Open it and read 20 rows of the biggest reason: the pattern you see is what to fix next.")


if __name__ == "__main__":
    main()
