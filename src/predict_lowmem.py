"""predict_lowmem.py - the same prediction as `run_pipeline.py predict`, in bounded memory.

    python src/predict_lowmem.py --data-dir dataset --work-dir work --out-dir output --n-jobs 4

Use it when the normal predict runs out of RAM.  It needs the model from a normal
`run_pipeline.py train` (work/model.pkl) and gives the same result; see "Why the result is the same".

How it keeps memory small
  PASS 1  Each test file is read in slices (--chunk rows, default 200k).  Each slice is cleaned
          and written to disk (work/lowmem/), split by country, then dropped from RAM.  While
          slices go by, the pass keeps only small global counts: word counts (for rarity),
          Source 1 name/address frequencies, and a sample of names/addresses (for the TF-IDF
          scorer).  Normaliser caches are kept small (--cache low).
  PASS 2  One country at a time (blocking keys contain the country, so pairs never cross
          countries: nothing is lost).
          2a  Blocking keys for Source 2/3 of that country, built slice by slice.  Only the keys
              and two short strings per record stay in memory, never the full cleaned rows.
          2b  Source 1 of that country in blocks of --block-chunk rows: join keys, cosine
              scores (TF-IDF computed only for the rows in the block), keep the top-K.
          2c  Features, one Source 2/3 slice at a time (only the full rows of that slice are
              loaded) -> stage-1 score; the features are written to disk.
          2d  Context features for the whole country (a small table: pair ids + scores),
              then stage-2 score slice by slice, then calibration.
  PASS 3  Decision rule on all pairs together (the "one owner" rule needs all of them),
          write output/*.tsv, self-validation.

Why the result is the same
  Every number the normal predict computes is computed here from the same inputs: word
  rarity and Source 1 frequencies are counted over ALL records, key block sizes are per key
  (keys contain the country), top-K is per Source 1 record, context features and the
  decision rule see all pairs of a country.  The only difference: when the test set has more
  than 2M records, the TF-IDF scorer learns its letter-chunk list from a different random
  2M-record sample.  This changes the shortlist very slightly.

Disk: about (candidate pairs x 400 bytes) of temporary files in work/lowmem/, deleted at the end.
"""
from __future__ import annotations

import argparse
import gc
import os
import pickle
import re
import shutil
import sys
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import normalization as N  # noqa: E402
from blocking import KT, CheapScorer, TokenStats, block_limits, record_keys, rowwise_cos  # noqa: E402
from config import Config  # noqa: E402
from decide import apply_decision, to_prediction  # noqa: E402
from features import build_features, context_features  # noqa: E402
from io_utils import self_validate, write_id_lists  # noqa: E402

T0 = time.time()
KEY_COLS = ["name_tokens", "addr_alpha", "addr_house", "name_compact", "name_initials",
            "country_norm", "name_key", "addr_postal", "name_core", "addr_norm"]


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def _mem():
    try:
        import psutil
        return f"{psutil.Process().memory_info().rss / 2**30:.1f} GB"
    except Exception:
        return "?"


def _slug(c: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", c) or "_none"


def _count_rows(path: str) -> int:
    with open(path, "rb") as f:
        return max(0, sum(buf.count(b"\n") for buf in iter(lambda: f.read(1 << 24), b"")) - 1)


def _chunks(rows, size):
    block = []
    for r in rows:
        block.append(r)
        if len(block) >= size:
            yield block
            block = []
    if block:
        yield block


def _clean_slice(block):
    """Worker: clean one slice of raw rows -> column dict (normalization._WORKER)."""
    return N.build_repr(block, N._WORKER)


# ----------------------------------------------------------------------------- PASS 1
def pass1(data_dir, work, norm_state, args, split="test"):
    """Clean every test file slice by slice; write slices to disk split by country."""
    tdir = os.path.join(data_dir, split)
    paths = {k: os.path.join(tdir, f"{split}_source{k}.tsv") for k in (1, 2, 3)}
    total = sum(_count_rows(p) for p in paths.values())
    p_sample = min(1.0, 2_000_000 / max(1, total))            # CheapScorer fits on <= 2M records
    rng = np.random.RandomState(args.seed)
    stats = TokenStats([])
    stats.n = 0
    freq_name, freq_addr = Counter(), Counter()
    sample_n, sample_a = [], []
    ids, parts = {}, {}                                         # parts[k][country] = [(file, n)]
    n_s1 = 0
    with Pool(args.n_jobs, initializer=N._init_worker, initargs=(norm_state, args.cache)) as pool:
        for k, path in paths.items():
            ids_k, parts_k, off = [], {}, 0
            log(f"PASS 1 source {k}: {os.path.basename(path)}")
            for pi, cols in enumerate(pool.imap(_clean_slice, _chunks(N.read_tsv(path), args.chunk))):
                df = N._to_frame(cols)
                df["gidx"] = np.arange(off, off + len(df), dtype=np.int64)
                off += len(df)
                ids_k.append(df["entity_id"].to_numpy(dtype=object))
                df = df.drop(columns=["business_name", "business_address"])
                stats.n += len(df)
                for t in df["name_tokens"]:
                    stats.name_df.update(set(t))
                for t in df["addr_tokens"]:
                    stats.addr_df.update(set(t))
                if k == 1:
                    freq_name.update(df["name_core"])
                    freq_addr.update(df["addr_norm"])
                    n_s1 += len(df)
                pick = rng.rand(len(df)) < p_sample
                sample_n += df["name_core"].values[pick].tolist()
                sample_a += df["addr_norm"].values[pick].tolist()
                for c, g in df.groupby("country_norm", sort=False):
                    fp = os.path.join(work, f"s{k}", _slug(c), f"part{pi:04d}.pkl")
                    os.makedirs(os.path.dirname(fp), exist_ok=True)
                    g.reset_index(drop=True).to_pickle(fp)
                    parts_k.setdefault(c, []).append((fp, len(g)))
                del df, cols
                print(f"\r   {off:,} rows cleaned, RAM {_mem()}", end="", flush=True)
            print()
            ids[k] = np.concatenate(ids_k) if ids_k else np.array([], dtype=object)
            parts[k] = parts_k
    freq = {"name": pd.Series(freq_name), "addr": pd.Series(freq_addr), "n": n_s1}
    scorer = CheapScorer(seed=args.seed).fit([pd.DataFrame({"name_core": sample_n, "addr_norm": sample_a})])
    del sample_n, sample_a
    for k in (1, 2, 3):
        log(f"  source {k}: {len(ids[k]):,} rows, countries "
            f"{ {c: sum(n for _, n in v) for c, v in parts[k].items()} }")
    return stats, freq, scorer, ids, parts


def _load(files, cols=None):
    dfs = []
    for fp, _ in files:
        d = pd.read_pickle(fp)
        dfs.append(d[cols] if cols else d)
    return pd.concat(dfs, ignore_index=True) if dfs else None


# ----------------------------------------------------------------------------- PASS 2
def blocking_country(s1c, tparts, stats, scorer, cfg):
    """Same as blocking.generate_candidates for one country; i = row in s1c, j = global row."""
    k1 = record_keys(s1c, stats, cfg.use_country_in_keys)
    k1_count = k1.groupby("key").size()
    out = []
    for src, files in tparts.items():
        if not files:
            continue
        kts, strs, off = [], [], 0                     # 2a: keys + strings only, slice by slice
        for fp, _ in files:
            d = pd.read_pickle(fp)[KEY_COLS + ["gidx"]]
            kt = record_keys(d, stats, cfg.use_country_in_keys)
            kt["pos"] += off
            kts.append(kt)
            strs.append(d[["gidx", "name_core", "addr_norm"]])
            off += len(d)
            del d
        kt = pd.concat(kts, ignore_index=True)
        T = pd.concat(strs, ignore_index=True)
        del kts, strs
        kt_count = kt.groupby("key").size()
        both = pd.concat([k1_count.rename("a"), kt_count.rename("b")], axis=1, join="inner")
        lim_t, lim_p = block_limits(cfg, stats.n)
        ok = both[(both["b"] <= lim_t) & (both["a"] * both["b"] <= lim_p)].index
        kt_ok = kt[kt["key"].isin(ok)][["key", "pos"]].rename(columns={"pos": "j"})
        k1_ok = k1[k1["key"].isin(ok)][["key", "pos", "kt"]].rename(columns={"pos": "i"})
        del kt, both
        log(f"    [s{src}] {len(T):,} records, usable keys {len(ok):,}, RAM {_mem()}")
        for s in range(0, len(s1c), cfg.block_chunk):   # 2b
            part = k1_ok[(k1_ok["i"] >= s) & (k1_ok["i"] < s + cfg.block_chunk)]
            m = part.merge(kt_ok, on="key", how="inner")
            if m.empty:
                continue
            m = m[["i", "j", "kt"]].drop_duplicates()
            m["bit"] = np.left_shift(np.int64(1), m["kt"].values.astype(np.int64))
            g = m.groupby(["i", "j"], sort=False)["bit"].agg(["sum", "size"]).reset_index()
            g.columns = ["i", "j", "key_mask", "n_keys"]
            del m
            ii, jj = g["i"].values, g["j"].values
            ui, ri = np.unique(ii, return_inverse=True)   # TF-IDF only for rows in this block
            uj, rj = np.unique(jj, return_inverse=True)
            A = s1c.iloc[ui]
            Bn = scorer.name_vec.transform(T["name_core"].values[uj]).tocsr()
            Ba = scorer.addr_vec.transform(T["addr_norm"].values[uj]).tocsr()
            An = scorer.name_vec.transform(A["name_core"].values).tocsr()
            Aa = scorer.addr_vec.transform(A["addr_norm"].values).tocsr()
            g["cos_name"] = rowwise_cos(An, Bn, ri, rj)
            g["cos_addr"] = rowwise_cos(Aa, Ba, ri, rj)
            g["cheap"] = 0.55 * g["cos_name"] + 0.45 * g["cos_addr"]
            grp = g.groupby("i", sort=False)
            r_joint = grp["cheap"].rank(ascending=False, method="first")
            r_name = grp["cos_name"].rank(ascending=False, method="first")
            r_addr = grp["cos_addr"].rank(ascending=False, method="first")
            keep = (r_joint <= cfg.topk_joint) | (r_name <= cfg.topk_name) | (r_addr <= cfg.topk_addr)
            keep &= g["cheap"] >= cfg.min_cheap
            g = g[keep].copy()
            g["j"] = T["gidx"].values[g["j"].values]      # position -> global row of the source
            g["src"] = np.int8(src)
            out.append(g)
        del kt_ok, T
        gc.collect()
    if not out:
        return None
    cand = pd.concat(out, ignore_index=True)
    cand["rank_cheap"] = cand.groupby(["i", "src"])["cheap"].rank(ascending=False, method="first")
    return cand


def score_country(cand, s1c, tparts, stats, freq, B, work, cfg):
    """2c + 2d: features slice by slice, stage 1, context, stage 2, calibration."""
    m1, m2 = B["m1"], B["m2"]
    groups, pieces, p1s = [], [], []
    fdir = os.path.join(work, "features")
    os.makedirs(fdir, exist_ok=True)
    for src, files in tparts.items():
        for fp, _ in files:
            d = pd.read_pickle(fp)
            sel = np.where((cand["src"].values == src) & np.isin(cand["j"].values, d["gidx"].values))[0]
            if not len(sel):
                continue
            cg = cand.iloc[sel].reset_index(drop=True)
            local = pd.Series(np.arange(len(d)), index=d["gidx"].values)
            cg_local = cg.assign(j=local.reindex(cg["j"].values).values)
            F = build_features(cg_local, s1c, {src: d}, stats, freq, cfg.n_jobs, log=lambda *_: None)
            p1 = m1.predict(F)
            if m2 is not None:
                path = os.path.join(fdir, f"F{len(groups):05d}.pkl")
                F.to_pickle(path)
            else:
                path = None
            groups.append(path)
            pieces.append(cg)
            p1s.append(p1)
            del d, F, cg_local
            print(f"\r    features: {sum(len(x) for x in pieces):,} / {len(cand):,} pairs, RAM {_mem()}",
                  end="", flush=True)
    print()
    cc = pd.concat(pieces, ignore_index=True)
    p1 = np.concatenate(p1s)
    if m2 is None:
        p = p1
    else:
        C = context_features(cc, p1)
        p, start = np.empty(len(cc), dtype=np.float32), 0
        for path, piece in zip(groups, pieces):
            F = pd.read_pickle(path)
            n = len(piece)
            Ci = C.iloc[start:start + n].reset_index(drop=True)
            p[start:start + n] = m2.predict(pd.concat([F, Ci], axis=1))
            os.remove(path)
            start += n
        del C
    pc = B["iso"].predict(p).astype(np.float32)
    return cc[["i", "j", "src"]].assign(p=pc)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=200_000, help="rows per cleaning slice (smaller = less RAM)")
    ap.add_argument("--block-chunk", type=int, default=100_000, help="Source 1 rows per blocking block")
    ap.add_argument("--cache", default="low", choices=["low", "full"], help="normaliser cache size per worker")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-temp", action="store_true", help="keep work/lowmem/ afterwards")
    ap.add_argument("--save-scores", metavar="PATH",
                    help="also save every candidate pair with its calibrated probability (for tune_on_dev.py, "
                         "or to re-apply a new decision rule later with --from-scores)")
    ap.add_argument("--from-scores", metavar="PATH",
                    help="skip cleaning/blocking/scoring: load pairs saved with --save-scores and only apply "
                         "the decision rule currently in model.pkl (takes minutes, not hours)")
    a = ap.parse_args()

    with open(os.path.join(a.work_dir, "model.pkl"), "rb") as f:
        B = pickle.load(f)
    if a.from_scores:
        log(f"loading saved pair scores from {a.from_scores}")
        S = pd.read_pickle(a.from_scores)
        finish(S["d"], {1: S["ids1"], 2: S["ids2"], 3: S["ids3"]}, B, a)
        return
    cfg = Config(**{k: v for k, v in B["cfg"].items() if k in Config.__dataclass_fields__})
    cfg.n_jobs, cfg.block_chunk = a.n_jobs, a.block_chunk
    work = os.path.join(a.work_dir, "lowmem")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    log(f"low-memory predict: slices of {a.chunk:,} rows, {a.n_jobs} workers, cache={a.cache}")
    if not cfg.use_country_in_keys:
        log("!! the model was trained with use_country_in_keys=False: all countries form ONE group "
            "(correct, but the memory saving from splitting by country is lost)")

    stats, freq, scorer, ids, parts = pass1(a.data_dir, work, B["norm_state"], a)
    gc.collect()
    log(f"PASS 1 done, RAM {_mem()}")

    countries = sorted(parts[1]) if cfg.use_country_in_keys else [None]
    decided = []
    for c in countries:
        if c is None:                                 # one group: merge all country files
            s1_files = [x for v in parts[1].values() for x in v]
            tparts = {k: [x for v in parts[k].values() for x in v] for k in (2, 3)}
        else:
            s1_files = parts[1][c]
            tparts = {k: parts[k].get(c, []) for k in (2, 3)}
        s1c = _load(s1_files)
        log(f"PASS 2 country={c}: S1 {len(s1c):,}, S2 {sum(n for _, n in tparts[2]):,}, "
            f"S3 {sum(n for _, n in tparts[3]):,}")
        cand = blocking_country(s1c, tparts, stats, scorer, cfg)
        if cand is None:
            log("    no candidates")
            continue
        log(f"    candidates {len(cand):,} ({len(cand) / len(s1c):.1f} per S1 record), RAM {_mem()}")
        d = score_country(cand, s1c, tparts, stats, freq, B, work, cfg)
        d["i"] = s1c["gidx"].values[d["i"].values]     # row in s1c -> global Source 1 row
        decided.append(d)
        del s1c, cand
        gc.collect()

    log("PASS 3 decision + output")
    d = pd.concat(decided, ignore_index=True) if decided else pd.DataFrame(columns=["i", "j", "src", "p"])
    if a.save_scores:
        os.makedirs(os.path.dirname(os.path.abspath(a.save_scores)), exist_ok=True)
        pd.to_pickle({"d": d, "ids1": ids[1], "ids2": ids[2], "ids3": ids[3]}, a.save_scores)
        log(f"saved {len(d):,} scored pairs to {a.save_scores}")
    finish(d, ids, B, a)
    if not a.keep_temp:
        shutil.rmtree(work, ignore_errors=True)


def finish(d, ids, B, a):
    """Decision rule from model.pkl -> output files -> self-validation."""
    log(f"decision rule: {B['params']}")
    keep = apply_decision(d, B["params"])
    s1_ids, tgt_ids = ids[1], {2: ids[2], 3: ids[3]}
    pred = to_prediction(d, keep, s1_ids, tgt_ids)
    allc = to_prediction(d, pd.Series(True, index=d.index), s1_ids, tgt_ids)
    os.makedirs(a.out_dir, exist_ok=True)
    mp_ = os.path.join(a.out_dir, "matching_results.tsv")
    cp = os.path.join(a.out_dir, "candidate_pairs.tsv")
    write_id_lists(mp_, "matched_entity_ids", s1_ids, pred)
    write_id_lists(cp, "candidate_entity_ids", s1_ids, allc)
    n_empty = sum(1 for e in s1_ids if not pred.get(e))
    log(f"wrote {mp_} ({len(s1_ids):,} rows, {n_empty / max(1, len(s1_ids)):.1%} empty) and {cp}")
    issues = self_validate(mp_, cp, s1_ids, tgt_ids[2], tgt_ids[3])
    log("self-validation: " + ("PASS" if not issues else "ISSUES\n  " + "\n  ".join(issues[:20])))


if __name__ == "__main__":
    main()
