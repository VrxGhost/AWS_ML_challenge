"""
run_normalization.py - fit the normaliser once, and LOOK at what it does.

    python src/run_normalization.py fit     --data-dir dataset            # -> work/norm_state.json
    python src/run_normalization.py fit     --data-dir dataset --report   # + held-out quality table
    python src/run_normalization.py pairs   --data-dir dataset --n 2000   # true pairs side by side
    python src/run_normalization.py preview --data-dir dataset --n 500    # raw vs cleaned, all 6 files

fit      Learns the Indian-script dictionary, abbreviation maps and protected look-alike words
         from ALL training true matches and saves work/norm_state.json.  run_pipeline.py train
         then reuses this file instead of fitting again.
pairs    Takes TRUE matches from train, cleans both sides and writes work/norm_pairs.tsv with
         core_eq / key_eq / state_eq / postal_eq / house_eq flags, disagreements first.
         Read the top 30 rows: every row is a true match the cleaning still fails to align,
         and the pattern you see is the next rule to add.  (The single most useful check.)
preview  First N rows of each source file: raw name/address next to the cleaned fields.

Replaces the old version, which wrote reps_*.tsv files that no other step read, joined
address components with spaces (so "12 mg road" + "bangalore" could not be split again)
and fitted on only 1/8 of the data.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import normalization as N  # noqa: E402

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def load_norm(state_path):
    if not os.path.exists(state_path):
        sys.exit(f"Missing {state_path}. Run first:  python src/run_normalization.py fit --data-dir dataset")
    return N.Normalizer.load(state_path)


def cmd_fit(a):
    if a.report:
        N.heldout_report(a.data_dir, a.report_mod, log=log)
    norm = N.fit_from_files(a.data_dir, a.sample_mod, log=log)
    os.makedirs(os.path.dirname(os.path.abspath(a.state)), exist_ok=True)
    norm.save(a.state)
    log(f"saved {a.state}: {len(norm.translit.map):,} Indic words, {len(norm.name_map)} name / "
        f"{len(norm.addr_map)} address abbreviations, {len(norm.protect)} protected words")
    log("examples of learned name abbreviations: " + str(dict(itertools.islice(norm.name_map.items(), 15))))
    log("examples of learned address abbreviations: " + str(dict(itertools.islice(norm.addr_map.items(), 15))))


PAIR_COLS = ["name_core", "name_key", "name_legal", "addr_comps", "addr_house", "addr_postal", "addr_state"]


def cmd_pairs(a):
    norm = load_norm(a.state)
    d = os.path.join(a.data_dir, "train")
    truth = N.read_truth(os.path.join(d, "train_ground_truth.tsv"))
    owner = {t: s for s, ts in truth.items() for t in ts}
    tg = {}
    for k in ("2", "3"):
        for r in N.read_tsv(os.path.join(d, f"train_source{k}.tsv")):
            if r[0] in owner:
                tg[r[0]] = r
                if len(tg) >= a.n * 2:
                    break
    need = {owner[t] for t in tg}
    s1 = {r[0]: r for r in N.read_tsv(os.path.join(d, "train_source1.tsv")) if r[0] in need}
    rows, agg = [], {c: 0 for c in ("core", "key", "state", "postal", "house")}
    both = {c: 0 for c in agg}
    for tid, b in itertools.islice(tg.items(), a.n):
        if owner[tid] not in s1:
            continue
        A = s1[owner[tid]]
        x, y = norm.record(A[1], A[2], A[3]), norm.record(b[1], b[2], b[3])
        fl = {"core": x["name_core"] == y["name_core"], "key": x["name_key"] == y["name_key"]}
        for f, c in (("addr_state", "state"), ("addr_postal", "postal"), ("addr_house", "house")):
            if x[f] and y[f]:
                fl[c] = x[f] == y[f]
        for c, v in fl.items():
            both[c] += 1
            agg[c] += v
        bad = sum(1 for v in fl.values() if not v)
        rows.append([bad, A[0], tid] + [int(fl.get(c, -1)) if c in fl else "" for c in agg]
                    + [A[1], b[1], A[2], b[2]]
                    + [N._fmt(x[c]) + "  <>  " + N._fmt(y[c]) for c in PAIR_COLS])
    rows.sort(key=lambda r: -r[0])
    out = os.path.join(a.work_dir, "norm_pairs.tsv")
    os.makedirs(a.work_dir, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\", lineterminator="\n")
        w.writerow(["n_disagree", "s1_id", "tgt_id"] + [c + "_eq" for c in agg]
                   + ["s1_name", "tgt_name", "s1_addr", "tgt_addr"] + [c + " (s1 <> tgt)" for c in PAIR_COLS])
        for r in rows:
            w.writerow([N._fmt(v) if not isinstance(v, int) else v for v in r])
    log(f"{len(rows):,} true pairs checked; share where the cleaned fields AGREE "
        f"(only counted when both sides have the field):")
    for c in agg:
        log(f"   {c + '_eq':10} {agg[c] / max(1, both[c]):.3f}   ({both[c]:,} pairs have it)")
    log(f"wrote {out}  <- open it, read the first 30 rows (worst first)")


PREVIEW_COLS = ["name_core", "name_key", "name_legal", "name_had_alias", "addr_comps", "addr_house",
                "addr_house_sfx", "addr_street", "addr_postal", "addr_state", "country_norm"]


def cmd_preview(a):
    norm = load_norm(a.state)
    os.makedirs(a.work_dir, exist_ok=True)
    for split in a.splits:
        for k in (1, 2, 3):
            inp = os.path.join(a.data_dir, split, f"{split}_source{k}.tsv")
            if not os.path.exists(inp):
                log(f"skip (not found): {inp}")
                continue
            out = os.path.join(a.work_dir, f"norm_preview_{split}_source{k}.tsv")
            with open(out, "w", encoding="utf-8", newline="") as f:
                f.write("\t".join(["entity_id", "raw_name", "raw_address", "country"] + PREVIEW_COLS) + "\n")
                for eid, name, addr, country in itertools.islice(N.read_tsv(inp), a.n):
                    r = norm.record(name, addr, country)
                    f.write("\t".join(N._fmt(v) for v in [eid, name, addr, country] + [r[c] for c in PREVIEW_COLS]) + "\n")
            log(f"wrote {out}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["fit", "pairs", "preview"])
    p.add_argument("--data-dir", default="dataset")
    p.add_argument("--work-dir", default="work")
    p.add_argument("--state", default=None, help="default: <work-dir>/norm_state.json")
    p.add_argument("--n", type=int, default=2000, help="pairs / preview rows")
    p.add_argument("--splits", nargs="+", default=["train", "test"])
    p.add_argument("--report", action="store_true", help="fit: also print the held-out quality table")
    p.add_argument("--report-mod", type=int, default=8)
    p.add_argument("--sample-mod", type=int, default=1, help="fit: 1 = all training data (recommended)")
    a = p.parse_args()
    a.state = a.state or os.path.join(a.work_dir, "norm_state.json")
    {"fit": cmd_fit, "pairs": cmd_pairs, "preview": cmd_preview}[a.cmd](a)


if __name__ == "__main__":
    main()
