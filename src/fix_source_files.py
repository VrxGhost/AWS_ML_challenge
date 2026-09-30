"""STEP 0 of the pipeline: make the raw TSV files loadable and complete.

What it does
  1. Reads train_source2.tsv + the "half" file for Source 2 (and the same for
     Source 3) line by line, repairing broken lines:
       - a line that does NOT start with an id (S1-/S2-/S3-) is the tail of the
         previous record (a newline inside a name/address)  -> glued back on
       - a record with too many tabs -> extra pieces are glued into the address
       - a record with too few tabs  -> missing fields are filled with ""
  2. Merges main + half, drops exact duplicate ids (keeps the first one).
     If no half file is found (you already merged them), it only repairs lines.
  3. Writes the merged file back as train_source{2,3}.tsv. The original is kept
     once as train_source{2,3}.orig.tsv (so running the script twice is safe).
  4. Checks that every id in train_ground_truth.tsv now exists.
  5. With --test-dir: repairs broken lines in the test files too (no merging),
     and checks that no test Source 1 row was lost.

Usage
  python src/fix_source_files.py --train-dir dataset/train \
         --s2-half dataset/train/source_2_half.tsv --s3-half dataset/train/source_3_half.tsv \
         --test-dir dataset/test
  (if --s2-half/--s3-half are omitted, files matching *2*half*.tsv / *3*half*.tsv
   inside --train-dir are used.)
"""
import argparse
import glob
import os
import re
import shutil
import sys

COLS = ["entity_id", "business_name", "business_address", "country"]
ID_START = re.compile(r"^S[123]-")
COUNTRY_TAIL = re.compile(r"[\s,]+(India|IN|IND|US|USA|United States|France|FR|FRA)\s*$", re.I)


class Stats:
    def __init__(self, name):
        self.name, self.n, self.glued, self.many, self.few = name, 0, 0, 0, 0

    def line(self):
        return (f"  {self.name}: {self.n:,} records (glued {self.glued}, "
                f"too many tabs {self.many}, too few tabs {self.few})")

    @property
    def changed(self):
        return self.glued + self.many + self.few > 0


def robust_read(path, report, max_show=5):
    """Stream 4-field records from a TSV that may contain broken lines.
    Yields lists [entity_id, business_name, business_address, country]."""
    st = Stats(os.path.basename(path))
    header, order, n_cols = None, None, 4

    def fix(g):
        f = g.split("\t")
        if order is not None and len(f) == n_cols:
            f = [f[i] for i in order]
        elif len(f) > 4:
            st.many += 1
            if st.many <= max_show:
                report.append(f"  {st.name}: {len(f)} fields -> extra pieces merged into address: {g[:90]!r}")
            f = [f[0], f[1], " ".join(x for x in f[2:-1] if x), f[-1]]
        elif len(f) < 4:
            st.few += 1
            if st.few <= max_show:
                report.append(f"  {st.name}: {len(f)} fields -> padded with empty fields: {g[:90]!r}")
            if len(f) == 3:   # tab before country lost? "...411001 India" -> split it back off
                m = COUNTRY_TAIL.search(f[2])
                if m:
                    f = [f[0], f[1], f[2][:m.start()].rstrip(" ,"), m.group(1)]
            f = f + [""] * (4 - len(f))
        st.n += 1
        return [x.strip() for x in f]

    pending = None
    with open(path, encoding="utf-8", errors="replace", newline="") as fh:
        for i, ln in enumerate(fh):
            ln = ln.rstrip("\r\n")
            if i == 0:
                ln = ln.lstrip("﻿")
                first = ln.split("\t")[0].strip().lower()
                if first == "entity_id":
                    header = [h.strip().lower() for h in ln.split("\t")]
                    n_cols = len(header)
                    if all(c in header for c in COLS):
                        order = [header.index(c) for c in COLS]
                    continue
            if not ln.strip():
                continue
            if ID_START.match(ln) or pending is None:
                if pending is not None:
                    yield fix(pending)
                pending = ln
            else:                      # tail of the previous record
                st.glued += 1
                if st.glued <= max_show:
                    report.append(f"  {st.name}: glued continuation line onto previous record: {ln[:90]!r}")
                pending = pending + " " + ln
    if pending is not None:
        yield fix(pending)
    report.append(st.line())
    robust_read.last = st


def clean_field(x):
    return x.replace("\t", " ").replace("\n", " ")


def backup_once(path):
    orig = path[:-4] + ".orig.tsv"
    if not os.path.exists(orig):
        shutil.copy2(path, orig)
    return orig


def find_half(train_dir, k):
    hits = [p for p in glob.glob(os.path.join(train_dir, "*.tsv"))
            if "half" in os.path.basename(p).lower() and str(k) in os.path.basename(p)]
    return hits[0] if len(hits) == 1 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="dataset/train")
    ap.add_argument("--s2-half")
    ap.add_argument("--s3-half")
    ap.add_argument("--test-dir")
    a = ap.parse_args()

    report = []
    ids = {}
    for k, half in ((2, a.s2_half), (3, a.s3_half)):
        main_path = os.path.join(a.train_dir, f"train_source{k}.tsv")
        # read from the untouched original if we already ran once
        orig = main_path[:-4] + ".orig.tsv"
        src_path = orig if os.path.exists(orig) else main_path
        half = half or find_half(a.train_dir, k)
        report.append(f"Source {k}: main = {src_path}, half = {half}")
        backup_once(main_path)
        seen, dup, bad_prefix, n_out = set(), 0, 0, 0
        tmp = main_path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="") as out:
            out.write("\t".join(COLS) + "\n")
            inputs = [src_path]
            if half and os.path.exists(half):
                inputs.append(half)
            else:
                report.append(f"  no half file found for Source {k} - only repairing lines")
            for path in inputs:
                for r in robust_read(path, report):
                    if r[0] in seen:
                        dup += 1
                        continue
                    seen.add(r[0])
                    if not r[0].startswith(f"S{k}-"):
                        bad_prefix += 1
                    out.write("\t".join(clean_field(x) for x in r) + "\n")
                    n_out += 1
        os.replace(tmp, main_path)
        if dup:
            report.append(f"  dropped {dup:,} duplicate ids (kept the first copy)")
        if bad_prefix:
            report.append(f"  !! {bad_prefix:,} ids do not start with S{k}-")
        report.append(f"  wrote {main_path}: {n_out:,} records")
        ids[k] = seen

    # Source 1: repair only
    s1_path = os.path.join(a.train_dir, "train_source1.tsv")
    ids[1] = {r[0] for r in robust_read(s1_path, report)}
    if robust_read.last.changed:
        report.append("  !! train_source1.tsv has broken lines - rewriting it (original kept as .orig.tsv)")
        backup_once(s1_path)
        tmp = s1_path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="") as out:
            out.write("\t".join(COLS) + "\n")
            for r in robust_read(s1_path[:-4] + ".orig.tsv", []):
                out.write("\t".join(clean_field(x) for x in r) + "\n")
        os.replace(tmp, s1_path)

    # ground-truth check
    gt_path = os.path.join(a.train_dir, "train_ground_truth.tsv")
    missing_s1, missing_t, total = 0, 0, 0
    with open(gt_path, encoding="utf-8") as f:
        next(f)
        for ln in f:
            p = ln.rstrip("\r\n").split("\t")
            if p[0] not in ids[1]:
                missing_s1 += 1
            for x in (p[1].split(",") if len(p) > 1 else []):
                x = x.strip()
                if not x:
                    continue
                total += 1
                if x not in ids[2] and x not in ids[3]:
                    missing_t += 1
    report.append(f"Ground truth: {total:,} matched ids, {missing_t:,} missing from S2/S3, "
                  f"{missing_s1:,} S1 ids missing from train_source1")

    # test files: repair only
    if a.test_dir:
        for k in (1, 2, 3):
            p = os.path.join(a.test_dir, f"test_source{k}.tsv")
            n = sum(1 for _ in robust_read(p, report))
            if robust_read.last.changed:
                orig = backup_once(p)
                tmp = p + ".tmp"
                with open(tmp, "w", encoding="utf-8", newline="") as out:
                    out.write("\t".join(COLS) + "\n")
                    for r in robust_read(orig, []):
                        out.write("\t".join(clean_field(x) for x in r) + "\n")
                os.replace(tmp, p)
                report.append(f"  repaired and rewrote {p} ({n:,} records; original kept as .orig.tsv)")

    print("\n".join(report))
    ok = missing_t == 0 and missing_s1 == 0
    print("\nRESULT:", "OK - every ground-truth id exists" if ok else "CHECK THE MISSING IDS ABOVE")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
