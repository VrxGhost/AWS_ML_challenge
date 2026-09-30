"""ONE COMMAND for the whole pipeline (v2: normalization.py).

Put this file in src/ next to the other .py files, then run from the project root
(the folder that has src/):

    # fastest first test (~50k Source 1 entities) - use this first to catch errors
    python src/run_all.py --quick

    # the real run
    python src/run_all.py --n-jobs 8

    # real run + quality report of the cleaning + build the final submission zip
    python src/run_all.py --n-jobs 8 --check --zip MyTeamName

What it runs, in order (every step prints its own log, everything is also
saved to work/run_all.log):

    STEP 0  fix_source_files.py        merge the *_half.tsv files into train S2/S3,
                                       repair broken lines (train + test)
    STEP 1  run_normalization.py fit   learn the Indian-script dictionary, abbreviations and
                                       protected words from ALL of train -> work/norm_state.json
                                       (--check: also the held-out table + work/norm_pairs.tsv)
    STEP 2  run_pipeline.py train      reuses work/norm_state.json; blocking, features,
                                       2-stage model, calibration, decision rule -> work/model.pkl
    STEP 3  run_pipeline.py predict    apply to test -> output/matching_results.tsv
                                       and output/candidate_pairs.tsv
    STEP 4  validate_submission.py     official format check (if the file is found)
    STEP 5  zip                        OPTIONAL (--zip TEAM): <TEAM>_submission.zip

Useful flags
    --skip-fix        step 0 was already done once (it is safe to repeat, just slower)
    --skip-norm       step 1 already done and normalization.py did not change
    --only train      run just one stage (fix | norm | train | predict | validate | zip)
    --sample-s1 N     Source 1 entities used for training (0 = all)
    --no-context      turn off stage-2 model (to compare scores)
    --lowmem          laptop mode: training uses small caches, and prediction runs through
                      predict_lowmem.py (same result, much less RAM, cleaned data goes via disk)
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time
import zipfile

SRC = os.path.dirname(os.path.abspath(__file__))
T0 = time.time()
LOG = None


def say(msg: str):
    line = f"[{time.time() - T0:7.1f}s] {msg}"
    print(line, flush=True)
    if LOG:
        LOG.write(line + "\n")
        LOG.flush()


def run(title: str, cmd: list[str]) -> None:
    """Run one step, stream its output to screen + log, stop everything on failure."""
    say("=" * 70)
    say(title)
    say("  $ " + " ".join(cmd))
    say("=" * 70)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace")
    for line in p.stdout:
        print(line, end="", flush=True)
        if LOG:
            LOG.write(line)
    code = p.wait()
    if code != 0:
        say(f"!! STEP FAILED (exit code {code}): {title}")
        say("   Read the lines just above this one - the error is there.")
        sys.exit(code)
    say(f"OK: {title}")


def find_validator(explicit: str | None) -> str | None:
    cands = [explicit] if explicit else []
    cands += ["utils/validate_submission.py", "../utils/validate_submission.py",
              "../../utils/validate_submission.py", os.path.join(SRC, "..", "utils", "validate_submission.py")]
    for c in cands:
        if c and os.path.exists(c):
            return c
    return None


def need(fname: str) -> str:
    path = os.path.join(SRC, fname)
    if not os.path.exists(path):
        say(f"!! {fname} not found in {SRC}. Put all the .py files in the same src/ folder.")
        sys.exit(2)
    return path


def make_zip(team: str, out_dir: str) -> None:
    """Builds the exact folder layout the problem statement asks for."""
    root = os.path.abspath(os.path.join(SRC, ".."))
    zpath = f"{team}_submission.zip"
    must = [os.path.join(out_dir, "matching_results.tsv"), os.path.join(out_dir, "candidate_pairs.tsv")]
    for m in must:
        if not os.path.exists(m):
            say(f"!! {m} missing - run predict first")
            sys.exit(3)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for m in must:
            z.write(m, f"output/{os.path.basename(m)}")
        for py in sorted(glob.glob(os.path.join(SRC, "**", "*.py"), recursive=True)):
            if "__pycache__" in py:
                continue
            z.write(py, "code/business_entity_resolution/src/" + os.path.relpath(py, SRC).replace(os.sep, "/"))
        for f in ("README.md", "requirements.txt"):
            p = os.path.join(root, f)
            if os.path.exists(p):
                z.write(p, f"code/business_entity_resolution/{f}")
            else:
                say(f"   warning: {f} not found next to src/ - add it before submitting")
        doc = next((p for p in ("Documentation_template.md", os.path.join(root, "Documentation_template.md"),
                                "../Documentation_template.md") if os.path.exists(p)), None)
        if doc:
            z.write(doc, "Documentation_template.md")
        else:
            say("   warning: Documentation_template.md not found - fill it in and add it to the zip")
    say(f"wrote {zpath}")


def main():
    global LOG
    ap = argparse.ArgumentParser(description="Run the whole entity-resolution pipeline with one command.")
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--sample-s1", type=int, default=400_000, help="0 = all Source 1 entities")
    ap.add_argument("--quick", action="store_true", help="fast test run: 50k entities")
    ap.add_argument("--no-context", action="store_true")
    ap.add_argument("--lowmem", action="store_true", help="laptop mode (see top of file)")
    ap.add_argument("--skip-fix", action="store_true")
    ap.add_argument("--skip-norm", action="store_true", help="reuse work/norm_state.json")
    ap.add_argument("--check", action="store_true", help="step 1 also prints the cleaning quality report")
    ap.add_argument("--only", choices=["fix", "norm", "train", "predict", "validate", "zip"])
    ap.add_argument("--s2-half", help="path of the Source 2 half file (auto-detected if omitted)")
    ap.add_argument("--s3-half", help="path of the Source 3 half file (auto-detected if omitted)")
    ap.add_argument("--validator", help="path of utils/validate_submission.py")
    ap.add_argument("--zip", metavar="TEAM", help="also build <TEAM>_submission.zip")
    a = ap.parse_args()

    os.makedirs(a.work_dir, exist_ok=True)
    LOG = open(os.path.join(a.work_dir, "run_all.log"), "a", encoding="utf-8")
    py = sys.executable
    sample = 50_000 if a.quick else a.sample_s1
    tdir, testdir = os.path.join(a.data_dir, "train"), os.path.join(a.data_dir, "test")
    say(f"run_all: data={a.data_dir} work={a.work_dir} out={a.out_dir} "
        f"sample_s1={sample or 'ALL'} n_jobs={a.n_jobs} quick={a.quick}")

    def want(stage):
        return a.only is None or a.only == stage

    # STEP 0 - fix + merge raw files
    if want("fix") and not a.skip_fix:
        cmd = [py, need("fix_source_files.py"), "--train-dir", tdir, "--test-dir", testdir]
        if a.s2_half:
            cmd += ["--s2-half", a.s2_half]
        if a.s3_half:
            cmd += ["--s3-half", a.s3_half]
        run("STEP 0 - merge half files + repair broken lines", cmd)

    # STEP 1 - fit the normaliser once (train then reuses work/norm_state.json)
    if want("norm") and not a.skip_norm:
        cmd = [py, need("run_normalization.py"), "fit", "--data-dir", a.data_dir, "--work-dir", a.work_dir]
        if a.check:
            cmd.append("--report")
        run("STEP 1 - fit normaliser (Indian scripts, abbreviations, protected words)", cmd)
        if a.check:
            run("STEP 1b - true pairs side by side -> work/norm_pairs.tsv",
                [py, need("run_normalization.py"), "pairs", "--data-dir", a.data_dir, "--work-dir", a.work_dir])

    common = ["--data-dir", a.data_dir, "--work-dir", a.work_dir, "--out-dir", a.out_dir,
              "--n-jobs", str(a.n_jobs)]
    if a.no_context:
        common.append("--no-context")

    # STEP 2 - train
    if want("train"):
        run("STEP 2 - train (blocking + features + model + decision)",
            [py, need("run_pipeline.py"), "train", *common, "--sample-s1", str(sample)]
            + (["--low-mem"] if a.lowmem else []))

    # STEP 3 - predict
    if want("predict"):
        if a.lowmem:
            run("STEP 3 - predict on test (low-memory, slice by slice)",
                [py, need("predict_lowmem.py"), "--data-dir", a.data_dir, "--work-dir", a.work_dir,
                 "--out-dir", a.out_dir, "--n-jobs", str(a.n_jobs)])
        else:
            run("STEP 3 - predict on test",
                [py, need("run_pipeline.py"), "predict", *common])

    # STEP 4 - official validator
    if want("validate"):
        v = find_validator(a.validator)
        if v:
            run("STEP 4 - official validation",
                [py, v, "--matching", os.path.join(a.out_dir, "matching_results.tsv"),
                 "--candidate", os.path.join(a.out_dir, "candidate_pairs.tsv"), "--test-dir", testdir])
        else:
            say("STEP 4 skipped: utils/validate_submission.py not found (pass --validator PATH). "
                "The built-in self-validation in STEP 3 still ran.")

    # STEP 5 - zip
    if a.zip and (a.only in (None, "zip")):
        say("STEP 5 - build submission zip")
        make_zip(a.zip, a.out_dir)
    elif a.only == "zip":
        say("--only zip needs --zip TEAM")

    say("ALL DONE. Upload output/matching_results.tsv to the portal.")


if __name__ == "__main__":
    main()
