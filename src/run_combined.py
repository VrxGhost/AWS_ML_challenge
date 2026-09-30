"""run_combined.py - one command for the final pipeline.

Run from the project root (the folder that has src/ and dataset/):

    python src/run_combined.py --n-jobs 8 --entities 300000 --cache full            # 64 GB machine
    python src/run_combined.py --n-jobs 4 --entities 100000                         # 16 GB machine
    python src/run_combined.py --n-jobs 8 --entities 300000 --cache full --tune --zip TEAM

What it runs (every step also goes to <work-dir>/run_combined.log):

  STEP 1  train_dense.py     in one go:
                             - dense training (every sampled entity against ALL Source 2/3 records)
                             - size-independent features and block limits (always on)
                             - the French-aware cleaner (fitted here if <work-dir>/norm_state.json is missing)
                             -> <work-dir>/model.pkl, prints DENSE HOLDOUT A / B
  STEP 2  predict_lowmem.py  test set -> <out-dir>/matching_results.tsv + candidate_pairs.tsv
                             (also saves every scored pair: <work-dir>/test_scores.pkl)
  STEP 3  validate           official utils/validate_submission.py if found
          => a first submission is READY HERE. Upload it.

  --tune adds the dense dev set + decision-rule tuning on the new model:
  STEP 4  make_devset.py     dev set from train, leaving out the entities train_dense used
                             -> <dev-dir>/  (a new folder: an old dataset_dev belongs to the OLD model)
  STEP 5  predict_lowmem.py  score the dev set -> <work-dir>/dev_scores.pkl
  STEP 6  tune_on_dev.py     224 rules; writes the best into model.pkl ONLY if it beats the current one
  STEP 7  predict_lowmem.py  --from-scores: apply the (maybe new) rule to the saved test pairs (minutes)
                             -> <out-dir>_tuned/
  STEP 8  validate the tuned output

  --stage3 (needs --tune) adds:
  STEP 9  stage3.py          small model on the dev scores WITH the reverse context features (full
                             competition there); used ONLY if it beats the tuned rule on a held-back
                             quarter of the dev set -> <out-dir>_s3/

  --zip TEAM builds TEAM_submission.zip from the last output that was produced and accepted.

Flags to resume after a crash:  --skip-train (model.pkl exists), --skip-predict (test_scores.pkl exists),
--skip-devset (dev folder exists), --skip-devscore (dev_scores.pkl exists).
Unattended runs: use run_failsafe.py, which resumes automatically after a crash.
"""
from __future__ import annotations

import argparse
import os
import sys

import run_all as R          # same folder: reuses run(), say(), make_zip(), find_validator()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--work-dir", default="work_dense")
    ap.add_argument("--out-dir", default="output_dense")
    ap.add_argument("--dev-dir", default="dataset_dev_dense")
    ap.add_argument("--entities", type=int, default=200_000, help="Source 1 entities with labelled pairs")
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--cache", default="low", choices=["low", "full"], help="full on a 64 GB machine")
    ap.add_argument("--chunk", type=int, default=200_000)
    ap.add_argument("--tune", action="store_true", help="also run steps 4-8 (dev set + rule tuning)")
    ap.add_argument("--dev-frac", type=float, default=1.0, help="share of dev Source 1 entities (0.5 = faster)")
    ap.add_argument("--stage3", action="store_true", help="after --tune: stage-3 model on the dev scores")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-predict", action="store_true")
    ap.add_argument("--skip-devset", action="store_true")
    ap.add_argument("--skip-devscore", action="store_true", help="dev_scores.pkl exists: do not score the dev set again")
    ap.add_argument("--validator", default=None, help="path to utils/validate_submission.py")
    ap.add_argument("--zip", metavar="TEAM", default=None)
    a = ap.parse_args()

    os.makedirs(a.work_dir, exist_ok=True)
    R.LOG = open(os.path.join(a.work_dir, "run_combined.log"), "a", encoding="utf-8")
    py = sys.executable
    common = ["--n-jobs", str(a.n_jobs), "--cache", a.cache, "--chunk", str(a.chunk)]
    test_scores = os.path.join(a.work_dir, "test_scores.pkl")
    dev_scores = os.path.join(a.work_dir, "dev_scores.pkl")
    tuned_dir = a.out_dir + "_tuned"

    for need in (os.path.join(a.data_dir, "train", "train_ground_truth.tsv"),
                 os.path.join(a.data_dir, "test", "test_source1.tsv")):
        if not os.path.exists(need):
            R.say(f"!! {need} not found. Run from the project root; dataset/ must hold train/ and test/.")
            sys.exit(2)

    def validate(out_dir, title):
        v = R.find_validator(a.validator)
        if v:
            R.run(title, [py, v, "--matching", os.path.join(out_dir, "matching_results.tsv"),
                          "--candidate", os.path.join(out_dir, "candidate_pairs.tsv"),
                          "--test-dir", os.path.join(a.data_dir, "test")])
        else:
            R.say(f"{title}: utils/validate_submission.py not found (pass --validator). "
                  "The built-in self-check inside predict still ran.")

    if not a.skip_train:
        R.run("STEP 1 - dense training (+ size-independent features + French cleaner)",
              [py, R.need("train_dense.py"), "--data-dir", a.data_dir, "--work-dir", a.work_dir,
               "--entities", str(a.entities), *common])
    if not a.skip_predict:
        R.run("STEP 2 - predict on test",
              [py, R.need("predict_lowmem.py"), "--data-dir", a.data_dir, "--work-dir", a.work_dir,
               "--out-dir", a.out_dir, *common, "--save-scores", test_scores])
    validate(a.out_dir, "STEP 3 - validate first submission")
    R.say(f"FIRST SUBMISSION READY: {os.path.join(a.out_dir, 'matching_results.tsv')}")
    final_dir = a.out_dir

    if a.tune:
        if not a.skip_devset:
            R.run("STEP 4 - build dense dev set (without the training entities)",
                  [py, R.need("make_devset.py"), "--data-dir", a.data_dir, "--work-dir", a.work_dir,
                   "--out", a.dev_dir, "--frac", str(a.dev_frac)])
        if a.skip_devscore and os.path.exists(dev_scores):
            R.say(f"STEP 5 skipped: {dev_scores} exists")
        else:
            R.run("STEP 5 - score the dev set",
                  [py, R.need("predict_lowmem.py"), "--data-dir", a.dev_dir, "--work-dir", a.work_dir,
                   "--out-dir", os.path.join(a.dev_dir, "output"), *common, "--save-scores", dev_scores])
        R.run("STEP 6 - tune the decision rule on the dev set",
              [py, R.need("tune_on_dev.py"), "--scores", dev_scores,
               "--truth", os.path.join(a.dev_dir, "dev_ground_truth.tsv"), "--work-dir", a.work_dir,
               "--by-country", os.path.join(a.dev_dir, "test", "test_source1.tsv"), "--write"])
        R.run("STEP 7 - apply the rule to the saved test pairs",
              [py, R.need("predict_lowmem.py"), "--data-dir", a.data_dir, "--work-dir", a.work_dir,
               "--out-dir", tuned_dir, "--from-scores", test_scores])
        validate(tuned_dir, "STEP 8 - validate tuned submission")
        R.say(f"TUNED SUBMISSION READY: {os.path.join(tuned_dir, 'matching_results.tsv')}")
        final_dir = tuned_dir

        if a.stage3:
            s3_work, s3_out = a.work_dir + "_s3", a.out_dir + "_s3"
            s3_scores = os.path.join(s3_work, "test_scores_s3.pkl")
            if os.path.exists(s3_scores):
                os.remove(s3_scores)
            R.run("STEP 9 - stage 3 on the dev scores (reverse context with full competition)",
                  [py, R.need("stage3.py"), "--work-dir", a.work_dir, "--dev-scores", dev_scores,
                   "--dev-truth", os.path.join(a.dev_dir, "dev_ground_truth.tsv"),
                   "--dev-s1", os.path.join(a.dev_dir, "test", "test_source1.tsv"),
                   "--test-scores", test_scores, "--test-s1", os.path.join(a.data_dir, "test", "test_source1.tsv"),
                   "--out-work", s3_work, "--n-jobs", str(a.n_jobs)])
            if os.path.exists(s3_scores):
                R.run("STEP 10 - stage-3 decisions on the test pairs",
                      [py, R.need("predict_lowmem.py"), "--data-dir", a.data_dir, "--work-dir", s3_work,
                       "--out-dir", s3_out, "--from-scores", s3_scores])
                validate(s3_out, "STEP 11 - validate stage-3 submission")
                R.say(f"STAGE-3 SUBMISSION READY: {os.path.join(s3_out, 'matching_results.tsv')}")
                final_dir = s3_out
            else:
                R.say("stage 3 did not beat the tuned rule on the check part: keeping the tuned submission")

    if a.zip:
        R.say(f"STEP 12 - build submission zip from {final_dir}")
        R.make_zip(a.zip, final_dir)
    R.say("ALL DONE.")


if __name__ == "__main__":
    main()
