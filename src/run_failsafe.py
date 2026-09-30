"""run_failsafe.py - run run_combined.py unattended and recover from crashes on its own.

    python src/run_failsafe.py --team TEAMNAME -- --work-dir work_v2 --out-dir output_v2 \
        --dev-dir dataset_dev_dense --skip-devset --n-jobs 4 --entities 100000 --tune --stage3

Everything after "--" is passed to run_combined.py.  After a crash it looks at which results already
exist and starts again from there (never redoing finished work):
  - work-dir/model.pkl exists                  -> --skip-train
  - test_scores.pkl + first output exist       -> --skip-predict
  - dev_scores.pkl exists                      -> --skip-devscore
  - the tuned output exists (so the crash was in stage 3, usually memory) -> next try WITHOUT
    --stage3: the tuned output is kept and the zip is built from it
  - otherwise (crash in training / predicting, usually memory) -> one worker fewer (min 2)
At most --max-tries attempts.  After --stop-heavy (clock time, default 22:30) no heavy step is
started again.  At the very end it ALWAYS makes sure a submission zip exists, built from the best
output that exists (stage 3 > tuned > first).  Everything is logged to <work-dir>/failsafe.log.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys

SRC = os.path.dirname(os.path.abspath(__file__))


def main():
    argv = sys.argv[1:]
    rest = []
    if "--" in argv:
        k = argv.index("--")
        argv, rest = argv[:k], argv[k + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--team", required=True)
    ap.add_argument("--max-tries", type=int, default=4)
    ap.add_argument("--stop-heavy", default="22:30", help="HH:MM: after this, no heavy step is started")
    a = ap.parse_args(argv)

    def opt(name, default):
        return rest[rest.index(name) + 1] if name in rest else default

    work, out = opt("--work-dir", "work_dense"), opt("--out-dir", "output_dense")
    n_jobs = int(opt("--n-jobs", "4"))
    tuned, s3 = out + "_tuned", out + "_s3"
    os.makedirs(work, exist_ok=True)
    logf = open(os.path.join(work, "failsafe.log"), "a", encoding="utf-8")

    def say(msg):
        line = f"[{dt.datetime.now():%H:%M:%S}] FAILSAFE: {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    hh, mm = map(int, a.stop_heavy.split(":"))
    stop_at = dt.datetime.now().replace(hour=hh, minute=mm, second=0, microsecond=0)
    base = [x for x in rest if x not in ("--skip-train", "--skip-predict", "--skip-devscore", "--zip")]
    if "--zip" in rest:                                    # drop "--zip TEAM" from base, we add our own
        i = rest.index("--zip")
        base = [x for j, x in enumerate(rest) if j not in (i, i + 1)
                and x not in ("--skip-train", "--skip-predict", "--skip-devscore")]
    use_stage3 = "--stage3" in base
    exists = lambda p: os.path.exists(p) and os.path.getsize(p) > 0
    ok = False

    for attempt in range(1, a.max_tries + 1):
        heavy_needed = not exists(os.path.join(work, "model.pkl")) or not exists(os.path.join(work, "test_scores.pkl"))
        if attempt > 1 and heavy_needed and dt.datetime.now() > stop_at:
            say(f"it is past {a.stop_heavy} and a heavy step would be needed: not starting it")
            break
        args = [x for x in base if x != "--stage3"]
        if "--n-jobs" not in args:
            args += ["--n-jobs", str(n_jobs)]
        else:
            args[args.index("--n-jobs") + 1] = str(n_jobs)
        if use_stage3:
            args.append("--stage3")
        if exists(os.path.join(work, "model.pkl")):
            args.append("--skip-train")
        if exists(os.path.join(work, "test_scores.pkl")) and exists(os.path.join(out, "matching_results.tsv")):
            args.append("--skip-predict")
        if exists(os.path.join(work, "dev_scores.pkl")):
            args.append("--skip-devscore")
        args += ["--zip", a.team]
        cmd = [sys.executable, "-u", os.path.join(SRC, "run_combined.py"), *args]
        say(f"attempt {attempt}/{a.max_tries}: {' '.join(cmd[2:])}")
        code = subprocess.call(cmd)
        if code == 0:
            say("run_combined finished OK")
            ok = True
            break
        say(f"attempt {attempt} FAILED with exit code {code}")
        if exists(os.path.join(tuned, "matching_results.tsv")) and use_stage3:
            use_stage3 = False
            say("the tuned output exists, so the crash was in stage 3: next try without stage 3")
        else:
            n_jobs = max(2, n_jobs - 1)
            say(f"next try with {n_jobs} workers (less memory)")

    # always leave a submission zip behind
    zpath = f"{a.team}_submission.zip"
    best = next((d for d in (s3, tuned, out) if exists(os.path.join(d, "matching_results.tsv"))), None)
    if best is None:
        say("NO output exists - use the backup (v1) submission")
        sys.exit(1)
    if not ok or not exists(zpath):
        say(f"building {zpath} from {best}")
        subprocess.call([sys.executable, os.path.join(SRC, "run_all.py"), "--only", "zip",
                         "--zip", a.team, "--out-dir", best])
    say(f"DONE. Best available output: {os.path.join(best, 'matching_results.tsv')}   zip: {zpath}")


if __name__ == "__main__":
    main()
