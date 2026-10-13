"""Push a DF evaluation to a Kaggle GPU as a private script kernel.

    python scripts/kaggle/push_job.py --dataset USER/wf-tb-main --collection tb-main \\
        --slug wf-tbmain-transport-s42 \\
        --cmd "src/evaluate_transport.py --collection tb-main --mode cells --length 10000
               --min-visits 30 --seeds 42 --out figures/metrics_transport_tb-main_cells_gpu.json"

Each --cmd is one script with its arguments, run from the repository root on
Kaggle with --device cuda appended. The kernel runs the pushed commit (HEAD by
default, which must already be on GitHub), so a result always names the code
that produced it. Follow and fetch with

    kaggle kernels status USER/SLUG
    kaggle kernels output USER/SLUG -p figures/kaggle/SLUG

The features go up once per collection as a private dataset (no manifest: it
holds the obfs4 bridge address, and the evaluation does not read it):

    cd tor_dataset/tb && zip -qr ~/wf-kaggle/tb-main-features.zip tb-main/features
    kaggle datasets init -p ~/wf-kaggle    # set "id" to USER/wf-tb-main, "title" to wf-tb-main
    kaggle datasets create -p ~/wf-kaggle  # private unless --public

A Kaggle session ends after 12 h, and a session cut off there may not keep its
output (the runner copies the per-seed .partial.json every 5 minutes in case it
does). Size each job to finish well inside 12 h: seeds are independent jobs,
and evaluate_budget.py on tb-main needs one job per seed (3 seeds took > 10 h).
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def git(*args):
    return subprocess.run(["git", *args], cwd=HERE, check=True, capture_output=True,
                          text=True).stdout.strip()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="USER/SLUG of the features dataset")
    p.add_argument("--collection", required=True)
    p.add_argument("--slug", required=True, help="kernel name; pushing again makes a new version")
    p.add_argument("--cmd", action="append", required=True, help="script and arguments, repeatable")
    p.add_argument("--commit", default="", help="default: HEAD, which must be pushed")
    p.add_argument("--dry-run", action="store_true", help="write the kernel folder, do not push")
    args = p.parse_args()

    commit = git("rev-parse", args.commit or "HEAD")
    if not args.dry_run and not git("branch", "-r", "--contains", commit):
        sys.exit(f"{commit[:7]} is not on GitHub yet; push it first")
    commands = [shlex.split(c) for c in args.cmd]
    for c in commands:
        if "--device" in c:
            sys.exit("leave --device out; the kernel adds --device cuda")

    user = args.dataset.split("/")[0]
    job = {"commit": commit, "collection": args.collection, "commands": commands}
    with open(os.path.join(HERE, "kaggle_run.py"), encoding="utf-8") as f:
        source = f.read()
    marker = "JOB = {}  # filled in by push_job.py"
    if marker not in source:
        sys.exit("kaggle_run.py lost its JOB marker")
    source = source.replace(marker, f"JOB = {json.dumps(job)}")

    out = tempfile.mkdtemp(prefix=f"kaggle-{args.slug}-")
    with open(os.path.join(out, "kaggle_run.py"), "w", encoding="utf-8") as f:
        f.write(source)
    meta = {
        "id": f"{user}/{args.slug}",
        "title": args.slug,
        "code_file": "kaggle_run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,  # to clone the repository
        "machine_shape": "NvidiaTeslaT4",  # current torch builds no longer run on the P100
        "dataset_sources": [args.dataset],
        "competition_sources": [],
        "kernel_sources": [],
    }
    with open(os.path.join(out, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"kernel {meta['id']} at {commit[:7]} in {out}")
    if not args.dry_run:
        subprocess.run(["kaggle", "kernels", "push", "-p", out], check=True)


if __name__ == "__main__":
    main()
