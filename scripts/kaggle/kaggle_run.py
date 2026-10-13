"""Run DF evaluations on a Kaggle GPU, from a pinned commit of this repository.

push_job.py fills in JOB and pushes this file as a private Kaggle script
kernel; it is not meant to be run anywhere else. The kernel clones the
repository at JOB["commit"], finds the collection's features in the attached
dataset (as a folder, or as the zip it was uploaded as), runs each command with
--device cuda, and copies the result JSONs and logs to /kaggle/working, which is
the kernel's output.
"""

import glob
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile

JOB = {}  # filled in by push_job.py

REPO = "https://github.com/palosferi/onlab.git"
INPUT = "/kaggle/input"
WORK = "/kaggle/working"
CODE = "/tmp/onlab"
TB_ROOT = "/tmp/tb"


def find_features(collection):
    """The dataset's <collection>/features folder, unzipping the upload if needed."""
    for root, dirs, _ in os.walk(INPUT):
        if root.endswith(os.path.join(collection, "features")) and "baseline" in dirs:
            return root
    for path in glob.glob(os.path.join(INPUT, "**", "*.zip"), recursive=True):
        print(f"unzipping {path}", flush=True)
        with zipfile.ZipFile(path) as z:
            z.extractall("/tmp/unzipped")
        for root, dirs, _ in os.walk("/tmp/unzipped"):
            if root.endswith(os.path.join(collection, "features")) and "baseline" in dirs:
                return root
    sys.exit(f"no {collection} features folder or zip under {INPUT}")


def collect_outputs(since):
    for path in glob.glob(os.path.join(CODE, "figures", "*.json")):
        if os.path.getmtime(path) >= since:
            shutil.copy2(path, WORK)


def main():
    import torch

    if not torch.cuda.is_available():
        smi = subprocess.run("nvidia-smi -L", shell=True, capture_output=True, text=True)
        net = subprocess.run(["git", "ls-remote", "-q", REPO, "HEAD"], capture_output=True, text=True)
        sys.exit(f"no GPU: torch {torch.__version__} (CUDA {torch.version.cuda}), "
                 f"nvidia-smi: {(smi.stdout or smi.stderr).strip()!r}, "
                 f"internet {'ok' if net.returncode == 0 else 'blocked'}, "
                 f"image {os.environ.get('KAGGLE_DOCKER_IMAGE', '?')}. Kaggle gives an "
                 "accelerator only to phone-verified accounts.")
    torch.zeros(1, device="cuda")  # fails here, not mid-run, if this torch lacks the GPU's arch
    print(f"GPU {torch.cuda.get_device_name(0)}, torch {torch.__version__}, "
          f"python {platform.python_version()}", flush=True)

    subprocess.run(["git", "clone", "-q", REPO, CODE], check=True)
    subprocess.run(["git", "-C", CODE, "checkout", "-q", JOB["commit"]], check=True)

    features = find_features(JOB["collection"])
    os.makedirs(os.path.join(TB_ROOT, JOB["collection"]), exist_ok=True)
    link = os.path.join(TB_ROOT, JOB["collection"], "features")
    if not os.path.exists(link):
        os.symlink(features, link)
    env = {**os.environ, "TOR_WF_TB_OUT": TB_ROOT}

    record = {"job": JOB, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "commands": []}
    start = time.time()
    for i, cmd in enumerate(JOB["commands"]):
        argv = [sys.executable, "-u", *cmd, "--device", "cuda"]
        name = os.path.splitext(os.path.basename(cmd[0]))[0]
        log_path = os.path.join(WORK, f"{i}_{name}.log")
        print(f"\n===== {' '.join(cmd)} =====", flush=True)
        t0 = time.time()
        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.Popen(argv, cwd=CODE, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True)
            last_copy = time.time()
            for line in proc.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
                # The scripts save a .partial.json after every seed; keep it in
                # the output, so a session cut off at 12 h still leaves the
                # finished seeds to resume from.
                if time.time() - last_copy > 300:
                    collect_outputs(start)
                    last_copy = time.time()
            proc.wait()
        record["commands"].append({"argv": cmd, "returncode": proc.returncode,
                                   "seconds": round(time.time() - t0)})
        collect_outputs(start)
        with open(os.path.join(WORK, "job.json"), "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        if proc.returncode != 0:
            sys.exit(f"{cmd[0]} exited with {proc.returncode}")


if __name__ == "__main__":
    main()
