"""Raw packet-direction sequences for Deep Fingerprinting.

The aggregated 21-feature pool in feature_pool.py collapses each trace to a
single row. DF needs the sequence itself, so this module is a second consumer
of exactly the same per-packet CSVs -- no new collection is required.
"""

import logging
import os
import re
from glob import glob

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

# Sirinam et al. use 5000 Tor cells. Our captures are TCP packets, so this is
# the packet-count analogue; see SEQUENCE_LENGTH_NOTE in the thesis notes.
SEQUENCE_LENGTH = 5000

FILENAME_RE = re.compile(r"^(?P<site>.+)_(?P<date>\d{8})_(?P<time>\d{6})$")


def parse_trace_name(file_path):
    """Return (site_name, timestamp) for a {site}_{YYYYMMDD}_{HHMMSS}.csv file.

    Falls back to (basename-prefix, None) so that oddly named files still load
    instead of aborting a whole run.
    """
    base = os.path.splitext(os.path.basename(file_path))[0]
    match = FILENAME_RE.match(base)
    if not match:
        return base.split("_")[0], None
    stamp = pd.to_datetime(
        match.group("date") + match.group("time"), format="%Y%m%d%H%M%S"
    )
    return match.group("site"), stamp


def _to_sequence(df, mode, length):
    dirs = np.sign(df["direction_size"].to_numpy(dtype=np.float32))

    if mode == "direction":
        seq = dirs
    elif mode == "tiktok":
        # Tik-Tok: keep direction but scale it by the packet's arrival time.
        seq = dirs * df["time_offset"].to_numpy(dtype=np.float32)
    elif mode == "iat":
        seq = dirs * df["inter_arrival_time"].to_numpy(dtype=np.float32)
    else:
        raise ValueError(f"unknown sequence mode: {mode}")

    out = np.zeros(length, dtype=np.float32)
    n = min(len(seq), length)
    out[:n] = seq[:n]
    return out


def load_sequences(
    directory,
    force_label=None,
    mode="direction",
    length=SEQUENCE_LENGTH,
    fail_on_high_skip=True,
    max_skip_ratio=0.05,
):
    """Load every CSV under `directory` as a fixed-length sequence.

    Returns (X, y, timestamps) where X is (n_traces, length) float32.
    """
    csv_files = sorted(glob(os.path.join(directory, "*.csv")))
    sequences, labels, stamps = [], [], []
    skipped = []

    for file_path in csv_files:
        site_name, stamp = parse_trace_name(file_path)
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                continue
            sequences.append(_to_sequence(df, mode, length))
            labels.append(force_label if force_label else site_name)
            stamps.append(stamp)
        except Exception as exc:
            skipped.append((file_path, str(exc)))

    if skipped:
        for file_path, err in skipped[:10]:
            LOGGER.warning("Skipping malformed trace %s: %s", file_path, err)
        if len(skipped) > 10:
            LOGGER.warning("... plus %d additional malformed files", len(skipped) - 10)

    total_files = len(csv_files)
    skip_ratio = (len(skipped) / total_files) if total_files else 0.0
    if fail_on_high_skip and total_files > 0 and skip_ratio > max_skip_ratio:
        raise RuntimeError(
            "Sequence loading skip ratio too high "
            f"({len(skipped)}/{total_files} = {skip_ratio:.2%})."
        )

    if not sequences:
        return (
            np.zeros((0, length), dtype=np.float32),
            np.array([], dtype=object),
            np.array([], dtype="datetime64[ns]"),
        )

    return (
        np.stack(sequences),
        np.array(labels, dtype=object),
        np.array(stamps, dtype="datetime64[ns]"),
    )


def chronological_split(y, timestamps, test_size=0.2, per_class=True):
    """Split indices by capture time: earliest traces train, latest test.

    A random stratified split lets same-session circuit and network conditions
    leak across the boundary, which is the inflation the Cherubin et al. online
    WF paper calls out. `per_class=True` splits within each label so that every
    class stays represented on both sides; `per_class=False` uses one global
    cut, which is stricter but can drop whole classes from train.
    """
    y = np.asarray(y)
    timestamps = np.asarray(timestamps)

    if np.isnat(timestamps).any():
        raise ValueError(
            "chronological_split needs a timestamp for every trace; "
            f"{int(np.isnat(timestamps).sum())} are missing."
        )

    if not per_class:
        order = np.argsort(timestamps, kind="stable")
        cut = int(round(len(order) * (1 - test_size)))
        return order[:cut], order[cut:]

    train_idx, test_idx = [], []
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        idx = idx[np.argsort(timestamps[idx], kind="stable")]
        cut = int(round(len(idx) * (1 - test_size)))
        # Never hand a class zero training traces.
        cut = max(cut, 1) if len(idx) > 1 else len(idx)
        train_idx.append(idx[:cut])
        test_idx.append(idx[cut:])

    return (
        np.concatenate(train_idx) if train_idx else np.array([], dtype=int),
        np.concatenate(test_idx) if test_idx else np.array([], dtype=int),
    )
