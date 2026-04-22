# Tor Website Fingerprinting – Summary

## Project Goal
This project compares classical ML and Deep Learning approaches for Tor website fingerprinting under:
- baseline Tor traffic
- obfs4-obfuscated traffic
- zero-shot setting (train on baseline, test on obfs4)

## Key Update: Why New Results Improved

The main reason for better accuracy is a pipeline redesign and minor tuning.

### 1. Feature Representation Changed
**Old pipeline**
- Used raw packet direction/size sequences
- Fixed length of 500 packets with truncation/zero-padding

**New pipeline**
- Uses engineered aggregate flow features (13 total), such as:
  - packet counts (incoming/outgoing/total)
  - byte statistics and in/out ratios
  - duration
  - inter-arrival-time mean/std
  - packet-size statistics

This made the signal easier for both models to learn.

### 2. Data Split Became More Stable
**Old pipeline**
- Regular train/test split (no class stratification)

**New pipeline**
- Stratified train/test split for baseline and obfs4 scenarios

This reduces class-imbalance.

### 3. Random Forest Setup Improved
- Increased trees (100 -> 200 in main evaluation setup)
- Combined with stronger feature set, produced a large accuracy jump.

### 4. Deep Learning Pipeline Was Reworked
**Old DL**
- 1D CNN on raw 500-length sequences
- Shorter training in zero-shot

**New DL**
- Triplet MLP on engineered aggregate features
- Standardization (feature scaling) before training
- BatchNorm + Dropout
- Longer training (50 epochs in main evaluation)

### Term Clarification
- **Raw packet direction/size sequences**:
  Each traffic sample is represented as an ordered packet list from one visit/flow. For each packet, we keep:
  - direction (incoming vs outgoing)
  - packet size (bytes)
  So the model sees a time-ordered sequence like: out-1200, in-300, in-1400, out-80, ...

- **Fixed length of 500 packets with truncation/zero-padding**:
  Neural models expect same-length input. So every sample is forced to length 500:
  - if a flow has more than 500 packets: keep first 500 (truncate the rest)
  - if a flow has fewer than 500 packets: append zeros until length is 500 (zero-padding)

- **Engineered aggregate flow features**:
  "Aggregate" means summary statistics computed over the whole flow (or large parts of it), not packet-by-packet raw sequence. Example aggregates:
  - total packets, incoming packets, outgoing packets
  - total bytes, in/out byte ratios
  - duration
  - average and std of inter-arrival times
  - average and std of packet sizes

- **1D CNN on raw 500-length sequences**:
  "Raw 500-length sequence" means the fixed-size vector from packet order (after truncation/padding), not summary stats. A 1D CNN applies sliding filters over this sequence to learn local traffic patterns.

- **Triplet MLP on engineered aggregate features**:
  - **MLP** = Multi-Layer Perceptron (a standard fully connected neural network).
  - **Triplet** here means the training setup uses triplets of samples:
    - anchor: one sample
    - positive: another sample from the same class/site
    - negative: sample from a different class/site
  The objective pulls anchor-positive closer and pushes anchor-negative farther in embedding space.

- **BatchNorm + Dropout**:
  - **BatchNorm (Batch Normalization)** normalizes intermediate activations during training, which usually stabilizes and speeds up learning.
  - **Dropout** randomly turns off a fraction of neurons during training to reduce overfitting and improve generalization.

---

## Results Comparison

### Old (archived scripts)
- **RF**: Baseline 35.11%, Obfs4 17.14%, Zero-shot 15.62%
- **DL**: Baseline 36.44%, Obfs4 25.24%, Zero-shot 15.14%

### New (current pipeline)
- **RF**: Baseline 80.89%, Obfs4 70.00%, Zero-shot 16.10%
- **DL**: Baseline 69.33%, Obfs4 54.29%, Zero-shot 19.81%

## Interpretation
- The largest gain came from switching to aggregate engineered features.
- In-distribution performance (baseline/obfs4 trained-and-tested on same domain) improved substantially.
- Zero-shot fingerprinting remains low in accuracy due to domain shift.

## Collection (Server-Friendly)
- Collector script: `scripts/collection/collector.py`
- Runner: `scripts/collection/run_collector.sh`
- Legacy archive (safety fallback):
  - `scripts/collection/archived/collector_legacy_from_fujitsu.py`
  - `scripts/collection/archived/run_collector_legacy_from_fujitsu.sh`

Features added for safer collection:
- Guard pre-check before each site capture (hardcoded guard IP still supported)
- Warm-up delay before timed capture window
- Structured JSONL events log: `logs/collection_events.jsonl`
- Output directory create/write validation at startup

Environment overrides (optional):
- `TOR_WF_PCAP_DIR`
- `TOR_WF_LOG_JSONL`
- `TOR_WF_INTERFACE`
- `TOR_WF_GUARD_IP`
- `TOR_WF_CAPTURE_DURATION`
- `TOR_WF_WARMUP_DURATION`

Example:
```bash
TOR_WF_PCAP_DIR=/mnt/tor_data TOR_WF_INTERFACE=enp3s0 \
TOR_WF_GUARD_IP=109.110.170.208 scripts/collection/run_collector.sh
```

## Overleaf Sync Workflow
1. In Overleaf: open your project -> Menu -> Git, and copy the Git URL.
2. On your machine: clone that Overleaf Git project once.
3. Sync results from this repo into the Overleaf clone:

```bash
OVERLEAF_REPO=/path/to/overleaf-clone ./scripts/sync_to_overleaf.sh
```

4. Then push from the Overleaf clone:

```bash
cd /path/to/overleaf-clone
git add .
git commit -m "Update thesis figures and metrics"
git push
```

Supervisor invite:
1. In Overleaf project -> Share.
2. Add your konzulens email with edit permission.

## Open-World (Other) Workflow
1. Collect Other pcaps on server with the two files below:
  - `scripts/collection/server_collect_other.py`
  - `scripts/collection/run_server_collect_other.sh`
2. Copy these files to server project (example: `/home/palos/tor_wf_project/scripts/`), then run:

```bash
cd /home/palos/tor_wf_project
source venv/bin/activate
bash scripts/collection/run_server_collect_other.sh
```

3. Move collected Other pcaps into this repo at `tor_dataset/other_tor/`.
4. Extract CSV features for all groups (baseline, obfs4, other):

```bash
python src/extract_all_features.py
```

5. Re-run evaluations (now includes open-world scenarios if `other_features/` exists):

```bash
python src/evaluate_all_rf.py
python src/evaluate_all_dl.py
python src/generate_all_figures.py
```

Outputs include:
- stable top-10 lock file: `figures/selected_top10_features.json`
- RF/DL metrics with macro-F1 and per-class recall: `figures/metrics_rf.json`, `figures/metrics_dl.json`
- side-by-side Baseline vs Obfs4 feature importance: `figures/02_feature_importance.png`
