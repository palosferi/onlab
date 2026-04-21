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
