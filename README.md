# Tor Website Fingerprinting – Project Summary

## Project Goal
This project investigates and compares classical Machine Learning (Random Forest) and Deep Learning (Triplet MLP) approaches for Tor website fingerprinting. The evaluation is conducted under four main scenarios to simulate real-world conditions:
* **Baseline Tor traffic:** Standard Tor browser routing within a closed-world setting.
* **Obfs4-obfuscated traffic:** Tor traffic masked by the obfs4 pluggable transport.
* **Zero-shot setting:** Models trained purely on baseline traffic but tested against obfs4 traffic to measure domain generalization.
* **Open-World Setting:** Introduces an "Other" category containing traffic from unmonitored websites to test the model's ability to handle unknown, real-world traffic.

## Methodology Improvements
The pipeline was recently overhauled, resulting in significant performance gains. 

* **Feature Representation:** Transitioned from raw packet sequences (fixed 500-length with truncation/padding) to 13 engineered aggregate flow features.
* **Feature Scope:** Features now include packet counts, byte statistics, in/out ratios, inter-arrival times, and duration.
* **Data Splitting:** Implemented a stratified train/test split to reduce class-imbalance issues.
* **Random Forest Tuning:** Increased estimators to 300 and stabilized the top-10 feature selection process across multiple seeds.
* **Deep Learning Rework:** Replaced the 1D CNN with a Triplet MLP trained on aggregate features, utilizing standardization, BatchNorm, and Dropout over 30 epochs.

## Results Summary
With the new aggregated features and stratified splitting, in-distribution performance improved substantially. Metrics were expanded to report macro-F1 and per-class recall alongside accuracy.

**Open-World Scenarios (Random Forest):**
* Open-World Baseline: 81.93% Accuracy | 81.02% Macro-F1
* Open-World Obfs4: 72.65% Accuracy | 70.84% Macro-F1
* Open-World Zero-Shot: 16.08% Accuracy | 12.13% Macro-F1

**Closed-World Scenarios (Random Forest):**
* Standard Baseline: 81.78% Accuracy
* Standard Obfs4: 72.38% Accuracy
* Standard Zero-Shot: 16.57% Accuracy

*(Note: Zero-shot fingerprinting remains challenging and low in accuracy due to the heavy domain shift introduced by obfuscation).*