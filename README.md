# Tor Website Fingerprinting – Project Summary

## Project Goal
This project investigates and compares classical Machine Learning (Random Forest) and Deep Learning (Triplet MLP) approaches for Tor website fingerprinting. The evaluation is conducted under four main scenarios to simulate real-world conditions:
* **Baseline Tor traffic:** Standard Tor browser routing within a closed-world setting.
* **Obfs4-obfuscated traffic:** Tor traffic masked by the obfs4 pluggable transport.
* **Zero-shot setting:** Models trained purely on baseline traffic but tested against obfs4 traffic to measure domain generalization.
* **Open-World Setting:** Introduces an "Other" category containing traffic from unmonitored websites to test the model's ability to handle unknown, real-world traffic.

## Methodology
The pipeline utilizes the following techniques for feature extraction and model training:
* **Feature Representation:** Extracts 21 engineered aggregate flow features per traffic sample, but statically selects and trains on only the top 10 features.
* **Feature Scope:** The 21 extracted features include total/incoming/outgoing packet counts, byte statistics, in/out ratios, inter-arrival time statistics (mean, std, median, 90th percentile), packet size statistics, duration, and direction changes.
* **Data Splitting:** Employs a stratified train/test split to maintain class distributions.
* **Random Forest Setup:** Utilizes 300 estimators and a stabilized top-10 feature selection process evaluated across multiple random seeds.
* **Deep Learning Setup:** Employs a Triplet MLP model trained on the top 10 aggregate features using Triplet Margin Loss over 30 epochs (with standardization, BatchNorm, and Dropout). The resulting embeddings are then classified using a K-Nearest Neighbors ($k=5$) classifier.

## Results Summary
The models are evaluated using accuracy and macro-F1 metrics based on the shared features track.

**Open-World Scenarios (Random Forest vs. Deep Learning):**
* Open-World Baseline: 80.93% Accuracy (RF) | 62.00% Accuracy (DL)
* Open-World Obfs4: 74.10% Accuracy (RF) | 48.90% Accuracy (DL)
* Open-World Zero-Shot: 22.73% Accuracy (RF) | 27.27% Accuracy (DL)

**Closed-World Scenarios (Random Forest vs. Deep Learning):**
* Standard Baseline: 80.44% Accuracy (RF) | 61.19% Accuracy (DL)
* Standard Obfs4: 73.33% Accuracy (RF) | 50.64% Accuracy (DL)
* Standard Zero-Shot: 15.08% Accuracy (RF) | 23.01% Accuracy (DL)

*(Note: Zero-shot fingerprinting remains challenging and yields low accuracy across both models due to the heavy domain shift introduced by obfuscation. The Triplet MLP shows better domain generalization in both standard and open-world zero-shot scenarios compared to the Random Forest.)*