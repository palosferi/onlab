# Tor Website Fingerprinting – User Guide

This document outlines the repository structure and provides step-by-step instructions for data collection, feature extraction, model evaluation, and reporting.

## Repository Structure
* `docs/` - Planning documents, written reports, and project summaries.
* `figures/` - Generated evaluation charts (.png) and metric reports (.json).
* `logs/` - Structured JSONL events and collection logs.
* `scripts/` - Bash and Python scripts for server-side data collection.
* `src/` - Core Python pipeline for feature extraction, model training, and evaluation.
* `tor_dataset/` - Raw PCAP files and extracted CSV features, organized by baseline, obfs4, and other. Alternatively, download the compiled archive directly from the [Zenodo Dataset Link](https://zenodo.org/records/20493234?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImQxMTQ4ZDUxLWNmN2QtNDY5ZS05OTczLTZjODFlYTY4OWYwOCIsImRhdGEiOnt9LCJyYW5kb20iOiI5MDdkMmIwYmYyNmE5NzIwZjlkMzQ2NzQ4NDg2MjU5NSJ9.QKqFF6V8eopaoBGEy8V2pVp4tq8eyAIEzPTbSW_Ch0AAewHHoVz_fMSj3uDic5G_era0gIZArKG0F1nEc9KjkQ).

## 1. Data Collection
Data collection uses automated headless browsers and `tcpdump`. A Tor SOCKS proxy and ControlPort must be active.

**Standard Closed-World Collection:**
* Script: `scripts/collection/collector.py`
* Runner: `scripts/collection/run_collector.sh`

**Open-World ("Other") Collection:**
* Script: `scripts/collection/server_collect_other.py`
* Runner: `scripts/collection/run_server_collect_other.sh`
* Execution: Run the bash script on the server. Collected pcaps will be saved to `tor_dataset/other_tor/`.

**Environment Overrides (Optional):**
Variables like `TOR_WF_PCAP_DIR`, `TOR_WF_INTERFACE`, `TOR_WF_GUARD_IP`, `TOR_WF_CAPTURE_DURATION`, and `TOR_WF_WARMUP_DURATION` can be passed to customize the collection environment.

## 2. Feature Extraction
Once PCAP files are populated in the `tor_dataset` subdirectories (manually or via Zenodo source expansion), extract the engineered flow features into CSV format.

* Run command: `python src/extract_all_features.py`
* Output: CSV files generated in `tor_dataset/extracted_features/` for baseline, obfs4, and other categories.

## 3. Evaluation
Train the models and generate evaluation metrics (accuracy, macro-F1, per-class recall, and confusion matrices). Both scripts execute shared and optimized feature tracks across multiple random seeds.

* Random Forest Evaluation: `python src/evaluate_all_rf.py`
* Deep Learning Evaluation: `python src/evaluate_all_dl.py`
* Output: Metrics are saved as `figures/metrics_rf.json` and `figures/metrics_dl.json`.

## 4. Figure Generation
Generate visualizations based on the trained models and output metrics.

* Run command: `python src/generate_all_figures.py`
* Output: PNG charts saved to the `figures/` directory, including closed-world summaries, macro-F1 comparisons, shared-vs-optimized deltas, open-world focus charts, feature importance plots, confusion matrices, and recall-normalized confusion matrices.