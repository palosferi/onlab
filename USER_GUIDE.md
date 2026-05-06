# Tor Website Fingerprinting – User Guide

This document outlines the repository structure and provides step-by-step instructions for data collection, feature extraction, model evaluation, and reporting.

## Repository Structure
* `docs/` - Planning documents, written reports, and project summaries.
* `figures/` - Generated evaluation charts (.png) and metric reports (.json).
* `logs/` - Structured JSONL events and collection logs.
* `scripts/` - Bash and Python scripts for server-side data collection and Overleaf synchronization.
* `src/` - Core Python pipeline for feature extraction, model training, and evaluation.
* `tor_dataset/` - Raw PCAP files and extracted CSV features, organized by baseline, obfs4, and other.

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
Once PCAP files are populated in the `tor_dataset` subdirectories, extract the engineered flow features into CSV format.

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

## 5. Overleaf Synchronization
To keep your thesis document updated with the latest figures and JSON metrics without manually uploading files:

* Clone your Overleaf Git repository to your local machine.
* Export the environment variable for your cloned repo path: `export OVERLEAF_REPO=/path/to/overleaf-clone`
* Run the sync script: `./scripts/sync_to_overleaf.sh`
* Navigate to your Overleaf clone, commit the changes, and push to the Overleaf remote repository.