# Tor Website Fingerprinting – User Guide

This document outlines the repository structure and provides step-by-step instructions for data collection, feature extraction, model evaluation, and reporting.

## Repository Structure
* `docs/` - Planning documents, written reports, and project summaries.
* `figures/` - Generated evaluation charts (.png) and metric reports (.json).
* `logs/` - Structured JSONL events and collection logs.
* `scripts/` - Bash and Python scripts for server-side data collection.
* `src/` - Core Python pipeline for feature extraction, model training, and evaluation.
* `tor_dataset/longitudinal/` - Weekly drift-study rounds: PCAPs, per-round feature CSVs, `manifest.csv` and `round_meta.json`.
* `tor_dataset/` - Raw PCAP files and extracted CSV features, organized by baseline, obfs4, and other. Alternatively, download the compiled archive directly from the [Zenodo Dataset Link](https://zenodo.org/records/20493234?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImQxMTQ4ZDUxLWNmN2QtNDY5ZS05OTczLTZjODFlYTY4OWYwOCIsImRhdGEiOnt9LCJyYW5kb20iOiI5MDdkMmIwYmYyNmE5NzIwZjlkMzQ2NzQ4NDg2MjU5NSJ9.QKqFF6V8eopaoBGEy8V2pVp4tq8eyAIEzPTbSW_Ch0AAewHHoVz_fMSj3uDic5G_era0gIZArKG0F1nEc9KjkQ).

## 0. Longitudinal Collection (concept drift study, 2026/27/1)

The concept drift study runs its own collection loop, separate from the spring
2026 scripts below. See `docs/LONGITUDINAL_SETUP.md` for the full host runbook
and `docs/SZAKDOLGOZAT_temavazlat.md` for the research plan.

```bash
cp scripts/collection/.env.collection.example .env.collection   # edit it
python scripts/collection/preflight.py                          # must pass first
scripts/collection/run_round.sh --repeats 5                     # one round
```

* `scripts/collection/preflight.py` - checks packages, sudo, interface, live
  capture, Tor bootstrap on both arms, real Tor exit through each SOCKS port,
  disk space and t0 availability.
* `scripts/collection/collect_round.py` - collects one round: both arms
  interleaved, randomised visit order, page-load validation, per-round manifest.
* `scripts/collection/run_round.sh` - locked runner for cron or systemd; runs
  preflight, collects, extracts features, logs everything.
* `scripts/collection/systemd/` - weekly timer units.

Analysis:

```bash
python src/extract_round_features.py --round latest
python src/drift_eval.py    --arm baseline    # decay curve and retraining policies
python src/drift_figures.py --arm baseline    # decay, per-group, cost figures
python src/drift_shift.py   --arm baseline    # model-free KS shift test
```

## 1. Data Collection (spring 2026, archived)
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