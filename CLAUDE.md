# Working in this repository

Tor website fingerprinting. A BSc szakdolgozat continuing a completed önlab: how
well a modern deep-learning attack generalizes across Tor transports (plain Tor,
obfs4, Snowflake). Thesis work happens on the `szakdolgozat-drift` branch.

## Machines, and which may reach the collection host

Collection runs on a home server (`fujitsu`), reached over Tailscale. **Server
work — deploys, rounds, preflight, reading logs — happens only from the Fedora
laptop or a GitHub Codespace.**

**The Windows work laptop is analysis-only and must never connect to the home
server**, by Tailscale or any other route. It has its own `.venv` and a local
copy of `tor_dataset/extracted_features`, which is everything the analysis needs.
Move code between machines through git, not by copying to the server from here.

The server's checkout is deployed by copying files, so its git HEAD lags. Treat
GitHub as the source of truth.

## Layout

- `src/` — analysis. `df_*.py` and `evaluate_df.py` are the Deep Fingerprinting
  track over raw packet-direction sequences; `feature_pool.py` and
  `evaluate_all_*.py` are the older 21-feature track; `drift_*.py` the
  longitudinal comparison; `sequence_budget.py` the window-length analysis.
- `scripts/collection/` — runs on the collection host only.
- `tests/` — plain scripts, no pytest. Run a file directly; it prints `n/n
  passed` and exits non-zero on failure. Cases needing `tcpdump` skip when it is
  absent, so the suite is clean on a workstation.
- `figures/`, `docs/` — results and the runbook (`docs/LONGITUDINAL_SETUP.md`).

## Conventions that have already caused bugs

- **Line endings are LF**, pinned in `.gitattributes`. A global
  `core.autocrlf=true` otherwise fills a Windows working tree with CRLF, and a
  shell script copied to a Linux host then fails to parse.
- **Quote DF results as mean ± sd over seeds, never a single run.** With ~30
  traces per class one run is not a measurement: the same code and seed once
  produced 73% and 28% on two machines. `evaluate_df.py` runs several seeds and
  records torch version, thread count and host.
- **One sequence length for every transport.** Snowflake carries ~4x the packets
  for the same pages, so a window that fits a baseline trace truncates a
  Snowflake one; `src/sequence_budget.py` measures the coverage a length buys.
- **Splits are chronological, never random stratified**, and open-world results
  are TPR/FPR, not accuracy.
- Never commit captures, extracted features, or bridge lines. `tor_dataset/` is
  ignored; bridge lines live outside the repository on both hosts.
