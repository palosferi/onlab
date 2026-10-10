# Working in this repository

Tor website fingerprinting. A BSc szakdolgozat continuing a completed önlab: how
well a modern deep-learning attack generalizes across Tor transports (plain Tor,
obfs4, Snowflake). Thesis work happens on the `szakdolgozat-drift` branch.

## Machines, and which may reach the collection host

Collection runs on a home server (`fujitsu`, Debian 13 since 2026-10-10, user
`ferencpalos`; it was Mint with user `palos` before), reached over Tailscale.
A fresh install is rebuilt with `scripts/collection/setup_host.sh`. **Server
work — deploys, rounds, preflight, reading logs — happens only from the Fedora
laptop or a GitHub Codespace.**

**The Windows work laptop is analysis-only and must never connect to the home
server**, by Tailscale or any other route. It has its own `.venv` and a local
copy of `tor_dataset/extracted_features`, which is everything the analysis needs.
Move code between machines through git, not by copying to the server from here.

The server's checkout is deployed by copying files (`rsync -a scripts src tests
ferencpalos@100.102.24.16:onlab/`, never with `--delete`), so its git HEAD lags. Treat
GitHub as the source of truth, and **diff the server's files against the repo
before deploying**: code has been edited on the server directly before, and a
blind copy would have overwritten it.

Long jobs on the server run as transient systemd user units
(`systemd-run --user --unit=wf-...`), so they survive the SSH session. DF
training belongs on a laptop: the server has four slow cores.

## Layout

- `src/` — analysis. `df_*.py` and `evaluate_df.py` are the Deep Fingerprinting
  track on the spring dataset; `evaluate_transport.py` is the thesis evaluation
  on a Tor Browser collection (3x3 cross-transport matrix, leave-one-out,
  pooled, background tab, open world); `extract_tb_features.py` turns its pcaps
  into per-packet CSVs. `feature_pool.py` and `evaluate_all_*.py` are the older
  21-feature track; `drift_*.py` the paused longitudinal comparison;
  `sequence_budget.py` the window-length analysis.
- `scripts/collection/` — runs on the collection host only. The thesis dataset
  comes from `collect_tb.py` via `run_tb.sh` (real Tor Browser through
  tbselenium, the three arms interleaved in one schedule). Site lists:
  `sites.py` (50 monitored) and `sites/*.txt` (500 unmonitored + 100
  background, from Tranco Y8YJG, rebuilt by `build_site_lists.py`).
  `collect_round.py` / `run_round.sh` are the old Chromium drift rounds; their
  weekly timer is disabled.
- `docs/THESIS_STATUS.md` — where the work stands: what is running, results so
  far, open tasks. Read it first and update it when that changes.
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
- **One sequence length for every transport, counted in cells.** Snowflake
  carries ~4x the packets for the same pages and host GRO coalesces TCP
  segments, so packets are not a comparable unit. DF's `cells` mode (514-byte
  units, ACKs drop out) makes the three transports agree within 15% and scores
  higher than `direction` on the spring data; it is the default in
  `evaluate_transport.py`, length 10000. `src/sequence_budget.py --unit cells`
  measures the coverage a length buys.
- **Capture filters are decided after the capture for baseline and Snowflake.**
  Snowflake's WebRTC ports and the unpinned baseline Tor's guard connections
  change mid-visit, so `collect_tb.py` captures all UDP/TCP and cuts the trace
  to the ports/peers the process actually used. Do not go back to a filter
  fixed at capture start.
- **The baseline guard is not pinned any more.** A pin with `StrictNodes 1`
  stops building circuits when the guard's descriptor changes (lost W38, W39
  and a site screen). The old torrc is kept on the server as
  `torrc.pinned-2026-09-28`.
- **Open-world pools are deduplicated on where a site lands**, not on its
  domain: `github.io` redirects to a monitored site and would have entered the
  open world as unmonitored.
- **Splits are chronological, never random stratified**, and open-world results
  are TPR/FPR, not accuracy.
- Never commit captures, extracted features, or bridge lines. `tor_dataset/` is
  ignored; bridge lines live outside the repository on both hosts.
