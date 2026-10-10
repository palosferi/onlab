# Thesis status

Last updated 2026-10-10. Update this file whenever a run starts or finishes, a
result comes in, or a task is done.

## Topic and deadlines

Topic posted on the BME portal by the konzulens (Dr. Sonkoly Balázs, TMIT):
*Mélytanulás-alapú weboldal-ujjlenyomatozás általánosítása a Tor hálózaton* /
*Generalization of Deep Learning-Based Website Fingerprinting on the Tor Network*.

| Date | What | Who |
|---|---|---|
| 2026-10-05 – 10-16 | adatlap on the portal | student |
| 2026-10-09 24:00 | feladatkiírás upload | konzulens (student sends him a draft first) |
| 2026-10-19 | feladatkiírás hitelesítés | |
| 2026-12-11 12:00 | thesis submission | student |

The portal also lists an *elbocsátó-befogadó nyilatkozat* awaiting upload; it
is only needed if the specialization is not at TMIT. Ask the konzulens.

## Scope

DF (Deep Fingerprinting) evaluated within and across three Tor transports
(baseline, obfs4, Snowflake), on traffic from the real Tor Browser. Plus an
open world (50 monitored / 500 unmonitored) and a background-tab ablation on
the baseline arm. Temporal drift is deferred; the weekly drift rounds are off.

## Running now

**Collection `tb-main` finished 2026-10-02 15:50** (unit `wf-tbmain` inactive).
Usable captures, from the manifest: monitored 1755 baseline / 1739 obfs4 / 1774
Snowflake, unmonitored 424 / 427 / 427, background 875 (baseline). Features
(1.8 GB) and manifest were copied to the Fedora laptop under
`tor_dataset/tb/tb-main/` on 2026-10-08.

Sites with fewer than 30 ok visits in some arm are dropped by
`--min-visits 30`: vimeo, medium, gnu, apnews, quora (reuters and imdb have
no usable visits at all). 43 of 50 monitored sites remain, 33+ visits each.
`sequence_budget --unit cells`: median trace is 7.5-8.5k cells, so length
10000 holds the whole median trace and about 60% of traces completely.

**DF on tb-main** started 2026-10-08 on the Fedora laptop:
`python src/evaluate_transport.py --collection tb-main --mode cells --length 10000
--min-visits 30 --threads 12`, log `logs/transport_tb-main.log`, output
`figures/metrics_transport_tb-main_cells.json`. **Killed 2026-10-10 ~06:00**
(the laptop lost power) during seed 44, so no JSON was written. Seeds 42 and 43
finished in the log, at about 13.5 h per seed. Accuracies from the log, mean ±
sd over those two seeds only:

| | |
|---|---|
| closed world baseline / obfs4 / Snowflake | 95.3 ± 0.4 / 93.9 ± 0.6 / 92.5 ± 0.6 |
| baseline -> obfs4 / -> Snowflake | 52.7 ± 1.9 / 63.8 ± 0.0 |
| obfs4 -> baseline / -> Snowflake | 49.9 ± 4.6 / 70.7 ± 3.5 |
| Snowflake -> baseline / -> obfs4 | 89.0 ± 1.3 / 62.5 ± 12.4 |
| leave-one-out: held-out baseline / obfs4 / Snowflake | 78.0 ± 3.4 / 50.1 ± 0.4 / 80.0 ± 1.2 |
| pooled on baseline / obfs4 / Snowflake | 95.2 ± 1.3 / 97.3 ± 0.8 / 93.5 ± 0.8 |
| background test: clean / background-aware model | 18.2 ± 1.7 / 30.3 ± 0.8 |
| open world TPR / FPR (argmax), baseline | 91.8 ± 0.6 / 32.9 ± 8.3 |
| open world TPR / FPR, obfs4 | 91.5 ± 0.2 / 34.7 ± 5.8 |
| open world TPR / FPR, Snowflake | 88.5 ± 4.2 / 43.5 ± 1.7 |

These need a rerun before they are quotable. That log is kept as
`logs/transport_tb-main_killed-2026-10-10.log`.

**Rerun started 2026-10-10 15:45** on the Fedora laptop as user unit
`wf-df-tbmain` (same command, wrapped in `systemd-inhibit`), log
`logs/transport_tb-main.log`, ~40 h. `evaluate_transport.py` now saves after
every seed to `figures/metrics_transport_tb-main_cells.partial.json`; if the run
dies, start the same command again and it skips the finished seeds.

The drift timer `tor-wf-round.timer` is disabled and stays off (drift deferred).

## Results so far (spring dataset, Chromium, 3 seeds, mean ± sd)

Same machine and thread count for both rows (`figures/metrics_df_cells.json`,
`figures/metrics_df_direction_laptop.json`):

| | direction | cells |
|---|---|---|
| closed world, baseline | 84.5 ± 2.0 | 87.7 ± 2.1 |
| closed world, obfs4 | 73.0 ± 1.3 | 79.5 ± 3.7 |
| zero-shot baseline -> obfs4 | 15.2 ± 1.9 | 17.6 ± 1.8 |
| open world TPR | 84.5 | 87.0 |

The cross-transport gap survives the change of representation, so it is not a
packetisation artefact. Spring open-world FPR rests on 13 unmonitored test
traces and is not meaningful; `tb-main` fixes that.

## Picking this up on another machine

- The DF run above lives on the Fedora laptop and writes
  `figures/metrics_transport_tb-main_cells.json` only when it ends. Check there
  first (`systemctl --user status wf-df-tbmain`, `tail logs/transport_tb-main.log`).
  If it was killed, rerun the same command; it resumes after the last finished seed.
- Features are not in git. From a Fedora laptop or Codespace:
  `rsync -a ferencpalos@100.102.24.16:onlab/tor_dataset/tb/tb-main/{features,manifest.csv} tor_dataset/tb/tb-main/`.
  The Windows laptop must not reach the server: move the folder there by other
  means (disk or cloud drive). The manifest holds the obfs4 bridge IP; do not commit it.
- `git pull` on `szakdolgozat-drift`; `.venv` needs torch, pandas, scikit-learn.

## Next tasks

1. Collect the DF results, write them up as mean ± sd and check the open-world FPR.
2. Feladatkiírás sent to the konzulens on 2026-10-09 (text as in
   `docs/FELADATKIIRAS.md`, background in `docs/FELADATKIIRAS_NOTES.md`); he
   uploads it. Fill in the adatlap by 2026-10-16.
3. Site sets, decided 2026-10-10 (`docs/SITE_SETS.md`): collect **S2**, a
   screened popular top-50 monitored set (coverage threat model) with a fresh
   500 unmonitored from further down Tranco, as the main dataset; tb-main stays
   as a second, targeted site set. Add R1-R4: target-transport data budget,
   monitored-set size, precision at realistic base rates, learning curve.
   Email to the konzulens drafted 2026-10-10.
4. Collection host moved from Mint to Debian 13 on 2026-10-10 (user
   `ferencpalos`, same Tailscale IP). Rebuilt with
   `scripts/collection/setup_host.sh`; pilot `tb-pilot-debian` 30/30 ok on all
   three arms. Arms now run the Expert Bundle tor 0.4.9.12 (tb-main: Mint's
   0.4.8.10) and the baseline guard is now Intrepid `8C7A9811` (tb-main:
   `F5612A75`); both are recorded per capture in the manifest. Backups: laptop
   `~/fujitsu-backup-2026-10-10/home`, full archive on the WD Elements drive,
   old system unpacked on the server at `/srv/mint-old`.
5. Delete the Azure resource group `wf-thesis` once no site needs recollecting
   (the obfs4 arm depends on that bridge). `az` is not installed on the laptop.
