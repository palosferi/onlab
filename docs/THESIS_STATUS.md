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
`figures/metrics_transport_tb-main_cells.json`. Results pending.

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
  first (`pgrep -f evaluate_transport`, `tail logs/transport_tb-main.log`). If it
  was killed, rerun the same command; it does not resume.
- Features are not in git. From a Fedora laptop or Codespace:
  `rsync -a palos@100.102.24.16:onlab/tor_dataset/tb/tb-main/{features,manifest.csv} tor_dataset/tb/tb-main/`.
  The Windows laptop must not reach the server: move the folder there by other
  means (disk or cloud drive). The manifest holds the obfs4 bridge IP; do not commit it.
- `git pull` on `szakdolgozat-drift`; `.venv` needs torch, pandas, scikit-learn.

## Next tasks

1. Collect the DF results, write them up as mean ± sd and check the open-world FPR.
2. Feladatkiírás sent to the konzulens on 2026-10-09 (text as in
   `docs/FELADATKIIRAS.md`, background in `docs/FELADATKIIRAS_NOTES.md`); he
   uploads it. Fill in the adatlap by 2026-10-16.
3. Decide the monitored/unmonitored site sets. The konzulens and the student had agreed
   on the 50 most popular sites; tb-main instead used the 35 spring sites plus
   15 new ones, and neither choice is settled. Options: keep tb-main and frame
   it as a targeted threat model (sensitive sites), or recollect a screened
   popular top-50 for a coverage threat model. Promised the konzulens a
   separate email on this on 2026-10-09, possibly asking his advice; not sent
   yet. The discussion so far is in `docs/SITE_SETS.md`.
4. Delete the Azure resource group `wf-thesis` once no site needs recollecting
   (the obfs4 arm depends on that bridge). `az` is not installed on the laptop.
