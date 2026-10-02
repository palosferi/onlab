# Thesis status

Last updated 2026-10-02. Update this file whenever a run starts or finishes, a
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

**Full Tor Browser collection `tb-main`** on fujitsu, started 2026-09-28 23:20
as systemd user unit `wf-tbmain` (restarts on failure, resumes from the
manifest). At 2026-10-02 it had finished batch 38 of 40.

- 50 monitored sites x 40 visits x 3 arms, 500 unmonitored x 3 arms, 50 x 20
  background-tab visits (baseline): 8500 captures.
- Output: `~/onlab/tor_dataset/tb/tb-main/<arm>/<kind>/*.pcap`, manifest at
  `.../tb-main/manifest.csv`, log `~/onlab/logs/tb-main.log`. Feature
  extraction runs automatically at the end into `.../tb-main/features/`.
- Status at batch 38: 7243 usable. Failures are concentrated on a few sites:
  gnu (179 browser network errors), reuters (137 blocked), medium (125),
  vimeo (124), imdb (97 blocked + 40 empty), quora (71), apnews (62).

Check it with:
`ssh palos@100.102.24.16 'systemctl --user is-active wf-tbmain; tail ~/onlab/logs/tb-main.log'`
(from the Fedora laptop or a Codespace only, see CLAUDE.md).

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

## Next tasks

1. When `wf-tbmain` finishes: copy `tor_dataset/tb/tb-main/features/` and the
   manifest to an analysis machine (the features, not the pcaps).
2. Decide which monitored sites to keep: drop or report separately those with
   too few usable visits per arm (see the list above). Count ok visits per
   site x arm from the manifest first.
3. `python src/sequence_budget.py --collection tb-main --unit cells` to confirm
   length 10000 on Tor Browser traffic.
4. `python src/evaluate_transport.py --collection tb-main --mode cells --length 10000`
   (several hours; run on a laptop, not the server).
5. Draft the feladatkiírás from the proven scope and send it to the konzulens
   before 2026-10-09.
6. Afterwards: re-enable or retire the drift timer (`tor-wf-round.timer`), and
   delete the Azure resource group `wf-thesis` when the obfs4 arm is no longer
   needed.
