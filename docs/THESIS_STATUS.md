# Thesis status

Last updated 2026-10-24. Update this file whenever a run starts or finishes, a
result comes in, or a task is done.

## Topic and deadlines

Topic posted on the BME portal by the konzulens (Dr. Sonkoly Balázs, TMIT):
*Mélytanulás-alapú weboldal-ujjlenyomatozás általánosítása a Tor hálózaton* /
*Generalization of Deep Learning-Based Website Fingerprinting on the Tor Network*.

| Date | What | Who |
|---|---|---|
| 2026-10-05 – 10-16 | adatlap on the portal (done 2026-10-13; no konzulens approval needed, no nyilatkozat or melléklet) | student |
| 2026-10-09 24:00 | feladatkiírás upload (done in time) | konzulens |
| 2026-10-19 | feladatkiírás hitelesítés | |
| 2026-12-11 12:00 | thesis submission | student |

No *elbocsátó-befogadó nyilatkozat* is needed (confirmed 2026-10-14).

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

**DF on tb-main: done 2026-10-12 10:38**, seeds 42/43/44, on the Fedora laptop
(torch 2.14.0+cpu, 12 threads): `python src/evaluate_transport.py --collection
tb-main --mode cells --length 10000 --min-visits 30 --threads 12`, output
`figures/metrics_transport_tb-main_cells.json`, log `logs/transport_tb-main.log`.
A first run was killed by a power loss during seed 44 (log kept as
`logs/transport_tb-main_killed-2026-10-10.log`); the rerun reproduced its seed
42 exactly. Accuracy, mean ± sd over 3 seeds, 43 classes:

| | |
|---|---|
| closed world baseline / obfs4 / Snowflake | 94.9 ± 0.7 / 93.4 ± 1.0 / 92.2 ± 0.7 |
| baseline -> obfs4 / -> Snowflake | 53.1 ± 1.5 / 63.1 ± 1.2 |
| obfs4 -> baseline / -> Snowflake | 49.2 ± 3.5 / 70.4 ± 2.6 |
| Snowflake -> baseline / -> obfs4 | 89.0 ± 0.9 / 63.3 ± 8.9 |
| leave-one-out: held-out baseline / obfs4 / Snowflake | 79.3 ± 3.3 / 52.4 ± 4.0 / 80.4 ± 1.1 |
| pooled on baseline / obfs4 / Snowflake | 95.2 ± 0.9 / 97.2 ± 0.6 / 93.5 ± 0.6 |
| background test: clean / background-aware model | 17.6 ± 1.6 / 30.1 ± 0.7 |
| open world TPR / FPR (argmax), baseline | 91.8 ± 0.5 / 34.1 ± 6.2 |
| open world TPR / FPR, obfs4 | 91.2 ± 0.6 / 32.9 ± 5.1 |
| open world TPR / FPR, Snowflake | 87.7 ± 3.2 / 41.6 ± 3.6 |

Open-world FPR is at the argmax threshold over ~85 unmonitored test traces per
arm; a threshold sweep (R3, `evaluate_budget.py --precision-from`) is the
quotable form.

**S2 screening started 2026-10-11 09:19** on the Debian server as user unit
`wf-screen-s2`: `build_site_lists.py --screen-only` over the first 6000 of
Tranco Y8YJG (`logs/tranco_Y8YJG_top6000.csv`, 5647 domains after the adult
filter and registrable dedup), through the baseline arm, 16 workers. Log
`logs/screen_s2.log`, output `scripts/collection/sites/screening_s2.csv`, written
only at the end. `--screen-only` keeps the tb-main monitored domains among the
candidates and builds no pools; `sites/screening.csv` (tb-main's screen, ranks
up to 2822) is left as it was. **Finished 10:36**: 2226 of 5647 passed (3206
http errors/timeouts, 171 too small, 33 challenge pages, 11 not HTML); 1985
distinct sites after deduplicating on where they land, so the 50th lands at rank
142 and 650 sites reach rank ~1900. The raw top 50 is full of infrastructure
domains whose landing page is a corporate site (cloudflare, fastly, gandi,
digicert, ntp.org, nginx.org, sentry, b-cdn, criteo, cloudflare-dns), so the
monitored set needs a stated filtering rule. **Rule decided and S2 built
2026-10-11** (`docs/SITE_SETS.md`, "S2 selection rule"): `sites/s2/` holds 50
monitored (ranks 1-220), 15 reserves, 500 unmonitored and 100 background, and
`dropped.csv` for the appendix. `collect_tb.py --site-set s2` collects it.

**S2 pilot started 2026-10-11 11:27** on the Debian server as user unit
`wf-s2-pilot`: `run_tb.sh --collection s2-pilot --site-set s2-pilot --repeats 3
--bg-repeats 0 --unmonitored-limit 5`, i.e. the 50 monitored plus 15 reserves,
3 visits on each of the three arms (600 captures, ~7 h), log
`logs/s2-pilot.log`. A monitored site that fails in Tor Browser on any arm goes
into `--pilot-failed` and the next reserve moves up; then rebuild `sites/s2/`
and start the full S2 run (`--site-set s2 --repeats 40 --bg-repeats 20`).
**Pilot finished 2026-10-11 17:47**: 491 of 585 monitored visits ok. Ten sites
failed only because they redirect (twitter, office, skype, zoom, t.me,
blogspot, amazonvideo, kaspersky, discord.gg, vk); `s2-pilot-b` (17:48-18:26,
`--only` those ten, listed at their landed URL since efe1c11) passes nine with
at least 2/3 visits on every arm. Still failing: **vk.com** (vk.ru / neterror,
1/9), **snapchat.com** (403 challenge page, 2/9), and **pinterest.com** is weak
(1/3 baseline, navigation timeouts). All 15 reserves passed 3/3.

**S2 rebuilt 2026-10-13** with `--pilot-failed vk.com,snapchat.com,pinterest.com`
(pinterest because 1/3 on baseline would leave it under 30 of 40 visits): the
reserves endpoints.news, ubuntu.com and twitch.tv move up, monitored ranks now
1-228, `dropped.csv` 57 rows. kaspersky.com is listed at kaspersky.fr and some
exits are sent to usa.kaspersky.com (8/9 ok in the pilot); check its count
after the run.

**Full S2 started 2026-10-13 09:34** on the Debian server as user unit `wf-s2`:
`run_tb.sh --collection s2 --site-set s2 --repeats 40 --bg-repeats 20`, 8500
captures (50 x 40 x 3 monitored, 500 x 3 unmonitored, 20 x 50 background on
baseline), ~94 h at 40 s each, so it should end around 2026-10-17. Log
`logs/s2.log` (the unit's stdout via `-p StandardOutput=append:`). Re-running
the same command resumes from the manifest.

**S2 incidents, 2026-10-14/15.** (1) From 2026-10-14 19:25 every browser
launch failed: Selenium leaked an ~830 KB profile copy into /tmp per visit and
the 3.9 GB tmpfs filled (5070 copies). Fixed in db1f010 (copies go to
`~/tor_wf_runtime/profile_tmp`, emptied before each launch), leaked copies
removed. The 206 launch failures used up their visits' attempts, so they were
moved to `tor_dataset/tb/s2/manifest_excluded_tmpfull_2026-10-14.csv` (full
manifest backed up as `manifest.csv.bak-2026-10-15`) and the run restarted
2026-10-15 ~10:35; retried visits come about a day after the rest of their
batch. (2) **The obfs4 bridge (Azure, `wfthesis`) has been down since
2026-10-14 21:57**: Tor marks the guard down and its address refuses TCP. Cause: the
Azure for Students credit expired and Microsoft disabled the subscription
(email 2026-10-14 22:34); its data is kept until 2027-01-17, and upgrading to
pay-as-you-go resumes it. It was the 12-month term, not the money: the subscription had
been created 2025-10-14 (an earlier course), so it expired on its first
anniversary with 6.57 EUR used. **Renewed by the user 2026-10-15** (same
subscription, new 12 months and $100); the VM restarted ~23:15 at the same
static IP (Standard SKU), Tor marks the bridge up again. The 3 obfs4 rows
written while it was down were set aside in
`manifest_excluded_bridgedown_2026-10-15.csv` (backup `manifest.csv.bak-2026-10-15b`)
and **S2 resumed 2026-10-15 ~23:20**, ~50 h to go, so it should end around
2026-10-18 late. The
run was paused (`wf-s2` stopped ~10:45) so obfs4 visits would not burn their
attempts and the arms stayed interleaved. Before resuming: bring the bridge back
(if the VM was deallocated its public IP may have changed, and the bridge line
on the server with it), set aside the 3 obfs4 `load_error` rows written after
the restart the same way, then rerun the same `run_tb.sh` command. 4011 ok
visits so far, 4474 to go (~50 h).

**S2 finished 2026-10-17 13:04** (features extracted by the same run, 2.2 GB
under `tor_dataset/tb/s2/features`; copied to the Fedora laptop 2026-10-18,
all 8385 ok captures). Ok
visits: monitored 1988 baseline / 1985 obfs4 / 1985 Snowflake, unmonitored
478 / 481 / 478, background 990. All 50 monitored sites have at least 33 ok
visits on every arm (lowest: base.org 33 on obfs4, skype and soundcloud 36), so
`--min-visits 30` keeps all 50; kaspersky.com came out at 39/40/39. Snowflake
had the most load timeouts (41 monitored).

**S2 DF on Kaggle: done 2026-10-19** at cbf7fd3, dataset `ferencpalos/wf-tb-s2`
(private, features only), Tesla T4, torch 2.11.0+cu128: `wf-s2-transport`
(evaluate_transport.py, 10947 s, `figures/metrics_transport_s2_cells_gpu.json`)
and `wf-s2-budget` (evaluate_budget.py, 15182 s,
`figures/metrics_budget_s2_cells_gpu.json` and its `_precision.json`). The first
push (version 1) went out before the dataset had finished processing and ran
without it. `--min-visits 30` kept all 50 sites. Mean ± sd over seeds 42/43/44,
50 classes, tb-main (GPU, 43 classes) beside it; the table is
`src/compare_collections.py` on the two transport files:

| | S2 | tb-main |
|---|---|---|
| closed world baseline / obfs4 / Snowflake | 97.7 ± 0.6 / 96.7 ± 0.8 / 92.0 ± 1.3 | 94.6 / 92.8 / 92.5 |
| baseline -> obfs4 / -> Snowflake | 39.2 ± 0.8 / 42.4 ± 2.1 | 55.2 / 64.8 |
| obfs4 -> baseline / -> Snowflake | 47.4 ± 4.6 / 51.0 ± 1.5 | 45.5 / 68.5 |
| Snowflake -> baseline / -> obfs4 | 71.4 ± 2.3 / 41.8 ± 3.8 | 88.5 / 61.3 |
| leave-one-out: held-out baseline / obfs4 / Snowflake | 73.4 ± 4.8 / 42.9 ± 9.0 / 54.9 ± 1.0 | 77.4 / 52.6 / 77.5 |
| pooled on baseline / obfs4 / Snowflake | 97.2 ± 0.1 / 98.5 ± 0.3 / 94.0 ± 0.5 | 94.6 / 96.8 / 92.7 |
| background test: clean / background-aware model | 16.0 ± 0.9 / 19.8 ± 1.0 | 19.0 / 28.2 |
| open world TPR / FPR (argmax), baseline | 94.3 ± 1.9 / 34.7 ± 1.2 | 92.8 / 30.6 |
| open world TPR / FPR, obfs4 | 92.7 ± 1.1 / 34.0 ± 5.3 | 89.5 / 32.5 |
| open world TPR / FPR, Snowflake | 89.1 ± 1.0 / 49.3 ± 7.4 | 88.7 / 42.0 |

So on the popular site set DF is at least as good within a transport and much
worse across them: zero-shot transfer drops 16-22 points in every direction
except obfs4 -> baseline (unchanged, already low), and Snowflake -> baseline is
no longer close to free (71 vs 88.5). Open-world FPR is over 96 unmonitored
test traces per arm.

| R1 on S2: baseline + k target traces per site | -> obfs4 | -> Snowflake |
|---|---|---|
| k = 0 (zero shot) | 39.2 ± 0.8 | 42.4 ± 2.1 |
| k = 1 | 79.7 ± 1.2 | 47.7 ± 2.9 |
| k = 3 | 88.8 ± 1.6 | 55.3 ± 1.8 |
| k = 5 | 91.9 ± 0.6 | 64.0 ± 0.4 |
| k = 10 | 93.8 ± 0.5 | 72.1 ± 1.1 |
| k = all (pair pooled) | 97.4 ± 0.5 | 92.9 ± 0.2 |

obfs4 recovers with a handful of target traces as on tb-main; Snowflake needs
far more than on tb-main (10 traces give 72 vs 81).

| R2 on S2: open world, baseline, k monitored sites | TPR | FPR, unseen pool | FPR, demoted sites | best precision at b = 1% |
|---|---|---|---|---|
| 5 | 90.0 ± 11.5 | 0.3 | 0.5 | 82.4% |
| 10 | 87.8 ± 0.8 | 4.5 | 0.8 | 38.6% |
| 20 | 94.5 ± 3.2 | 16.0 | 1.3 | 46.6% |
| 50 (all) | 94.3 ± 1.9 | 34.7 | - | 17.6% |

R4 on S2, baseline closed world on n training traces per site: 5 -> 80.7 ±
1.4, 10 -> 91.0 ± 0.9, 15 -> 94.0 ± 0.4, 20 -> 95.6 ± 1.5, all -> 97.7 ± 0.6.

**R3 on S2 done 2026-10-21** (`figures/metrics_transport_s2_cells_gpu_precision.json`):
best precision with TPR >= 50% at base rate 5% / 1% / 0.1% is 50.9 / 17.6 ±
8.0 / 2.1% baseline, 53.7 / 18.3 ± 2.0 / 2.2% obfs4, 53.8 / 21.2 ± 15.2 / 2.9%
Snowflake (upper bounds, WPR taken as 0), about twice tb-main's but with a wide
spread. **Caveat for both collections:** the sweep in `evaluate_transport.py`
stops at threshold 0.95, and at b = 1% the best point sits at 0.95 in 16 of
the 18 seed-arm runs of the two collections, with TPR still 57-80%, so a stricter threshold would likely give more
precision; and one false positive in ~96 pool traces is already ~1% FPR. Both
limits come from the run (confidences are not saved), so a finer grid needs a
rerun on Kaggle.

**Open-world rerun, decided 2026-10-21.** Since then `evaluate_transport.py`
and `evaluate_budget.py` save every open-world test trace's confidence (taken in
float64, as float32 rounds a confident model's top scores to 1.0), sweep finer
above 0.95, and record WPR; `--precision-from` tries every distinct confidence as
a threshold and records the FPR at the best point (`best_fprs`: 0 means the
precision is only as high as the ~96-trace pool can show). Training is
unchanged, so the threshold-0 TPR/FPR of the rerun must equal the old files
exactly: that is the check. Jobs, per collection, only the open-world parts:
`evaluate_transport.py --scenarios open_world --out
figures/metrics_openworld_<c>_cells_gpu.json` and `evaluate_budget.py
--scenarios set_size --out figures/metrics_setsize_<c>_cells_gpu.json`.
`thesis_figures.py` takes R3 from `metrics_openworld_<c>_cells_gpu_precision.json`
when it exists. Until then the R3 numbers above and the R2 precision column
are provisional: held down by the grid, held up by WPR taken as 0.

**Open-world rerun done 2026-10-24** (Kaggle `wf-s2-openworld` and
`wf-tbmain-openworld` at fb4e34b, T4). Check passed: all 24 runs repeat the old
ones exactly (best epoch, threshold-0 TPR/FPR, TPR/FPR at every old grid
point). Files `figures/metrics_{openworld,setsize}_{s2,tb-main}_cells_gpu.json`
and their `_precision.json`. The grid was holding precision down, but with
exact thresholds the best point at b = 1% is decided by whether 0, 1 or 2 of
the 85-96 pool traces get through, so the seeds scatter over almost the whole
range:

| best precision, b = 1% (per seed: FPR at the best point) | S2 | tb-main |
|---|---|---|
| open world baseline | 50.7 ± 42.7 (2.1, 2.1, 0%) | 25.1 ± 10.1 (1.2, 1.2, 3.5%) |
| open world obfs4 | 100 ± 0 (0, 0, 0%) | 69.8 ± 52.3 (5.9, 0, 0%) |
| open world Snowflake | 43.6 ± 49.0 (0, 2.1, 4.2%) | 9.9 ± 0.4 (4.7, 5.9, 4.7%) |

A pool this size can only place precision at b = 1% between ~35% (one false
positive) and 100% (none). **Undecided: how R3 is reported** (see Next tasks);
the R3 lines and the R2 precision column above, and the precision figures, are
not updated until then.

Thesis figures for S2 written 2026-10-21 by `src/thesis_figures.py --collection
s2`: `figures/thesis/s2_{cross_matrix,budget,curve,precision}.pdf`.

**Baseline guard, checked 2026-10-21** on the server manifests: tb-main and S2
both went through `F5612A75` (S2: 3551 of 3553 baseline rows, the other 2 via
`1F322949` right after the 10-15 restart). `8C7A9811` was used only by
`tb-pilot-debian` (10-10) and `s2-pilot-b` (10-11 evening), with `s2-pilot`
(10-11 morning) on `F5612A75` in between: the unpinned Tor moves between its
primary guards, so the guard is a per-capture fact from the manifest, not a
setting.

**R3 done 2026-10-13** on the final tb-main results
(`figures/metrics_transport_tb-main_cells_precision.json`): best precision with
TPR >= 50% at base rate 5% / 1% / 0.1% is 29.2 / 7.3 / 0.8% baseline, 34.1 /
9.1 / 1.0% obfs4, 28.1 / 7.0 / 0.7% Snowflake (WPR not recorded by
evaluate_transport.py, taken as 0, so these are upper bounds).

**R1, R2, R4 not yet run:** `src/evaluate_budget.py` (budget = R1, set_size =
R2, curve = R4), smoke-tested on 4 sites. Run on the Fedora laptop:
`python src/evaluate_budget.py --collection tb-main --min-visits 30 --threads 12`.
Default grid is 16 trainings per seed, ~26k training traces at ~95 epochs; at
the ~20 ms per trace-epoch of tb-main seed 44 that is ~14 h per seed, ~2 days
for three seeds on an idle laptop (seed 43, with the laptop in use, ran 65%
slower).

**Plan since 2026-10-13: all thesis DF runs on a Kaggle T4** (`scripts/kaggle/`,
`--device cuda`): first tb-main seed 42 as a timing and agreement check against
the CPU run above, then tb-main transport + R1/R2/R4 for three seeds, later S2.
The CPU tb-main results stay as a cross-hardware reproduction. Waiting on: the
Kaggle CLI login on the laptop (`kaggle auth login`, CLI in
`~/.local/share/kaggle-cli`) and the features uploaded as a private dataset
(steps in `scripts/kaggle/push_job.py`).

**Kaggle check done 2026-10-13**: dataset `ferencpalos/wf-tb-main` (private),
kernel `wf-tbmain-transport-s42` at 40c5d2a ran tb-main seed 42 in 53 min on a
Tesla T4 (torch 2.11.0+cu128), against 10-16 h per seed on the laptop CPU.
`figures/metrics_transport_tb-main_cells_gpu_s42.json`. Its numbers differ from
CPU seed 42 by about as much as CPU seeds differ from each other (e.g. closed
world 94.1 / 94.1 / 92.6 vs 94.9 ± 0.7 / 93.4 ± 1.0 / 92.2 ± 0.7; leave-out
obfs4 46.0 vs 52.4 ± 4.0), and every qualitative finding is the same. So same
seed on other hardware is just another draw: no table may mix the two. Next:
tb-main transport (3 seeds, ~3 h) and R1/R2/R4 (3 seeds, ~6 h est.) as two
Kaggle jobs. Pushing to Kaggle needs the user (the permission classifier blocks
it for Claude).

**Pushed 2026-10-13 ~14:30** at affd553: `wf-tbmain-budget` (evaluate_budget.py,
3 seeds, out `figures/metrics_budget_tb-main_cells_gpu.json`) running, and
`wf-tbmain-transport` (evaluate_transport.py, 3 seeds, out
`figures/metrics_transport_tb-main_cells_gpu.json`) queued behind it. Fetch with
`kaggle kernels output ferencpalos/<slug> -p <dir>`.

**tb-main transport on Kaggle: done 2026-10-13 21:22** (9537 s on a T4, at
affd553), `figures/metrics_transport_tb-main_cells_gpu.json`. Its seed 42 equals
the `wf-tbmain-transport-s42` test in all 20 scenarios, so GPU runs repeat.
**These are the thesis numbers** (mean ± sd over seeds 42/43/44, 43 classes):

| | |
|---|---|
| closed world baseline / obfs4 / Snowflake | 94.6 ± 0.9 / 92.8 ± 1.3 / 92.5 ± 0.3 |
| baseline -> obfs4 / -> Snowflake | 55.2 ± 5.3 / 64.8 ± 2.0 |
| obfs4 -> baseline / -> Snowflake | 45.5 ± 5.8 / 68.5 ± 4.9 |
| Snowflake -> baseline / -> obfs4 | 88.5 ± 1.0 / 61.3 ± 6.6 |
| leave-one-out: held-out baseline / obfs4 / Snowflake | 77.4 ± 2.8 / 52.6 ± 6.3 / 77.5 ± 0.6 |
| pooled on baseline / obfs4 / Snowflake | 94.6 ± 0.2 / 96.8 ± 1.0 / 92.7 ± 0.8 |
| background test: clean / background-aware model | 19.0 ± 1.2 / 28.2 ± 1.7 |
| open world TPR / FPR (argmax), baseline | 92.8 ± 0.9 / 30.6 ± 2.4 |
| open world TPR / FPR, obfs4 | 89.5 ± 1.2 / 32.5 ± 5.6 |
| open world TPR / FPR, Snowflake | 88.7 ± 1.3 / 42.0 ± 3.8 |

Every row agrees with the CPU run above within the seed spread; the CPU run is
the cross-hardware reproduction. R3 from it
(`figures/metrics_transport_tb-main_cells_gpu_precision.json`): best precision
with TPR >= 50% at base rate 5% / 1% / 0.1% is 33.1 / 8.7 / 0.9% baseline, 41.0
/ 12.1 / 1.4% obfs4, 34.1 / 9.0 / 1.0% Snowflake (upper bounds, WPR taken as 0).

**R1/R2/R4 on Kaggle: done 2026-10-14 08:56** (13523 s on a T4; the kernel
showed RUNNING while it waited for a GPU, so wall time is no guide),
`figures/metrics_budget_tb-main_cells_gpu.json`. Mean ± sd over 3 seeds. Its
k = 0, n = all and k = all points equal the transport run above exactly.

| R1: baseline + k target traces per site | -> obfs4 | -> Snowflake |
|---|---|---|
| k = 0 (zero shot) | 55.2 ± 5.3 | 64.8 ± 2.0 |
| k = 1 | 82.3 ± 1.1 | 65.5 ± 1.5 |
| k = 3 | 86.5 ± 1.3 | 73.2 ± 0.8 |
| k = 5 | 86.6 ± 0.5 | 74.0 ± 2.2 |
| k = 10 | 89.6 ± 1.1 | 81.4 ± 1.1 |
| k = all (pair pooled) | 94.3 ± 0.5 | 94.2 ± 0.3 |

| R2: open world, baseline, k monitored sites | TPR | FPR, unseen pool | FPR, demoted sites | best precision at b = 1% |
|---|---|---|---|---|
| 5 | 86.3 ± 5.9 | 3.5 | 0.3 | 100% (0 false positives in ~85 pool traces: FPR < ~3.5%) |
| 10 | 93.3 ± 1.9 | 8.6 | 1.7 | 19.2% |
| 20 | 93.1 ± 3.1 | 19.6 | 3.1 | 19.2% |
| 43 (all) | 92.8 ± 0.9 | 30.6 | - | 8.7% |

R4, baseline closed world on n training traces per site: 5 -> 73.1 ± 2.3,
10 -> 83.0 ± 1.0, 15 -> 89.0 ± 0.5, 20 -> 89.7 ± 1.1, all (~28) -> 94.6 ± 0.9;
still rising at the end, so more traces per site would still help.

Precision (R3) now uses the FPR on the held-out unmonitored pool when a sweep
records it (`fpr_pool`): the summary's pooled FPR is pulled down by the demoted
sites, which the model saw in training, and gave 46/38% at b = 1% for 10/20
sites before the fix.

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

- DF runs live on the Fedora laptop and write their JSON only when they end
  (a `.partial.json` after each seed). If one was killed, rerun the same
  command; it resumes after the last finished seed.
- Features are not in git. From a Fedora laptop or Codespace:
  `rsync -a ferencpalos@100.102.24.16:onlab/tor_dataset/tb/tb-main/{features,manifest.csv} tor_dataset/tb/tb-main/`.
  The Windows laptop must not reach the server: move the folder there by other
  means (disk or cloud drive). The manifest holds the obfs4 bridge IP; all of `tor_dataset/` is git-ignored
  for that reason (so are `run_*.json` and `round_meta.json`).
- `git pull` on `szakdolgozat` (renamed from `szakdolgozat-drift` on 2026-10-13: `git fetch origin; git branch -m szakdolgozat-drift szakdolgozat; git branch -u origin/szakdolgozat szakdolgozat`); `.venv` needs torch, pandas, scikit-learn.

## Next tasks

Rewritten 2026-10-21. Done before that: tb-main and S2 collected, DF transport
+ R1/R2/R4 on both on Kaggle, R3 and figures on both, admin (feladatkiírás,
adatlap; the 10-19 hitelesítés needed nothing from the student).

1. Push `szakdolgozat` after each commit (user: Claude's `git push` has no
   GitHub credentials).
2. **Decide how R3 is reported** (open-world rerun done 2026-10-24, see
   above): the ~96-trace unmonitored pool cannot resolve precision at b = 1%.
   Then update the R3 lines, the R2 precision column and the precision figures
   (`thesis_figures.py` already reads `metrics_openworld_*_precision.json`).
3. **Write the thesis** (submission 2026-12-11 12:00). Nothing written yet;
   `docs/SZAKDOLGOZAT_temavazlat.md` still describes the drift plan, and its
   schedule had a zeroth draft to the konzulens by the end of November. Points
   the text must carry:
   - S2 vs tb-main differs in more than the site set: S2 ran on the Debian
     server with tor 0.4.9.12, tb-main on Mint with 0.4.8.10 (same baseline
     guard). The cross-transport drop between them cannot be put on the site
     set alone.
   - R3's "best threshold" is chosen on the test set (an oracle operating
     point), and the ~96-trace unmonitored pool cannot show an FPR below ~1%.
   - Guard, tor version and bridge are per capture in the manifest; the bridge
     address never goes into the thesis or the repository.
4. **Status email to the konzulens** with the final results (user's call;
   the 2026-10-14 draft predates S2, so it needs the S2 tables).
5. **Delete the Azure resource group `wf-thesis`** once no site needs
   recollecting (the obfs4 arm depends on that bridge; the renewed
   subscription runs to 2027-10). `az` is not installed on the laptop.

Collection host: Debian 13 since 2026-10-10 (user `ferencpalos`, same
Tailscale IP), rebuilt with `scripts/collection/setup_host.sh`. Arms run the
Expert Bundle tor 0.4.9.12 (tb-main: Mint's 0.4.8.10); the baseline guard
varies (see "Baseline guard, checked 2026-10-21"). Backups: laptop
`~/fujitsu-backup-2026-10-10/home`, full archive on the WD Elements drive, old
system unpacked on the server at `/srv/mint-old`. The drift timer stays off.
