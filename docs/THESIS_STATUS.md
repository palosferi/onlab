# Thesis status

Last updated 2026-10-13. Update this file whenever a run starts or finishes, a
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
  means (disk or cloud drive). The manifest holds the obfs4 bridge IP; do not commit it.
- `git pull` on `szakdolgozat` (renamed from `szakdolgozat-drift` on 2026-10-13: `git fetch origin; git branch -m szakdolgozat-drift szakdolgozat; git branch -u origin/szakdolgozat szakdolgozat`); `.venv` needs torch, pandas, scikit-learn.

## Next tasks

1. Write up the tb-main DF results (done, in the table above) and run R1, R2, R4.
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
