# Notes behind the feladatkiírás draft

Background for `FELADATKIIRAS.md`: what each term means and why the draft is
phrased the way it is. The kiírás itself stays non-technical; these details
belong in the thesis.

## Deep Fingerprinting

Sirinam et al., CCS 2018. A 1D CNN (four convolutional blocks, two fully
connected layers) whose input is a visit as a sequence of +1/-1 values, one per
Tor cell, signed by direction; output is the visited site. 98% closed world on
undefended Tor in the paper. `src/df_model.py` follows the paper's
hyperparameters; the one adaptation is the input: we capture TCP/UDP packets,
not Tor cells, so `cells` mode turns bytes back into cell counts (514 bytes on
the wire per cell), length 10000. The thesis uses DF only; the önlab's
21-feature random forest and the `tiktok`/`iat` modes are not part of it.
Other attacks to cite: k-FP, CUMUL, Var-CNN, Tik-Tok.

## Why tbselenium, and why it is not named

tbselenium is a Selenium wrapper made for Tor Browser and the usual tool in WF
research; Playwright/Puppeteer do not support Tor Browser properly. The kiírás
says "a valódi Tor Browsert vezérelve": what matters is the real browser (the
spring data came from Chromium through a Tor proxy), not the library.

## Why obfs4 and Snowflake

Tor Browser ships obfs4, Snowflake, meek and WebTunnel. Snowflake and obfs4
are the most used (verify on Tor Metrics, "Bridge users by transport", before
citing). They disguise traffic differently: obfs4 makes bytes look random over
TCP to a fixed bridge; Snowflake runs over WebRTC (UDP) through volunteer
proxies that change during a visit. WebTunnel is a candidate for future work.
"Híd" is the term the Hungarian Tor Browser uses for bridge.

## Sites

- 50 monitored: the sites the attacker wants to recognize. The 36 spring
  sites minus Amazon (always a bot check) plus 15 new; international and
  Hungarian (wikipedia, github, bbc, reddit, telex, index, hvg, bme, 444, ...).
  `scripts/collection/sites.py`.
- 500 unmonitored: top of the Tranco ranking (pinned list Y8YJG), screened to
  sites that load over Tor and are real pages. They stand for "everything else".
- 100 background: a separate Tranco pool used only for the second tab.

## Second tab ("háttérfül" is not Hungarian; the draft says "második böngészőlap")

Baseline arm only. The collector opens a second tab with a random background
site, waits 0.5-3 s, then loads the monitored site in the first tab; the
capture holds both. 20 per monitored site, 875 usable. Tested two ways: the
model trained on clean visits scored on these, and a model trained with part of
them included.

## Evaluation terms

- Closed world: the user only visits the 50 sites; which one?
- Open world: the user may visit anything; flag monitored visits (TPR) without
  false alarms on the 500 (FPR). Accuracy is meaningless here.
- 3x3 cross-transport matrix: train on one connection type, test on each.
- Leave-one-out: train on two types, test on the unseen third.
- Pooled: train on all three together.
- Chronological split: test on each site's latest visits, so no future leaks
  into training.
- Seeds: the same training with different random starts; mean ± sd.

## Analysis terms

- Packet vs cell representation: Snowflake splits the same data into ~4x the
  packets and the OS merges TCP segments, so packets are not comparable across
  transports; 514-byte cells are (within 15%).
- Sequence length: DF reads a fixed number of entries; too short and it sees
  less of one transport, an artefact rather than a finding.
- Per-transport traffic characteristics: trace length, load time and similar.

All three check that the cross-transport drop is real. The kiírás says only
"a forgalom jellemzői alapján".

## Removed from the draft

- "Ez a cenzúrát megkerülő felhasználók esetében gyakorlati kérdés" became the
  sentence on who uses pluggable transports.
- "(pl. időbeli drift)": not needed; future work is a standard closing task.
- Selenium, TPR/FPR, leave-one-out, pooled, seeds, cells: too technical for a
  kiírás.
