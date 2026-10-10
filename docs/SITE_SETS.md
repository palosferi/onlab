# Monitored / unmonitored site sets

Decided 2026-10-10: **S2 + R1-R4** (see "Decision" at the end). Task 3 in
`THESIS_STATUS.md`. The email to the konzulens (Balázs), promised for
2026-10-09, was drafted on 2026-10-10.

## What happened

- About a month before, the student and Balázs agreed on the **50 most popular
  sites** as the monitored set.
- `tb-main` instead used the **35 spring önlab sites** (amazon dropped: it always
  got a bot check) **plus 15 new ones** added on 2026-09-27
  (`scripts/collection/sites.py`, comment "more news, more Hungarian, and the
  privacy sites a Tor user is plausibly monitored for visiting"). The reason
  for not using the top 50: the top of the popularity list is full of domains
  that are not real pages (CDNs, APIs, trackers) or don't load over Tor. The
  screened popular sites went into the 500 unmonitored instead.
- The student now thinks the top 50 isn't ideal either. Balázs has only been
  told that the sets might change (in the feladatkiírás email, 2026-10-09).
  He has not been told which sites were collected.

The 15 new sites:

| Group | Sites | Reason |
|---|---|---|
| Privacy / anti-censorship | torproject, eff, signal | What a Tor user is most plausibly watched for visiting |
| International news | npr, apnews, dw, spiegel | More news |
| Tech news | arstechnica, theverge | More news (goes with wired, techcrunch from spring) |
| Hungarian news | 444, 24hu, portfolio | More Hungarian |
| Open source / dev | python, kernel, archlinux | No reason recorded; probably more of the gnu/debian/mozilla category |

`--min-visits 30` drops apnews (new) and reuters, imdb, vimeo, medium, gnu,
quora (spring), leaving 43.

## The student's goal and the analysis so far

Goal as the student stated it: the most threat for the least resources, e.g.
only about 1/10 of sites monitored.

- "Threat" needs a number. Accuracy and F1 are closed-world numbers. Open-world
  threat is **TPR at a low fixed FPR** and **precision at a realistic base rate**.
  Precision decides whether the attack is usable, since real traffic is mostly
  unmonitored.
- That goal is a **coverage threat model**: identify as large a share of all
  visits as possible. Its answer is the **screened popular top-N**. Visits are
  Zipf-distributed, so the top sites carry most page loads, which also gives
  the best base rate. So the agreed top-50 plan does fit this goal.
- The objection: learning that someone visited Google is worth little. Most WF
  papers use a **targeted threat model** instead: watch sensitive sites (news,
  political, privacy, censored) and report precision/TPR at low FPR.

| | Coverage model | Targeted model |
|---|---|---|
| Monitored | Screened top-N by popularity | Sensitive sites |
| Unmonitored | Next popular sites plus some long tail | As many different sites as possible |
| Main number | Share of all visits identified | Precision / TPR at low FPR |
| Literature | DF closed world (Alexa top 100) | Most open-world WF work |

The unmonitored set works the same under both: many different sites, one visit
each. tb-main already does that (~425 sites per transport, ~1 visit each).
Adding sites from further down Tranco would make the FPR more honest.

## Recommendation so far (Claude's, not decided)

The research question is cross-transport transfer, which probably doesn't
depend much on the threat model. So:

1. Name the threat model explicitly in the thesis. That matters more than which
   one is picked.
2. The current set fits neither model cleanly: 35 of the sites are there only
   because the önlab used them. Calling it targeted is defensible (news,
   Hungarian, privacy, tech), but only about half was chosen for that reason.
3. Switching to coverage means screening sites and a new multi-day collection
   on the server. Suggested: keep tb-main, frame it as targeted, and put the
   coverage model in future work or a short follow-up run if time allows.

That trade-off is the question for Balázs. Next step: discuss with the student,
then draft the second email to him.

Keep the Azure resource group `wf-thesis` (obfs4 bridge) until this is decided:
a recollection needs it.

## Decision (2026-10-10)

The student chose to collect a new main dataset and to add the attacker-cost
analyses, all on the time and compute available before 2026-12-11.

- **S2**: screened popular top-50 as the monitored set (coverage threat model,
  the plan agreed with the konzulens). The 500 unmonitored are the screened top
  of Tranco today, so they are redrawn from further down the list and
  collected again; S2 is a full collection (~8500 captures, 4-5 days).
- **tb-main stays** as a second, targeted site set, so the cross-transport
  result is tested on two site sets.
- **R1** target-transport data budget: train on baseline plus k traces per
  site from obfs4 / Snowflake, k = 0, 1, 3, 5, 10, all.
- **R2** monitored-set size: k = 5, 10, 20, 50 monitored, the rest unmonitored.
- **R3** precision at realistic base rates, from the open-world threshold sweep.
- **R4** learning curve: traces per site against accuracy.

The feladatkiírás says "50 kiválasztott (megfigyelt)", so either set satisfies it.

