# Longitudinal collection setup

Runbook for the capture host. Every step here exists because skipping it
produces a round of PCAPs that look valid and contain nothing usable.

## Why this replaces the spring collector

`scripts/collection/collector.py` was written for a single machine and a single
burst of collection. Three things in it break over a multi-month series:

- The capture interface and guard IP were hardcoded literals. Once the guard
  rotates, the tcpdump filter matches no packets and every capture is empty.
- Nothing checked that the page loaded. A Cloudflare challenge produces a
  well-formed PCAP containing none of the target site's traffic, and counting
  those as samples fabricates drift that is not there.
- Baseline and obfs4 were collected in different months, so protocol and time
  were perfectly confounded.

`collect_round.py` resolves the peer address from the running Tor process,
validates every page load, and interleaves both arms inside one round.

## 1. Two Tor instances

The obfs4 arm needs its own Tor, otherwise the two arms cannot be interleaved
without reconfiguring Tor between captures.

```bash
sudo tor-instance-create baseline
sudo tor-instance-create obfs4
sudo cp scripts/collection/torrc.baseline.example /etc/tor/instances/baseline/torrc
sudo cp scripts/collection/torrc.obfs4.example    /etc/tor/instances/obfs4/torrc
# edit the obfs4 torrc: paste a bridge line from https://bridges.torproject.org/
sudo systemctl enable --now tor@baseline tor@obfs4
```

Add yourself to the Tor control groups so stem can authenticate by cookie:

```bash
sudo usermod -aG debian-tor "$USER"   # group name differs on Fedora: toranon
```

Pin the guard in the baseline torrc and record the fingerprint. A guard change
mid-study shifts latency for every site at once and is indistinguishable from
drift in the results.

## 2. Passwordless sudo for capture

```
your_user ALL=(root) NOPASSWD: /usr/bin/tcpdump, /usr/bin/pkill, /usr/bin/mv, /usr/bin/chown, /usr/bin/rm
```

On Fedora and Ubuntu, SELinux or AppArmor may block tcpdump writing to `/tmp`.
The preflight check catches this.

## 3. Configure and verify

```bash
cp scripts/collection/.env.collection.example .env.collection
$EDITOR .env.collection
python scripts/collection/preflight.py
```

Preflight must pass before the first round. It checks packages, binaries, sudo,
the interface, a live one-packet capture, Tor bootstrap on both control ports,
that each SOCKS port actually exits the Tor network, disk space, and that the
spring t0 features are present.

## 4. First round

```bash
scripts/collection/run_round.sh --repeats 5
```

One round is 36 sites, 5 repeats, 2 arms, so 360 captures at roughly 37 seconds
each: about 3.7 hours. The runner takes a lock, runs preflight, collects,
extracts features, and logs to `logs/round_<year>-W<week>.log`.

Safe to interrupt. `--resume` skips site and arm pairs that already have enough
captures, so a killed round can be finished later the same week.

## 5. Weekly schedule

```bash
mkdir -p ~/.config/systemd/user
cp scripts/collection/systemd/tor-wf-round.* ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now tor-wf-round.timer
sudo loginctl enable-linger "$USER"     # timers run without an active login
systemctl --user list-timers tor-wf-round.timer
```

The timer fires at a fixed weekday and a fixed time of day on purpose. News
sites change through the day, so holding time-of-day constant keeps week-to-week
differences attributable to drift rather than to diurnal variation. `Persistent=true`
catches up a round missed to a reboot, because a missing week is a permanent
hole in the series.

Cron equivalent if systemd is not available:

```cron
0 2 * * 3 /home/YOUR_USER/onlab/scripts/collection/run_round.sh --repeats 5
```

## 6. What lands on disk

```
tor_dataset/longitudinal/2026-W37/
├── round_meta.json          host, Tor version, guard, consensus, status counts
├── manifest.csv             one row per capture attempt, including failures
├── baseline/                accepted PCAPs
│   └── _rejected/           blocked, empty or redirected captures
├── baseline_features/       per-packet CSVs
├── obfs4/
└── obfs4_features/
```

Rejected captures are kept rather than deleted. The reject rate per site per
week is itself a result: a site that starts serving CAPTCHAs to Tor exit nodes
is a real-world finding, not just a collection problem.

## 7. Analysis

```bash
python src/extract_round_features.py --round latest   # run_round.sh does this
python src/drift_eval.py    --arm baseline            # decay curve, policies
python src/drift_figures.py --arm baseline            # figures
python src/drift_shift.py   --arm baseline            # model-free shift test
```

## 8. Weekly check, two minutes

```bash
tail -3 logs/rounds.log
python - <<'EOF'
import csv, glob, collections
for m in sorted(glob.glob("tor_dataset/longitudinal/*/manifest.csv"))[-2:]:
    c = collections.Counter(r["status"] for r in csv.DictReader(open(m)))
    total = sum(c.values())
    print(m, dict(c), f"usable {c['ok']}/{total}")
EOF
```

If the usable fraction drops below about 80 percent, stop and diagnose before
the next round. A round that is 40 percent blocked pages is worse than a missing
round, because it silently biases the drift estimate.

## Things that must not change mid-study

Changing any of these makes rounds incomparable and costs the series:

- the 36-site target list and the URLs in it
- `TOR_WF_CAPTURE_DURATION` and `TOR_WF_WARMUP_DURATION`
- the feature extraction code in `src/extract_all_features.py`
- the pinned guard and the obfs4 bridge
- `TOR_WF_REMOTE_DNS`, which is off by default to match the spring t0
- the capture host and its network, if at all avoidable

If one of them has to change, note the round it changed in and treat it as a
covariate in the analysis rather than pretending it did not happen.
