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

## 1. Two Tor instances, no root required

The obfs4 arm needs its own Tor, otherwise the two arms cannot be interleaved
without reconfiguring Tor between captures. Tor runs fine as an unprivileged
user with its own DataDirectory, so the system Tor is left alone:

```bash
scripts/collection/setup_tor_instances.sh
```

This creates `~/tor_wf_runtime/{baseline,obfs4}`, each with its own torrc, data
directory and log. Baseline listens on SOCKS 9060 and control 9061, obfs4 on
9062 and 9063. Use `--status` to see what is running and `--stop` to stop them.

### Guard pinning

The baseline arm uses one fixed guard for the whole study, so a change in
measured accuracy is drift and not a different route through the network. The
guard is discovered on the first run, written to
`~/tor_wf_runtime/baseline/pinned_guard.txt`, then pinned with `EntryNodes`.

Two things make this fiddly, and the script handles both:

- A Tor pinned to one guard from a cold start cannot bootstrap. It needs
  microdescriptors before it can use the guard, and a circuit through the guard
  to fetch them. So it bootstraps unrestricted first, then restarts pinned.
- A pin to a relay that has left the consensus leaves Tor stuck at 5 percent,
  unable to build any circuit at all. The script validates the pin against the
  live consensus and discards it if the relay is gone.

The spring t0 ran over guard `th4r` (27A06581, 57.129.38.230), which is no
longer in the consensus. The new series uses a different guard and records it
as a covariate rather than pretending the path is unchanged.

A running instance does not pick up a rewritten torrc. If the script reports
"already running" while you are changing the pin, restart that instance or the
pin silently does nothing.

## 1b. The obfs4 arm needs a bridge you control

Do not build the obfs4 arm on a volunteer bridge. The spring t0 bridge stopped
answering mid-study, and five replacements handed out by bridges.torproject.org
were all unreachable within the same afternoon. This is a known issue: the Tor
Project's own tracker carries "Stop BridgeDB from handing out offline bridges",
because reachability data lags and stale entries stay in the pool. A study that
must collect identically every week until December cannot rest on that.

Run a private bridge instead, on any host with a public IP:

```bash
scripts/collection/setup_obfs4_bridge.sh <public_ip> 9010
```

It installs tor and obfs4proxy, writes a bridge torrc with
`PublishServerDescriptor 0` and `BridgeDistribution none` so the bridge is never
published or distributed, sets `AssumeReachable 1` because only the obfs4 port
is exposed, and prints the finished bridge line.

Two things are easy to miss. On a cloud VM the provider's firewall is separate
from the host's, so the obfs4 port must be opened in the security group too.
And leave the source unrestricted: obfs4 is probe-resistant and ignores anyone
without the certificate, so the certificate is the access control, not the
firewall. Restricting the source to the collection host's address breaks the
moment a residential IP rotates.

Put the printed line in `~/tor_wf_runtime/obfs4/torrc` on the collection host,
restart that instance, and set `TOR_WF_COLLECT_OBFS4=1`. Keep the line out of
git: it lives outside the repository tree on both machines.

## 2. Let tcpdump capture

Run once as root:

```bash
sudo setcap cap_net_raw,cap_net_admin=eip "$(command -v tcpdump)"
```

This is the only step that needs root, and it is required: a weekly timer
firing at 02:00 has nobody to answer a sudo password prompt. With capabilities
set, tcpdump writes captures straight to their final path, so no privileged
move or ownership fix happens at all.

Passwordless sudo also works and preflight accepts it, but check that the path
in your sudoers rule actually exists. A rule naming `/usr/sbin/tcpdump` on a
system whose binary is `/usr/bin/tcpdump` silently matches nothing. Note that
`tcpdump -D` succeeds without capture privileges, so it cannot be used to test
this.


## 3. Configure and verify

```bash
cp scripts/collection/.env.collection.example .env.collection
$EDITOR .env.collection      # ports 9060-9063 to match step 1
set -a; . ./.env.collection; set +a
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

Three user units: one timer, one round service, and one Tor instance service
per arm so both instances come back after a reboot.

```bash
mkdir -p ~/.config/systemd/user
cp scripts/collection/systemd/tor-wf-round.* ~/.config/systemd/user/
sed "s|%h|$HOME|g; s|%i|baseline|g" scripts/collection/systemd/tor-wf@.service \
	> ~/.config/systemd/user/tor-wf-baseline.service
sed "s|%h|$HOME|g; s|%i|obfs4|g" scripts/collection/systemd/tor-wf@.service \
	> ~/.config/systemd/user/tor-wf-obfs4.service
systemctl --user daemon-reload
systemctl --user enable tor-wf-baseline.service tor-wf-obfs4.service
systemctl --user enable --now tor-wf-round.timer
loginctl enable-linger "$USER"          # timers run without an active login
systemctl --user list-timers tor-wf-round.timer
```

Without linger the timer only runs while you are logged in, which for a study
that fires at 02:00 means it never runs at all.

The timer fires at a fixed weekday and a fixed time of day on purpose. News
sites change through the day, so holding time-of-day constant keeps week-to-week
differences attributable to drift rather than to diurnal variation. `Persistent=true`
catches up a round missed to a reboot, because a missing week is a permanent
hole in the series.

Cron equivalent if systemd is not available:

```cron
0 2 * * 3 /home/YOUR_USER/onlab/scripts/collection/run_round.sh --repeats 5
```

## 5b. What a round does, and what protects it

`run_round.sh` is not just the collector. Each round runs five steps, and the
middle three exist because the series is unattended for thirteen weeks and the
dangerous failure is silent degradation, not a crash.

1. **`ensure_guard.sh`** verifies the pinned guard is still in the consensus. If
   it has left, the pin is replaced automatically, the change is appended to
   `~/tor_wf_runtime/baseline/guard_history.log`, and collection continues. A
   recorded guard change is a covariate you can control for. A pin to a departed
   relay leaves Tor stuck at 5 percent and costs every remaining week, which is
   not recoverable. The spring t0 guard left the network exactly this way.
2. **`preflight.py`** must pass or the round aborts rather than collecting junk.
3. **`collect_round.py --resume`** collects. Resume also tops up sites whose
   earlier captures were rejected, since quarantined captures are not counted,
   so a site converges toward its full quota across reruns in the same week.
4. **`round_health.py`** scores the round and writes the result into
   `round_meta.json`. It fails the round when the usable fraction drops below 55
   percent or fewer than 20 sites reach 3 usable captures. The verdict is
   appended to `logs/rounds.log`, so the weekly check is one line.
5. **`backup_round.sh`** archives manifests, metadata and feature CSVs offsite
   via rclone. About 8 MB per round, so roughly 110 MB for the whole study.
   PCAPs are excluded by default at 1 to 2 GB per round; set
   `TOR_WF_BACKUP_PCAPS=1` to include them.

Thirteen weeks of irreplaceable measurements on a single laptop is the largest
risk to the thesis and the one the collection code cannot otherwise reduce.

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
tail -3 logs/rounds.log          # each line carries a healthy / NEEDS ATTENTION verdict
python src/round_health.py --round latest
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
- the pinned guard, and the obfs4 bridge including its host and region
- `TOR_WF_REMOTE_DNS`, which is off by default to match the spring t0
- the capture host and its network, if at all avoidable

If one of them has to change, note the round it changed in and treat it as a
covariate in the analysis rather than pretending it did not happen.
