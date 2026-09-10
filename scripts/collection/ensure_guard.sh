#!/bin/bash
# Keep the baseline arm's guard alive and pinned, before a round starts.
#
# The guard is pinned so that week-to-week differences are drift rather than a
# different route. But relays leave the network: the spring t0 guard did exactly
# that, and a pin to a departed relay leaves Tor stuck at 5% unable to build any
# circuit, which would silently cost every remaining week of the study.
#
# A recorded guard change is a covariate. A missing month of data is not
# recoverable. So this repairs rather than fails.

set -euo pipefail

RUNTIME_DIR="${TOR_WF_RUNTIME_DIR:-$HOME/tor_wf_runtime}"
DIR="$RUNTIME_DIR/baseline"
CONTROL="${TOR_WF_BASELINE_CONTROL_PORT:-9061}"
HISTORY="$DIR/guard_history.log"
PY_BIN="${TOR_WF_PYTHON:-$(command -v python3)}"

running() { [ -f "$DIR/tor.pid" ] && kill -0 "$(cat "$DIR/tor.pid")" 2>/dev/null; }
stop_tor() { running && kill "$(cat "$DIR/tor.pid")" 2>/dev/null && sleep 3 || true; }

start_tor() {
	: >"$DIR/tor.log"
	/usr/bin/tor -f "$DIR/torrc" >"$DIR/start.log" 2>&1 || return 1
	for _ in $(seq 1 60); do
		grep -q "Bootstrapped 100%" "$DIR/tor.log" && return 0
		sleep 2
	done
	return 1
}

if ! running; then
	echo "[ensure_guard] baseline instance not running, starting it"
	start_tor || echo "[ensure_guard] WARNING: did not reach 100% bootstrap"
fi

PINNED="$(cut -d' ' -f1 "$DIR/pinned_guard.txt" 2>/dev/null || echo "")"
if [ -z "$PINNED" ]; then
	echo "[ensure_guard] no pin recorded, nothing to verify"
	exit 0
fi

IN_CONSENSUS="$("$PY_BIN" - "$CONTROL" "$PINNED" <<'PYCHECK' 2>/dev/null || echo unknown
import sys
from stem.control import Controller
port, fp = int(sys.argv[1]), sys.argv[2].upper()
try:
    with Controller.from_port(port=port) as c:
        c.authenticate()
        print("yes" if any(d.fingerprint == fp for d in c.get_network_statuses()) else "no")
except Exception:
    print("unknown")
PYCHECK
)"

case "$IN_CONSENSUS" in
yes)
	echo "[ensure_guard] pinned guard $PINNED still in the consensus"
	exit 0
	;;
unknown)
	echo "[ensure_guard] could not query the consensus; leaving the pin alone"
	exit 0
	;;
esac

echo "[ensure_guard] pinned guard $PINNED has LEFT the consensus, repinning"
stop_tor
grep -vE "^(EntryNodes|StrictNodes)" "$DIR/torrc" >"$DIR/torrc.tmp" && mv "$DIR/torrc.tmp" "$DIR/torrc"
start_tor || { echo "[ensure_guard] FAILED to bootstrap unpinned"; exit 1; }

NEW="$("$PY_BIN" - "$CONTROL" <<'PYPICK'
import sys
from stem.control import Controller
with Controller.from_port(port=int(sys.argv[1])) as c:
    c.authenticate()
    circs = [x for x in c.get_circuits() if x.status == "BUILT" and x.path]
    if not circs:
        c.new_circuit(await_build=True, timeout=90)
        circs = [x for x in c.get_circuits() if x.status == "BUILT" and x.path]
    fp = circs[0].path[0][0]
    ns = c.get_network_status(fp, default=None)
    print(f"{fp} {ns.nickname if ns else '?'} {ns.address if ns else '?'}")
PYPICK
)"

echo "$(date -Is) replaced $PINNED with $NEW" >>"$HISTORY"
echo "$NEW" >"$DIR/pinned_guard.txt"
stop_tor
printf 'EntryNodes %s\nStrictNodes 1\n' "$(echo "$NEW" | cut -d' ' -f1)" >>"$DIR/torrc"
start_tor || { echo "[ensure_guard] FAILED to bootstrap with the new pin"; exit 1; }

echo "[ensure_guard] now pinned to $NEW"
echo "[ensure_guard] RECORD THIS: the guard changed mid-study, see $HISTORY"
