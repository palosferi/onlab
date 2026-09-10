#!/bin/bash
# Same-day control round through a DIFFERENT guard.
#
# The t0-to-W37 comparison showed a large shift that was near-identical across
# site categories, which points at the network path rather than at websites
# changing. This separates the two: same day, same sites, same settings, only
# the guard differs. If the shift here matches the shift against t0, the effect
# is path. If it sits at the sampling-noise level, the t0 shift is real.
#
#   scripts/collection/run_guard_control.sh [repeats]

set -euo pipefail

cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"

REPEATS="${1:-5}"
RUNTIME_DIR="${TOR_WF_RUNTIME_DIR:-$HOME/tor_wf_runtime}"
DIR="$RUNTIME_DIR/guardctl"
SOCKS=9064
CONTROL=9065
PY_BIN="${TOR_WF_PYTHON:-$REPO_ROOT/.venv/bin/python}"

BASE_GUARD="$(cut -d' ' -f1 "$RUNTIME_DIR/baseline/pinned_guard.txt" 2>/dev/null || echo "")"
echo "=== guard control round (baseline arm guard: ${BASE_GUARD:-unknown}) ==="

mkdir -p "$DIR/data"
chmod 700 "$DIR/data"

write_torrc() {
	local pin="$1"
	cat >"$DIR/torrc" <<EOF
SocksPort $SOCKS
ControlPort $CONTROL
CookieAuthentication 1
DataDirectory $DIR/data
PidFile $DIR/tor.pid
Log notice file $DIR/tor.log
RunAsDaemon 1
$( [ -n "$pin" ] && printf 'EntryNodes %s\nStrictNodes 1\n' "$pin" )
EOF
}

start_tor() {
	: >"$DIR/tor.log"
	/usr/bin/tor -f "$DIR/torrc" >"$DIR/start.log" 2>&1
	for _ in $(seq 1 60); do
		grep -q "Bootstrapped 100%" "$DIR/tor.log" && return 0
		sleep 2
	done
	echo "  NOT bootstrapped:"; grep -oE "Bootstrapped [0-9]+%.*" "$DIR/tor.log" | tail -1
	return 1
}

stop_tor() {
	[ -f "$DIR/tor.pid" ] && kill "$(cat "$DIR/tor.pid")" 2>/dev/null || true
	sleep 3
}

stop_tor

# Bootstrap unrestricted to prime the directory cache, then pick a guard that is
# NOT the one the baseline arm uses.
echo "[*] priming and selecting a different guard"
write_torrc ""
start_tor

CTRL_GUARD="$("$PY_BIN" - "$CONTROL" "$BASE_GUARD" <<'PYPICK'
import random, sys
from stem.control import Controller
port, avoid = int(sys.argv[1]), sys.argv[2].upper()
with Controller.from_port(port=port) as c:
    c.authenticate()
    cands = [d for d in c.get_network_statuses()
             if "Guard" in d.flags and "Running" in d.flags and "Fast" in d.flags
             and "Stable" in d.flags and d.fingerprint.upper() != avoid]
    random.seed(20260910)
    random.shuffle(cands)
    d = cands[0]
    print(f"{d.fingerprint} {d.nickname} {d.address}")
PYPICK
)"
echo "  control guard: $CTRL_GUARD"
echo "$CTRL_GUARD" >"$DIR/pinned_guard.txt"

stop_tor
write_torrc "$(echo "$CTRL_GUARD" | cut -d' ' -f1)"
start_tor
echo "[*] control instance pinned and bootstrapped"

# Everything except the guard is held identical to the main round.
export TOR_WF_ROUND_ID="$(date -u +%G-W%V)-guardctl"
export TOR_WF_COLLECT_BASELINE=1
export TOR_WF_COLLECT_OBFS4=0
export TOR_WF_BASELINE_SOCKS_PORT=$SOCKS
export TOR_WF_BASELINE_CONTROL_PORT=$CONTROL

echo "[*] collecting round $TOR_WF_ROUND_ID with $REPEATS repeats"
"$PY_BIN" scripts/collection/collect_round.py --resume --repeats "$REPEATS"
"$PY_BIN" src/extract_round_features.py --round "$TOR_WF_ROUND_ID"

echo
echo "=== compare: same day, different guard ==="
"$PY_BIN" src/drift_shift.py --arm baseline \
	--reference "$(date -u +%G-W%V)" --target "$TOR_WF_ROUND_ID" || true
