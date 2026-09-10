#!/bin/bash
# Run one longitudinal round.  Safe to invoke from cron or a systemd timer:
# it refuses to start a second copy, resumes a partially finished round, and
# writes everything it does to logs/.
set -euo pipefail

cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"

mkdir -p logs
LOCK="/tmp/tor_wf_round.lock"

exec 9>"$LOCK"
if ! flock -n 9; then
	echo "$(date -Is) another round is already running, exiting." >>logs/rounds.log
	exit 0
fi

if [ -f .env.collection ]; then
	set -a
	# shellcheck disable=SC1091
	source .env.collection
	set +a
fi

if [ -d .venv ]; then
	# shellcheck disable=SC1091
	source .venv/bin/activate
fi

PY_BIN="$(command -v python3 || command -v python)"
ROUND_LOG="logs/round_$(date +%Y-W%V).log"

# Leftovers from a killed run hold the capture interface and the SOCKS ports.
sudo -n pkill -x tcpdump 2>/dev/null || true
pkill -f "chrome --headless" 2>/dev/null || true
sleep 2

{
	echo "===== round start $(date -Is) ====="
	"$PY_BIN" scripts/collection/preflight.py || {
		echo "PREFLIGHT FAILED, round aborted"
		exit 1
	}
	"$PY_BIN" scripts/collection/collect_round.py --resume "$@"
	echo "===== extracting features $(date -Is) ====="
	"$PY_BIN" src/extract_round_features.py --round latest
	echo "===== round end $(date -Is) ====="
} >>"$ROUND_LOG" 2>&1

echo "$(date -Is) round finished, log: $REPO_ROOT/$ROUND_LOG" >>logs/rounds.log
