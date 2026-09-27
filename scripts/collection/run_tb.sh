#!/bin/bash
# Run (or resume) a Tor Browser collection, then extract its features.
#
#   scripts/collection/run_tb.sh --collection tb-2026-10 --repeats 40 --bg-repeats 20
#
# Collection runs in the tbselenium venv (tbselenium needs Selenium < 4.29);
# feature extraction runs in the main venv. Re-running with the same arguments
# resumes from the manifest.
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs

exec 9>/tmp/tor_wf_tb.lock
if ! flock -n 9; then
	echo "another Tor Browser collection is already running" >&2
	exit 1
fi

# .env.collection supplies defaults; anything already set by the caller wins,
# so e.g. TOR_WF_COLLECT_BASELINE=0 run_tb.sh ... really drops that arm.
if [ -f .env.collection ]; then
	while IFS='=' read -r key value; do
		[[ "$key" =~ ^[A-Z_][A-Z0-9_]*$ ]] || continue
		[ -n "${!key+x}" ] || export "$key=$value"
	done <.env.collection
fi
export TOR_WF_COLLECT_SNOWFLAKE="${TOR_WF_COLLECT_SNOWFLAKE:-1}"

TB_PY="${TOR_WF_TB_PYTHON:-$HOME/tor_wf_runtime/tbvenv/bin/python}"
MAIN_PY="${TOR_WF_PYTHON:-.venv/bin/python}"

COLLECTION=""
args=("$@")
for i in "${!args[@]}"; do
	[ "${args[$i]}" = "--collection" ] && COLLECTION="${args[$((i + 1))]}"
done
[ -n "$COLLECTION" ] || { echo "--collection is required" >&2; exit 2; }

# Leftovers from a killed run hold the capture and the browser profile dirs.
pkill -x tcpdump 2>/dev/null || true
pkill -f "tor-browser/Browser/firefox" 2>/dev/null || true
pkill -x geckodriver 2>/dev/null || true
sleep 1

echo "===== collection start $(date -Is) ====="
"$TB_PY" scripts/collection/collect_tb.py "$@"
echo "===== extracting features $(date -Is) ====="
"$MAIN_PY" src/extract_tb_features.py --collection "$COLLECTION"
echo "===== done $(date -Is) ====="
