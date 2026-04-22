#!/bin/bash
set -euo pipefail

# Run on server in tor_wf_project root (or from any subdirectory).
# Example:
#   bash scripts/collection/run_server_collect_other.sh
#   bash scripts/run_server_collect_other.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ -f "$SCRIPT_DIR/../venv/bin/activate" ]]; then
	PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
elif [[ -f "$SCRIPT_DIR/../../venv/bin/activate" ]]; then
	PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
else
	echo "ERROR: Could not find venv/bin/activate relative to script location."
	exit 1
fi

cd "$PROJECT_ROOT"
source venv/bin/activate

if [[ -z "${TOR_WF_PCAP_DIR:-}" ]]; then
	TOR_WF_PCAP_DIR="$PROJECT_ROOT/data/pcaps/other_tor"
	export TOR_WF_PCAP_DIR
fi

if [[ -z "${TOR_WF_LOG_JSONL:-}" ]]; then
	TOR_WF_LOG_JSONL="$PROJECT_ROOT/logs/collection_other_events.jsonl"
	export TOR_WF_LOG_JSONL
fi

mkdir -p "$TOR_WF_PCAP_DIR" "$PROJECT_ROOT/logs"

if command -v python >/dev/null 2>&1; then
	PY_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
	PY_BIN="python3"
else
	echo "ERROR: Neither python nor python3 found in PATH."
	exit 1
fi

sudo pkill -f "chrome" || true
sudo pkill -x "tcpdump" || true

if [[ -f "scripts/collection/server_collect_other.py" ]]; then
	PYTHONUNBUFFERED=1 "$PY_BIN" scripts/collection/server_collect_other.py >> logs/collection_other.log 2>&1
elif [[ -f "scripts/server_collect_other.py" ]]; then
	PYTHONUNBUFFERED=1 "$PY_BIN" scripts/server_collect_other.py >> logs/collection_other.log 2>&1
else
	echo "ERROR: Could not find server_collect_other.py in scripts/collection or scripts/."
	exit 1
fi
