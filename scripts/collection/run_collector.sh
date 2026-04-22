#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

source .venv/bin/activate

if command -v python >/dev/null 2>&1; then
	PY_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
	PY_BIN="python3"
else
	echo "ERROR: Neither python nor python3 found in PATH."
	exit 1
fi

# Cleanup before run (headless collector machine)
sudo pkill -f "chrome" || true
sudo pkill -x "tcpdump" || true

"$PY_BIN" scripts/collection/collector.py >> logs/collection.log 2>&1
