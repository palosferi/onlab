#!/bin/bash
# Back up a round's analysable data offsite.
#
# Thirteen weeks of irreplaceable measurements living on one laptop is the
# largest single risk to this thesis, and it is not a risk the collection code
# can otherwise mitigate. Feature CSVs plus manifests are a few tens of MB per
# round, so there is no reason not to.
#
# PCAPs are excluded by default at roughly 1-2 GB per round. Set
# TOR_WF_BACKUP_PCAPS=1 to include them.
#
#   scripts/collection/backup_round.sh [round_id|latest]

set -euo pipefail

cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"

LONGITUDINAL_DIR="${TOR_WF_LONGITUDINAL_DIR:-$REPO_ROOT/tor_dataset/longitudinal}"
REMOTE="${TOR_WF_BACKUP_REMOTE:-gdrive:}"
REMOTE_PATH="${TOR_WF_BACKUP_PATH:-tor_wf_longitudinal}"
STAGING="${TOR_WF_BACKUP_STAGING:-$HOME/wf_backups}"

SELECTOR="${1:-latest}"
if [ "$SELECTOR" = "latest" ]; then
	ROUND="$(ls -1 "$LONGITUDINAL_DIR" | sort | tail -1)"
else
	ROUND="$SELECTOR"
fi
[ -d "$LONGITUDINAL_DIR/$ROUND" ] || {
	echo "no such round: $ROUND"
	exit 1
}

mkdir -p "$STAGING"
TAR_ARGS=(--exclude="_rejected")
if [ "${TOR_WF_BACKUP_PCAPS:-0}" != "1" ]; then
	TAR_ARGS+=(--exclude="*.pcap")
fi

echo "[backup] packing $ROUND"
if command -v zstd >/dev/null; then
	ARCHIVE="$STAGING/${ROUND}.tar.zst"
	tar "${TAR_ARGS[@]}" -C "$LONGITUDINAL_DIR" -cf - "$ROUND" |
		zstd -q -19 -T0 -o "$ARCHIVE" -f
else
	ARCHIVE="$STAGING/${ROUND}.tar.gz"
	tar "${TAR_ARGS[@]}" -C "$LONGITUDINAL_DIR" -czf "$ARCHIVE" "$ROUND"
fi
echo "[backup] $ARCHIVE ($(du -h "$ARCHIVE" | cut -f1))"

if command -v rclone >/dev/null && rclone listremotes 2>/dev/null | grep -q "^${REMOTE}$"; then
	echo "[backup] uploading to ${REMOTE}${REMOTE_PATH}/"
	rclone copy "$ARCHIVE" "${REMOTE}${REMOTE_PATH}/" --no-traverse
	echo "[backup] verifying"
	rclone check "$ARCHIVE" "${REMOTE}${REMOTE_PATH}/" --one-way 2>&1 | tail -2
else
	echo "[backup] remote '$REMOTE' not configured; archive kept locally only."
	echo "[backup] Configure it with: rclone config"
	exit 0
fi

# Keep the last few local archives, drop older ones.
ls -1t "$STAGING"/*.tar.* 2>/dev/null | tail -n +6 | xargs -r rm -f
echo "[backup] done"
