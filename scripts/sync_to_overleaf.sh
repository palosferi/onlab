#!/bin/bash
set -euo pipefail

# Usage:
#   OVERLEAF_REPO=/path/to/overleaf_git_clone ./scripts/sync_to_overleaf.sh
# Optional:
#   OVERLEAF_RESULTS_SUBDIR=results

if [[ -z "${OVERLEAF_REPO:-}" ]]; then
  echo "ERROR: set OVERLEAF_REPO to your Overleaf git clone path"
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST_SUBDIR="${OVERLEAF_RESULTS_SUBDIR:-results}"
DEST_DIR="${OVERLEAF_REPO%/}/${DEST_SUBDIR}"

mkdir -p "$DEST_DIR/figures"
mkdir -p "$DEST_DIR/docs"

cp -f "$REPO_ROOT/figures/metrics_rf.json" "$DEST_DIR/figures/" 2>/dev/null || true
cp -f "$REPO_ROOT/figures/metrics_dl.json" "$DEST_DIR/figures/" 2>/dev/null || true
cp -f "$REPO_ROOT/figures/02_top10_features_rf.json" "$DEST_DIR/figures/" 2>/dev/null || true
cp -f "$REPO_ROOT/figures/01_final_comparison.png" "$DEST_DIR/figures/" 2>/dev/null || true
cp -f "$REPO_ROOT/figures/02_feature_importance.png" "$DEST_DIR/figures/" 2>/dev/null || true
cp -f "$REPO_ROOT/figures/03_confusion_matrices.png" "$DEST_DIR/figures/" 2>/dev/null || true
cp -f "$REPO_ROOT/README.md" "$DEST_DIR/docs/project_summary.md" 2>/dev/null || true

echo "Synced results to: $DEST_DIR"
echo "Next: cd $OVERLEAF_REPO && git add . && git commit -m 'Update results' && git push"
