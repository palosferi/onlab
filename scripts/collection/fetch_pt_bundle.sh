#!/bin/bash
# Fetch lyrebird, Tor's pluggable-transport client, from the Tor Expert Bundle.
#
# Debian and Mint package obfs4proxy but no Snowflake client, and obfs4proxy is
# the retired predecessor of lyrebird. The Expert Bundle carries the same
# lyrebird build Tor Browser ships, which also supplies the Snowflake bridge
# lines Tor Browser itself uses -- both of which belong in a study that claims to
# measure what a real Tor user's traffic looks like.
#
#   scripts/collection/fetch_pt_bundle.sh            # latest release
#   TOR_WF_TB_VERSION=15.0.23 scripts/collection/fetch_pt_bundle.sh
#
# The version actually installed is recorded in $PT_DIR/VERSION, because the
# transport implementation is a covariate: a lyrebird update can change traffic
# shape without anything in this repository changing.

set -euo pipefail

RUNTIME_DIR="${TOR_WF_RUNTIME_DIR:-$HOME/tor_wf_runtime}"
PT_DIR="$RUNTIME_DIR/pt"
ARCH="${TOR_WF_TB_ARCH:-linux-x86_64}"
BASE="https://dist.torproject.org/torbrowser"
VERSION="${TOR_WF_TB_VERSION:-}"

if [ -z "$VERSION" ]; then
	echo "=== discovering the latest Tor Browser version ==="
	VERSION="$(curl -fsS --max-time 30 "$BASE/" |
		grep -oE 'href="[0-9]+\.[0-9]+(\.[0-9]+)?/"' |
		grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | sort -V | tail -1)"
	[ -n "$VERSION" ] || {
		echo "Could not determine a version. Set TOR_WF_TB_VERSION." >&2
		exit 1
	}
fi

TARBALL="tor-expert-bundle-$ARCH-$VERSION.tar.gz"
SUMS="sha256sums-signed-build.txt"

mkdir -p "$PT_DIR"
cd "$PT_DIR"

echo "=== downloading $TARBALL ==="
curl -fsS --max-time 300 -O "$BASE/$VERSION/$TARBALL"
curl -fsS --max-time 60 -O "$BASE/$VERSION/$SUMS"
curl -fsS --max-time 60 -O "$BASE/$VERSION/$SUMS.asc" || echo "  (no signature available)"

echo "=== verifying checksum ==="
grep " $TARBALL\$" "$SUMS" | sha256sum -c -

# The checksum file is itself signed. Checking that signature is what proves the
# sums came from Tor rather than from whoever answered the HTTPS request, so it
# is attempted whenever the signing key is already present.
if [ -f "$SUMS.asc" ] && command -v gpg >/dev/null 2>&1; then
	if gpg --verify "$SUMS.asc" "$SUMS" >/dev/null 2>&1; then
		echo "  signature on $SUMS verified"
	else
		echo "  NOTE: signature not verified (signing key absent or untrusted)."
		echo "        gpg --auto-key-locate nodefault,wkd --locate-keys torbrowser@torproject.org"
	fi
fi

echo "=== extracting ==="
tar xzf "$TARBALL"
LYREBIRD="$PT_DIR/tor/pluggable_transports/lyrebird"
[ -x "$LYREBIRD" ] || {
	echo "Expected lyrebird at $LYREBIRD but it is missing or not executable." >&2
	exit 1
}

echo "$VERSION" >"$PT_DIR/VERSION"
rm -f "$TARBALL"

echo
echo "lyrebird:  $LYREBIRD"
echo "version:   $("$LYREBIRD" -version 2>&1 | head -1) (Tor Browser $VERSION)"
echo "bundled tor: $("$PT_DIR/tor/tor" --version 2>/dev/null | head -1)"
echo
echo "Next: scripts/collection/setup_tor_instances.sh  (it picks lyrebird up from here)"
