#!/bin/bash
# Rebuild the collection host after an OS reinstall. Written for Debian 13.
#
#   scripts/collection/setup_host.sh
#
# It expects ~/onlab and ~/tor_wf_runtime restored from the backup of the old
# install. Their torrcs (with the obfs4 bridge line, which never enters the
# repository), Tor data directories, Tor Browser, geckodriver and the Tor Expert
# Bundle are used as they are, so collection runs on the same software as
# before. What this script adds is everything that lived outside the home
# directory or is tied to the old Python:
#
#   - system packages: obfs4proxy, tcpdump, and the libraries Tor Browser needs;
#   - capture capabilities on tcpdump, so collection runs without sudo;
#   - linger, so the user units keep running without a login;
#   - fresh venvs: tbvenv for collection (tbselenium needs Selenium < 4.29) and
#     ~/onlab/.venv for feature extraction;
#   - the tor-wf@ user units for the three arms, started and checked.
#
# The arms run the tor from the Expert Bundle, the same version Tor Browser
# ships, not the distribution's: what a Tor Browser user runs, and an unattended
# upgrade cannot change it in the middle of a collection. tb-main (Mint, until
# 2026-10) ran the distribution's tor 0.4.8.10; record the version as a
# covariate. Safe to run again.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
RUNTIME_DIR="${TOR_WF_RUNTIME_DIR:-$HOME/tor_wf_runtime}"
TOR_DIR="$RUNTIME_DIR/pt/tor"
TB_DIR="$RUNTIME_DIR/tb"
ARMS="baseline obfs4 snowflake"
TBSELENIUM="git+https://github.com/webfp/tor-browser-selenium@b2169f880ac250bc0f469b238fda8c92eb83cd10"

step() { printf '\n=== %s ===\n' "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

step "checking what was restored from the backup"
[ -x "$TOR_DIR/tor" ] || die "$TOR_DIR/tor missing; restore ~/tor_wf_runtime (or run fetch_pt_bundle.sh)"
[ -x "$TOR_DIR/pluggable_transports/lyrebird" ] || die "lyrebird missing from $TOR_DIR"
[ -x "$TB_DIR/tor-browser/Browser/firefox" ] || die "Tor Browser missing under $TB_DIR/tor-browser"
[ -x "$TB_DIR/geckodriver" ] || die "geckodriver missing at $TB_DIR/geckodriver"
for arm in $ARMS; do
	[ -f "$RUNTIME_DIR/$arm/torrc" ] || die "$RUNTIME_DIR/$arm/torrc missing"
done
# setup_tor_instances.sh would regenerate the torrcs and re-pin the baseline
# guard from pinned_guard.txt; the restored, unpinned ones are what tb-main used.
grep -q '^EntryNodes' "$RUNTIME_DIR/baseline/torrc" && die "baseline torrc pins a guard; see CLAUDE.md"
echo "  ok"

step "system packages (sudo)"
sudo apt-get update -qq
# firefox-esr is there for its dependencies: Tor Browser needs the same GTK and
# X libraries even when it runs headless.
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
	obfs4proxy tcpdump libcap2-bin firefox-esr \
	python3 python3-venv python3-dev build-essential git curl gpg xz-utils zstd rsync ca-certificates

step "tcpdump capture capabilities (sudo)"
TCPDUMP="$(PATH="$PATH:/usr/sbin:/sbin" command -v tcpdump)"
sudo setcap cap_net_raw,cap_net_admin=eip "$TCPDUMP"
/usr/sbin/getcap "$TCPDUMP"

step "linger (sudo)"
sudo loginctl enable-linger "$USER"

step "collection venv $RUNTIME_DIR/tbvenv"
rm -rf "$RUNTIME_DIR/tbvenv"
python3 -m venv "$RUNTIME_DIR/tbvenv"
"$RUNTIME_DIR/tbvenv/bin/pip" install -q --upgrade pip
"$RUNTIME_DIR/tbvenv/bin/pip" install -q selenium==4.28.1 stem==1.8.2 PySocks==1.7.1 "tbselenium @ $TBSELENIUM"

step "analysis venv $REPO_DIR/.venv"
rm -rf "$REPO_DIR/.venv"
python3 -m venv "$REPO_DIR/.venv"
"$REPO_DIR/.venv/bin/pip" install -q --upgrade pip
# The CPU build of torch; the default wheel pulls gigabytes of CUDA libraries.
"$REPO_DIR/.venv/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cpu
"$REPO_DIR/.venv/bin/pip" install -q -r "$REPO_DIR/requirements.txt"

step "tor-wf@ user units"
units="$HOME/.config/systemd/user"
mkdir -p "$units"
for arm in $ARMS; do
	# The per-arm units of the Mint install ran /usr/bin/tor.
	systemctl --user disable --now "tor-wf-$arm.service" 2>/dev/null || true
	rm -f "$units/tor-wf-$arm.service"
done
cp "$REPO_DIR/scripts/collection/systemd/tor-wf@.service" "$units/"
systemctl --user daemon-reload
for arm in $ARMS; do
	: >"$RUNTIME_DIR/$arm/tor.log"
	systemctl --user enable --now "tor-wf@$arm.service"
done

step "waiting for the three arms to bootstrap"
# A Tor started on a restored DataDirectory reads the old last-activity time
# from its state file and goes dormant at once, stuck at 25%. A request through
# its SOCKS port counts as activity and wakes it.
for arm in $ARMS; do
	port="$(awk '/^SocksPort/ {print $2}' "$RUNTIME_DIR/$arm/torrc")"
	curl -s -o /dev/null --max-time 170 --socks5-hostname "127.0.0.1:$port" https://check.torproject.org/ &
done
failed=0
for arm in $ARMS; do
	for _ in $(seq 1 90); do
		grep -q "Bootstrapped 100%" "$RUNTIME_DIR/$arm/tor.log" && break
		sleep 2
	done
	if grep -q "Bootstrapped 100%" "$RUNTIME_DIR/$arm/tor.log"; then
		echo "  $arm: bootstrapped"
	else
		echo "  $arm: NOT bootstrapped: $(grep -oE 'Bootstrapped [0-9]+%.*' "$RUNTIME_DIR/$arm/tor.log" | tail -1)"
		failed=1
	fi
done

step "versions"
echo "  $(LD_LIBRARY_PATH="$TOR_DIR" "$TOR_DIR/tor" --version | head -1)"
echo "  obfs4proxy $(dpkg-query -W -f='${Version}' obfs4proxy)"
echo "  $(sed -n 1p "$TB_DIR/tor-browser/Browser/TorBrowser/Docs/ChangeLog.txt" 2>/dev/null)"
echo "  $("$TB_DIR/geckodriver" --version | head -1)"

step "tests"
cd "$REPO_DIR"
for t in tests/test_*.py; do "$REPO_DIR/.venv/bin/python" "$t" | tail -1; done

[ "$failed" = 0 ] || die "an arm did not bootstrap; see $RUNTIME_DIR/<arm>/tor.log"
cat <<EOF

Done. Next, a pilot over all three arms before any real collection:
  scripts/collection/run_tb.sh --collection tb-pilot-debian --only wikipedia,bbc,telex --repeats 2 --bg-repeats 1 --unmonitored-limit 3
EOF
