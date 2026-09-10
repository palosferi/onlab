"""Shared configuration, environment discovery and validation for longitudinal
Tor website-fingerprinting collection.

The spring 2026 collector hardcoded the capture interface and the guard IP of a
single machine.  A longitudinal series has to survive host changes, guard
rotation and bridge changes across three months, so everything here is either
resolved at runtime from the live Tor process or overridable from the
environment.
"""

import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
from datetime import datetime, timezone

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LONGITUDINAL_DIR = os.getenv(
    "TOR_WF_LONGITUDINAL_DIR", os.path.join(REPO_ROOT, "tor_dataset", "longitudinal")
)

# Same 36 monitored sites as the spring 2026 collection.  Do not edit this list
# during the study: a changed target list makes rounds incomparable.
TARGET_SITES = {
    # Static / low expected drift
    "wikipedia": "https://en.wikipedia.org/wiki/Main_Page",
    "duckduckgo": "https://duckduckgo.com",
    "github": "https://github.com",
    "stackoverflow": "https://stackoverflow.com",
    "mit_ocw": "https://ocw.mit.edu",
    "gnu": "https://www.gnu.org",
    "debian": "https://www.debian.org",
    "w3c": "https://www.w3.org",
    # News / high expected drift
    "bbc": "https://www.bbc.com",
    "cnn": "https://www.cnn.com",
    "reuters": "https://www.reuters.com",
    "theguardian": "https://www.theguardian.com/international",
    "aljazeera": "https://www.aljazeera.com",
    "hacker_news": "https://news.ycombinator.com",
    "wired": "https://www.wired.com",
    "techcrunch": "https://techcrunch.com",
    # Hungarian sites
    "telex": "https://telex.hu",
    "index": "https://index.hu",
    "hvg": "https://hvg.hu",
    "origo": "https://www.origo.hu",
    "hwsw": "https://www.hwsw.hu",
    "bme": "https://www.bme.hu",
    # E-commerce and entertainment
    "amazon": "https://www.amazon.com",
    "ebay": "https://www.ebay.com",
    "aliexpress": "https://www.aliexpress.com",
    "imdb": "https://www.imdb.com",
    "reddit": "https://old.reddit.com",
    "twitch": "https://www.twitch.tv",
    "vimeo": "https://vimeo.com",
    "soundcloud": "https://soundcloud.com",
    # Structured sites
    "wordpress": "https://wordpress.org",
    "mozilla": "https://www.mozilla.org",
    "archive": "https://archive.org",
    "medium": "https://medium.com",
    "quora": "https://www.quora.com",
    "coursera": "https://www.coursera.org",
}

# Unmonitored sites for the open-world "other" class.
OTHER_SITES = {
    "other1": "https://www.nasa.gov",
    "other2": "https://www.nature.com",
    "other3": "https://www.lonelyplanet.com",
    "other4": "https://www.khanacademy.org",
    "other5": "https://www.openstreetmap.org",
    "other6": "https://www.goodreads.com",
    "other7": "https://www.britannica.com",
    "other8": "https://www.udemy.com",
    "other9": "https://www.airbnb.com",
    "other10": "https://www.booking.com",
    "other11": "https://www.canva.com",
    "other12": "https://www.slashdot.org",
    "other13": "https://www.behance.net",
    "other14": "https://www.flickr.com",
    "other15": "https://www.tripadvisor.com",
    "other16": "https://www.accuweather.com",
    "other17": "https://www.loc.gov",
    "other18": "https://www.wiktionary.org",
    "other19": "https://www.timeanddate.com",
    "other20": "https://www.ietf.org",
}


def env_int(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def env_bool(name, default):
    return os.getenv(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "y"}


class Arm:
    """One traffic condition inside a round: plain Tor, or Tor over an obfs4 bridge.

    Each arm needs its own Tor instance so that both can be captured inside the
    same round without reconfiguring Tor mid-collection.
    """

    def __init__(self, name, socks_port, control_port, enabled=True):
        self.name = name
        self.socks_port = socks_port
        self.control_port = control_port
        self.enabled = enabled
        self.peer_ip = None
        self.guard_fingerprint = None
        self.guard_nickname = None

    def __repr__(self):
        return f"<Arm {self.name} socks={self.socks_port} ctrl={self.control_port} peer={self.peer_ip}>"


def configured_arms():
    arms = [
        Arm(
            "baseline",
            env_int("TOR_WF_BASELINE_SOCKS_PORT", 9050),
            env_int("TOR_WF_BASELINE_CONTROL_PORT", 9051),
            env_bool("TOR_WF_COLLECT_BASELINE", True),
        ),
        Arm(
            "obfs4",
            env_int("TOR_WF_OBFS4_SOCKS_PORT", 9052),
            env_int("TOR_WF_OBFS4_CONTROL_PORT", 9053),
            env_bool("TOR_WF_COLLECT_OBFS4", True),
        ),
    ]
    return [a for a in arms if a.enabled]


def detect_interface():
    """Return the interface carrying the default route, unless overridden."""
    override = os.getenv("TOR_WF_INTERFACE")
    if override:
        return override
    try:
        out = subprocess.run(
            ["ip", "route", "get", "1.1.1.1"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        m = re.search(r"\bdev\s+(\S+)", out)
        if m:
            return m.group(1)
    except Exception:
        pass
    return "eth0"


def resolve_peer_ip(controller, arm):
    """Find the IP that Tor traffic for this arm actually goes to.

    For the baseline arm that is the entry guard.  For the obfs4 arm it is the
    bridge, which is not in the consensus at all.  A pinned literal from the
    spring collector silently produces empty captures once the guard rotates,
    so both are read back from the running Tor process.
    """
    override = os.getenv(f"TOR_WF_{arm.name.upper()}_PEER_IP")
    if override:
        arm.peer_ip = override
        return override

    if arm.name == "obfs4":
        try:
            bridges = controller.get_conf("Bridge", multiple=True) or []
        except Exception:
            bridges = []
        for line in bridges:
            m = re.search(r"(\d{1,3}(?:\.\d{1,3}){3}):(\d+)", line)
            if m:
                arm.peer_ip = m.group(1)
                fp = re.search(r"\b([0-9A-Fa-f]{40})\b", line)
                arm.guard_fingerprint = fp.group(1).upper() if fp else None
                arm.guard_nickname = "obfs4-bridge"
                return arm.peer_ip
        raise RuntimeError(
            f"No Bridge line found on control port {arm.control_port}. "
            "Is this Tor instance actually configured with UseBridges 1?"
        )

    # Baseline arm: first hop of any built circuit is the guard.
    circuits = [c for c in controller.get_circuits() if c.status == "BUILT" and c.path]
    if not circuits:
        controller.new_circuit(await_build=True, timeout=90)
        circuits = [c for c in controller.get_circuits() if c.status == "BUILT" and c.path]
    if not circuits:
        raise RuntimeError(f"Tor on control port {arm.control_port} has no built circuit.")

    fingerprint = circuits[0].path[0][0]
    arm.guard_fingerprint = fingerprint
    status = controller.get_network_status(fingerprint)
    arm.guard_nickname = status.nickname
    arm.peer_ip = status.address
    return arm.peer_ip


def round_id(now=None):
    """ISO year-week label, e.g. 2026-W37.  One round per calendar week."""
    override = os.getenv("TOR_WF_ROUND_ID")
    if override:
        return override
    now = now or datetime.now(timezone.utc)
    iso = now.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def round_dir(rid, arm_name):
    return os.path.join(LONGITUDINAL_DIR, rid, arm_name)


def tor_metadata(controller):
    """Version and consensus facts that explain drift later.

    A Tor release or a consensus parameter change shifts traffic shape without
    any website changing, so these are recorded per round as covariates.
    """
    meta = {}
    try:
        meta["tor_version"] = str(controller.get_version())
    except Exception:
        meta["tor_version"] = None
    for key, info in (
        ("consensus_valid_after", "consensus/valid-after"),
        ("consensus_fresh_until", "consensus/fresh-until"),
        ("traffic_read", "traffic/read"),
        ("traffic_written", "traffic/written"),
    ):
        try:
            meta[key] = controller.get_info(info)
        except Exception:
            meta[key] = None
    return meta


def host_metadata():
    def run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception:
            return None

    return {
        "capture_host": os.getenv("TOR_WF_HOST_LABEL", socket.gethostname()),
        "kernel": run(["uname", "-r"]),
        "tcpdump_version": (run(["tcpdump", "--version"]) or "").splitlines()[:1],
        "interface": detect_interface(),
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
    }


# Interstitials that produce a perfectly valid PCAP containing none of the
# target site's traffic.  Counting these as real samples is the single easiest
# way to fabricate drift that is not there.
BLOCK_PATTERNS = [
    r"just a moment",
    r"attention required",
    r"checking your browser",
    r"cloudflare",
    r"captcha",
    r"are you a robot",
    r"unusual traffic",
    r"access denied",
    r"403 forbidden",
    r"429 too many requests",
    r"blocked",
    r"verify you are human",
    r"enable javascript and cookies",
]
_BLOCK_RE = re.compile("|".join(BLOCK_PATTERNS), re.IGNORECASE)

MIN_PAGE_SOURCE_BYTES = env_int("TOR_WF_MIN_PAGE_BYTES", 2000)
MIN_PCAP_BYTES = env_int("TOR_WF_MIN_PCAP_BYTES", 5000)


def classify_page(driver, expected_host):
    """Decide whether a capture is a usable sample of the target site."""
    result = {
        "final_url_host": None,
        "title": None,
        "title_hash": None,
        "page_bytes": 0,
        "ready_state": None,
        "load_time_ms": None,
        "status": "ok",
        "detail": "",
    }
    try:
        source = driver.page_source or ""
        title = driver.title or ""
        result["title"] = title[:200]
        result["title_hash"] = hashlib.sha256(title.encode("utf-8", "replace")).hexdigest()[:16]
        result["page_bytes"] = len(source)
        result["ready_state"] = driver.execute_script("return document.readyState")
        try:
            from urllib.parse import urlparse

            result["final_url_host"] = urlparse(driver.current_url).hostname
        except Exception:
            pass
        try:
            timing = driver.execute_script(
                "var e = performance.getEntriesByType('navigation')[0];"
                "return e ? Math.round(e.duration) : null;"
            )
            result["load_time_ms"] = timing
        except Exception:
            pass

        probe = (title + " " + source[:4000]).lower()
        if _BLOCK_RE.search(probe):
            result["status"] = "blocked"
            result["detail"] = "interstitial or challenge page detected"
        elif len(source) < MIN_PAGE_SOURCE_BYTES:
            result["status"] = "empty"
            result["detail"] = f"page source only {len(source)} bytes"
        elif result["final_url_host"] and expected_host:
            # A redirect off the target registrable domain means we captured
            # something else entirely.
            if not _same_site(result["final_url_host"], expected_host):
                result["status"] = "redirected"
                result["detail"] = f"landed on {result['final_url_host']}"
    except Exception as exc:
        result["status"] = "browser_error"
        result["detail"] = str(exc)[:300]
    return result


def _same_site(host_a, host_b):
    a = ".".join(host_a.lower().split(".")[-2:])
    b = ".".join(host_b.lower().split(".")[-2:])
    return a == b


def pcap_ok(path):
    if not os.path.exists(path):
        return False, 0
    size = os.path.getsize(path)
    return size >= MIN_PCAP_BYTES, size


_tcpdump_mode = None


def tcpdump_mode():
    """How tcpdump can be run here: "direct", "sudo", or None if not at all.

    "direct" means the binary carries cap_net_raw, so no sudo is involved and
    the capture can be written straight to its final path. That is the only
    mode that works for an unattended weekly timer: a sudo password prompt has
    nobody to answer it at 02:00.
    """
    global _tcpdump_mode
    if _tcpdump_mode is not None:
        return _tcpdump_mode
    override = os.getenv("TOR_WF_TCPDUMP_MODE")
    if override in ("direct", "sudo"):
        _tcpdump_mode = override
        return _tcpdump_mode
    td = shutil.which("tcpdump")
    if td:
        iface = detect_interface()
        # "tcpdump -D" succeeds without capture privileges, so it proves
        # nothing. The only reliable probe is opening a real capture socket.
        for mode, prefix in (("direct", []), ("sudo", ["sudo", "-n"])):
            if _can_capture(prefix + [td, "-i", iface, "-c", "1", "-w", os.devnull]):
                _tcpdump_mode = mode
                return _tcpdump_mode
    _tcpdump_mode = None
    return _tcpdump_mode


def _can_capture(argv):
    """True if this command opens a capture socket rather than being refused."""
    try:
        r = subprocess.run(argv, capture_output=True, timeout=4, text=True)
    except subprocess.TimeoutExpired:
        # Still waiting for a matching packet, so the socket opened fine.
        return True
    except Exception:
        return False
    err = (r.stderr or "").lower()
    if "permitted" in err or "permission" in err or "password" in err:
        return False
    return r.returncode == 0


def tcpdump_argv(args):
    mode = tcpdump_mode()
    td = shutil.which("tcpdump") or "tcpdump"
    if mode == "sudo":
        return ["sudo", "-n", td] + args
    return [td] + args


def pkill_tcpdump_argv():
    """Kill a stuck tcpdump, with sudo only if the capture needed sudo."""
    if tcpdump_mode() == "sudo":
        return ["sudo", "-n", "pkill", "-x", "tcpdump"]
    return ["pkill", "-x", "tcpdump"]


def setcap_hint():
    td = shutil.which("tcpdump") or "/usr/bin/tcpdump"
    return f"sudo setcap cap_net_raw,cap_net_admin=eip {td}"


def browser_binary():
    """Path to the Chrome-family browser, or None to let Selenium decide.

    The collection host runs Chromium rather than Google Chrome, and Selenium
    will not find it by itself.
    """
    override = os.getenv("TOR_WF_BROWSER_BINARY")
    if override:
        return override
    for candidate in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        path = shutil.which(candidate)
        if path:
            return path
    return None


def chromedriver_binary():
    override = os.getenv("TOR_WF_CHROMEDRIVER")
    if override:
        return override
    return shutil.which("chromedriver")


def have(binary):
    return shutil.which(binary) is not None


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
