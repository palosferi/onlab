"""Environment check for the longitudinal collection host.

Run this once on the capture machine before the first round, and again after
any reboot or Tor upgrade.  Every failure it reports is a failure that would
otherwise show up as a round of silently empty PCAPs.

    python scripts/collection/preflight.py
"""

import os
import shutil
import socket
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import wf_config as cfg  # noqa: E402

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
results = []


def check(name, fn):
    try:
        state, detail = fn()
    except Exception as exc:
        state, detail = FAIL, f"{type(exc).__name__}: {exc}"
    results.append((state, name, detail))
    symbol = {PASS: "ok  ", FAIL: "FAIL", WARN: "warn"}[state]
    print(f"[{symbol}] {name}: {detail}")
    return state


def check_python_packages():
    missing = []
    for mod in ("selenium", "stem", "scapy", "pandas", "numpy", "sklearn"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        return FAIL, f"missing: {', '.join(missing)} (pip install selenium stem scapy pandas numpy scikit-learn)"
    return PASS, "selenium, stem, scapy, pandas, numpy, sklearn present"


def check_binaries():
    missing = [b for b in ("tcpdump", "ip") if not cfg.have(b)]
    chrome = next((b for b in ("google-chrome", "chromium", "chromium-browser") if cfg.have(b)), None)
    driver = cfg.have("chromedriver")
    if missing:
        return FAIL, f"missing binaries: {', '.join(missing)}"
    if not chrome:
        return FAIL, "no chrome/chromium binary found"
    if not driver:
        return WARN, f"{chrome} found, chromedriver not on PATH (Selenium Manager may fetch it)"
    return PASS, f"{chrome} + chromedriver + tcpdump"


def check_sudo():
    r = subprocess.run(["sudo", "-n", "tcpdump", "--version"],
                       capture_output=True, text=True, timeout=10)
    if r.returncode != 0:
        return FAIL, ("passwordless sudo for tcpdump not available. Add to sudoers: "
                      f"{os.getenv('USER', 'user')} ALL=(root) NOPASSWD: /usr/bin/tcpdump, /usr/bin/pkill, /usr/bin/mv, /usr/bin/chown")
    return PASS, "sudo -n tcpdump works"


def check_interface():
    iface = cfg.detect_interface()
    r = subprocess.run(["ip", "link", "show", iface], capture_output=True, text=True, timeout=5)
    if r.returncode != 0:
        return FAIL, f"interface {iface} not found"
    if "state UP" not in r.stdout and "UNKNOWN" not in r.stdout:
        return WARN, f"interface {iface} exists but is not UP"
    return PASS, f"capture interface {iface}"


def check_capture():
    iface = cfg.detect_interface()
    tmp = "/tmp/wf_preflight.pcap"
    p = subprocess.Popen(["sudo", "-n", "tcpdump", "-i", iface, "-w", tmp, "-c", "1", "tcp"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        socket.create_connection(("1.1.1.1", 443), timeout=4).close()
    except OSError:
        pass
    try:
        p.wait(timeout=8)
    except subprocess.TimeoutExpired:
        p.kill()
    ok = os.path.exists(tmp) and os.path.getsize(tmp) > 0
    subprocess.run(["sudo", "-n", "rm", "-f", tmp], check=False)
    if not ok:
        return FAIL, f"tcpdump produced no data on {iface} (AppArmor or SELinux may block writes to /tmp)"
    return PASS, f"tcpdump captured on {iface}"


def make_arm_check(arm):
    def fn():
        from stem.control import Controller

        try:
            c = Controller.from_port(port=arm.control_port)
        except Exception as exc:
            return FAIL, (f"control port {arm.control_port} unreachable ({exc}). "
                          f"See docs/LONGITUDINAL_SETUP.md for the two-instance Tor config.")
        with c:
            c.authenticate()
            version = c.get_version()
            try:
                boot = c.get_info("status/bootstrap-phase")
            except Exception:
                boot = "unknown"
            if "PROGRESS=100" not in str(boot):
                return FAIL, f"tor {version} not bootstrapped: {boot}"
            peer = cfg.resolve_peer_ip(c, arm)
            return PASS, (f"tor {version} bootstrapped, socks {arm.socks_port}, "
                          f"peer {peer} ({arm.guard_nickname})")

    return fn


def make_socks_check(arm):
    def fn():
        try:
            socket.create_connection(("127.0.0.1", arm.socks_port), timeout=4).close()
        except OSError:
            return FAIL, (f"nothing listening on SOCKS port {arm.socks_port}. "
                          f"Start the '{arm.name}' Tor instance "
                          "(see docs/LONGITUDINAL_SETUP.md step 1).")
        try:
            import requests
            import socks  # noqa: F401
        except ImportError:
            return WARN, (f"socks port {arm.socks_port} accepts connections "
                          "(pip install requests pysocks for a full end-to-end proxy test)")

        proxies = {
            "http": f"socks5h://127.0.0.1:{arm.socks_port}",
            "https": f"socks5h://127.0.0.1:{arm.socks_port}",
        }
        r = requests.get("https://check.torproject.org/api/ip", proxies=proxies, timeout=45)
        data = r.json()
        if not data.get("IsTor"):
            return FAIL, f"traffic through socks {arm.socks_port} is NOT going over Tor"
        return PASS, f"socks {arm.socks_port} exits Tor at {data.get('IP')}"

    return fn


def check_storage():
    path = cfg.LONGITUDINAL_DIR
    os.makedirs(path, exist_ok=True)
    probe = os.path.join(path, ".write_probe")
    with open(probe, "w") as f:
        f.write("ok")
    os.remove(probe)
    free_gb = shutil.disk_usage(path).free / 1e9
    arms = len(cfg.configured_arms())
    repeats = cfg.env_int("TOR_WF_REPEATS", 5)
    # Spring 2026 baseline captures averaged 5.6 MB per 18 second trace
    # (6309 MB over 1123 PCAPs), rounded up for obfs4 overhead.
    per_round_gb = 36 * repeats * arms * 0.006
    need = per_round_gb * 13
    state = PASS if free_gb > need else (WARN if free_gb > per_round_gb * 4 else FAIL)
    return state, (f"{free_gb:.0f} GB free, ~{per_round_gb:.1f} GB per round, "
                   f"~{need:.0f} GB for 13 rounds")


def check_t0():
    spring = os.path.join(cfg.REPO_ROOT, "tor_dataset", "extracted_features", "baseline_features")
    if not os.path.isdir(spring):
        return WARN, "spring baseline features not found, t0 comparison unavailable"
    n = len([f for f in os.listdir(spring) if f.endswith(".csv")])
    return PASS, f"{n} spring t0 feature files available"


def main():
    print("=== longitudinal collection preflight ===\n")
    check("python packages", check_python_packages)
    check("binaries", check_binaries)
    check("passwordless sudo", check_sudo)
    check("capture interface", check_interface)
    check("live capture", check_capture)
    for arm in cfg.configured_arms():
        check(f"tor arm '{arm.name}'", make_arm_check(arm))
        check(f"socks arm '{arm.name}'", make_socks_check(arm))
    check("storage", check_storage)
    check("spring t0 features", check_t0)

    fails = [r for r in results if r[0] == FAIL]
    warns = [r for r in results if r[0] == WARN]
    print(f"\n{len(results) - len(fails) - len(warns)} passed, {len(warns)} warnings, {len(fails)} failures")
    if fails:
        print("\nBlocking issues:")
        for _, name, detail in fails:
            print(f"  - {name}: {detail}")
        sys.exit(1)
    print("\nReady. Start a round with: scripts/collection/run_round.sh")


if __name__ == "__main__":
    main()
