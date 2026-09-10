"""Collect one longitudinal round.

A round is one sweep of the fixed 36-site target list, captured under every
enabled arm (plain Tor and Tor over obfs4), with the visit order randomised and
the two arms interleaved.

Interleaving is the point.  In the spring 2026 dataset all baseline traffic was
captured in March and all obfs4 traffic in April, so protocol and time were
perfectly confounded and no baseline-versus-obfs4 difference could be separated
from drift.  Collecting both arms inside the same round removes that.

Usage:
    python scripts/collection/collect_round.py [--repeats N] [--sites all|targets|other]
"""

import argparse
import csv
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import wf_config as cfg  # noqa: E402

MANIFEST_FIELDS = [
    "round_id",
    "arm",
    "site",
    "url",
    "repeat_index",
    "started_at",
    "pcap",
    "pcap_bytes",
    "status",
    "detail",
    "final_url_host",
    "title",
    "title_hash",
    "page_bytes",
    "ready_state",
    "load_time_ms",
    "peer_ip",
    "guard_fingerprint",
    "guard_nickname",
    "tor_version",
    "browser_version",
    "capture_seconds",
]


def build_driver(socks_port):
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service

    opts = Options()
    binary = cfg.browser_binary()
    if binary:
        opts.binary_location = binary
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument(f"--proxy-server=socks5://127.0.0.1:{socks_port}")
    # Chrome resolves hostnames locally even behind a SOCKS5 proxy, so the
    # target name leaks outside Tor.  Forcing remote resolution is the correct
    # behaviour but it perturbs load timing, and the spring 2026 t0 was
    # collected without it.  Comparability with t0 wins by default; enable it
    # only if the whole series is recollected under the new setting.
    if cfg.env_bool("TOR_WF_REMOTE_DNS", False):
        opts.add_argument("--host-resolver-rules=MAP * ~NOTFOUND , EXCLUDE 127.0.0.1")
    # Page load strategy is deliberately left at Selenium's default ("normal"),
    # matching the spring collector: get() returns on the load event and the
    # warmup plus capture sleep runs after it.
    # Use the system chromedriver when present so Selenium Manager does not
    # try to fetch one over the network mid-round.
    driver_path = cfg.chromedriver_binary()
    service = Service(executable_path=driver_path) if driver_path else None
    driver = webdriver.Chrome(options=opts, service=service) if service else webdriver.Chrome(options=opts)
    driver.set_page_load_timeout(cfg.env_int("TOR_WF_PAGE_TIMEOUT", 90))
    return driver


def new_circuit(control_port):
    try:
        from stem import Signal
        from stem.control import Controller

        with Controller.from_port(port=control_port) as c:
            c.authenticate()
            c.signal(Signal.NEWNYM)
        time.sleep(cfg.env_int("TOR_WF_NEWNYM_SLEEP", 6))
        return True
    except Exception as exc:
        print(f"[-] NEWNYM failed on {control_port}: {exc}")
        return False


def open_controller(control_port):
    from stem.control import Controller

    c = Controller.from_port(port=control_port)
    c.authenticate()
    return c


def capture_one(item, arm, interface, out_dir, browser_version, tor_version):
    site, url, repeat_index = item["site"], item["url"], item["repeat"]
    started_at = datetime.now()
    ts = started_at.strftime("%Y%m%d_%H%M%S")
    # Filename convention must stay <site>_<date>_<time>.pcap: the existing
    # feature pipeline recovers the class label by splitting on it.
    final_pcap = os.path.join(out_dir, f"{site}_{ts}.pcap")
    tmp_pcap = f"/tmp/wf_{arm.name}_{site}_{ts}.pcap"

    capture_seconds = cfg.env_int("TOR_WF_CAPTURE_DURATION", 15)
    warmup = cfg.env_int("TOR_WF_WARMUP_DURATION", 3)

    direct = cfg.tcpdump_mode() == "direct"
    if direct:
        # tcpdump carries cap_net_raw, so it writes straight to the final path
        # and no privileged move or ownership fix is needed afterwards.
        tmp_pcap = final_pcap

    row = {f: "" for f in MANIFEST_FIELDS}
    row.update(
        {
            "round_id": item["round_id"],
            "arm": arm.name,
            "site": site,
            "url": url,
            "repeat_index": repeat_index,
            "started_at": started_at.isoformat(),
            "pcap": os.path.relpath(final_pcap, cfg.REPO_ROOT),
            "peer_ip": arm.peer_ip,
            "guard_fingerprint": arm.guard_fingerprint or "",
            "guard_nickname": arm.guard_nickname or "",
            "tor_version": tor_version or "",
            "browser_version": browser_version or "",
            "capture_seconds": capture_seconds,
        }
    )

    tcpdump = subprocess.Popen(
        cfg.tcpdump_argv(["-i", interface, "-w", tmp_pcap, "-U",
                          "tcp", "and", "host", arm.peer_ip]),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    time.sleep(1)

    driver = None
    page = {"status": "browser_error", "detail": "driver not started"}
    try:
        driver = build_driver(arm.socks_port)
        row["browser_version"] = driver.capabilities.get("browserVersion", "")
        load_failed = False
        try:
            driver.get(url)
        except Exception as exc:
            load_failed = True
            page = {"status": "load_timeout", "detail": str(exc)[:300]}
        # Same post-load window as the spring collector, so trace length stays
        # comparable across the whole series.
        time.sleep(warmup + capture_seconds)
        if not load_failed:
            page = cfg.classify_page(driver, urlparse(url).hostname)
    except Exception as exc:
        page = {"status": "browser_error", "detail": str(exc)[:300]}
    finally:
        if driver is not None:
            try:
                driver.quit()
            except Exception:
                pass

    try:
        tcpdump.terminate()
        tcpdump.wait(timeout=5)
    except Exception:
        subprocess.run(cfg.pkill_tcpdump_argv(), check=False)
    time.sleep(1)

    if not direct and os.path.exists(tmp_pcap):
        subprocess.run(["sudo", "-n", "mv", tmp_pcap, final_pcap], check=False)
        subprocess.run(
            ["sudo", "-n", "chown", f"{os.getuid()}:{os.getgid()}", final_pcap], check=False
        )

    ok, size = cfg.pcap_ok(final_pcap)
    row["pcap_bytes"] = size
    for key in ("final_url_host", "title", "title_hash", "page_bytes",
                "ready_state", "load_time_ms"):
        if page.get(key) is not None:
            row[key] = page.get(key)

    status = page.get("status", "ok")
    detail = page.get("detail", "")
    if status == "ok" and not ok:
        status = "empty_capture"
        detail = f"pcap only {size} bytes, guard filter may be stale"
    row["status"] = status
    row["detail"] = detail

    # Rejected samples are kept on disk but quarantined, so the round can be
    # audited later and the reject rate itself is a measurable quantity.
    if status != "ok" and os.path.exists(final_pcap):
        reject_dir = os.path.join(out_dir, "_rejected")
        os.makedirs(reject_dir, exist_ok=True)
        moved = os.path.join(reject_dir, os.path.basename(final_pcap))
        os.replace(final_pcap, moved)
        row["pcap"] = os.path.relpath(moved, cfg.REPO_ROOT)

    marker = {"ok": "+", "blocked": "!", "empty_capture": "!"}.get(status, "-")
    print(
        f"[{marker}] {arm.name:8s} {site:14s} r{repeat_index} "
        f"{status:14s} pcap={size:>8d}B page={row['page_bytes'] or 0}B {detail}"
    )
    return row


def existing_counts(out_dir):
    counts = {}
    if not os.path.isdir(out_dir):
        return counts
    for name in os.listdir(out_dir):
        if name.endswith(".pcap"):
            site = name.rsplit("_", 2)[0]
            counts[site] = counts.get(site, 0) + 1
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=cfg.env_int("TOR_WF_REPEATS", 5),
                        help="captures per site per arm in this round")
    parser.add_argument("--sites", choices=["targets", "other", "all"], default="targets")
    parser.add_argument("--resume", action="store_true",
                        help="skip site/arm pairs that already have enough captures")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rid = cfg.round_id()
    interface = cfg.detect_interface()
    arms = cfg.configured_arms()
    if not arms:
        sys.exit("No arms enabled. Set TOR_WF_COLLECT_BASELINE or TOR_WF_COLLECT_OBFS4.")

    site_map = {}
    if args.sites in ("targets", "all"):
        site_map.update(cfg.TARGET_SITES)
    if args.sites in ("other", "all"):
        site_map.update(cfg.OTHER_SITES)

    print(f"=== round {rid} | interface {interface} | arms {[a.name for a in arms]} ===")

    if args.dry_run:
        total = len(site_map) * args.repeats * len(arms)
        print(f"    dry run: {total} captures would be queued "
              f"({len(site_map)} sites x {args.repeats} repeats x {len(arms)} arms)")
        print(f"    estimated duration: {total * 37 / 3600:.1f} h")
        for arm in arms:
            print(f"    {arm.name}: socks {arm.socks_port}, control {arm.control_port}, "
                  f"output {cfg.round_dir(rid, arm.name)}")
        return

    controllers = {}
    tor_versions = {}
    for arm in arms:
        try:
            c = open_controller(arm.control_port)
        except Exception as exc:
            sys.exit(
                f"Cannot reach Tor control port {arm.control_port} for arm '{arm.name}': {exc}\n"
                "Run scripts/collection/preflight.py for a full environment check."
            )
        controllers[arm.name] = c
        cfg.resolve_peer_ip(c, arm)
        meta = cfg.tor_metadata(c)
        tor_versions[arm.name] = meta.get("tor_version")
        print(f"    {arm.name}: peer={arm.peer_ip} guard={arm.guard_nickname} tor={meta.get('tor_version')}")

    work = []
    for arm in arms:
        out_dir = cfg.round_dir(rid, arm.name)
        have = existing_counts(out_dir) if args.resume else {}
        for site, url in site_map.items():
            done = have.get(site, 0)
            for repeat in range(done + 1, args.repeats + 1):
                work.append({"arm": arm.name, "site": site, "url": url,
                             "repeat": repeat, "round_id": rid})

    # Deterministic shuffle: reproducible from the round id, but a network
    # outage no longer wipes one site's entire round.
    random.Random(rid).shuffle(work)
    print(f"    {len(work)} captures queued "
          f"({len(site_map)} sites x {args.repeats} repeats x {len(arms)} arms)")

    arm_by_name = {a.name: a for a in arms}
    for arm in arms:
        os.makedirs(cfg.round_dir(rid, arm.name), exist_ok=True)

    meta_path = os.path.join(cfg.LONGITUDINAL_DIR, rid, "round_meta.json")
    cfg.write_json(meta_path, {
        "round_id": rid,
        "host": cfg.host_metadata(),
        "arms": {a.name: {"peer_ip": a.peer_ip, "guard_fingerprint": a.guard_fingerprint,
                          "guard_nickname": a.guard_nickname, "socks_port": a.socks_port,
                          "tor": cfg.tor_metadata(controllers[a.name])} for a in arms},
        "repeats": args.repeats,
        "site_count": len(site_map),
        "queued_captures": len(work),
        "started_at": datetime.now().isoformat(),
    })

    manifest_path = os.path.join(cfg.LONGITUDINAL_DIR, rid, "manifest.csv")
    write_header = not os.path.exists(manifest_path)
    browser_version = None
    summary = {}

    with open(manifest_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        if write_header:
            writer.writeheader()

        for i, item in enumerate(work, start=1):
            arm = arm_by_name[item["arm"]]
            out_dir = cfg.round_dir(rid, arm.name)
            row = capture_one(item, arm, interface, out_dir, browser_version,
                              tor_versions.get(arm.name))
            browser_version = row.get("browser_version") or browser_version
            writer.writerow(row)
            fh.flush()
            summary[row["status"]] = summary.get(row["status"], 0) + 1

            # A run of empty captures almost always means the guard rotated
            # mid-round and the tcpdump filter is now pointing at nothing.
            if row["status"] == "empty_capture":
                try:
                    cfg.resolve_peer_ip(controllers[arm.name], arm)
                    print(f"    re-resolved {arm.name} peer -> {arm.peer_ip}")
                except Exception as exc:
                    print(f"    peer re-resolve failed: {exc}")

            new_circuit(arm.control_port)
            if i % 25 == 0:
                print(f"--- {i}/{len(work)} done | {summary} ---")

    for c in controllers.values():
        try:
            c.close()
        except Exception:
            pass

    total = sum(summary.values())
    ok = summary.get("ok", 0)
    print(f"\n=== round {rid} finished: {ok}/{total} usable ({ok / max(total, 1):.1%}) ===")
    print(f"    {summary}")
    print(f"    manifest: {manifest_path}")

    meta = {}
    if os.path.exists(meta_path):
        import json
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    meta["finished_at"] = datetime.now().isoformat()
    meta["status_counts"] = summary
    meta["usable_ratio"] = round(ok / max(total, 1), 4)
    cfg.write_json(meta_path, meta)


if __name__ == "__main__":
    main()
