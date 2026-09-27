"""Collect the thesis dataset with the real Tor Browser.

The spring and longitudinal collectors drove headless Chromium through a SOCKS
proxy. That is not what a Tor user runs: Tor Browser differs in its HTTP stack,
its first-party circuit isolation, its fingerprinting defences and its request
pattern, and a classifier trained on Chromium traffic says little about it.
This collector drives Tor Browser itself (tbselenium), pointed at our own
per-transport Tor instances so the three arms differ in transport only.

What one run collects, all interleaved in the same schedule:

  monitored    50 sites x --repeats visits, under every arm;
  unmonitored  the open-world pool, one visit per site under every arm;
  background   baseline arm only: a monitored visit with a second, unrelated
               site loading in another tab (the background-tab ablation).

Interleaving is the methodological point. The schedule is cut into --repeats
batches. Batch k holds the k-th visit of every monitored site, a k-th share of
the unmonitored pool and of the background visits, shuffled; within a slot the
arms are visited back to back in random order. So no arm, and no kind of
sample, is concentrated at any one time, and a chronological split on repeat
index is a split in time for all of them alike.

Two collection details replace fixed sleeps:

  * the capture ends when the page has settled -- the load event has fired and
    no new resource has started for --quiet seconds -- instead of after a
    fixed 18 s, so a slow Snowflake load is not cut short and a fast page is
    not padded with idle keep-alives;
  * Tor Browser is launched before the capture starts and quit after it stops,
    so browser start-up and shutdown traffic stays out of the trace.

Usage (via run_tb.sh, which sets the environment):
    collect_tb.py --collection tb-2026-10 --repeats 40 --bg-repeats 20
    collect_tb.py --collection tb-pilot --repeats 2 --only wikipedia,bbc --unmonitored-limit 5
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

import sites  # noqa: E402
import wf_config as cfg  # noqa: E402

TB_DIR = os.path.expanduser(os.getenv("TOR_WF_TB_DIR", "~/tor_wf_runtime/tb/tor-browser"))
GECKODRIVER = os.path.expanduser(os.getenv("TOR_WF_GECKODRIVER", "~/tor_wf_runtime/tb/geckodriver"))
TB_ROOT = os.getenv("TOR_WF_TB_OUT", os.path.join(cfg.REPO_ROOT, "tor_dataset", "tb"))

PAGE_TIMEOUT = cfg.env_int("TOR_WF_PAGE_TIMEOUT", 90)
# Statuses caused by the network rather than the site: worth one more try.
TRANSIENT = {"load_timeout", "browser_error", "empty_capture", "load_error"}

MANIFEST_FIELDS = [
    "collection", "batch", "kind", "arm", "site", "url", "repeat", "attempt",
    "started_at", "pcap", "pcap_bytes", "status", "detail",
    "final_url_host", "title", "page_bytes", "ready_state",
    "load_time_ms", "settle_s", "capture_s", "resources",
    "bg_site", "bg_url", "bg_lead_s",
    "peer_ip", "guard_fingerprint", "tor_version", "browser_version",
]


# --------------------------------------------------------------------------
# schedule


def build_schedule(monitored, unmonitored, background, arms, repeats, bg_repeats, seed):
    """Return a list of batches; each batch is a shuffled list of slots.

    A slot is one site visit under one or more arms, visited back to back.
    Pure function of its inputs, so a resumed run rebuilds the same schedule.
    """
    rng = random.Random(seed)
    arm_names = [a for a in arms]
    unmon_items = list(unmonitored.items())
    bg_items = list(background.items())
    batches = [[] for _ in range(repeats)]

    for k in range(repeats):
        for site, url in monitored.items():
            batches[k].append({"kind": "monitored", "site": site, "url": url,
                               "repeat": k + 1, "arms": list(arm_names)})

    # Unmonitored sites are visited once each, spread evenly over the batches.
    for i, (site, url) in enumerate(unmon_items):
        batches[i * repeats // max(len(unmon_items), 1)].append(
            {"kind": "unmonitored", "site": site, "url": url, "repeat": 1,
             "arms": list(arm_names)})

    if bg_repeats and bg_items and "baseline" in arm_names:
        for b in range(bg_repeats):
            k = b * repeats // bg_repeats
            for site, url in monitored.items():
                bg_site, bg_url = bg_items[rng.randrange(len(bg_items))]
                batches[k].append({"kind": "background", "site": site, "url": url,
                                   "repeat": b + 1, "arms": ["baseline"],
                                   "bg_site": bg_site, "bg_url": bg_url,
                                   "bg_lead_s": round(rng.uniform(0.5, 3.0), 2)})

    for k, batch in enumerate(batches):
        rng_k = random.Random(f"{seed}-{k}")
        rng_k.shuffle(batch)
        for slot in batch:
            rng_k.shuffle(slot["arms"])
    return batches


def capture_key(kind, arm, site, repeat):
    return (kind, arm, site, int(repeat))


def read_progress(manifest_path):
    """(done keys, attempts per key) from an existing manifest."""
    done, attempts = set(), {}
    if not os.path.exists(manifest_path):
        return done, attempts
    with open(manifest_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = capture_key(row["kind"], row["arm"], row["site"], row["repeat"])
            attempts[key] = attempts.get(key, 0) + 1
            if row["status"] == "ok":
                done.add(key)
    return done, attempts


def needs_capture(key, done, attempts, max_attempts):
    return key not in done and attempts.get(key, 0) < max_attempts


# --------------------------------------------------------------------------
# browser and capture


def launch_browser(arm):
    from tbselenium.tbdriver import TorBrowserDriver
    import tbselenium.common as cm

    driver = TorBrowserDriver(
        TB_DIR,
        tor_cfg=cm.USE_RUNNING_TOR,
        socks_port=arm.socks_port,
        control_port=arm.control_port,
        headless=True,
        executable_path=GECKODRIVER,
    )
    driver.set_page_load_timeout(PAGE_TIMEOUT)
    return driver


def wait_settled(driver, quiet, max_wait):
    """Wait until no new resource has started for `quiet` seconds.

    The load event alone is too early for pages that keep fetching after it
    (lazy images, ad auctions), and a fixed sleep is wrong in both directions.
    Returns (seconds waited, resource count).
    """
    start = time.time()
    last_count, last_change = -1, start
    count = 0
    while time.time() - start < max_wait:
        try:
            count = driver.execute_script("return performance.getEntriesByType('resource').length")
        except Exception:
            break
        if count != last_count:
            last_count, last_change = count, time.time()
        elif time.time() - last_change >= quiet:
            break
        time.sleep(0.5)
    return round(time.time() - start, 2), count


class PortWatch:
    """Poll lyrebird's local UDP ports in the background; keep the union."""

    def __init__(self, interval=0.5):
        import threading

        self.seen = set(cfg.snowflake_ports())
        self.changes = 0
        self._last = set(self.seen)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(interval,), daemon=True)
        self._thread.start()

    def _run(self, interval):
        while not self._stop.wait(interval):
            now = set(cfg.snowflake_ports())
            if now and now != self._last:
                self.changes += 1
                self._last = now
            self.seen |= now

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)
        return sorted(self.seen)


def cut_to_ports(raw, out, ports):
    """Keep only packets on `ports` from the all-UDP scratch capture."""
    if os.path.exists(raw) and ports:
        subprocess.run(["tcpdump", "-r", raw, "-w", out] + cfg.capture_filter([], udp_ports=ports),
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    if os.path.exists(raw):
        os.remove(raw)


def stop_tcpdump(proc):
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        subprocess.run(cfg.pkill_tcpdump_argv(), check=False)


def capture_one(slot, arm, attempt, ctx):
    from selenium.common.exceptions import TimeoutException

    kind, site, url = slot["kind"], slot["site"], slot["url"]
    started = datetime.now()
    ts = started.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(ctx["root"], arm.name, kind)
    os.makedirs(out_dir, exist_ok=True)
    # <label>_<date>_<time>.pcap: the feature pipeline recovers label and time
    # from the name.
    pcap = os.path.join(out_dir, f"{site}_{ts}.pcap")

    row = {f: "" for f in MANIFEST_FIELDS}
    row.update({
        "collection": ctx["collection"], "batch": slot["batch"], "kind": kind,
        "arm": arm.name, "site": site, "url": url, "repeat": slot["repeat"],
        "attempt": attempt, "started_at": started.isoformat(),
        "pcap": os.path.relpath(pcap, cfg.REPO_ROOT),
        "guard_fingerprint": arm.guard_fingerprint or "",
        "tor_version": ctx["tor_versions"].get(arm.name, ""),
        "bg_site": slot.get("bg_site", ""), "bg_url": slot.get("bg_url", ""),
        "bg_lead_s": slot.get("bg_lead_s", ""),
    })

    driver = None
    try:
        driver = launch_browser(arm)
        row["browser_version"] = driver.capabilities.get("browserVersion", "")
    except Exception as exc:
        row.update(status="browser_error", detail=f"launch: {str(exc)[:250]}", pcap="")
        return row

    ports = None
    if arm.name == "snowflake":
        # The WebRTC ports move whenever the volunteer proxy changes, and the
        # pilot saw that mid-capture in one visit out of four. A filter fixed
        # at the start would silently drop the rest of the page load, so all
        # UDP is captured to a scratch file while lyrebird's ports are polled,
        # and the trace is cut down to the union of them afterwards.
        ports = PortWatch()
        raw = pcap + ".raw"
        bpf = ["udp"]
    else:
        raw = pcap
        row["peer_ip"] = ",".join(arm.peer_ips or [arm.peer_ip])
        bpf = cfg.capture_filter(arm.peer_ips or [arm.peer_ip])

    tcpdump = subprocess.Popen(
        cfg.tcpdump_argv(["-i", ctx["interface"], "-w", raw, "-U"] + bpf),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(1)
    t0 = time.time()

    page = None
    try:
        main_tab = driver.current_window_handle
        if kind == "background":
            driver.switch_to.new_window("tab")
            # Assigning location returns at once, unlike get(), so the noise
            # page keeps loading while the target loads in the first tab.
            driver.execute_script("window.location.href = arguments[0]", slot["bg_url"])
            driver.switch_to.window(main_tab)
            time.sleep(slot["bg_lead_s"])
        try:
            driver.get(url)
        except TimeoutException as exc:
            page = {"status": "load_timeout", "detail": str(exc).splitlines()[0][:200]}
            try:
                driver.execute_script("window.stop()")
            except Exception:
                pass
        except Exception:
            # Firefox raises on a network error but still shows its error
            # page, which classify_page below recognises and names.
            pass
        settle_s, resources = wait_settled(driver, ctx["quiet"], ctx["max_settle"])
        row["settle_s"], row["resources"] = settle_s, resources
        time.sleep(ctx["tail"])
    except Exception as exc:
        page = {"status": "browser_error", "detail": str(exc)[:250]}

    stop_tcpdump(tcpdump)
    row["capture_s"] = round(time.time() - t0, 2)
    if ports is not None:
        seen = ports.stop()
        row["peer_ip"] = "udp:" + ",".join(str(p) for p in seen)
        cut_to_ports(raw, pcap, seen)

    try:
        if page is None:
            driver.switch_to.window(driver.window_handles[0])
            page = cfg.classify_page(driver, urlparse(url).hostname)
    except Exception as exc:
        page = {"status": "browser_error", "detail": str(exc)[:250]}
    finally:
        try:
            driver.quit()
        except Exception:
            pass

    for key in ("final_url_host", "title", "page_bytes", "ready_state", "load_time_ms"):
        if page.get(key) is not None:
            row[key] = page[key]
    status, detail = page.get("status", "ok"), page.get("detail", "")

    ok, size = cfg.pcap_ok(pcap)
    row["pcap_bytes"] = size
    if ports is not None and ports.changes:
        detail = (detail + "; " if detail else "") + f"snowflake ports changed {ports.changes}x, all kept"
    if status == "ok" and not ok:
        status, detail = "empty_capture", f"pcap only {size} bytes"
    row["status"], row["detail"] = status, detail

    if status != "ok" and os.path.exists(pcap):
        reject_dir = os.path.join(out_dir, "_rejected")
        os.makedirs(reject_dir, exist_ok=True)
        moved = os.path.join(reject_dir, os.path.basename(pcap))
        os.replace(pcap, moved)
        row["pcap"] = os.path.relpath(moved, cfg.REPO_ROOT)
    return row


def new_identity(arm):
    """NEWNYM on this arm's Tor, so the next visit gets fresh circuits."""
    try:
        from stem import Signal
        from stem.control import Controller

        with Controller.from_port(port=arm.control_port) as c:
            c.authenticate()
            c.signal(Signal.NEWNYM)
    except Exception as exc:
        print(f"    NEWNYM failed on {arm.name}: {exc}")


# --------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--collection", required=True, help="output folder name under tor_dataset/tb/")
    p.add_argument("--repeats", type=int, default=40, help="visits per monitored site per arm")
    p.add_argument("--bg-repeats", type=int, default=20,
                   help="background-tab visits per monitored site (baseline only; 0 = off)")
    p.add_argument("--unmonitored-limit", type=int, default=0, help="use only the first N (0 = all)")
    p.add_argument("--only", default="", help="comma separated monitored sites, for pilots")
    p.add_argument("--seed", default="thesis", help="schedule seed; keep it fixed to resume")
    p.add_argument("--max-attempts", type=int, default=2)
    p.add_argument("--quiet", type=float, default=2.0, help="seconds without a new resource = settled")
    p.add_argument("--max-settle", type=float, default=15.0)
    p.add_argument("--tail", type=float, default=1.0, help="seconds captured after settling")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    monitored = dict(sites.MONITORED)
    if args.only:
        wanted = [s.strip() for s in args.only.split(",") if s.strip()]
        unknown = [s for s in wanted if s not in monitored]
        if unknown:
            sys.exit(f"unknown monitored site(s): {', '.join(unknown)}")
        monitored = {s: monitored[s] for s in wanted}
    unmonitored = sites.load_pool("unmonitored")
    background = sites.load_pool("background")
    if args.unmonitored_limit:
        unmonitored = dict(list(unmonitored.items())[:args.unmonitored_limit])
    if not unmonitored:
        print("warning: sites/unmonitored.txt is empty; run build_site_lists.py for the open world")
    if args.bg_repeats and not background:
        sys.exit("background-tab visits need sites/background.txt; run build_site_lists.py or pass --bg-repeats 0")

    arms = cfg.configured_arms()
    if not arms:
        sys.exit("no arms enabled")
    arm_by_name = {a.name: a for a in arms}

    batches = build_schedule(monitored, unmonitored, background, list(arm_by_name),
                             args.repeats, args.bg_repeats, args.seed)
    root = os.path.join(TB_ROOT, args.collection)
    manifest_path = os.path.join(root, "manifest.csv")
    done, attempts = read_progress(manifest_path)

    todo = sum(1 for batch in batches for slot in batch for a in slot["arms"]
               if needs_capture(capture_key(slot["kind"], a, slot["site"], slot["repeat"]),
                                done, attempts, args.max_attempts))
    total = sum(len(slot["arms"]) for batch in batches for slot in batch)
    print(f"=== {args.collection}: {len(monitored)} monitored x {args.repeats}, "
          f"{len(unmonitored)} unmonitored, background {args.bg_repeats} x {len(monitored)}, "
          f"arms {list(arm_by_name)} ===")
    print(f"    {total} captures in {len(batches)} batches, {len(done)} already done, {todo} to go "
          f"(~{todo * 40 / 3600:.0f} h at 40 s each)")
    if args.dry_run:
        return

    for path, what in ((TB_DIR, "Tor Browser"), (GECKODRIVER, "geckodriver")):
        if not os.path.exists(path):
            sys.exit(f"{what} not found at {path}")
    if cfg.tcpdump_mode() is None:
        sys.exit(f"tcpdump cannot capture here; {cfg.setcap_hint()}")

    from stem.control import Controller

    tor_versions = {}
    for arm in arms:
        with Controller.from_port(port=arm.control_port) as c:
            c.authenticate()
            cfg.resolve_peer_ip(c, arm)
            tor_versions[arm.name] = cfg.tor_metadata(c).get("tor_version")
        print(f"    {arm.name}: socks {arm.socks_port}, peer {arm.peer_ip}, guard {arm.guard_nickname}")

    os.makedirs(root, exist_ok=True)
    meta_path = os.path.join(root, f"run_{datetime.now():%Y%m%d_%H%M%S}.json")
    cfg.write_json(meta_path, {
        "collection": args.collection, "args": vars(args), "host": cfg.host_metadata(),
        "tor_browser": TB_DIR, "tor_versions": tor_versions,
        "arms": {a.name: {"socks_port": a.socks_port, "peer_ip": a.peer_ip,
                          "guard_fingerprint": a.guard_fingerprint} for a in arms},
        "counts": {"monitored": len(monitored), "unmonitored": len(unmonitored),
                   "background_pool": len(background), "captures_total": total},
    })

    ctx = {"collection": args.collection, "root": root, "interface": cfg.detect_interface(),
           "tor_versions": tor_versions, "quiet": args.quiet, "max_settle": args.max_settle,
           "tail": args.tail}

    write_header = not os.path.exists(manifest_path)
    tally, streak, n = {}, {a: 0 for a in arm_by_name}, 0
    with open(manifest_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        if write_header:
            writer.writeheader()
        for k, batch in enumerate(batches):
            for slot in batch:
                slot["batch"] = k
                for arm_name in slot["arms"]:
                    arm = arm_by_name[arm_name]
                    key = capture_key(slot["kind"], arm_name, slot["site"], slot["repeat"])
                    while needs_capture(key, done, attempts, args.max_attempts):
                        attempts[key] = attempts.get(key, 0) + 1
                        row = capture_one(slot, arm, attempts[key], ctx)
                        writer.writerow(row)
                        fh.flush()
                        new_identity(arm)
                        n += 1
                        status = row["status"]
                        tally[status] = tally.get(status, 0) + 1
                        mark = "+" if status == "ok" else "-"
                        print(f"[{mark}] b{k:02d} {slot['kind'][:5]} {arm_name:9s} {slot['site'][:22]:22s} "
                              f"r{slot['repeat']:<2} {status:13s} load={row['load_time_ms'] or '-'}ms "
                              f"settle={row['settle_s'] or '-'}s pcap={row['pcap_bytes'] or 0}B "
                              f"{row['detail'][:60]}", flush=True)
                        if status == "ok":
                            done.add(key)
                            streak[arm_name] = 0
                            break
                        streak[arm_name] += 1
                        if status == "empty_capture" or streak[arm_name] % 5 == 0:
                            # A run of failures on one arm usually means its
                            # guard or bridge address moved under the filter.
                            try:
                                with Controller.from_port(port=arm.control_port) as c:
                                    c.authenticate()
                                    cfg.resolve_peer_ip(c, arm)
                                print(f"    re-resolved {arm_name} peer -> {arm.peer_ip}")
                            except Exception as exc:
                                print(f"    {arm_name} re-resolve failed: {exc}")
                        if streak[arm_name] >= 10:
                            print(f"    {arm_name}: {streak[arm_name]} failures in a row, pausing 5 min")
                            time.sleep(300)
                        if status not in TRANSIENT:
                            break
                    if n and n % 50 == 0:
                        print(f"--- {n} captures this run | {tally} ---", flush=True)
            print(f"=== batch {k + 1}/{len(batches)} done | {tally} ===", flush=True)

    ok = tally.get("ok", 0)
    print(f"\n=== finished: {ok}/{n} usable this run | {tally} ===")


if __name__ == "__main__":
    main()
