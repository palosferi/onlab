import json
import os
import socket
import subprocess
import time
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from stem import Signal
from stem.control import Controller

# --- CONFIGURATION ---
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

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PCAP_DIR = os.getenv(
    "TOR_WF_PCAP_DIR",
    os.path.join(REPO_ROOT, "tor_dataset", "other_tor"),
)
LOG_JSONL = os.getenv(
    "TOR_WF_LOG_JSONL",
    os.path.join(REPO_ROOT, "logs", "collection_other_events.jsonl"),
)
NETWORK_INTERFACE = os.getenv("TOR_WF_INTERFACE", "enp3s0")
GUARD_IP = os.getenv("TOR_WF_GUARD_IP", "109.110.170.208")
CAPTURE_DURATION = int(os.getenv("TOR_WF_CAPTURE_DURATION", "15"))
WARMUP_DURATION = int(os.getenv("TOR_WF_WARMUP_DURATION", "3"))
ENABLE_NEWNYM = os.getenv("TOR_WF_ENABLE_NEWNYM", "1") == "1"
MIN_PCAP_BYTES = int(os.getenv("TOR_WF_MIN_PCAP_BYTES", "2000"))
TOR_SOCKS_HOST = os.getenv("TOR_WF_SOCKS_HOST", "127.0.0.1")
TOR_SOCKS_PORT = int(os.getenv("TOR_WF_SOCKS_PORT", "9050"))
TOR_CONTROL_HOST = os.getenv("TOR_WF_CONTROL_HOST", "127.0.0.1")
TOR_CONTROL_PORT = int(os.getenv("TOR_WF_CONTROL_PORT", "9051"))

_newnym_disabled_logged = False


def append_event(event):
    os.makedirs(os.path.dirname(LOG_JSONL), exist_ok=True)
    with open(LOG_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=True) + "\n")


def verify_pcap_dir(path):
    os.makedirs(path, exist_ok=True)
    probe = os.path.join(path, ".write_probe")
    with open(probe, "w", encoding="utf-8") as f:
        f.write("ok")
    os.remove(probe)


def is_guard_active(guard_ip, timeout_sec=2):
    try:
        socket.create_connection((guard_ip, 443), timeout=timeout_sec).close()
        return True
    except OSError:
        return False


def is_port_open(host, port, timeout_sec=2):
    try:
        socket.create_connection((host, port), timeout=timeout_sec).close()
        return True
    except OSError:
        return False


def get_tor_browser():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument(f"--proxy-server=socks5://{TOR_SOCKS_HOST}:{TOR_SOCKS_PORT}")
    return webdriver.Chrome(options=chrome_options)


def request_new_tor_circuit():
    global _newnym_disabled_logged
    if not ENABLE_NEWNYM:
        if not _newnym_disabled_logged:
            print("[!] NEWNYM disabled (TOR_WF_ENABLE_NEWNYM=0).")
            _newnym_disabled_logged = True
        return

    try:
        with Controller.from_port(address=TOR_CONTROL_HOST, port=TOR_CONTROL_PORT) as controller:
            controller.authenticate()
            controller.signal(Signal.NEWNYM)
            print("[+] Requested new Tor circuit.")
            time.sleep(5)
    except Exception as e:
        print(f"[-] Could not request new circuit: {e}")


def capture_site(site_name, url):
    print(f"\n[*] Capturing {site_name}: {url}")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_pcap = f"/tmp/{site_name}_{ts}.pcap"
    final_pcap = os.path.join(PCAP_DIR, f"{site_name}_{ts}.pcap")

    if not is_guard_active(GUARD_IP):
        print(f"[-] Guard pre-check failed for {GUARD_IP}; skipping {site_name}.")
        append_event({"time": datetime.now().isoformat(), "site": site_name, "status": "guard_inactive"})
        return

    cmd = [
        "sudo",
        "tcpdump",
        "-i",
        NETWORK_INTERFACE,
        "-w",
        tmp_pcap,
        "tcp",
        "and",
        "host",
        GUARD_IP,
    ]

    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1)

    status = "ok"
    error_message = ""

    driver = get_tor_browser()
    try:
        print(f"[*] Navigating: {url}")
        driver.get(url)
        time.sleep(WARMUP_DURATION)
        time.sleep(CAPTURE_DURATION)
    except Exception as e:
        status = "browser_error"
        error_message = str(e)
        print(f"[-] Browser error for {site_name}: {e}")
    finally:
        driver.quit()

    subprocess.run(["sudo", "pkill", "-x", "tcpdump"], check=False)
    time.sleep(1)

    pcap_saved = False
    pcap_size_bytes = 0
    if os.path.exists(tmp_pcap):
        subprocess.run(["sudo", "mv", tmp_pcap, final_pcap], check=False)
        subprocess.run(["sudo", "chown", f"{os.getuid()}:{os.getgid()}", final_pcap], check=False)
        pcap_saved = os.path.exists(final_pcap)
        if pcap_saved:
            pcap_size_bytes = os.path.getsize(final_pcap)
            if pcap_size_bytes < MIN_PCAP_BYTES and status == "ok":
                status = "pcap_too_small"
                error_message = f"small pcap ({pcap_size_bytes} bytes)"
                print(f"[!] {site_name} capture too small: {pcap_size_bytes} bytes")
    else:
        status = "capture_error"

    append_event(
        {
            "time": datetime.now().isoformat(),
            "site": site_name,
            "url": url,
            "status": status,
            "pcap": final_pcap,
            "pcap_saved": pcap_saved,
            "pcap_size_bytes": pcap_size_bytes,
            "warmup_sec": WARMUP_DURATION,
            "capture_sec": CAPTURE_DURATION,
            "error": error_message,
        }
    )


def main():
    global ENABLE_NEWNYM

    verify_pcap_dir(PCAP_DIR)
    print(f"[*] Saving Other pcaps to: {PCAP_DIR}")

    if not is_port_open(TOR_SOCKS_HOST, TOR_SOCKS_PORT):
        print(
            f"[-] Tor SOCKS proxy is not reachable at {TOR_SOCKS_HOST}:{TOR_SOCKS_PORT}. "
            "Start Tor before collection."
        )
        return

    if ENABLE_NEWNYM and not is_port_open(TOR_CONTROL_HOST, TOR_CONTROL_PORT):
        print(
            f"[!] Tor ControlPort is not reachable at {TOR_CONTROL_HOST}:{TOR_CONTROL_PORT}. "
            "Continuing with TOR_WF_ENABLE_NEWNYM=0."
        )
        ENABLE_NEWNYM = False

    for site_name, url in OTHER_SITES.items():
        capture_site(site_name, url)
        request_new_tor_circuit()


if __name__ == "__main__":
    main()
