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
TARGET_SITES = {
    # Static/low drift group
    "wikipedia": "https://en.wikipedia.org/wiki/Main_Page",
    "duckduckgo": "https://duckduckgo.com",
    "github": "https://github.com",
    "stackoverflow": "https://stackoverflow.com",
    "mit_ocw": "https://ocw.mit.edu",
    "gnu": "https://www.gnu.org",
    "debian": "https://www.debian.org",
    "w3c": "https://www.w3.org",
    # News/high drift group
    "bbc": "https://www.bbc.com",
    "cnn": "https://www.cnn.com",
    "reuters": "https://www.reuters.com",
    "theguardian": "https://www.theguardian.com/international",
    "aljazeera": "https://www.aljazeera.com",
    "hacker_news": "https://news.ycombinator.com",
    "wired": "https://www.wired.com",
    "techcrunch": "https://techcrunch.com",
    # HU sites
    "telex": "https://telex.hu",
    "index": "https://index.hu",
    "hvg": "https://hvg.hu",
    "origo": "https://www.origo.hu",
    "hwsw": "https://www.hwsw.hu",
    "bme": "https://www.bme.hu",
    # E-commerce + entertainment
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

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PCAP_DIR = os.getenv("TOR_WF_PCAP_DIR", os.path.join(REPO_ROOT, "tor_dataset", "baseline_tor"))
LOG_JSONL = os.getenv("TOR_WF_LOG_JSONL", os.path.join(REPO_ROOT, "logs", "collection_events.jsonl"))
NETWORK_INTERFACE = os.getenv("TOR_WF_INTERFACE", "enp3s0")
GUARD_IP = os.getenv("TOR_WF_GUARD_IP", "109.110.170.208")
CAPTURE_DURATION = int(os.getenv("TOR_WF_CAPTURE_DURATION", "15"))
WARMUP_DURATION = int(os.getenv("TOR_WF_WARMUP_DURATION", "3"))


def append_event(event):
    os.makedirs(os.path.dirname(LOG_JSONL), exist_ok=True)
    with open(LOG_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=True) + "\n")


def verify_pcap_dir(path):
    os.makedirs(path, exist_ok=True)
    probe_path = os.path.join(path, ".write_probe")
    with open(probe_path, "w", encoding="utf-8") as f:
        f.write("ok")
    os.remove(probe_path)


def is_guard_active(guard_ip, timeout_sec=2):
    try:
        socket.create_connection((guard_ip, 443), timeout=timeout_sec).close()
        return True
    except OSError:
        return False


def get_tor_browser():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--proxy-server=socks5://127.0.0.1:9050")
    return webdriver.Chrome(options=chrome_options)


def request_new_tor_circuit():
    try:
        with Controller.from_port(port=9051) as controller:
            controller.authenticate()
            controller.signal(Signal.NEWNYM)
            print("[+] Requested new Tor circuit.")
            time.sleep(5)
            return True
    except Exception as e:
        print(f"[-] Could not request new circuit: {e}")
        return False


def capture_site(site_name, url):
    started_at = datetime.now().isoformat()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_pcap = f"/tmp/{site_name}_{ts}.pcap"
    final_pcap = os.path.join(PCAP_DIR, f"{site_name}_{ts}.pcap")

    if not is_guard_active(GUARD_IP):
        msg = f"[-] Guard pre-check failed for {GUARD_IP}; skipping {site_name}."
        print(msg)
        append_event(
            {
                "time": started_at,
                "site": site_name,
                "url": url,
                "status": "skipped_guard_inactive",
                "guard_ip": GUARD_IP,
                "pcap": final_pcap,
            }
        )
        return

    tcpdump_cmd = [
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

    print(f"\n[*] Starting capture for {site_name} -> {final_pcap}")
    capture_proc = subprocess.Popen(tcpdump_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1)

    status = "ok"
    error_message = ""
    driver = get_tor_browser()
    try:
        print(f"[*] Navigating to {url}...")
        driver.get(url)
        time.sleep(WARMUP_DURATION)
        time.sleep(CAPTURE_DURATION)
    except Exception as e:
        status = "browser_error"
        error_message = str(e)
        print(f"[-] Error loading {url}: {e}")
    finally:
        driver.quit()

    print(f"[*] Stopping capture for {site_name}")
    # Keeping pkill intentionally as a robust fallback from earlier process-stop issues.
    subprocess.run(["sudo", "pkill", "-x", "tcpdump"], check=False)
    try:
        capture_proc.terminate()
        capture_proc.wait(timeout=3)
    except Exception:
        pass
    time.sleep(1)

    pcap_saved = False
    if os.path.exists(tmp_pcap):
        subprocess.run(["sudo", "mv", tmp_pcap, final_pcap], check=False)
        subprocess.run(["sudo", "chown", f"{os.getuid()}:{os.getgid()}", final_pcap], check=False)
        pcap_saved = os.path.exists(final_pcap)
    else:
        status = "capture_error"
        if not error_message:
            error_message = f"tcpdump failed to create {tmp_pcap}"
        print(f"[-] Error: {error_message}")

    append_event(
        {
            "time": started_at,
            "site": site_name,
            "url": url,
            "status": status,
            "guard_ip": GUARD_IP,
            "pcap": final_pcap,
            "pcap_saved": pcap_saved,
            "warmup_sec": WARMUP_DURATION,
            "capture_sec": CAPTURE_DURATION,
            "error": error_message,
        }
    )


def main():
    verify_pcap_dir(PCAP_DIR)
    print(f"[*] Output dir: {PCAP_DIR}")
    print(f"[*] Structured log: {LOG_JSONL}")

    for site_name, url in TARGET_SITES.items():
        capture_site(site_name, url)
        request_new_tor_circuit()


if __name__ == "__main__":
    main()
