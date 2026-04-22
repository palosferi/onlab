import os
import subprocess
import time
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from stem import Signal
from stem.control import Controller

# --- CONFIGURATION ---
TARGET_SITES = {
    "wikipedia": "https://en.wikipedia.org/wiki/Main_Page",
    "duckduckgo": "https://duckduckgo.com",
    "github": "https://github.com",
    "stackoverflow": "https://stackoverflow.com",
    "mit_ocw": "https://ocw.mit.edu",
    "gnu": "https://www.gnu.org",
    "debian": "https://www.debian.org",
    "w3c": "https://www.w3.org",
    "bbc": "https://www.bbc.com",
    "cnn": "https://www.cnn.com",
    "reuters": "https://www.reuters.com",
    "theguardian": "https://www.theguardian.com/international",
    "aljazeera": "https://www.aljazeera.com",
    "hacker_news": "https://news.ycombinator.com",
    "wired": "https://www.wired.com",
    "techcrunch": "https://techcrunch.com",
    "telex": "https://telex.hu",
    "index": "https://index.hu",
    "hvg": "https://hvg.hu",
    "origo": "https://www.origo.hu",
    "hwsw": "https://www.hwsw.hu",
    "bme": "https://www.bme.hu",
    "amazon": "https://www.amazon.com",
    "ebay": "https://www.ebay.com",
    "aliexpress": "https://www.aliexpress.com",
    "imdb": "https://www.imdb.com",
    "reddit": "https://old.reddit.com",
    "twitch": "https://www.twitch.tv",
    "vimeo": "https://vimeo.com",
    "soundcloud": "https://soundcloud.com",
    "wordpress": "https://wordpress.org",
    "mozilla": "https://www.mozilla.org",
    "archive": "https://archive.org",
    "medium": "https://medium.com",
    "quora": "https://www.quora.com",
    "coursera": "https://www.coursera.org",
}

PCAP_DIR = "/mnt/tor_data/"
NETWORK_INTERFACE = "enp3s0"
GUARD_IP = "109.110.170.208"
CAPTURE_DURATION = 15


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
    except Exception as e:
        print(f"[-] Could not request new circuit: {e}")


def capture_site(site_name, url):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_pcap = f"/tmp/{site_name}_{timestamp}.pcap"
    final_pcap = os.path.join(PCAP_DIR, f"{site_name}_{timestamp}.pcap")

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

    subprocess.Popen(tcpdump_cmd, stdout=subprocess.DEVNULL)
    time.sleep(1)

    driver = get_tor_browser()
    try:
        print(f"[*] Navigating to {url}...")
        driver.get(url)
        time.sleep(CAPTURE_DURATION)
    except Exception as e:
        print(f"[-] Error loading {url}: {e}")
    finally:
        driver.quit()

    print(f"[*] Stopping capture for {site_name}")
    subprocess.run(["sudo", "pkill", "tcpdump"], check=False)
    time.sleep(1)

    if os.path.exists(tmp_pcap):
        subprocess.run(["sudo", "mv", tmp_pcap, final_pcap], check=False)
        subprocess.run(["sudo", "chown", f"{os.getuid()}:{os.getgid()}", final_pcap], check=False)
    else:
        print(f"[-] Error: tcpdump failed to create {tmp_pcap}!")


def main():
    global PCAP_DIR
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pcap_abspath = os.path.join(script_dir, PCAP_DIR)

    if not os.path.exists(pcap_abspath):
        os.makedirs(pcap_abspath)

    PCAP_DIR = pcap_abspath

    for site_name, url in TARGET_SITES.items():
        capture_site(site_name, url)
        request_new_tor_circuit()


if __name__ == "__main__":
    main()
