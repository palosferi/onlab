import glob
import ipaddress
import json
import os

import pandas as pd
from scapy.all import IP, TCP, rdpcap

# --- CONFIGURATION ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tor_dataset"))
BASELINE_PCAP_DIR = os.path.join(BASE_DIR, "baseline_tor")
OBFS4_PCAP_DIR = os.path.join(BASE_DIR, "obfs4_tor")
OTHER_PCAP_DIR = os.path.join(BASE_DIR, "other_tor")
OUTPUT_DIR = os.path.join(BASE_DIR, "extracted_features")
FEATURE_SCHEMA_VERSION = "v2_direction_configurable"


def is_local_source(ip_src):
    # Configurable via env so extraction remains portable across labs/hosts.
    raw_prefixes = os.environ.get("LOCAL_IP_PREFIXES", "auto").strip().lower()
    if raw_prefixes and raw_prefixes != "auto":
        prefixes = [p.strip() for p in raw_prefixes.split(",") if p.strip()]
        return any(ip_src.startswith(prefix) for prefix in prefixes)

    try:
        return ipaddress.ip_address(ip_src).is_private
    except ValueError:
        return False


def should_overwrite_outputs():
    val = os.environ.get("EXTRACT_OVERWRITE", "0").strip().lower()
    return val in {"1", "true", "yes", "y"}


def should_skip_existing_output(out_path, overwrite=False):
    return (not overwrite) and os.path.exists(out_path)


def extract_features(pcap_path):
    try:
        packets = rdpcap(pcap_path)
    except Exception as e:
        print(f"[-] Error reading {pcap_path}: {e}")
        return pd.DataFrame()

    rows = []
    first_ts = None
    last_ts = None

    for pkt in packets:
        if not (pkt.haslayer(IP) and pkt.haslayer(TCP)):
            continue

        timestamp = float(pkt.time)
        if first_ts is None:
            first_ts = timestamp

        pkt_len = int(len(pkt))
        direction_size = pkt_len if is_local_source(pkt[IP].src) else -pkt_len
        inter_arrival = 0.0 if last_ts is None else max(0.0, timestamp - last_ts)
        time_offset = timestamp - first_ts
        last_ts = timestamp

        rows.append(
            {
                "time_offset": time_offset,
                "direction_size": direction_size,
                "inter_arrival_time": inter_arrival,
            }
        )

    return pd.DataFrame(rows)


def process_directory(input_dir, output_subdir, overwrite=False):
    os.makedirs(os.path.join(OUTPUT_DIR, output_subdir), exist_ok=True)
    pcap_files = glob.glob(os.path.join(input_dir, "*.pcap"))

    print(f"[+] Processing {len(pcap_files)} files from {input_dir} -> {output_subdir}")
    for i, pcap_path in enumerate(pcap_files, start=1):
        filename = os.path.basename(pcap_path)
        csv_name = filename.replace(".pcap", ".csv")
        out_path = os.path.join(OUTPUT_DIR, output_subdir, csv_name)

        if should_skip_existing_output(out_path, overwrite=overwrite):
            continue

        df = extract_features(pcap_path)
        if not df.empty:
            df.to_csv(out_path, index=False)

        if i % 100 == 0:
            print(f"  processed {i}/{len(pcap_files)}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    overwrite = should_overwrite_outputs()

    if os.path.isdir(BASELINE_PCAP_DIR):
        process_directory(BASELINE_PCAP_DIR, "baseline_features", overwrite=overwrite)

    if os.path.isdir(OBFS4_PCAP_DIR):
        process_directory(OBFS4_PCAP_DIR, "obfs4_features", overwrite=overwrite)

    if os.path.isdir(OTHER_PCAP_DIR):
        process_directory(OTHER_PCAP_DIR, "other_features", overwrite=overwrite)

    metadata_path = os.path.join(OUTPUT_DIR, "_extraction_metadata.json")
    metadata = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "local_ip_prefixes": os.environ.get("LOCAL_IP_PREFIXES", "auto"),
        "overwrite_mode": overwrite,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Done extracting all feature CSV files.")


if __name__ == "__main__":
    main()
