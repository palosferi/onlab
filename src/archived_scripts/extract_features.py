import os
import glob
import pandas as pd
from scapy.all import rdpcap, IP, TCP

# --- CONFIGURATION ---
BASE_DIR = os.path.expanduser("~/Desktop/tor_dataset")
BASELINE_DIR = os.path.join(BASE_DIR, "baseline_tor")
OBFS4_DIR = os.path.join(BASE_DIR, "obfs4_tor")
OUTPUT_DIR = os.path.join(BASE_DIR, "extracted_features")


def extract_features(pcap_path):
    try:
        packets = rdpcap(pcap_path)
    except Exception as e:
        print(f"[-] Error reading {pcap_path}: {e}")
        return None

    if len(packets) == 0:
        return None

    features = []

    # Detect client IP from the first TCP/IP packet
    client_ip = None
    for pkt in packets:
        if IP in pkt and TCP in pkt:
            client_ip = pkt[IP].src
            break

    if not client_ip:
        return None

    first_packet_time = float(packets[0].time)
    last_time = first_packet_time

    for pkt in packets:
        if IP in pkt and TCP in pkt:
            current_time = float(pkt.time)
            inter_arrival_time = current_time - last_time
            last_time = current_time

            # TCP payload size (application data only)
            packet_size = len(pkt[TCP].payload)

            if packet_size > 0:
                if pkt[IP].src == client_ip:
                    direction_size = packet_size  # Outgoing packet (+)
                else:
                    direction_size = -packet_size  # Incoming packet (-)

                features.append(
                    {
                        "time_offset": round(current_time - first_packet_time, 4),
                        "inter_arrival_time": round(inter_arrival_time, 6),
                        "direction_size": direction_size,
                    }
                )

    if not features:
        return None

    return pd.DataFrame(features)


def process_directory(input_dir, output_subdir):
    print(f"\n[*] Starting extraction for directory: {input_dir}")

    out_path = os.path.join(OUTPUT_DIR, output_subdir)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    pcap_files = glob.glob(os.path.join(input_dir, "*.pcap"))
    total_files = len(pcap_files)

    print(f"[+] Found {total_files} PCAP files in {output_subdir}.")

    for i, pcap_path in enumerate(pcap_files, 1):
        filename = os.path.basename(pcap_path)
        csv_filename = filename.replace(".pcap", ".csv")
        csv_path = os.path.join(out_path, csv_filename)

        # Skip existing outputs so interrupted runs can resume.
        if os.path.exists(csv_path):
            continue

        print(f"[*] Processing [{i}/{total_files}]: {filename}...")
        df = extract_features(pcap_path)

        if df is not None:
            df.to_csv(csv_path, index=False)
        else:
            print(f"[-] Warning: No valid TCP payload data in {filename}")


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"[+] Created output directory: {OUTPUT_DIR}")

    # 1. Process baseline captures
    if os.path.exists(BASELINE_DIR):
        process_directory(BASELINE_DIR, "baseline_features")
    else:
        print(f"[-] Directory not found: {BASELINE_DIR}")

    # 2. Process obfs4 captures
    if os.path.exists(OBFS4_DIR):
        process_directory(OBFS4_DIR, "obfs4_features")
    else:
        print(f"[-] Directory not found: {OBFS4_DIR}")

    print("\n[+] Feature extraction complete!")


if __name__ == "__main__":
    main()
