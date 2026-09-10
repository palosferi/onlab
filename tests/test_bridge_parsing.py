"""Bridge address parsing and capture filter construction.

A bridge commonly has both an IPv4 and an IPv6 address and Tor falls back
between them silently. Filtering on the wrong family captures nothing while
the page still loads, which produces an empty PCAP that looks like a
collection glitch rather than the configuration error it is.

    python tests/test_bridge_parsing.py
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                "scripts", "collection"))

import wf_config as cfg

PARSE_CASES = [
    ("obfs4 109.110.170.208:29323 B93BAE4F17CEACD9E491920C5D283C0D4C3D6D3D cert=abc iat-mode=0",
     "109.110.170.208"),
    ("obfs4 [2a05:d019:a97:d01:f6e6:41b8:c34e:e77e]:443 DC12DAE7037DCEBE7AEB80ABA854F1887ABD0C3C cert=x",
     "2a05:d019:a97:d01:f6e6:41b8:c34e:e77e"),
    ("obfs4 [2a01:4f8:241:4d16::2]:8080 DA96CA0221FC90A47E228B5090C8E1BFB0B2F7C9 cert=y iat-mode=0",
     "2a01:4f8:241:4d16::2"),
    ("garbage line with no address", None),
]

FILTER_CASES = [
    ["51.81.93.109"],
    ["109.110.170.208", "2a01:4f8:241:4d16::2"],
    ["2a05:d019:a97:d01:f6e6:41b8:c34e:e77e"],
]


def main():
    failures = 0
    for line, expected in PARSE_CASES:
        got = cfg.parse_bridge_address(line)
        ok = got == expected
        failures += not ok
        print(f"[{'ok  ' if ok else 'FAIL'}] parse {str(expected):40s} got {got}")

    for peers in FILTER_CASES:
        argv = cfg.capture_filter(peers)
        # tcpdump itself is the authority on whether the expression is valid.
        r = subprocess.run(["tcpdump", "-d", "-y", "EN10MB"] + argv,
                           capture_output=True, text=True)
        ok = r.returncode == 0
        failures += not ok
        detail = "compiles" if ok else r.stderr.strip()[:60]
        print(f"[{'ok  ' if ok else 'FAIL'}] filter {' '.join(argv):58s} {detail}")

    total = len(PARSE_CASES) + len(FILTER_CASES)
    print(f"\n{total - failures}/{total} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
