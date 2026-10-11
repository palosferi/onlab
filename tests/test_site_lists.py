"""Site-set selection from a screening (build_site_lists.py --site-set).

The monitored set is the screened Tranco order minus infrastructure and
duplicates, and the open-world pools come from below it. A domain that slips
through as both, or a pool site that lands on a monitored one, would put the
same page under two labels, so the split is checked on a small fake screening.

    python tests/test_site_lists.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                "scripts", "collection"))

import build_site_lists as bsl  # noqa: E402
import sites  # noqa: E402

REGISTRABLE_CASES = [
    ("www.bbc.co.uk", "bbc.co.uk"),
    ("news.bbc.co.uk", "bbc.co.uk"),
    ("www.gov.uk", "gov.uk"),
    ("en.wikipedia.org", "wikipedia.org"),
    ("www.google.com", "google.com"),
    ("store.steampowered.com", "steampowered.com"),
]


def row(rank, domain, final=None, ok=True):
    return (rank, domain), ok, "ok" if ok else "http 0", final or f"https://{domain}/"


SCREENING = [
    row(1, "google.com"),
    row(2, "cloudflare.com"),                      # excluded: cdn
    row(3, "youtube.com"),
    row(4, "youtu.be", "https://www.youtube.com/"),  # lands on a chosen site
    row(5, "down.example", ok=False),              # failed the screen
    row(6, "vk.com"),
    row(7, "vk.ru"),                               # excluded: duplicate
    row(8, "bbc.com"),
    row(9, "reserve.example"),
    row(10, "pool-a.example"),
    row(11, "fastly.net"),                         # infrastructure is fine in a pool
    row(12, "bbc.co.uk"),                          # duplicate stays out of the pools too
    row(13, "telex.hu"),                           # tb-main monitored: kept out of the pools
    row(14, "gmail.com", "https://mail.google.com/"),
    row(15, "pool-b.example"),
    row(16, "pool-c.example"),
]

EXCLUDED_CSV = """# comment lines are skipped
domain,category,note
cloudflare.com,cdn,
fastly.net,cdn,
vk.ru,duplicate,of vk.com
bbc.co.uk,duplicate,of bbc.com
"""


def check(name, got, expected):
    ok = got == expected
    print(f"[{'ok  ' if ok else 'FAIL'}] {name:34s} {got if not ok else ''}")
    return ok


def main():
    results = []
    for host, expected in REGISTRABLE_CASES:
        results.append(check(f"registrable {host}", sites.registrable(host), expected))

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as fh:
        fh.write(EXCLUDED_CSV)
    try:
        excluded = bsl.read_excluded(fh.name)
    finally:
        os.unlink(fh.name)
    results.append(check("excluded.csv read", sorted(excluded),
                         ["bbc.co.uk", "cloudflare.com", "fastly.net", "vk.ru"]))

    mon, res, unmon, bg, dropped = bsl.select_site_set(
        SCREENING, 4, 1, 2, 1, excluded, pool_excluded={"telex.hu"})
    doms = lambda chunk: [d for _, d in chunk]  # noqa: E731
    results.append(check("monitored", doms(mon), ["google.com", "youtube.com", "vk.com", "bbc.com"]))
    results.append(check("reserves", doms(res), ["reserve.example"]))
    results.append(check("unmonitored", doms(unmon), ["pool-a.example", "fastly.net"]))
    results.append(check("background", doms(bg), ["pool-b.example"]))
    results.append(check("dropped with reasons", [(d, c) for _, d, _, c, _ in dropped],
                         [("cloudflare.com", "cdn"), ("youtu.be", "same landing site"),
                          ("vk.ru", "duplicate")]))

    excluded["vk.com"] = ("pilot", "failed in Tor Browser")
    mon, res, *_ = bsl.select_site_set(SCREENING, 4, 1, 2, 1, excluded, pool_excluded={"telex.hu"})
    results.append(check("pilot failure moves a reserve up", doms(mon),
                         ["google.com", "youtube.com", "bbc.com", "reserve.example"]))

    print(f"\n{sum(results)}/{len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
