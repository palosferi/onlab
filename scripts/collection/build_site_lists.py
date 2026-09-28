"""Build the unmonitored and background pools from a pinned Tranco list.

The open world needs sites the classifier was never trained on, in quantity.
Taking them from the Tranco ranking makes the choice citable and repeatable;
screening them over Tor first keeps out domains that are not web pages at all
(CDN and API hosts rank high in Tranco) and sites that only ever answer Tor with
a challenge, which would otherwise surface later as a pile of rejected captures.

A site passes when, fetched through the baseline Tor instance, it ends in HTTP
200 with an HTML body of a plausible size and no challenge marker. The first
--unmonitored passing sites form the open-world pool; the next --background
form a separate pool for the background-tab ablation, so the noise tab never
loads a site that is also an open-world test sample.

    python scripts/collection/build_site_lists.py --tranco top-1m.csv --list-id Y8YJG
"""

import argparse
import csv
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sites  # noqa: E402
import wf_config as cfg  # noqa: E402

# Adult and gambling sites are left out: they add nothing a news or shop site
# does not, and a thesis appendix listing them helps nobody.
EXCLUDE_RE = re.compile(r"porn|xxx|sex|xvideo|xhamster|xnxx|onlyfans|chaturbate|casino|bet365|"
                        r"stripchat|livejasmin|hentai|escort|f95zone|rule34|nhentai", re.IGNORECASE)
MIN_BYTES = 5000


def screen(domain, socks_port, timeout):
    """Fetch https://domain over Tor; return (ok, reason, final_url)."""
    # A distinct SOCKS username per domain puts every fetch on its own circuit,
    # so one slow exit does not stall the whole screen.
    argv = ["curl", "-sL", "--max-time", str(timeout), "--max-redirs", "5",
            "--socks5-hostname", f"127.0.0.1:{socks_port}", "--proxy-user", f"screen-{domain}:x",
            "-A", "Mozilla/5.0 (Windows NT 10.0; rv:140.0) Gecko/20100101 Firefox/140.0",
            "-w", "\n%{http_code} %{content_type} %{url_effective}",
            f"https://{domain}"]
    try:
        out = subprocess.run(argv, capture_output=True, timeout=timeout + 5).stdout
    except subprocess.TimeoutExpired:
        return False, "timeout", ""
    text = out.decode("utf-8", "replace")
    body, _, trailer = text.rpartition("\n")
    # "<code> <content type> <url>"; the content type may itself contain a
    # space ("text/html; charset=utf-8"), a URL never does.
    parts = trailer.split()
    if len(parts) < 2 or not parts[0].isdigit():
        return False, "no response", ""
    code, ctype, final = int(parts[0]), " ".join(parts[1:-1]), parts[-1]
    if code != 200:
        return False, f"http {code}", final
    if "html" not in ctype:
        return False, f"type {ctype or '?'}", final
    if len(body) < MIN_BYTES:
        return False, f"only {len(body)} bytes", final
    m = re.search(r"<title[^>]*>(.*?)</title>", body[:20000], re.IGNORECASE | re.DOTALL)
    title = m.group(1).strip() if m else ""
    if cfg._TITLE_BLOCK_RE.search(title):
        return False, f"challenge title {title[:40]!r}", final
    if len(body) < cfg.INTERSTITIAL_MAX_BYTES and cfg._BODY_BLOCK_RE.search(body[:6000]):
        return False, "challenge phrase", final
    return True, "ok", final


def select(results, unmonitored, background):
    """Pick the pools from screening results, deduplicated on where sites land.

    Two domains that redirect to the same site (youtu.be and youtube.com) are
    one page, not two; and a domain that redirects onto a monitored site would
    put a monitored page into the open world under the unmonitored label. Both
    are decided on the final URL, which only the screen knows.
    """
    seen = set(sites.monitored_domains())
    picked = []
    for (rank, dom), ok, _, final in results:
        if not ok or EXCLUDE_RE.search(dom):
            continue
        landed = sites.registrable(urlparse(final).hostname or dom)
        if landed in seen or sites.registrable(dom) in seen:
            continue
        seen.update({landed, sites.registrable(dom)})
        picked.append((rank, dom))
    need = unmonitored + background
    if len(picked) < need:
        sys.exit(f"only {len(picked)} distinct passing sites, need {need}; raise --candidates")
    return picked[:unmonitored], picked[unmonitored:need]


def read_screening(path):
    with open(path, newline="", encoding="utf-8") as fh:
        # split()[-1]: screens before the content-type fix stored
        # "charset=utf-8 https://..." in this column.
        return [((int(r["rank"]), r["domain"]), r["ok"] == "1", r["reason"],
                 (r["final_url"].split() or [""])[-1])
                for r in csv.DictReader(fh)]


def label_for(domain):
    return re.sub(r"[^a-z0-9.-]", "-", domain.lower())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tranco", help="Tranco top-1m.csv")
    p.add_argument("--from-screening", action="store_true",
                   help="rebuild the pools from sites/screening.csv without fetching again")
    p.add_argument("--list-id", required=True, help="Tranco list id, recorded in the output")
    p.add_argument("--unmonitored", type=int, default=500)
    p.add_argument("--background", type=int, default=100)
    p.add_argument("--candidates", type=int, default=1500,
                   help="how far down the ranking to screen")
    p.add_argument("--socks-port", type=int, default=cfg.env_int("TOR_WF_BASELINE_SOCKS_PORT", 9060))
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--timeout", type=int, default=45)
    args = p.parse_args()

    screening_csv = os.path.join(sites.SITES_DIR, "screening.csv")
    if args.from_screening:
        results = read_screening(screening_csv)
    else:
        if not args.tranco:
            sys.exit("--tranco is required unless --from-screening")
        taken = sites.monitored_domains()
        candidates = []
        with open(args.tranco, encoding="utf-8") as fh:
            for rank, domain in csv.reader(fh):
                reg = sites.registrable(domain)
                if reg in taken or EXCLUDE_RE.search(domain):
                    continue
                taken.add(reg)
                candidates.append((int(rank), domain))
                if len(candidates) >= args.candidates:
                    break

        print(f"screening {len(candidates)} Tranco {args.list_id} domains over socks "
              f"{args.socks_port}", flush=True)
        with ThreadPoolExecutor(args.workers) as pool:
            results = list(pool.map(lambda c: (c, *screen(c[1], args.socks_port, args.timeout)),
                                    candidates))
        os.makedirs(sites.SITES_DIR, exist_ok=True)
        with open(screening_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["rank", "domain", "ok", "reason", "final_url"])
            for (rank, dom), ok, reason, final in results:
                w.writerow([rank, dom, int(ok), reason, final])

    reasons = {}
    for _, ok, reason, _ in results:
        key = "ok" if ok else reason.split(" ")[0]
        reasons[key] = reasons.get(key, 0) + 1
    print(f"{reasons.get('ok', 0)}/{len(results)} passed the screen; {reasons}")
    unmon, bg = select(results, args.unmonitored, args.background)

    header = (f"# Tranco list {args.list_id} (https://tranco-list.eu/list/{args.list_id}), "
              f"screened over Tor by build_site_lists.py.\n# label url  # tranco rank\n")
    for name, chunk in (("unmonitored", unmon), ("background", bg)):
        with open(os.path.join(sites.SITES_DIR, f"{name}.txt"), "w", encoding="utf-8") as fh:
            fh.write(header)
            for rank, dom in chunk:
                fh.write(f"{label_for(dom)} https://{dom}  # {rank}\n")

    print(f"wrote {len(unmon)} unmonitored and {len(bg)} background sites "
          f"(ranks up to {bg[-1][0] if bg else unmon[-1][0]})")


if __name__ == "__main__":
    main()
