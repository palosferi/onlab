"""Regression tests for capture validation.

The first test round rejected a 1.4 MB Guardian front page as a challenge
interstitial, because the body scan matched a Cloudflare script reference.
Both failure directions corrupt the drift measurement: counting challenge
pages as samples invents drift, and discarding real pages invents it too.

    python tests/test_classify_page.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                "scripts", "collection"))

import wf_config as cfg


class FakeDriver:
    def __init__(self, title, source, url):
        self.title = title
        self.page_source = source
        self.current_url = url

    def execute_script(self, js):
        return "complete" if "readyState" in js else 1234


CASES = [
    # (name, title, source, url, expected_host, expected_status)
    ("large real page with a cloudflare script tag",
     "News, sport and opinion from the Guardian",
     "<html>" + "x" * 1_400_000 + "<script src='https://cdn.cloudflare.com/a.js'></script>",
     "https://www.theguardian.com/international", "www.theguardian.com", "ok"),
    ("cloudflare challenge interstitial",
     "Just a moment...",
     "<html>Checking your browser... Ray ID: 8ab</html>",
     "https://www.reuters.com", "www.reuters.com", "blocked"),
    ("effectively empty response",
     "", "<html></html>", "https://www.imdb.com", "www.imdb.com", "empty"),
    ("small page carrying a captcha prompt",
     "Sign in",
     "<html>" + "y" * 3000 + " please complete the captcha to continue</html>",
     "https://x.com", "x.com", "blocked"),
    ("large page whose javascript merely mentions captcha",
     "SoundCloud",
     "<html>" + "z" * 250_000 + " captcha_widget_loader </html>",
     "https://soundcloud.com", "soundcloud.com", "ok"),
    ("http status page as the title",
     "403 Forbidden", "<html>" + "q" * 5000 + "</html>",
     "https://a.com", "a.com", "blocked"),
    ("redirect away from the target site",
     "Login", "<html>" + "w" * 120_000 + "</html>",
     "https://accounts.example.net/login", "www.reddit.com", "redirected"),
]


def main():
    failures = 0
    for name, title, source, url, host, expected in CASES:
        got = cfg.classify_page(FakeDriver(title, source, url), host)["status"]
        ok = got == expected
        failures += not ok
        print(f"[{'ok  ' if ok else 'FAIL'}] {name:48s} expected={expected:11s} got={got}")
    print(f"\n{len(CASES) - failures}/{len(CASES)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
