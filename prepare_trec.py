#!/usr/bin/env python3
"""Download the four official TREC judgment files used by the paper."""

from __future__ import annotations

from pathlib import Path
import urllib.request


ROOT = Path(__file__).resolve().parent
FILES = (
    ("2009.qrels.gz", "https://trec.nist.gov/data/web/09/qrels.diversity.gz"),
    ("2010.qrels", "https://trec.nist.gov/data/web/10/10.diversity-qrels.final"),
    ("2011.qrels", "https://trec.nist.gov/data/web/11/qrels-for-ndeval"),
    ("2012.qrels", "https://trec.nist.gov/data/web/12/qrels-for-ndeval"),
)


def main() -> None:
    destination = ROOT / "data" / "trec_diversity"
    destination.mkdir(parents=True, exist_ok=True)
    for name, url in FILES:
        target = destination / name
        if not target.exists():
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
            target.write_bytes(data)
        print(f"ready: {name}", flush=True)


if __name__ == "__main__":
    main()
