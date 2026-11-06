#!/usr/bin/env python3
"""Extract the three MIND-small files used by the runtime experiment."""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _extract(archive: Path, member: str, destination: Path) -> None:
    with zipfile.ZipFile(archive) as handle:
        matches = [name for name in handle.namelist() if name.rstrip("/").endswith(member)]
        if len(matches) != 1:
            raise RuntimeError(f"expected one {member} in {archive}, found {len(matches)}")
        with handle.open(matches[0]) as source, destination.open("wb") as target:
            shutil.copyfileobj(source, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-zip", type=Path, required=True)
    parser.add_argument("--dev-zip", type=Path, required=True)
    args = parser.parse_args()

    destination = ROOT / "data" / "mind"
    destination.mkdir(parents=True, exist_ok=True)
    _extract(args.train_zip, "news.tsv", destination / "train_news.tsv")
    _extract(args.train_zip, "behaviors.tsv", destination / "train_behaviors.tsv")
    _extract(args.dev_zip, "news.tsv", destination / "dev_news.tsv")

    print(f"MIND-small files prepared in {destination}")


if __name__ == "__main__":
    main()
