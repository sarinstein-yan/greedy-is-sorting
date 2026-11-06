#!/usr/bin/env python3
"""Single entry point for reproducing the experiments in the paper."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments import study  # noqa: E402

SEEDS = [20260814, 20260815, 20260816]


def _run(*arguments: str) -> None:
    subprocess.run([sys.executable, *arguments], cwd=ROOT, check=True)


def _qrels() -> list[Path]:
    paths = sorted((ROOT / "data" / "trec_diversity").glob("*.qrels*"))
    if len(paths) != 4:
        raise RuntimeError(f"expected four TREC qrels files, found {len(paths)}")
    return paths


def synthetic() -> None:
    rows = []
    original_seed = study.SEED
    try:
        for seed in SEEDS:
            study.SEED = seed
            rows.append(study.exhaustive_audit(instances=1000))
    finally:
        study.SEED = original_seed
    aggregate = {
        "seeds": SEEDS,
        "instances_per_seed_per_metric": 1000,
        "alpha_dcg_instances": sum(int(row["alpha_dcg_instances"]) for row in rows),
        "alpha_dcg_failures": sum(int(row["alpha_dcg_failures"]) for row in rows),
        "err_ia_instances": sum(int(row["err_ia_instances"]) for row in rows),
        "err_ia_failures": sum(int(row["err_ia_failures"]) for row in rows),
        "per_seed": rows,
    }
    output = ROOT / "outputs" / "synthetic"
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2))


def numeric() -> None:
    missing = [
        ROOT / "data" / "mind" / name
        for name in ("train_news.tsv", "train_behaviors.tsv", "dev_news.tsv")
        if not (ROOT / "data" / "mind" / name).exists()
    ]
    if missing:
        raise RuntimeError("MIND-small data are missing; follow README.md")

    qrels = _qrels()
    output = ROOT / "outputs" / "numeric"
    output.mkdir(parents=True, exist_ok=True)
    structure = study.qrels_structure(qrels)
    full_lp = study.trec_full_lp(qrels, k=20, exact_certificates=False)
    runtime = study.mind_runtime()
    erria = study.erria_runtime(n=65_238)
    study.write_csv(output / "trec_structure.csv", structure)
    study.write_csv(output / "trec_full_lp_numeric.csv", full_lp)
    study.write_csv(output / "mind_runtime.csv", runtime)
    study.write_csv(output / "erria_runtime.csv", erria)

    summary = {
        "trec_queries": sum(int(row["queries"]) for row in structure),
        "trec_positive_documents": sum(int(row["positive_documents"]) for row in structure),
        "trec_support_one_documents": sum(int(row["exclusive_documents"]) for row in structure),
        "trec_support_one_queries": sum(int(row["fully_exclusive_queries"]) for row in structure),
        "trec_numeric_lp_median_seconds": statistics.median(
            float(row["lp_seconds"]) for row in full_lp
        ),
        "mind_candidates": max(int(row["n"]) for row in runtime),
        "erria_speedup_range": [
            min(float(row["speedup"]) for row in erria),
            max(float(row["speedup"]) for row in erria),
        ],
        "runtime_note": "wall-clock values depend on hardware and system load",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def exact(workers: int) -> None:
    _run("formal/recompute_exact_trec.py", "--workers", str(workers))


def erria() -> None:
    _run(
        "experiments/trec_support_one_erria.py",
        "--qrels-dir",
        str(ROOT / "data" / "trec_diversity"),
        "--repeats",
        "200",
    )


def leaderboard() -> None:
    exact_csv = ROOT / "outputs" / "exact" / "full_query_exact.csv"
    if not exact_csv.exists():
        raise RuntimeError("run the exact stage before the leaderboard stage")
    _run(
        "formal/trec_leaderboard_robustness.py",
        "--exact-csv",
        str(exact_csv),
        "--qrels-dir",
        str(ROOT / "data" / "trec_diversity"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=("synthetic", "numeric", "exact", "erria", "leaderboard", "all"),
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    stages = {
        "synthetic": synthetic,
        "numeric": numeric,
        "exact": lambda: exact(args.workers),
        "erria": erria,
        "leaderboard": leaderboard,
    }
    if args.stage == "all":
        for name in ("synthetic", "numeric", "exact", "erria", "leaderboard"):
            stages[name]()
    else:
        stages[args.stage]()


if __name__ == "__main__":
    main()
