#!/usr/bin/env python3
"""Recompute the reported TREC LP decisions with exact rational arithmetic."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from exact_certificate import (  # noqa: E402
    LOG_DISCOUNT_HEX_K20,
    exact_overlap_certificate,
    exact_overlap_dp,
    exact_projection_certificate,
)
from study import load_qrels, overlap_greedy  # noqa: E402




def _overlap_tasks(
    paths: list[Path], seeds: list[int], trials_per_query: int
) -> list[tuple[int, str, str, int, list[set[str]], int]]:
    tasks: list[tuple[int, str, str, int, list[set[str]], int]] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for path in paths:
            year = path.name[:4]
            for qid, docs in load_qrels(path).items():
                doc_items = sorted(docs.items())
                overlap = [entry for entry in doc_items if len(entry[1]) > 1]
                if not overlap or len(doc_items) < 5:
                    continue
                for trial in range(trials_per_query):
                    anchor = overlap[trial % len(overlap)]
                    rest = [entry for entry in doc_items if entry[0] != anchor[0]]
                    chosen = rng.choice(
                        len(rest), size=min(9, len(rest)), replace=False
                    )
                    sample = [anchor] + [rest[int(i)] for i in chosen]
                    label_sets = [set(labels) for _, labels in sample]
                    tasks.append(
                        (seed, year, qid, trial, label_sets, min(5, len(label_sets)))
                    )
    return tasks


def _full_tasks(
    paths: list[Path], k: int
) -> list[tuple[str, str, list[set[str]], int]]:
    tasks: list[tuple[str, str, list[set[str]], int]] = []
    for path in paths:
        year = path.name[:4]
        for qid, docs in load_qrels(path).items():
            label_sets = [set(labels) for _, labels in sorted(docs.items())]
            tasks.append((year, qid, label_sets, min(k, len(label_sets))))
    return tasks


def _certify_overlap(
    task: tuple[int, str, str, int, list[set[str]], int]
) -> dict[str, object]:
    seed, year, qid, trial, label_sets, k = task
    greedy = overlap_greedy(label_sets, k)
    certificate = exact_overlap_certificate(label_sets, greedy)
    optimum = exact_overlap_dp(label_sets, k)
    projection = exact_projection_certificate(label_sets, k)
    if certificate.upper < optimum:
        raise RuntimeError("exact LP bound lies below exact subset-DP optimum")
    if projection.upper < optimum:
        raise RuntimeError("exact projection bound lies below exact subset-DP optimum")
    best_projection_lower = max(certificate.lower, projection.lower)
    row = {
        "seed": seed,
        "year": year,
        "qid": qid,
        "trial": trial,
        "n": len(label_sets),
        "k": k,
        "greedy_exact": str(certificate.lower),
        "optimum_exact": str(optimum),
        "lp_upper_exact": str(certificate.upper),
        "greedy_strictly_suboptimal": certificate.lower < optimum,
        "lp_equals_exact_dp": certificate.upper == optimum,
        "projection_lower_exact": str(projection.lower),
        "projection_upper_exact": str(projection.upper),
        "projection_relative_regret": float(
            (optimum - projection.lower) / optimum
        ),
        "projection_best_relative_width": float(
            (projection.upper - best_projection_lower) / optimum
        ),
        "projection_certified_optimal": projection.upper == best_projection_lower,
        "solver": certificate.solver,
    }
    return row


def _certify_full(
    task: tuple[str, str, list[set[str]], int]
) -> dict[str, object]:
    year, qid, label_sets, k = task
    greedy = overlap_greedy(label_sets, k)
    certificate = exact_overlap_certificate(label_sets, greedy)
    gap = certificate.gap
    row = {
        "year": year,
        "qid": qid,
        "n": len(label_sets),
        "intents": len(set().union(*label_sets)),
        "k": k,
        "greedy_lower_exact": str(certificate.lower),
        "lp_upper_exact": str(certificate.upper),
        "absolute_gap_exact": str(gap),
        "relative_gap_to_upper": float(gap / certificate.upper),
        "certified_optimal": certificate.certified_optimal,
        "exact_lp_seconds": certificate.solve_seconds,
        "solver": certificate.solver,
    }
    return row


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[20260814, 20260815, 20260816]
    )
    parser.add_argument("--trials-per-query", type=int, default=50)
    parser.add_argument("--k-full", type=int, default=20)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs" / "exact"
    )
    args = parser.parse_args()

    paths = sorted((ROOT / "data" / "trec_diversity").glob("*.qrels*"))
    if len(paths) != 4:
        raise RuntimeError(f"expected four TREC qrels files, found {len(paths)}")
    overlap_tasks = _overlap_tasks(paths, args.seeds, args.trials_per_query)
    full_tasks = _full_tasks(paths, args.k_full)
    print(
        f"exact certification: {len(overlap_tasks)} small pools, "
        f"{len(full_tasks)} full queries, {args.workers} workers",
        flush=True,
    )

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        overlap_rows = []
        for index, row in enumerate(
            executor.map(_certify_overlap, overlap_tasks, chunksize=4), start=1
        ):
            overlap_rows.append(row)
            if index % 1000 == 0:
                print(f"small pools: {index}/{len(overlap_tasks)}", flush=True)
        full_rows = []
        for index, row in enumerate(
            executor.map(_certify_full, full_tasks, chunksize=1), start=1
        ):
            full_rows.append(row)
            if index % 20 == 0:
                print(f"full queries: {index}/{len(full_tasks)}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    exception_rows = [
        row
        for row in overlap_rows
        if row["greedy_strictly_suboptimal"] or not row["lp_equals_exact_dp"]
    ]
    _write_csv(args.output_dir / "overlap_exact_exceptions.csv", exception_rows)
    _write_csv(args.output_dir / "full_query_exact.csv", full_rows)

    per_seed: dict[str, object] = {}
    for seed in args.seeds:
        rows = [row for row in overlap_rows if row["seed"] == seed]
        per_seed[str(seed)] = {
            "overlap_pools": len(rows),
            "strict_greedy_failures": sum(
                bool(row["greedy_strictly_suboptimal"]) for row in rows
            ),
            "lp_exact_small_pools": sum(
                bool(row["lp_equals_exact_dp"]) for row in rows
            ),
            "lp_bound_failures": sum(
                Fraction(str(row["lp_upper_exact"]))
                < Fraction(str(row["optimum_exact"]))
                for row in rows
            ),
            "projection_certified_optimal": sum(
                bool(row["projection_certified_optimal"]) for row in rows
            ),
            "projection_mean_relative_regret": statistics.fmean(
                float(row["projection_relative_regret"]) for row in rows
            ),
            "projection_mean_best_relative_width": statistics.fmean(
                float(row["projection_best_relative_width"]) for row in rows
            ),
            "projection_max_best_relative_width": max(
                float(row["projection_best_relative_width"]) for row in rows
            ),
        }
    solver = full_rows[0]["solver"] if full_rows else overlap_rows[0]["solver"]
    summary = {
        "config": {
            "seeds": args.seeds,
            "trials_per_query": args.trials_per_query,
            "small_pool_n": 10,
            "small_pool_k": 5,
            "full_query_k": args.k_full,
        },
        "arithmetic": {
            "discount_hex": list(LOG_DISCOUNT_HEX_K20),
            "discount_interpretation": "exact rationals represented by pinned binary64 values",
            "alpha": "1/2",
            "comparison": "exact rational equality",
            "solver": solver,
        },
        "per_seed": per_seed,
        "aggregate": {
            "overlap_pools": len(overlap_rows),
            "strict_greedy_failures": sum(
                bool(row["greedy_strictly_suboptimal"]) for row in overlap_rows
            ),
            "lp_exact_small_pools": sum(
                bool(row["lp_equals_exact_dp"]) for row in overlap_rows
            ),
            "lp_bound_failures": sum(
                Fraction(str(row["lp_upper_exact"]))
                < Fraction(str(row["optimum_exact"]))
                for row in overlap_rows
            ),
            "projection_certified_optimal": sum(
                bool(row["projection_certified_optimal"]) for row in overlap_rows
            ),
            "projection_mean_relative_regret": statistics.fmean(
                float(row["projection_relative_regret"]) for row in overlap_rows
            ),
            "projection_mean_best_relative_width": statistics.fmean(
                float(row["projection_best_relative_width"])
                for row in overlap_rows
            ),
            "projection_max_best_relative_width": max(
                float(row["projection_best_relative_width"])
                for row in overlap_rows
            ),
            "full_queries": len(full_rows),
            "full_query_lp_certified_optimal": sum(
                bool(row["certified_optimal"]) for row in full_rows
            ),
            "full_query_lp_mean_relative_gap": statistics.fmean(
                float(row["relative_gap_to_upper"]) for row in full_rows
            ),
            "full_query_exact_lp_median_seconds": statistics.median(
                float(row["exact_lp_seconds"]) for row in full_rows
            ),
        },

    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
