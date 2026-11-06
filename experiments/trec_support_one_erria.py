#!/usr/bin/env python3
"""Validate exact ERR-IA ChainMerge on real support-one TREC queries.

The experiment follows the public ``ndeval`` semantics: relevance is binary,
alpha is 0.5, intents have equal weight, and ERR uses reciprocal-rank discount.
On a support-one query this is ERR-IA with item satisfaction probability 0.5.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chainmerge import ErrCandidate, erria, erria_chainmerge, erria_naive_greedy  # noqa: E402


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def load_binary_qrels(path: Path) -> dict[str, dict[str, set[str]]]:
    result: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    with _open_text(path) as handle:
        for raw in handle:
            fields = raw.split()
            if len(fields) >= 4 and float(fields[3]) > 0:
                result[fields[0]][fields[2]].add(fields[1])
    return {qid: dict(docs) for qid, docs in result.items()}


def support_one_instances(
    qrels_dir: Path,
) -> list[tuple[str, str, list[ErrCandidate], dict[str, float]]]:
    instances = []
    paths = sorted(qrels_dir.glob("*.qrels*"))
    if len(paths) != 4:
        raise RuntimeError(f"expected four qrels files in {qrels_dir}, found {len(paths)}")
    for path in paths:
        year = path.name[:4]
        for qid, documents in sorted(load_binary_qrels(path).items()):
            if not documents or any(len(intents) != 1 for intents in documents.values()):
                continue
            candidates = [
                ErrCandidate(docid, next(iter(intents)), 0.5)
                for docid, intents in sorted(documents.items())
            ]
            groups = sorted({candidate.group for candidate in candidates})
            priors = {group: 1.0 / len(groups) for group in groups}
            instances.append((year, qid, candidates, priors))
    return instances


def _median_runtime(function, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns() - started) / 1e6)
    return statistics.median(samples)


def run_audit(
    qrels_dir: Path, repeats: int
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    for year, qid, candidates, priors in support_one_instances(qrels_dir):
        k = min(20, len(candidates))
        discounts = [1.0 / (rank + 1) for rank in range(k)]
        merge_answer = erria_chainmerge(candidates, priors, k=k)
        scan_answer = erria_naive_greedy(candidates, priors, k=k)
        merge_value = erria(merge_answer, priors, discounts=discounts)
        scan_value = erria(scan_answer, priors, discounts=discounts)
        if abs(merge_value - scan_value) > 1e-12:
            raise RuntimeError(f"objective disagreement for {year} topic {qid}")
        # Warm both paths once before collecting independent cold-call timings.
        erria_chainmerge(candidates, priors, k=k)
        erria_naive_greedy(candidates, priors, k=k)
        merge_ms = _median_runtime(
            lambda: erria_chainmerge(candidates, priors, k=k), repeats
        )
        scan_ms = _median_runtime(
            lambda: erria_naive_greedy(candidates, priors, k=k), repeats
        )
        scan_evaluations = sum(len(candidates) - rank for rank in range(k))
        rows.append(
            {
                "year": year,
                "qid": qid,
                "candidates": len(candidates),
                "intents": len(priors),
                "k": k,
                "merge_objective": merge_value,
                "scan_objective": scan_value,
                "absolute_objective_difference": abs(merge_value - scan_value),
                "same_document_sequence": [x.item_id for x in merge_answer]
                == [x.item_id for x in scan_answer],
                "merge_median_ms": merge_ms,
                "scan_median_ms": scan_ms,
                "speedup": scan_ms / merge_ms,
                "scan_candidate_evaluations": scan_evaluations,
                "merge_heap_pops": k,
            }
        )
    if not rows:
        raise RuntimeError("no complete support-one queries found")
    summary: dict[str, object] = {
        "semantics": {
            "source": "official TREC Web diversity qrels, positive judgments",
            "ndeval_relevance": "binary (judgments > 1 are clipped to 1)",
            "alpha": 0.5,
            "intent_prior": "uniform over judged positive subtopics",
            "item_satisfaction": 0.5,
            "discount": "1/rank",
            "cutoff": 20,
        },
        "queries": len(rows),
        "queries_by_year": {
            year: sum(row["year"] == year for row in rows)
            for year in ("2009", "2010", "2011", "2012")
        },
        "objective_matches": sum(
            float(row["absolute_objective_difference"]) <= 1e-12 for row in rows
        ),
        "sequence_matches": sum(bool(row["same_document_sequence"]) for row in rows),
        "candidate_min_median_max": [
            min(int(row["candidates"]) for row in rows),
            statistics.median(int(row["candidates"]) for row in rows),
            max(int(row["candidates"]) for row in rows),
        ],
        "intent_min_median_max": [
            min(int(row["intents"]) for row in rows),
            statistics.median(int(row["intents"]) for row in rows),
            max(int(row["intents"]) for row in rows),
        ],
        "median_query_speedup": statistics.median(
            float(row["speedup"]) for row in rows
        ),
        "geometric_mean_speedup": statistics.geometric_mean(
            float(row["speedup"]) for row in rows
        ),
        "min_max_speedup": [
            min(float(row["speedup"]) for row in rows),
            max(float(row["speedup"]) for row in rows),
        ],
        "repeats_per_algorithm_per_query": repeats,
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrels-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "trec_support_one_erria",
    )
    args = parser.parse_args()
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    rows, summary = run_audit(args.qrels_dir, args.repeats)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_query.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
