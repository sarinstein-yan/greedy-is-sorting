#!/usr/bin/env python3
"""Conservative robustness audit of official TREC diversity leaderboards.

NIST publishes aggregate alpha-nDCG@20 values for submitted runs but not the
run files themselves. This script combines those values with the paper's exact
per-query lower/upper ideal bounds. The resulting intervals are rigorous even
without submitted rankings: replacing the historical greedy ideal can decrease
any query score by at most U/L - 1.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from urllib.parse import urljoin, urlsplit


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from exact_certificate import exact_overlap_value  # noqa: E402
from study import load_qrels  # noqa: E402

USER_AGENT = "ChainMerge reproducibility audit (academic use)"
INDEX_URLS = {
    "2010": "https://trec.nist.gov/pubs/trec19/appendices/web-adhoc.diversity.html",
    "2011": "https://trec.nist.gov/pubs/trec20/appendices/web.html",
    "2012": "https://trec.nist.gov/pubs/trec21/appendices/web.html",
}
APPENDIX_2009 = (
    "https://trec.nist.gov/pubs/trec18/appendices/web-diversity/appendix.pdf"
)


@dataclass(frozen=True)
class OfficialRun:
    year: str
    run_id: str
    category: str
    num_topics: int
    alpha_ndcg20: float
    published_decimals: int
    source_url: str


def _request(url: str, retries: int = 4) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return response.read()
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def _cached_download(url: str, cache_dir: Path) -> tuple[Path, dict[str, object]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    name = urlsplit(url).path.strip("/").replace("/", "_")
    path = cache_dir / name
    if not path.exists():
        path.write_bytes(_request(url))
    return path, {
        "url": url,
        "cache_file": path.name,
        "bytes": path.stat().st_size,
    }


def _pdf_text(path: Path, *, pages: bool = False) -> str | list[str]:
    try:
        import fitz
    except ImportError as error:
        raise RuntimeError("PyMuPDF is required: pip install PyMuPDF==1.26.4") from error
    document = fitz.open(path)
    texts = [page.get_text() for page in document]
    return texts if pages else "\n".join(texts)


def _decimals(token: str) -> int:
    return len(token.rsplit(".", 1)[1]) if "." in token else 0


def parse_2009_pages(pages: list[str], source_url: str) -> list[OfficialRun]:
    runs: list[OfficialRun] = []
    average_pattern = re.compile(
        r"Average\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+"
        r"([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)",
        re.S,
    )
    for page_number, text in enumerate(pages, start=1):
        run_match = re.search(r"Novelty and Diversity Track Run\s*-\s*(\S+)", text)
        category_match = re.search(r"Collection:\s*Category\s*([AB])", text, re.I)
        average_match = average_pattern.search(text)
        if not (run_match and category_match and average_match):
            raise RuntimeError(f"could not parse 2009 appendix page {page_number}")
        score_token = average_match.group(3)
        runs.append(
            OfficialRun(
                year="2009",
                run_id=run_match.group(1),
                category=category_match.group(1).upper(),
                num_topics=50,
                alpha_ndcg20=float(score_token),
                published_decimals=_decimals(score_token),
                source_url=source_url,
            )
        )
    return runs


def parse_summary_blocks(text: str, year: str, source_url: str) -> list[OfficialRun]:
    runs: list[OfficialRun] = []
    for block_number, block in enumerate(re.split(r"Summary Statistics", text)[1:], start=1):
        run_match = re.search(r"Run ID:\s*(\S+)", block)
        task_match = re.search(r"Task\s*:\s*(\S+)", block, re.I)
        if not run_match or not task_match:
            continue
        if task_match.group(1).lower() != "diversity":
            continue
        category_match = re.search(
            r"(?:Document collection category|Category)\s*:\s*([AB])\b",
            block,
            re.I,
        )
        topics_match = re.search(r"Number of Topics\s*:\s*(\d+)", block, re.I)
        score_match = re.search(r"(?:α|alpha)-nDCG@20\s*([0-9]+(?:\.[0-9]+)?)", block, re.I)
        if not (category_match and topics_match and score_match):
            raise RuntimeError(
                f"incomplete diversity block {block_number} in {source_url}: "
                f"run={run_match.group(1)}"
            )
        score_token = score_match.group(1)
        runs.append(
            OfficialRun(
                year=year,
                run_id=run_match.group(1),
                category=category_match.group(1).upper(),
                num_topics=int(topics_match.group(1)),
                alpha_ndcg20=float(score_token),
                published_decimals=_decimals(score_token),
                source_url=source_url,
            )
        )
    return runs


def _pdf_links(index_url: str, html: str) -> list[str]:
    links = re.findall(r"HREF=[\"']?([^\"' >]+\.pdf)", html, re.I)
    return sorted({urljoin(index_url, link) for link in links})


def collect_official_runs(cache_dir: Path) -> tuple[list[OfficialRun], list[dict[str, object]]]:
    manifest: list[dict[str, object]] = []
    appendix_path, appendix_record = _cached_download(APPENDIX_2009, cache_dir)
    manifest.append(appendix_record)
    page_text = _pdf_text(appendix_path, pages=True)
    assert isinstance(page_text, list)
    runs = parse_2009_pages(page_text, APPENDIX_2009)

    for year, index_url in INDEX_URLS.items():
        index_path, index_record = _cached_download(index_url, cache_dir)
        manifest.append(index_record)
        html = index_path.read_text(encoding="latin1")
        links = _pdf_links(index_url, html)
        if not links:
            raise RuntimeError(f"no PDF links found at {index_url}")
        for url in links:
            pdf_path, pdf_record = _cached_download(url, cache_dir)
            manifest.append(pdf_record)
            runs.extend(parse_summary_blocks(str(_pdf_text(pdf_path)), year, url))

    unique: dict[tuple[str, str], OfficialRun] = {}
    for run in runs:
        key = (run.year, run.run_id)
        if key in unique and unique[key] != run:
            raise RuntimeError(f"conflicting official records for {key}")
        unique[key] = run
    counts = {
        year: sum(run.year == year for run in unique.values())
        for year in ("2009", "2010", "2011", "2012")
    }
    if counts["2009"] != 48 or any(
        counts[year] < 10 for year in ("2010", "2011", "2012")
    ):
        raise RuntimeError(f"implausible parsed diversity-run counts: {counts}")
    return sorted(unique.values(), key=lambda run: (run.year, run.run_id)), manifest


def _ndeval_greedy_value(
    documents: list[tuple[str, set[str]]], k: int
) -> Fraction:
    """Reproduce ``ndeval`` greedy, including its larger-docid tie-break."""

    seen: Counter[str] = Counter()
    remaining = list(range(len(documents)))
    order: list[int] = []
    for _ in range(min(k, len(remaining))):
        best = max(
            remaining,
            key=lambda index: (
                sum(Fraction(1, 2) ** seen[label] for label in documents[index][1]),
                documents[index][0],
            ),
        )
        order.append(best)
        remaining.remove(best)
        seen.update(documents[best][1])
    return exact_overlap_value(order, [labels for _, labels in documents])


def load_query_perturbations(
    exact_csv: Path, qrels_dir: Path
) -> tuple[dict[str, list[float]], dict[str, object]]:
    exact_rows: dict[tuple[str, str], dict[str, str]] = {}
    with exact_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            exact_rows[(row["year"], row["qid"])] = row

    values: dict[str, list[float]] = defaultdict(list)
    tie_changed = 0
    maximum_lower_difference = 0.0
    qrel_paths = sorted(qrels_dir.glob("*.qrels*"))
    if len(qrel_paths) != 4:
        raise RuntimeError(f"expected four qrels files in {qrels_dir}")
    for path in qrel_paths:
        year = path.name[:4]
        for qid, docs in load_qrels(path).items():
            row = exact_rows.get((year, qid))
            if row is None:
                raise RuntimeError(f"missing exact row for {year}/{qid}")
            documents = [(docid, set(labels)) for docid, labels in sorted(docs.items())]
            official_lower = _ndeval_greedy_value(documents, min(20, len(documents)))
            recorded_lower = Fraction(row["greedy_lower_exact"])
            upper = Fraction(row["lp_upper_exact"])
            if official_lower <= 0 or upper < official_lower:
                raise RuntimeError(
                    f"invalid official-greedy interval for {year}/{qid}: "
                    f"{official_lower} <= {upper}"
                )
            if official_lower != recorded_lower:
                tie_changed += 1
                maximum_lower_difference = max(
                    maximum_lower_difference,
                    abs(float(official_lower - recorded_lower)),
                )
            values[year].append(float(upper / official_lower - 1))
    if {year: len(rows) for year, rows in values.items()} != {
        "2009": 50,
        "2010": 48,
        "2011": 50,
        "2012": 50,
    }:
        raise RuntimeError("exact CSV does not contain the expected 198 TREC queries")
    return dict(values), {
        "queries": len(exact_rows),
        "official_tie_break": "maximum docid among equal marginal gains",
        "queries_whose_greedy_value_differs_from_paper_reference": tie_changed,
        "maximum_absolute_greedy_value_difference": maximum_lower_difference,
    }


def conservative_mean_bound(perturbations: list[float], num_topics: int) -> float:
    """Worst-case mean loss for any unknown subset of ``num_topics`` topics."""

    if not 0 < num_topics <= len(perturbations):
        raise ValueError("number of evaluated topics is incompatible with qrels")
    return sum(sorted(perturbations, reverse=True)[:num_topics]) / num_topics


def _strictly_greater(left: float, right: float) -> bool:
    """Strict comparison insulated from decimal-to-binary representation noise."""

    return left > right + 1e-15


def audit_runs(
    runs: list[OfficialRun], perturbations: dict[str, list[float]]
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    interval_rows: list[dict[str, object]] = []
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for run in runs:
        bound = conservative_mean_bound(perturbations[run.year], run.num_topics)
        halfwidth = 0.5 * 10 ** (-run.published_decimals)
        row: dict[str, object] = {
            **asdict(run),
            "rounding_halfwidth": halfwidth,
            "denominator_loss_bound": bound,
            "true_score_lower": max(0.0, run.alpha_ndcg20 - halfwidth - bound),
            "true_score_upper": min(1.0, run.alpha_ndcg20 + halfwidth),
        }
        interval_rows.append(row)
        grouped[(run.year, run.category, run.num_topics)].append(row)

    pair_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    for (year, category, num_topics), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: (-float(row["alpha_ndcg20"]), str(row["run_id"])))
        for ordinal, row in enumerate(rows, start=1):
            row["published_ordinal_rank"] = ordinal
            row["certified_best_rank"] = 1 + sum(
                _strictly_greater(
                    float(other["true_score_lower"]),
                    float(row["true_score_upper"]),
                )
                for other in rows
                if other is not row
            )
            row["certified_worst_rank"] = 1 + sum(
                _strictly_greater(
                    float(other["true_score_upper"]),
                    float(row["true_score_lower"]),
                )
                for other in rows
                if other is not row
            )

        strict_pairs = 0
        certified_pairs = 0
        denominator_only_certified = 0
        rounding_only_certified = 0
        for left_index, left in enumerate(rows):
            for right in rows[left_index + 1 :]:
                if float(left["alpha_ndcg20"]) == float(right["alpha_ndcg20"]):
                    continue
                strict_pairs += 1
                high, low = (
                    (left, right)
                    if float(left["alpha_ndcg20"]) > float(right["alpha_ndcg20"])
                    else (right, left)
                )
                gap = float(
                    Decimal(str(high["alpha_ndcg20"]))
                    - Decimal(str(low["alpha_ndcg20"]))
                )
                bound = max(
                    float(high["denominator_loss_bound"]),
                    float(low["denominator_loss_bound"]),
                )
                rounding = float(high["rounding_halfwidth"]) + float(low["rounding_halfwidth"])
                certified = _strictly_greater(
                    float(high["true_score_lower"]),
                    float(low["true_score_upper"]),
                )
                certified_pairs += certified
                denominator_only_certified += _strictly_greater(gap, bound)
                rounding_only_certified += _strictly_greater(gap, rounding)
                pair_rows.append(
                    {
                        "year": year,
                        "category": category,
                        "num_topics": num_topics,
                        "higher_run": high["run_id"],
                        "lower_run": low["run_id"],
                        "published_gap": gap,
                        "denominator_loss_bound": bound,
                        "rounding_pair_halfwidth": rounding,
                        "ordering_certified": certified,
                    }
                )
        leader = rows[0]
        leader_certified = all(
            _strictly_greater(
                float(leader["true_score_lower"]),
                float(other["true_score_upper"]),
            )
            for other in rows[1:]
        )
        group_rows.append(
            {
                "year": year,
                "category": category,
                "num_topics": num_topics,
                "runs": len(rows),
                "strict_published_pairs": strict_pairs,
                "certified_pairs": certified_pairs,
                "certified_pair_fraction": certified_pairs / strict_pairs if strict_pairs else 1.0,
                "denominator_only_certified_pairs": denominator_only_certified,
                "rounding_only_certified_pairs": rounding_only_certified,
                "published_leader": leader["run_id"],
                "leader_certified": leader_certified,
                "denominator_loss_bound": leader["denominator_loss_bound"],
            }
        )
    return interval_rows, pair_rows, group_rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exact-csv",
        type=Path,
        default=ROOT / "outputs" / "exact" / "full_query_exact.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "leaderboard",
    )
    parser.add_argument("--qrels-dir", type=Path, required=True)
    args = parser.parse_args()

    cache_dir = args.output_dir / "source_cache"
    runs, manifest = collect_official_runs(cache_dir)
    perturbations, tie_diagnostics = load_query_perturbations(
        args.exact_csv, args.qrels_dir
    )
    intervals, pairs, groups = audit_runs(runs, perturbations)
    _write_csv(args.output_dir / "run_score_intervals.csv", intervals)
    _write_csv(args.output_dir / "pairwise_orderings.csv", pairs)
    _write_csv(args.output_dir / "group_summary.csv", groups)
    (args.output_dir / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "method": {
            "historical_score": "mean_q numerator_q / greedy_ideal_q",
            "replacement": "greedy_ideal_q <= true_ideal_q <= LP_upper_q",
            "per_query_loss_bound": "LP_upper_q / greedy_ideal_q - 1",
            "unknown_topic_subset": "average of the largest num_topics per-query bounds",
            "comparison_scope": "same year, collection category, and evaluated-topic count",
            "published_rounding": "included as a symmetric half-unit in the last printed decimal",
        },
        "official_runs": len(runs),
        "evaluator_tie_diagnostics": tie_diagnostics,
        "runs_by_year": {
            year: sum(run.year == year for run in runs)
            for year in ("2009", "2010", "2011", "2012")
        },
        "comparison_groups": groups,
        "aggregate_strict_pairs": sum(int(row["strict_published_pairs"]) for row in groups),
        "aggregate_certified_pairs": sum(int(row["certified_pairs"]) for row in groups),
    }
    total = int(summary["aggregate_strict_pairs"])
    summary["aggregate_certified_pair_fraction"] = (
        int(summary["aggregate_certified_pairs"]) / total if total else 1.0
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
