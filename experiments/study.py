#!/usr/bin/env python3
"""Experiment functions for the ChainMerge paper."""

from __future__ import annotations

import csv
import gzip
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import permutations
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chainmerge import (  # noqa: E402
    Candidate,
    ErrCandidate,
    alpha_chainmerge,
    alpha_chainmerge_presorted,
    alpha_dcg,
    alpha_ideal_from_group_counts,
    alpha_naive_greedy,
    erria,
    erria_chainmerge,
    erria_naive_greedy,
    log_discounts,
)
from exact_certificate import (  # noqa: E402
    exact_overlap_certificate,
)

SEED = 20260814


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def load_qrels(path: Path) -> dict[str, dict[str, set[str]]]:
    """Return qid -> docid -> positive subtopic ids."""

    result: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    with _open_text(path) as handle:
        for raw in handle:
            fields = raw.split()
            if len(fields) < 4:
                continue
            qid, subtopic, docid, relevance = fields[:4]
            if float(relevance) > 0:
                result[qid][docid].add(subtopic)
    return {qid: dict(docs) for qid, docs in result.items()}


def qrels_structure(paths: Iterable[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        year = path.name[:4]
        qrels = load_qrels(path)
        num_docs = sum(len(docs) for docs in qrels.values())
        exclusive_docs = sum(
            len(labels) == 1 for docs in qrels.values() for labels in docs.values()
        )
        exclusive_queries = sum(
            all(len(labels) == 1 for labels in docs.values()) for docs in qrels.values()
        )
        rows.append(
            {
                "year": year,
                "queries": len(qrels),
                "positive_documents": num_docs,
                "exclusive_documents": exclusive_docs,
                "exclusive_document_fraction": exclusive_docs / num_docs,
                "fully_exclusive_queries": exclusive_queries,
                "fully_exclusive_query_fraction": exclusive_queries / len(qrels),
                "max_intents_per_document": max(
                    len(labels) for docs in qrels.values() for labels in docs.values()
                ),
            }
        )
    return rows


def overlap_alpha_dcg(
    order: tuple[int, ...] | list[int],
    label_sets: list[set[str]],
    *,
    alpha: float = 0.5,
) -> float:
    seen: Counter[str] = Counter()
    value = 0.0
    for rank, index in enumerate(order):
        gain = sum((1.0 - alpha) ** seen[label] for label in label_sets[index])
        value += log_discounts(len(order))[rank] * gain
        seen.update(label_sets[index])
    return value


def overlap_greedy(label_sets: list[set[str]], k: int, alpha: float = 0.5) -> list[int]:
    seen: Counter[str] = Counter()
    remaining = list(range(len(label_sets)))
    order: list[int] = []
    for _ in range(min(k, len(remaining))):
        best = max(
            remaining,
            key=lambda i: (sum((1.0 - alpha) ** seen[x] for x in label_sets[i]), -i),
        )
        order.append(best)
        remaining.remove(best)
        seen.update(label_sets[best])
    return order




def overlap_lp_numeric(
    label_sets: list[set[str]], k: int, alpha: float = 0.5
) -> tuple[float, float]:
    """Numerical HiGHS solution of the nested-prefix LP, used for timing.

    x[t,i] is fractional membership of document i in the length-(t+1)
    prefix.  u[t,g] lies below every tangent of the concave geometric coverage
    function h_g.  Certificate decisions use the separate exact-rational
    implementation; this routine reports solver performance only.
    """

    started = time.perf_counter()
    n = len(label_sets)
    k = min(k, n)
    intents = sorted({label for labels in label_sets for label in labels})
    g_index = {label: j for j, label in enumerate(intents)}
    g_count = len(intents)
    memberships: list[list[int]] = [[] for _ in intents]
    for i, labels in enumerate(label_sets):
        for label in labels:
            memberships[g_index[label]].append(i)
    x_vars = n * k
    total_vars = x_vars + g_count * k

    def x_index(t: int, i: int) -> int:
        return t * n + i

    def u_index(t: int, g: int) -> int:
        return x_vars + t * g_count + g

    discounts = log_discounts(k)
    discounts.append(0.0)
    objective = np.zeros(total_vars)
    for t in range(k):
        delta = discounts[t] - discounts[t + 1]
        for g in range(g_count):
            objective[u_index(t, g)] = -delta

    eq_r: list[int] = []
    eq_c: list[int] = []
    eq_v: list[float] = []
    b_eq = np.arange(1, k + 1, dtype=float)
    for t in range(k):
        for i in range(n):
            eq_r.append(t)
            eq_c.append(x_index(t, i))
            eq_v.append(1.0)
    a_eq = coo_matrix((eq_v, (eq_r, eq_c)), shape=(k, total_vars)).tocsr()

    ub_r: list[int] = []
    ub_c: list[int] = []
    ub_v: list[float] = []
    b_ub: list[float] = []
    row = 0
    # Nested prefixes: x[t,i] <= x[t+1,i].
    for t in range(k - 1):
        for i in range(n):
            ub_r.extend([row, row])
            ub_c.extend([x_index(t, i), x_index(t + 1, i)])
            ub_v.extend([1.0, -1.0])
            b_ub.append(0.0)
            row += 1
    retention = 1.0 - alpha
    for t in range(k):
        for g, members in enumerate(memberships):
            h_value = 0.0
            for r in range(len(members)):
                slope = retention**r
                # u <= h(r) + slope * (sum_i A_ig x_i - r)
                ub_r.append(row)
                ub_c.append(u_index(t, g))
                ub_v.append(1.0)
                for i in members:
                    ub_r.append(row)
                    ub_c.append(x_index(t, i))
                    ub_v.append(-slope)
                b_ub.append(h_value - slope * r)
                h_value += slope
                row += 1
    a_ub = coo_matrix((ub_v, (ub_r, ub_c)), shape=(row, total_vars)).tocsr()
    bounds = [(0.0, 1.0)] * x_vars
    # Coverage variables are nonnegative and capped by their full intent value.
    for _t in range(k):
        for members in memberships:
            cap = sum(retention**r for r in range(len(members)))
            bounds.append((0.0, cap))
    solution = linprog(
        objective,
        A_ub=a_ub,
        b_ub=np.asarray(b_ub),
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
        options={"presolve": True},
    )
    if not solution.success:
        raise RuntimeError(f"prefix LP failed: {solution.message}")
    return -float(solution.fun), time.perf_counter() - started




def trec_full_lp(
    paths: Iterable[Path], *, k: int = 20, exact_certificates: bool = True
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        year = path.name[:4]
        for qid, docs in load_qrels(path).items():
            items = sorted(docs.items())
            label_sets = [set(labels) for _, labels in items]
            actual_k = min(k, len(label_sets))
            greedy = overlap_greedy(label_sets, actual_k)
            numeric_objective, seconds = overlap_lp_numeric(label_sets, actual_k)
            if exact_certificates:
                certificate = exact_overlap_certificate(label_sets, greedy)
                exact_lower = certificate.lower
                exact_upper = certificate.upper
                lower = float(exact_lower)
                upper = float(exact_upper)
                gap = exact_upper - exact_lower
                certified_optimal = gap == 0
                exact_lp_seconds = certificate.solve_seconds
                exact_solver = certificate.solver
            else:
                lower = overlap_alpha_dcg(greedy, label_sets)
                upper = numeric_objective
                tolerance = 1e-8 * max(1.0, upper)
                if upper + tolerance < lower:
                    raise RuntimeError("numerical LP objective lies below feasible score")
                gap = Fraction(max(0.0, upper - lower))
                certified_optimal = None
                exact_lower = exact_upper = None
                exact_lp_seconds = None
                exact_solver = ""
            rows.append(
                {
                    "year": year,
                    "qid": qid,
                    "n": len(label_sets),
                    "intents": len({x for labels in label_sets for x in labels}),
                    "k": actual_k,
                    "fully_exclusive": all(len(labels) == 1 for labels in label_sets),
                    "greedy_lower": lower,
                    "lp_upper": upper,
                    "absolute_gap": float(gap),
                    "relative_gap_to_upper": float(gap / exact_upper)
                    if exact_upper is not None
                    else float(gap) / upper,
                    "certified_optimal": certified_optimal,
                    "greedy_lower_exact": str(exact_lower) if exact_lower is not None else "",
                    "lp_upper_exact": str(exact_upper) if exact_upper is not None else "",
                    "absolute_gap_exact": str(gap) if exact_upper is not None else "",
                    "lp_numeric_objective": numeric_objective,
                    "lp_seconds": seconds,
                    "exact_lp_seconds": exact_lp_seconds,
                    "exact_solver": exact_solver,
                }
            )
    return rows


def load_mind_news() -> dict[str, tuple[str, str]]:
    news: dict[str, tuple[str, str]] = {}
    for filename in ["train_news.tsv", "dev_news.tsv"]:
        with (ROOT / "data" / "mind" / filename).open(
            "r", encoding="utf-8", errors="replace"
        ) as handle:
            for raw in handle:
                fields = raw.rstrip("\n").split("\t")
                if len(fields) >= 3:
                    news[fields[0]] = (fields[1] or "missing", fields[2] or "missing")
    return news


def load_mind_clicks() -> Counter[str]:
    clicks: Counter[str] = Counter()
    path = ROOT / "data" / "mind" / "train_behaviors.tsv"
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            fields = raw.rstrip("\n").split("\t")
            if len(fields) < 5:
                continue
            for token in fields[4].split():
                if token.endswith("-1"):
                    clicks[token.rsplit("-", 1)[0]] += 1
    return clicks


def _median_time(callable_, repeats: int = 3) -> tuple[float, object]:
    times: list[float] = []
    answer = None
    for _ in range(repeats):
        start = time.perf_counter()
        answer = callable_()
        times.append(time.perf_counter() - start)
    return statistics.median(times), answer


def mind_runtime() -> list[dict[str, object]]:
    news = load_mind_news()
    clicks = load_mind_clicks()
    ordered_ids = sorted(news, key=lambda item: (-clicks[item], item))
    rows: list[dict[str, object]] = []
    for granularity, group_position in [("category", 0), ("subcategory", 1)]:
        all_candidates = [
            Candidate(item, news[item][group_position], 1.0 + math.log1p(clicks[item]))
            for item in ordered_ids
        ]
        # ordered_ids is click-sorted, hence also gain-sorted.
        for n in [5_000, 10_000, 20_000, 30_000, 45_000, len(all_candidates)]:
            candidates = all_candidates[:n]
            counts = Counter(candidate.group for candidate in candidates)
            for k in [10, 100, 500]:
                if k > n:
                    continue
                chain_seconds, chain_answer = _median_time(
                    lambda: alpha_chainmerge_presorted(candidates, k=k, alpha=0.5)
                )
                naive_seconds, naive_answer = _median_time(
                    lambda: alpha_naive_greedy(candidates, k=k, alpha=0.5)
                )
                count_seconds, count_answer = _median_time(
                    lambda: alpha_ideal_from_group_counts(counts, k=k, alpha=0.5)
                )
                binary_candidates = [
                    Candidate(candidate.item_id, candidate.group, 1.0)
                    for candidate in candidates
                ]
                binary_naive_seconds, binary_naive_answer = _median_time(
                    lambda: alpha_naive_greedy(binary_candidates, k=k, alpha=0.5)
                )
                chain_value = alpha_dcg(chain_answer, alpha=0.5)
                naive_value = alpha_dcg(naive_answer, alpha=0.5)
                assert abs(chain_value - naive_value) <= 1e-9 * max(1.0, chain_value)
                binary_naive_value = alpha_dcg(binary_naive_answer, alpha=0.5)
                assert abs(binary_naive_value - count_answer[1]) <= 1e-9 * max(
                    1.0, binary_naive_value
                )
                rows.extend(
                    [
                        {
                            "dataset": "MIND-small",
                            "granularity": granularity,
                            "n": n,
                            "groups": len(counts),
                            "k": k,
                            "method": "sequential-greedy",
                            "seconds": naive_seconds,
                            "objective": naive_value,
                        },
                        {
                            "dataset": "MIND-small",
                            "granularity": granularity,
                            "n": n,
                            "groups": len(counts),
                            "k": k,
                            "method": "ChainMerge-graded",
                            "seconds": chain_seconds,
                            "objective": chain_value,
                        },
                        {
                            "dataset": "MIND-small",
                            "granularity": granularity,
                            "n": n,
                            "groups": len(counts),
                            "k": k,
                            "method": "ChainMerge-counts",
                            "seconds": count_seconds,
                            "objective": count_answer[1],
                        },
                        {
                            "dataset": "MIND-small",
                            "granularity": granularity,
                            "n": n,
                            "groups": len(counts),
                            "k": k,
                            "method": "sequential-greedy-binary",
                            "seconds": binary_naive_seconds,
                            "objective": binary_naive_value,
                        },
                    ]
                )
    return rows


def erria_runtime(n: int = 65_238) -> list[dict[str, object]]:
    """MIND-sized synthetic ERR-IA scaling with nonuniform priors."""

    rows: list[dict[str, object]] = []
    configurations = [(18, "category-sized"), (270, "subcategory-sized")]
    for group_count, granularity in configurations:
        rng = np.random.default_rng(SEED + group_count)
        group_mass = rng.lognormal(mean=0.0, sigma=1.0, size=group_count)
        group_probability = group_mass / group_mass.sum()
        groups = rng.choice(group_count, size=n, p=group_probability)
        satisfactions = rng.beta(2.0, 8.0, size=n)
        raw_priors = rng.lognormal(mean=0.0, sigma=1.0, size=group_count)
        raw_priors /= raw_priors.sum()
        priors = {group: float(raw_priors[group]) for group in range(group_count)}
        candidates = [
            ErrCandidate(index, int(groups[index]), float(satisfactions[index]))
            for index in range(n)
        ]
        for k in [10, 100, 500]:
            chain_seconds, chain_answer = _median_time(
                lambda: erria_chainmerge(candidates, priors, k=k)
            )
            scan_seconds, scan_answer = _median_time(
                lambda: erria_naive_greedy(candidates, priors, k=k)
            )
            chain_value = erria(chain_answer, priors)
            scan_value = erria(scan_answer, priors)
            if abs(chain_value - scan_value) > 1e-11 * max(1.0, chain_value):
                raise RuntimeError("ERR-IA ChainMerge and marginal scan disagree")
            rows.append(
                {
                    "dataset": "MIND-sized synthetic",
                    "granularity": granularity,
                    "n": n,
                    "groups": group_count,
                    "k": k,
                    "scan_seconds": scan_seconds,
                    "chainmerge_seconds": chain_seconds,
                    "speedup": scan_seconds / chain_seconds,
                    "objective": chain_value,
                }
            )
    return rows


def exhaustive_audit(instances: int = 250) -> dict[str, object]:
    rng = np.random.default_rng(SEED)
    alpha_failures = 0
    err_failures = 0
    alpha_checked = 0
    err_checked = 0
    for _ in range(instances):
        n = int(rng.integers(3, 9))
        k = int(rng.integers(1, min(5, n) + 1))
        groups = rng.integers(0, min(4, n), size=n)
        gains = rng.uniform(0.01, 4.0, size=n)
        alpha = float(rng.uniform())
        candidates = [Candidate(i, int(groups[i]), float(gains[i])) for i in range(n)]
        answer = alpha_chainmerge(candidates, k=k, alpha=alpha)
        value = alpha_dcg(answer, alpha=alpha)
        optimum = max(alpha_dcg(order, alpha=alpha) for order in permutations(candidates, k))
        alpha_failures += abs(value - optimum) > 1e-10
        alpha_checked += 1

        satisfactions = rng.uniform(0.001, 0.999, size=n)
        raw_priors = rng.uniform(0.01, 1.0, size=min(4, n))
        raw_priors /= raw_priors.sum()
        priors = {i: float(x) for i, x in enumerate(raw_priors)}
        err_candidates = [
            ErrCandidate(i, int(groups[i]), float(satisfactions[i])) for i in range(n)
        ]
        err_answer = erria_chainmerge(err_candidates, priors, k=k)
        err_value = erria(err_answer, priors)
        err_optimum = max(erria(order, priors) for order in permutations(err_candidates, k))
        err_failures += abs(err_value - err_optimum) > 1e-10
        err_checked += 1
    return {
        "seed": SEED,
        "alpha_dcg_instances": alpha_checked,
        "alpha_dcg_failures": alpha_failures,
        "err_ia_instances": err_checked,
        "err_ia_failures": err_failures,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
