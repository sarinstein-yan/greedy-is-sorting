"""Exact chain-merge optimizers for partition-separable novelty metrics."""

from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass
from math import log2
from typing import Hashable, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class Candidate:
    item_id: Hashable
    group: Hashable
    gain: float


def log_discounts(k: int) -> list[float]:
    return [1.0 / log2(rank + 2.0) for rank in range(k)]


def _grouped_sorted(candidates: Iterable[Candidate]) -> dict[Hashable, list[Candidate]]:
    grouped: dict[Hashable, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.gain < 0:
            raise ValueError("gains must be nonnegative")
        grouped[candidate.group].append(candidate)
    for values in grouped.values():
        values.sort(key=lambda x: (-x.gain, str(x.item_id)))
    return dict(grouped)


def alpha_chainmerge(
    candidates: Iterable[Candidate], *, k: int, alpha: float = 0.5
) -> list[Candidate]:
    """Globally optimize exclusive-intent alpha-DCG for any decreasing discount."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    if k <= 0:
        return []
    retention = 1.0 - alpha
    grouped = _grouped_sorted(candidates)
    heap: list[tuple[float, str, int, Hashable]] = []
    for group, values in grouped.items():
        if values:
            heapq.heappush(heap, (-values[0].gain, str(group), 0, group))
    result: list[Candidate] = []
    while heap and len(result) < k:
        _, _, index, group = heapq.heappop(heap)
        values = grouped[group]
        result.append(values[index])
        next_index = index + 1
        if next_index < len(values):
            virtual_gain = values[next_index].gain * retention**next_index
            heapq.heappush(heap, (-virtual_gain, str(group), next_index, group))
    return result


def alpha_chainmerge_presorted(
    candidates: Iterable[Candidate], *, k: int, alpha: float = 0.5
) -> list[Candidate]:
    """Chain merge when the input is already nonincreasing in intrinsic gain.

    A stable partition of a globally score-sorted retrieval list leaves every
    group internally sorted, so no additional comparison sort is required.
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    retention = 1.0 - alpha
    grouped: dict[Hashable, list[Candidate]] = defaultdict(list)
    previous = float("inf")
    for candidate in candidates:
        if candidate.gain > previous + 1e-15:
            raise ValueError("input is not sorted by nonincreasing gain")
        previous = candidate.gain
        grouped[candidate.group].append(candidate)
    heap: list[tuple[float, str, int, Hashable]] = []
    for group, values in grouped.items():
        if values:
            heapq.heappush(heap, (-values[0].gain, str(group), 0, group))
    result: list[Candidate] = []
    while heap and len(result) < k:
        _, _, index, group = heapq.heappop(heap)
        values = grouped[group]
        result.append(values[index])
        next_index = index + 1
        if next_index < len(values):
            score = values[next_index].gain * retention**next_index
            heapq.heappush(heap, (-score, str(group), next_index, group))
    return result


def alpha_ideal_from_group_counts(
    group_counts: Mapping[Hashable, int], *, k: int, alpha: float = 0.5
) -> tuple[list[Hashable], float]:
    """Exact binary exclusive-intent alpha-IDCG from group counts only."""

    retention = 1.0 - alpha
    heap: list[tuple[float, str, int, Hashable]] = []
    for group, count in group_counts.items():
        if count > 0:
            heapq.heappush(heap, (-1.0, str(group), 0, group))
    groups: list[Hashable] = []
    value = 0.0
    disc = log_discounts(k)
    while heap and len(groups) < k:
        negative_gain, _, index, group = heapq.heappop(heap)
        groups.append(group)
        value += disc[len(groups) - 1] * (-negative_gain)
        next_index = index + 1
        if next_index < group_counts[group]:
            heapq.heappush(heap, (-(retention**next_index), str(group), next_index, group))
    return groups, value




def alpha_dcg(
    ranking: Sequence[Candidate],
    *,
    alpha: float = 0.5,
    discounts: Sequence[float] | None = None,
) -> float:
    if discounts is None:
        discounts = log_discounts(len(ranking))
    seen: dict[Hashable, int] = defaultdict(int)
    value = 0.0
    retention = 1.0 - alpha
    for position, candidate in enumerate(ranking[: len(discounts)]):
        value += discounts[position] * candidate.gain * retention ** seen[candidate.group]
        seen[candidate.group] += 1
    return value


def alpha_naive_greedy(
    candidates: Sequence[Candidate], *, k: int, alpha: float = 0.5
) -> list[Candidate]:
    """Reference O(nk) marginal greedy with deterministic tie-breaking."""

    remaining = list(candidates)
    seen: dict[Hashable, int] = defaultdict(int)
    retention = 1.0 - alpha
    result: list[Candidate] = []
    while remaining and len(result) < k:
        best_index = max(
            range(len(remaining)),
            key=lambda idx: (
                remaining[idx].gain * retention ** seen[remaining[idx].group],
                -idx,
            ),
        )
        candidate = remaining.pop(best_index)
        result.append(candidate)
        seen[candidate.group] += 1
    return result


@dataclass(frozen=True)
class ErrCandidate:
    item_id: Hashable
    group: Hashable
    satisfaction: float


def erria_chainmerge(
    candidates: Iterable[ErrCandidate],
    priors: Mapping[Hashable, float],
    *,
    k: int,
) -> list[ErrCandidate]:
    grouped: dict[Hashable, list[ErrCandidate]] = defaultdict(list)
    for candidate in candidates:
        if not 0.0 <= candidate.satisfaction <= 1.0:
            raise ValueError("satisfaction must lie in [0, 1]")
        grouped[candidate.group].append(candidate)
    for values in grouped.values():
        values.sort(key=lambda x: (-x.satisfaction, str(x.item_id)))
    heap: list[tuple[float, str, int, float, Hashable]] = []
    for group, values in grouped.items():
        if values:
            prior = float(priors.get(group, 0.0))
            score = prior * values[0].satisfaction
            heapq.heappush(heap, (-score, str(group), 0, 1.0 - values[0].satisfaction, group))
    result: list[ErrCandidate] = []
    while heap and len(result) < k:
        _, _, index, survival_after, group = heapq.heappop(heap)
        values = grouped[group]
        result.append(values[index])
        next_index = index + 1
        if next_index < len(values):
            score = float(priors.get(group, 0.0)) * survival_after * values[next_index].satisfaction
            next_survival = survival_after * (1.0 - values[next_index].satisfaction)
            heapq.heappush(heap, (-score, str(group), next_index, next_survival, group))
    return result


def erria_naive_greedy(
    candidates: Sequence[ErrCandidate],
    priors: Mapping[Hashable, float],
    *,
    k: int,
) -> list[ErrCandidate]:
    """Reference O(nk) ERR-IA marginal scan with deterministic tie-breaking."""

    remaining = list(candidates)
    survival: dict[Hashable, float] = defaultdict(lambda: 1.0)
    result: list[ErrCandidate] = []
    while remaining and len(result) < k:
        best_index = max(
            range(len(remaining)),
            key=lambda index: (
                float(priors.get(remaining[index].group, 0.0))
                * survival[remaining[index].group]
                * remaining[index].satisfaction,
                -index,
            ),
        )
        candidate = remaining.pop(best_index)
        result.append(candidate)
        survival[candidate.group] *= 1.0 - candidate.satisfaction
    return result


def erria(
    ranking: Sequence[ErrCandidate],
    priors: Mapping[Hashable, float],
    *,
    discounts: Sequence[float] | None = None,
) -> float:
    if discounts is None:
        discounts = log_discounts(len(ranking))
    survival: dict[Hashable, float] = defaultdict(lambda: 1.0)
    value = 0.0
    for position, candidate in enumerate(ranking[: len(discounts)]):
        group = candidate.group
        value += (
            discounts[position]
            * float(priors.get(group, 0.0))
            * survival[group]
            * candidate.satisfaction
        )
        survival[group] *= 1.0 - candidate.satisfaction
    return value
