"""Exact rational certificates for binary overlapping alpha-DCG.

The reference evaluator stores logarithmic discounts as IEEE-754 binary64
constants.  Every such constant is a dyadic rational.  This module pins those
constants by hexadecimal representation, builds the nested-prefix LP entirely
over :class:`fractions.Fraction`, and asks SoPlex to solve the rational LP
exactly.  No floating-point tolerance participates in a certificate decision.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Sequence


# Python/C binary64 values of 1 / log2(rank + 1), pinned for the paper's
# maximum certificate cutoff K=20.  float.fromhex() recovers the bit pattern
# exactly and Fraction.from_float() then exposes its exact dyadic value.
LOG_DISCOUNT_HEX_K20: tuple[str, ...] = (
    "0x1.0000000000000p+0",
    "0x1.430939835353ep-1",
    "0x1.0000000000000p-1",
    "0x1.b903469050f73p-2",
    "0x1.8c23246dc0aa0p-2",
    "0x1.6cc193acea9b5p-2",
    "0x1.5555555555555p-2",
    "0x1.430939835353ep-2",
    "0x1.34413509f79ffp-2",
    "0x1.28009c1dd6454p-2",
    "0x1.1da3383416064p-2",
    "0x1.14b94f8d9641fp-2",
    "0x1.0cf3ffed2d6acp-2",
    "0x1.0619dc46d3e15p-2",
    "0x1.0000000000000p-2",
    "0x1.f50b57eac5885p-3",
    "0x1.eb22cc68aa6e3p-3",
    "0x1.e21e1180c5dabp-3",
    "0x1.d9dcd21439834p-3",
    "0x1.d244c78367a0dp-3",
)


@dataclass(frozen=True)
class ExactCertificate:
    """Exact LP upper bound paired with an exact feasible-ranking value."""

    lower: Fraction
    upper: Fraction
    solve_seconds: float
    solver: str

    @property
    def gap(self) -> Fraction:
        return self.upper - self.lower

    @property
    def certified_optimal(self) -> bool:
        return self.gap == 0


@dataclass(frozen=True)
class ExactProjectionCertificate:
    """Exact solver-free projection bound and its feasible projected ranking."""

    lower: Fraction
    upper: Fraction
    deleted_bound: Fraction
    order: tuple[int, ...]
    primary: tuple[str, ...]

    @property
    def gap(self) -> Fraction:
        return self.upper - self.lower


def binary64_log_discounts(k: int) -> tuple[Fraction, ...]:
    """Return the pinned evaluator discounts as exact dyadic rationals."""

    if not 0 <= k <= len(LOG_DISCOUNT_HEX_K20):
        raise ValueError(
            f"exact certificate supports 0 <= k <= {len(LOG_DISCOUNT_HEX_K20)}"
        )
    return tuple(
        Fraction.from_float(float.fromhex(value))
        for value in LOG_DISCOUNT_HEX_K20[:k]
    )


def exact_overlap_value(
    order: Sequence[int],
    label_sets: Sequence[set[str]],
    *,
    alpha: Fraction = Fraction(1, 2),
) -> Fraction:
    """Evaluate a feasible ranking exactly under the pinned discounts."""

    if not 0 <= alpha <= 1:
        raise ValueError("alpha must lie in [0, 1]")
    discounts = binary64_log_discounts(len(order))
    retention = 1 - alpha
    seen: Counter[str] = Counter()
    value = Fraction(0)
    for rank, index in enumerate(order):
        gain = sum(
            (retention ** seen[label] for label in label_sets[index]),
            Fraction(0),
        )
        value += discounts[rank] * gain
        seen.update(label_sets[index])
    return value


def exact_overlap_dp(
    label_sets: Sequence[set[str]],
    k: int,
    *,
    alpha: Fraction = Fraction(1, 2),
) -> Fraction:
    """Exact subset DP for the small overlap audit, evaluated over rationals."""

    n = len(label_sets)
    k = min(k, n)
    discounts = binary64_log_discounts(k)
    retention = 1 - alpha
    counts: list[Counter[str]] = [Counter() for _ in range(1 << n)]
    for mask in range(1, 1 << n):
        low = mask & -mask
        i = low.bit_length() - 1
        counts[mask] = counts[mask ^ low] + Counter(label_sets[i])
    dp: list[Fraction | None] = [None] * (1 << n)
    dp[0] = Fraction(0)
    for mask in range(1 << n):
        rank = mask.bit_count()
        if rank >= k or dp[mask] is None:
            continue
        for i in range(n):
            if mask & (1 << i):
                continue
            gain = sum(
                (retention ** counts[mask][label] for label in label_sets[i]),
                Fraction(0),
            )
            nxt = mask | (1 << i)
            candidate = dp[mask] + discounts[rank] * gain
            if dp[nxt] is None or candidate > dp[nxt]:
                dp[nxt] = candidate
    answers = [
        value
        for mask, value in enumerate(dp)
        if mask.bit_count() == k and value is not None
    ]
    return max(answers, default=Fraction(0))


def exact_projection_certificate(
    label_sets: Sequence[set[str]],
    k: int,
    *,
    alpha: Fraction = Fraction(1, 2),
) -> ExactProjectionCertificate:
    """Compute the paper's solver-free projection bound over exact rationals."""

    if any(not labels for labels in label_sets):
        raise ValueError("projection requires at least one positive intent per document")
    k = min(k, len(label_sets))
    discounts = binary64_log_discounts(k)
    retention = 1 - alpha
    degrees = Counter(label for labels in label_sets for label in labels)
    primary = [
        min(labels, key=lambda label: (degrees[label], label))
        for labels in label_sets
    ]

    def residual_counts(choice: Sequence[str]) -> Counter[str]:
        return Counter(
            label
            for index, labels in enumerate(label_sets)
            for label in labels
            if label != choice[index]
        )

    def residual_bound(counts: Counter[str]) -> Fraction:
        return sum(
            (
                sum(
                    (
                        discounts[rank] * retention**rank
                        for rank in range(min(count, k))
                    ),
                    Fraction(0),
                )
                for count in counts.values()
            ),
            Fraction(0),
        )

    for _ in range(10):
        changed = False
        current_bound = residual_bound(residual_counts(primary))
        for index, labels in enumerate(label_sets):
            best_label = primary[index]
            best_bound = current_bound
            for label in sorted(labels):
                if label == primary[index]:
                    continue
                trial = list(primary)
                trial[index] = label
                trial_bound = residual_bound(residual_counts(trial))
                if trial_bound < best_bound:
                    best_label = label
                    best_bound = trial_bound
            if best_label != primary[index]:
                primary[index] = best_label
                current_bound = best_bound
                changed = True
        if not changed:
            break

    chains: dict[str, list[int]] = {}
    for index, label in enumerate(primary):
        chains.setdefault(label, []).append(index)
    for indices in chains.values():
        indices.sort(key=str)
    positions = {label: 0 for label in chains}
    order: list[int] = []
    for _ in range(k):
        available = [
            (
                retention ** positions[label],
                label,
                chains[label][positions[label]],
            )
            for label in chains
            if positions[label] < len(chains[label])
        ]
        if not available:
            break
        _, label, index = min(
            available, key=lambda entry: (-entry[0], entry[1], entry[2])
        )
        order.append(index)
        positions[label] += 1

    projected_sets = [{label} for label in primary]
    repaired_optimum = exact_overlap_value(order, projected_sets, alpha=alpha)
    lower = exact_overlap_value(order, label_sets, alpha=alpha)
    deleted_bound = residual_bound(residual_counts(primary))
    independent_intent_bound = sum(
        (
            sum(
                (
                    discounts[rank] * retention**rank
                    for rank in range(min(count, k))
                ),
                Fraction(0),
            )
            for count in degrees.values()
        ),
        Fraction(0),
    )
    upper = min(repaired_optimum + deleted_bound, independent_intent_bound)
    if upper < lower:
        raise RuntimeError("exact projection bound lies below its feasible ranking")
    return ExactProjectionCertificate(
        lower=lower,
        upper=upper,
        deleted_bound=deleted_bound,
        order=tuple(order),
        primary=tuple(primary),
    )


def _dyadic_decimal(value: Fraction) -> str:
    """Render a dyadic rational as an exact terminating decimal string."""

    denominator = value.denominator
    if denominator & (denominator - 1):
        raise ValueError(f"non-dyadic LP coefficient: {value}")
    if value == 0:
        return "0"
    sign = "-" if value < 0 else ""
    numerator = abs(value.numerator)
    exponent = denominator.bit_length() - 1
    if exponent == 0:
        return f"{sign}{numerator}"
    scaled = numerator * (5**exponent)
    digits = str(scaled).rjust(exponent + 1, "0")
    integer = digits[:-exponent] or "0"
    fractional = digits[-exponent:].rstrip("0")
    return f"{sign}{integer}.{fractional}" if fractional else f"{sign}{integer}"


def _linear_expression(terms: Iterable[tuple[Fraction, str]]) -> str:
    pieces: list[str] = []
    for coefficient, variable in terms:
        if coefficient == 0:
            continue
        magnitude = _dyadic_decimal(abs(coefficient))
        if not pieces:
            prefix = "- " if coefficient < 0 else ""
        else:
            prefix = " - " if coefficient < 0 else " + "
        pieces.append(f"{prefix}{magnitude} {variable}")
    return "".join(pieces) or "0"


def build_exact_overlap_lp(
    label_sets: Sequence[set[str]],
    k: int,
    *,
    alpha: Fraction = Fraction(1, 2),
) -> tuple[str, dict[str, Fraction]]:
    """Build the paper's prefix LP with exact binary64/rational coefficients."""

    if alpha != Fraction(1, 2):
        raise ValueError("the pinned formal certificate uses alpha = 0.5")
    n = len(label_sets)
    k = min(k, n)
    if k <= 0:
        return "Maximize\n obj: 0\nSubject To\n empty: 0 = 0\nEnd\n", {}

    intents = sorted({label for labels in label_sets for label in labels})
    memberships = {
        label: [i for i, labels in enumerate(label_sets) if label in labels]
        for label in intents
    }
    discounts = list(binary64_log_discounts(k)) + [Fraction(0)]
    deltas = [discounts[t] - discounts[t + 1] for t in range(k)]
    retention = 1 - alpha
    objective = {
        f"u_{t}_{g}": deltas[t]
        for t in range(k)
        for g in range(len(intents))
    }

    lines = ["Maximize", f" obj: {_linear_expression((c, v) for v, c in objective.items())}"]
    lines.append("Subject To")
    for t in range(k):
        lines.append(
            f" prefix_{t}: "
            f"{_linear_expression((Fraction(1), f'x_{t}_{i}') for i in range(n))} = {t + 1}"
        )
    for t in range(k - 1):
        for i in range(n):
            lines.append(f" nested_{t}_{i}: 1 x_{t}_{i} - 1 x_{t + 1}_{i} <= 0")

    caps: dict[int, Fraction] = {}
    for g, label in enumerate(intents):
        members = memberships[label]
        h_value = Fraction(0)
        for r in range(len(members)):
            slope = retention**r
            for t in range(k):
                terms = [(Fraction(1), f"u_{t}_{g}")]
                terms.extend((-slope, f"x_{t}_{i}") for i in members)
                rhs = h_value - slope * r
                lines.append(
                    f" tangent_{t}_{g}_{r}: {_linear_expression(terms)} <= {_dyadic_decimal(rhs)}"
                )
            h_value += slope
        caps[g] = h_value

    lines.append("Bounds")
    for t in range(k):
        for i in range(n):
            lines.append(f" 0 <= x_{t}_{i} <= 1")
        for g in range(len(intents)):
            lines.append(f" 0 <= u_{t}_{g} <= {_dyadic_decimal(caps[g])}")
    lines.append("End")
    return "\n".join(lines) + "\n", objective


def _read_rational_solution(path: Path) -> dict[str, Fraction]:
    values: dict[str, Fraction] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2 or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", fields[0]):
            continue
        try:
            values[fields[0]] = Fraction(fields[1])
        except (ValueError, ZeroDivisionError):
            continue
    return values


def _verify_primal_solution(
    label_sets: Sequence[set[str]],
    k: int,
    values: dict[str, Fraction],
    *,
    alpha: Fraction = Fraction(1, 2),
) -> None:
    """Independently check every primal constraint over exact rationals."""

    n = len(label_sets)
    k = min(k, n)
    intents = sorted({label for labels in label_sets for label in labels})
    memberships = {
        label: [i for i, labels in enumerate(label_sets) if label in labels]
        for label in intents
    }
    retention = 1 - alpha
    for t in range(k):
        prefix = sum((values.get(f"x_{t}_{i}", Fraction(0)) for i in range(n)), Fraction(0))
        if prefix != t + 1:
            raise RuntimeError(f"exact primal violates prefix {t}: {prefix} != {t + 1}")
        for i in range(n):
            x_value = values.get(f"x_{t}_{i}", Fraction(0))
            if not Fraction(0) <= x_value <= Fraction(1):
                raise RuntimeError(f"exact primal violates x bound at ({t}, {i})")
            if t + 1 < k and x_value > values.get(f"x_{t + 1}_{i}", Fraction(0)):
                raise RuntimeError(f"exact primal violates nesting at ({t}, {i})")
        for g, label in enumerate(intents):
            members = memberships[label]
            u_value = values.get(f"u_{t}_{g}", Fraction(0))
            cap = sum((retention**r for r in range(len(members))), Fraction(0))
            if not Fraction(0) <= u_value <= cap:
                raise RuntimeError(f"exact primal violates u bound at ({t}, {g})")
            prefix_mass = sum(
                (values.get(f"x_{t}_{i}", Fraction(0)) for i in members),
                Fraction(0),
            )
            h_value = Fraction(0)
            for r in range(len(members)):
                slope = retention**r
                tangent = h_value + slope * (prefix_mass - r)
                if u_value > tangent:
                    raise RuntimeError(
                        f"exact primal violates tangent at ({t}, {g}, {r})"
                    )
                h_value += slope


def exact_overlap_certificate(
    label_sets: Sequence[set[str]],
    order: Sequence[int],
    *,
    executable: str = "soplex",
    extra_solver_options: Sequence[str] = (),
) -> ExactCertificate:
    """Solve the exact prefix LP and compare it with an exact feasible score."""

    resolved = shutil.which(executable)
    if resolved is None:
        raise RuntimeError(
            "SoPlex is required for exact certificates; install SoPlex >= 8 "
            "and ensure `soplex` is on PATH"
        )
    k = len(order)
    lp_text, objective = build_exact_overlap_lp(label_sets, k)
    lower = exact_overlap_value(order, label_sets)
    with tempfile.TemporaryDirectory(prefix="chainmerge-exact-") as directory:
        root = Path(directory)
        lp_path = root / "certificate.lp"
        solution_path = root / "primal.sol"
        dual_path = root / "dual.sol"
        lp_path.write_text(lp_text, encoding="utf-8")
        required_checks = (
            "problem is solved [optimal]",
            "Primal solution feasible in original problem (max. violation = 0)",
            "Dual solution feasible in original problem (max. violation = 0)",
        )
        base_options = tuple(extra_solver_options)
        attempts = [("default", base_options)]
        if "-r0" not in base_options:
            # SoPlex's bound-flipping ratio test can rarely report a loop and
            # abort. The textbook ratio test changes the simplex pivot rule,
            # not the rational LP or its exact validation.
            attempts.append(("textbook-ratio", (*base_options, "-r0")))
        elapsed = 0.0
        failures: list[str] = []
        transcript = ""
        used_attempt = ""
        for attempt, solver_options in attempts:
            solution_path.unlink(missing_ok=True)
            dual_path.unlink(missing_ok=True)
            started = time.perf_counter()
            process = subprocess.run(
                [
                    resolved,
                    "--readmode=1",
                    "--solvemode=2",
                    "-f0",
                    "-o0",
                    "-l0",
                    "-v3",
                    *solver_options,
                    "-c",
                    f"-X={solution_path}",
                    f"-Y={dual_path}",
                    str(lp_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            elapsed += time.perf_counter() - started
            transcript = process.stdout + "\n" + process.stderr
            passed = all(check in transcript for check in required_checks)
            wrote_solutions = solution_path.exists() and dual_path.exists()
            if process.returncode == 0 and passed and wrote_solutions:
                used_attempt = attempt
                break
            failures.append(f"[{attempt}]\n{transcript[-4000:]}")
        else:
            raise RuntimeError(
                "exact SoPlex solve failed after all certified modes:\n"
                + "\n".join(failures)
            )
        values = _read_rational_solution(solution_path)
    # SoPlex omits exact zeros from sparse solution files.
    _verify_primal_solution(label_sets, k, values)
    upper = sum(
        (coefficient * values.get(name, Fraction(0)) for name, coefficient in objective.items()),
        Fraction(0),
    )
    if upper < lower:
        raise RuntimeError(
            f"exact LP upper bound {upper} lies below feasible score {lower}"
        )
    version = next(
        (line.strip() for line in transcript.splitlines() if line.startswith("SoPlex version")),
        "SoPlex exact rational",
    )
    return ExactCertificate(
        lower=lower,
        upper=upper,
        solve_seconds=elapsed,
        solver=(
            f"{version}; readmode=1, solvemode=2, feastol=0, opttol=0, "
            f"valtol=0, solve_attempt={used_attempt}"
        ),
    )
