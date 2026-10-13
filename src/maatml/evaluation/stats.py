"""Interval arithmetic the evidence layer shares: slices now, derived gates next.

A rate on its own says nothing about how much evidence stands behind it; the
Wilson lower bound is what a floor is derived from, so it lives here once.
"""

from __future__ import annotations

import math
import random
from typing import Sequence

# Two-sided 95 % normal quantile.
Z95 = 1.959963984540054


def wilson_interval(successes: int, n: int, *, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for ``successes`` out of ``n`` Bernoulli trials.

    Raises on ``n <= 0`` rather than returning ``(0.0, 0.0)``: a bound on no
    evidence is not a bound, and a caller that wants "no rate" must say so.
    """
    if n <= 0:
        raise ValueError(f"wilson_interval needs n > 0; got n={n}")
    if not 0 <= successes <= n:
        raise ValueError(f"successes must lie in [0, n]; got {successes}/{n}")
    # At the boundaries the algebra gives exactly 0 or 1; float rounding does not.
    if successes == 0:
        return 0.0, _wilson_upper(0, n, z)
    if successes == n:
        return _wilson_lower_raw(n, n, z), 1.0
    return _wilson_lower_raw(successes, n, z), _wilson_upper(successes, n, z)


def _wilson_centre_half(successes: int, n: int, z: float) -> tuple[float, float]:
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return centre, half


def _wilson_lower_raw(successes: int, n: int, z: float) -> float:
    centre, half = _wilson_centre_half(successes, n, z)
    return max(0.0, centre - half)


def _wilson_upper(successes: int, n: int, z: float) -> float:
    centre, half = _wilson_centre_half(successes, n, z)
    return min(1.0, centre + half)


def wilson_lower(successes: int, n: int, *, z: float = Z95) -> float:
    return wilson_interval(successes, n, z=z)[0]


def floor2(value: float) -> float:
    """Floor to two places: a written floor never rounds up past the evidence."""
    return math.floor(value * 100.0 + 1e-9) / 100.0


def quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("quantile of no values")
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def cluster_bootstrap_lower(
    groups: Sequence[tuple[int, int]],
    *,
    iters: int = 1000,
    seed: str | int = 42,
    alpha: float = 0.05,
) -> float:
    """One-sided lower bound on a pooled rate, resampling whole groups.

    ``groups`` are ``(successes, n)`` per cluster (a camera, a family, a
    document). Rows inside a cluster are not independent, so a row-level
    Wilson bound overstates the evidence; drawing clusters with replacement
    and taking the ``alpha`` quantile of the pooled rate respects that.
    """
    pieces = [(int(k), int(n)) for k, n in groups if n > 0]
    if not pieces:
        raise ValueError("cluster_bootstrap_lower needs at least one group with n > 0")
    rng = random.Random(str(seed))
    samples: list[float] = []
    for _ in range(iters):
        drawn = rng.choices(pieces, k=len(pieces))
        total = sum(n for _k, n in drawn)
        samples.append(sum(k for k, _n in drawn) / total if total else 0.0)
    samples.sort()
    return quantile(samples, alpha)
