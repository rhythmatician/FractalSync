"""Compare cancellation-free radial evaluations with high-precision references."""

from __future__ import annotations

from itertools import product
from math import asinh, exp, frexp, hypot, inf, isfinite, log, log1p, ulp
from sys import float_info

from mpmath import mp


LOG2 = log(2.0)


def log_hypot(log_x: float, log_y: float) -> float:
    larger, smaller = max(log_x, log_y), min(log_x, log_y)
    return larger + log1p(exp(2 * (smaller - larger))) / 2


def log_asinh_exp(log_x: float) -> float:
    """Return log(asinh(exp(log_x))) without making an unbounded argument."""
    if log_x < -20:
        return log_x
    if log_x > 20:
        return log(log_x + LOG2)
    return log(asinh(exp(log_x)))


def radial_log(distance: float, floor: float, anisotropy: float) -> float:
    if not all(map(isfinite, (distance, floor, anisotropy))):
        raise ValueError("Inputs must be finite")
    if distance < 0 or floor <= 0 or anisotropy < 0:
        raise ValueError("Expected nonnegative distance and anisotropy and a positive floor")
    if distance == 0:
        return 0.0
    distance_mantissa, distance_exp = frexp(distance)
    floor_mantissa, floor_exp = frexp(floor)
    log_q = log(distance_mantissa / floor_mantissa) + (distance_exp - floor_exp) * LOG2
    b = hypot(1, anisotropy)
    log_b = log(b)
    log_c = -log1p(anisotropy / b)
    first = log_c - log_b + log_asinh_exp(log_b + log_q)
    total = first
    if anisotropy:
        log_z = log_q + log_c + log_hypot(-log_b, log_q) - log_hypot(0, log_q)
        second = log(anisotropy) + log_asinh_exp(log_z)
        larger, smaller = max(first, second), min(first, second)
        total = larger + log1p(exp(smaller - larger))
    try:
        return exp(total)
    except OverflowError:
        return inf


def reference(distance: float, floor: float, anisotropy: float) -> float:
    with mp.workdps(780):
        q, a = mp.mpf(distance) / mp.mpf(floor), mp.mpf(anisotropy)
        b = mp.sqrt(1 + a * a)
        return float(b * mp.asinh(b * q) - a * mp.asinh(a * q / mp.sqrt(1 + q * q)))


def main() -> None:
    distances = (0.0, 5e-324, 1e-300, 1e-10, 1.0, 1e100, 1e300)
    floors = (5e-324, 1e-150, 1e-4, 1.0, 1e150, 1e300)
    anisotropies = (0.0, 1e-100, 1.0, 1 / LOG2, 1e8, 1e150, 1e300, float_info.max)
    worst_relative = 0.0
    finite = 0
    for distance, floor, a in product(distances, floors, anisotropies):
        expected = reference(distance, floor, a)
        actual = radial_log(distance, floor, a)
        if isfinite(expected):
            finite += 1
            assert abs(actual - expected) <= max(8 * ulp(expected), 1e-12 * expected)
            if expected >= 1e-300:
                worst_relative = max(worst_relative, abs(actual - expected) / expected)
        else:
            assert actual == inf
    probes = len(distances) * len(floors) * len(anisotropies)
    print(f"Checked {finite} finite and {probes - finite} overflow probes")
    print(f"Worst relative discrepancy for values >= 1e-300: {worst_relative:.3e}")

    for q, a in ((1e-300, 1e300), (1.0, 1e8)):
        b = hypot(1, a)
        direct = b * asinh(b * q) - a * asinh(a * q / hypot(1, q))
        print(f"q={q:g}, a={a:g}: direct={direct:.12g}, scaled={radial_log(q, 1, a):.12g}")


if __name__ == "__main__":
    main()
