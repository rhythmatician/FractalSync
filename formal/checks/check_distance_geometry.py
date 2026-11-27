"""Independent synthetic checks for research issue 160; no runtime authority.

The analytic proofs are in docs/research/160-distance-geometry.md. These finite
floating-point checks are neither formal proofs nor GeometryProvider certificates.
Only the Python standard library is required.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable


def radial_closed(q: float, a: float) -> float:
    """Equation (6), evaluated only on the moderate parameters below."""
    b = math.hypot(1.0, a)
    return b * math.asinh(b * q) - a * math.atanh(a * q / math.hypot(1.0, b * q))


def integrate(f: Callable[[float], float], lo: float, hi: float) -> float:
    """Adaptive Simpson quadrature, independently of the closed primitive."""
    def recurse(
        left: float, right: float, fl: float, fm: float, fr: float,
        whole: float, tol: float, depth: int,
    ) -> float:
        mid = (left + right) / 2
        ql, qr = (left + mid) / 2, (mid + right) / 2
        fql, fqr = f(ql), f(qr)
        first = (mid - left) * (fl + 4 * fql + fm) / 6
        second = (right - mid) * (fm + 4 * fqr + fr) / 6
        delta = first + second - whole
        if abs(delta) <= 15 * tol:
            return first + second + delta / 15
        if depth == 0:
            raise AssertionError("quadrature did not converge")
        return recurse(left, mid, fl, fql, fm, first, tol / 2, depth - 1) + recurse(
            mid, right, fm, fqr, fr, second, tol / 2, depth - 1
        )

    if lo == hi:
        return 0.0
    mid = (lo + hi) / 2
    fl, fm, fr = f(lo), f(mid), f(hi)
    whole = (hi - lo) * (fl + 4 * fm + fr) / 6
    return recurse(lo, hi, fl, fm, fr, whole, 1e-11, 24)


def main() -> None:
    a_values = (0.0, 0.1, 0.5, 1.0, 1 / math.log(2), 3.0, 10.0)
    q_values = (0.0, 1e-6, 0.1, 1.0, 10.0, 1e3, 1e6)
    worst_primitive = 0.0
    for a in a_values:
        b = math.hypot(1.0, a)
        sharp_constant = a * math.asinh(a) - b * math.log(b)
        for q in q_values:
            closed = radial_closed(q, a)
            quad = integrate(lambda u: math.hypot(1.0, a * math.tanh(u)), 0, math.asinh(q))
            worst_primitive = max(worst_primitive, abs(closed - quad))
            assert math.isclose(closed, quad, rel_tol=2e-12, abs_tol=2e-10)
            gap = b * math.asinh(q) - closed
            assert -2e-10 <= gap <= sharp_constant + 2e-10

    rng = random.Random(160)
    worst_remainder = 0.0
    for _ in range(2000):
        a, r = rng.random() * 4, rng.random() * 3
        epsilon = 10 ** rng.uniform(-3, 1)
        w = rng.uniform(-2, 2)
        speed = math.hypot(w, rng.random() * 3)
        den = r * r + epsilon * epsilon
        graph = speed * speed / den + a * a * r * r * w * w / den ** 2
        radial = (epsilon * epsilon + (1 + a * a) * r * r) * w * w / den ** 2
        remainder = (speed * speed - w * w) / den
        scale = max(1.0, abs(graph), abs(radial), abs(remainder))
        worst_remainder = max(worst_remainder, abs(graph - radial - remainder) / scale)
        assert abs(graph - radial - remainder) <= 2e-12 * scale
        assert graph + 2e-12 * scale >= radial

    # S={0} on the line: the positive half-line has exact limiting distance
    # sqrt(1+a^2)*log(y/x). Check the quantitative theorem against this independent
    # explicit unregularized metric, for epsilon chosen inside its stated regime.
    for a in a_values:
        exact_limit = math.hypot(1.0, a) * math.log(3)
        for ratio in (0.1, 0.25, 0.5):
            epsilon = ratio * math.exp(-exact_limit)
            regularized = radial_closed(3 / epsilon, a) - radial_closed(1 / epsilon, a)
            assert (1 - ratio ** 2) * exact_limit - 2e-10 <= regularized
            assert regularized <= exact_limit + 2e-10

    print(f"49 closed-form/quadrature cases: maximum absolute discrepancy {worst_primitive:.3e}")
    print(f"2000 pointwise cases: maximum scaled remainder discrepancy {worst_remainder:.3e}")
    print("21 explicit singular-limit cases: all within the proved relative bound")

    # Opposite points of the unit circle around a point Shore. At zero floor,
    # the angle change costs π even if a competitor varies the radius.
    # The semicircle supplies an upper bound at positive floor.
    clearance = math.exp(-math.pi)
    crossings: list[float] = []
    shore_endpoints: list[float] = []
    for epsilon in (1e-2, 1e-3, 1e-4):
        detour_upper = math.pi / math.hypot(1.0, epsilon)
        detour_lower = (1 - (epsilon / clearance)**2) * math.pi
        assert epsilon < clearance
        assert detour_lower <= detour_upper <= math.pi

        # Parallel Shores at x=0 and x=2. The path from x=-1 to x=1/2
        # attains the two radial costs F(1) + F(1/2). The path from x=0
        # to x=2 must reach distance one, and costs 2F(1).
        radial = radial_closed(1 / epsilon, 1)
        crossings.append(radial + radial_closed(0.5 / epsilon, 1))
        shore_endpoints.append(2 * radial)
        assert radial_closed(0, 1) == 0

    assert all(a < b for a, b in zip(crossings, crossings[1:]))
    assert all(a < b for a, b in zip(shore_endpoints, shore_endpoints[1:]))
    print("Fixed-pair checks: diagonal, point-Shore detour, split components, distinct Shore endpoints")


if __name__ == "__main__":
    main()
