"""Deterministic synthetic checks for research #162, not runtime authority.

Uses Python's standard library only. Analytic statements and assumptions are in
normalized_distance_fields.md. Floating-point checks are not formal proofs.
"""

from collections.abc import Callable
from math import hypot, isclose, sin
from random import Random


def normalized(field: Callable[[float], float], x: float, u: float, h: float) -> float:
    return (field(x + h * u) - field(x)) / h


def require_close(actual: float, expected: float, label: str) -> None:
    if not isclose(actual, expected, rel_tol=2e-11, abs_tol=2e-11):
        raise AssertionError(f"{label}: {actual} != {expected}")


def main() -> None:
    rng = Random(162)
    field: Callable[[float], float] = lambda t: abs(t - 1.0) - 1.0
    maximum_cocycle_error = 0.0
    maximum_budget_excess = 0.0
    for _ in range(2000):
        x, u, v = (rng.uniform(-2.0, 2.0) for _ in range(3))
        h, alpha = 10 ** rng.uniform(-1.0, 1.0), 10 ** rng.uniform(-1.0, 1.0)
        left = normalized(field, x + h * v, u, alpha * h)
        right = (normalized(field, x, v + alpha * u, h) - normalized(field, x, v, h)) / alpha
        require_close(left, right, "chart transition")
        maximum_cocycle_error = max(maximum_cocycle_error, abs(left - right))

        epsilon, q = 0.3, 10 ** rng.uniform(-1.0, 1.0)
        bias, ripple = rng.uniform(-0.2, 0.2), rng.uniform(-0.1, 0.1)

        def error(t: float) -> float:
            return bias + ripple * sin(3 * t)

        def approximate(t: float) -> float:
            return field(t) + error(t)

        rho = hypot(field(x), epsilon)
        rho_hat = hypot(approximate(x), epsilon)
        h_hat = q * rho_hat
        eta = abs(error(x + h_hat * u) - error(x))
        actual_error = abs(normalized(approximate, x, u, h_hat) - normalized(field, x, u, q * rho))
        bound = eta / h_hat + 2 * abs(u) * abs(rho_hat - rho) / max(rho, rho_hat)
        maximum_budget_excess = max(maximum_budget_excess, actual_error - bound)
        if actual_error > bound + 2e-12:
            raise AssertionError("computed-scale provider error bound failed")

    # Genuine signed distance to [0, 2a] attains both constants of two.
    a, delta, h, k = 1.0, 0.2, 1.0, 1.7
    require_close(abs(normalized(field, 0.0, 1.0, h) - normalized(field, 0.0, 1.0, k)),
                  2 * (k - h) / k, "sharp scale bound")
    require_close(abs(normalized(field, 0.0, 1.0, k) - normalized(field, delta, 1.0, k)),
                  2 * delta / k, "sharp center bound")

    # Common bias cancels the numerator exactly; the crest ruler moves quadratically.
    epsilon = 0.3
    for bias in (-0.1, -0.01, 0.0, 0.01, 0.1):
        rho_hat = hypot(bias, epsilon)
        ratio = (rho_hat - epsilon) / rho_hat
        if ratio > bias * bias / (2 * epsilon * epsilon) + 2e-15:
            raise AssertionError("quadratic crest bound failed")
        biased: Callable[[float], float] = lambda t: field(t) + bias
        require_close(normalized(biased, 0.0, 0.8, 0.7), normalized(field, 0.0, 0.8, 0.7),
                      "constant offset at fixed scale")

    # Halfplane versus halfplane plus a disconnected disk. All 81 exact samples
    # agree, although the signed distance differs at the center of the disk.
    disk_x, disk_y, disk_r = 0.25, 0.01, 0.002

    def halfplane_with_disk(x: float, y: float) -> float:
        return y if y <= 0 else min(y, hypot(x - disk_x, y - disk_y) - disk_r)

    for ix in range(-4, 5):
        for iy in range(-4, 5):
            x, y = ix / 2, iy / 2
            require_close(halfplane_with_disk(x, y), y, "finite-sample topology ambiguity")
    require_close(halfplane_with_disk(disk_x, disk_y), -disk_r, "disconnected component")

    # Exact nonunique dyadic tangents, approximated by enough points of S.
    points = [0.0] + [2.0 ** (-n) for n in range(60)]

    def distance_to_dyadic(t: float) -> float:
        return min(abs(t - point) for point in points)

    for exponent in (8, 12, 20, 32):
        h = 2.0 ** (-exponent)
        require_close(normalized(distance_to_dyadic, 0.0, 1.0, h), 0.0, "dyadic tangent")
        require_close(normalized(distance_to_dyadic, 0.0, 1.0, 1.5 * h), 1 / 3,
                      "shifted dyadic tangent")

    print("PASS: 2000 chart transitions and combined provider/ruler budgets")
    print(f"Maximum chart-transition absolute error: {maximum_cocycle_error:.3e}")
    print(f"Maximum numerical error-budget excess: {maximum_budget_excess:.3e}")
    print("PASS: both sharp constants, offset cancellation, quadratic crest bound")
    print("PASS: 81 identical signed-distance samples with different connectivity")
    print("PASS: nonunique, nonconical dyadic blow-up example")


if __name__ == "__main__":
    main()
