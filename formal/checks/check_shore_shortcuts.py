"""Evaluate flat-Shore shortcut references and a local circular perturbation."""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from math import atan, cos, fsum, hypot, isclose, pi, sin, sqrt

QUADRATURE_PANELS = 256
MetricPath = Callable[[float], tuple[float, float, float]]


def integrate(
    integrand: Callable[[float], float], stop: float, panels: int = QUADRATURE_PANELS
) -> float:
    """Composite Simpson quadrature on an even number of panels."""
    assert panels > 0 and panels % 2 == 0
    h = stop / panels
    interior = fsum(
        (4 if index % 2 else 2) * integrand(h * index) for index in range(1, panels)
    )
    return h * (integrand(0.0) + interior + integrand(stop)) / 3


def span_excess(height: float, b_squared: float, panels: int) -> float:
    """Normalized excursion span minus pi, evaluated without cancellation."""
    def integrand(u: float) -> float:
        increment = b_squared * (height * sin(u)) ** 2
        return increment / (sqrt(1.0 + increment) + 1.0)

    return 2 * integrate(integrand, pi / 2, panels)


def excursion_height(excess: float, b_squared: float, panels: int) -> float:
    assert excess > 0.0 and b_squared >= 1.0
    lo, hi = 0.0, 1.0
    while span_excess(hi, b_squared, panels) < excess:
        hi *= 2.0
    for _ in range(64):
        mid = (lo + hi) / 2
        if mid == lo or mid == hi:
            break
        if span_excess(mid, b_squared, panels) < excess:
            lo = mid
        else:
            hi = mid
    assert span_excess(lo, b_squared, panels) <= excess <= span_excess(hi, b_squared, panels)
    return (lo + hi) / 2


def excursion_arc(height: float, b_squared: float, panels: int) -> float:
    def integrand(u: float) -> float:
        z = (height * sin(u)) ** 2
        return sqrt(1.0 + b_squared * z) / (1.0 + z)

    return 2 * hypot(1.0, height) * integrate(integrand, pi / 2, panels)


def shortcut_gap(height: float, b_squared: float, panels: int) -> float:
    """Compute span minus arc after removing the term with zero integral."""
    center = height**2 / (hypot(1.0, height) + 1.0)

    def integrand(u: float) -> float:
        z = (height * sin(u)) ** 2
        return (b_squared * z / (sqrt(1.0 + b_squared * z) + 1.0)) * (
            (z - center) / (1.0 + z)
        )

    return 2 * integrate(integrand, pi / 2, panels)


def path_length(
    path: MetricPath,
    stop: float,
    epsilon: float,
    b_squared: float,
    curvature: float = 0.0,
    panels: int = QUADRATURE_PANELS,
) -> float:
    """Integrate metric speed from a signed normal trace and two velocities."""
    def speed(u: float) -> float:
        r, dr, dy = path(u)
        rho = hypot(epsilon, r)
        radial = hypot(epsilon, sqrt(b_squared) * r) / rho**2 * dr
        tangent = (1.0 + curvature * r) / rho * dy
        return hypot(radial, tangent)

    return integrate(speed, stop, panels)


def excursion_path(height: float, b_squared: float, epsilon: float) -> MetricPath:
    def path(u: float) -> tuple[float, float, float]:
        normal = height * sin(u)
        return (
            epsilon * normal,
            epsilon * height * cos(u),
            epsilon * sqrt(1.0 + b_squared * normal**2),
        )

    return path


def sine_path(height: float, span: float, epsilon: float) -> MetricPath:
    def path(t: float) -> tuple[float, float, float]:
        return epsilon * height * sin(pi * t), epsilon * height * pi * cos(pi * t), (
            epsilon * span
        )

    return path


def shore_travel(vertices: tuple[float, ...], epsilon: float) -> float:
    return fsum(abs(end - start) / epsilon for start, end in zip(vertices, vertices[1:]))


def check_below_threshold() -> None:
    # The path-length integral and shoreline variation are separate evaluators.
    for epsilon in (0.007, 1.0, 250.0):
        for b_squared in (1.0, 2.0, 10.0):
            for span in (0.5, pi - 0.1, pi):
                start = -0.375 * epsilon
                end = start + epsilon * span
                waypoint = start + 0.2 * (end - start)
                assert isclose(shore_travel((start, waypoint, waypoint, end), epsilon), span)
                assert shore_travel((start, end, start, end), epsilon) > span
                competitor = path_length(sine_path(0.125, span, epsilon), 1.0, epsilon, b_squared)
                assert competitor - span > 1e-6
        moved = path_length(sine_path(0.125, 0.0, epsilon), 1.0, epsilon, 1.0)
        assert shore_travel((epsilon, epsilon, epsilon), epsilon) == 0.0 and moved > 0.0


def check_rational_sine_margin() -> None:
    # Concavity of sqrt and pi < 22/7 yield a rational upper bound on the
    # sine excursion's length minus span at a=1, span=4, amplitude=1/4.
    amplitude_squared = Fraction(1, 16)
    span = Fraction(4)
    gap_upper = amplitude_squared * span / 4 * (
        (1 + 2 * amplitude_squared) * (Fraction(22, 7) / span) ** 2
        - 1 / (1 + amplitude_squared)
    )
    assert gap_upper < 0
    fine = path_length(sine_path(0.25, 4.0, 1.0), 1.0, 1.0, 2.0, panels=512)
    assert fine - 4.0 < float(gap_upper)
    print(f"Sine fixture span 4: length {fine:.12f}, shortening at least {-gap_upper}")


def check_sine_second_variation() -> None:
    for epsilon in (0.007, 1.0, 250.0):
        for b_squared in (1.0, 2.0, 10.0):
            for span in (pi - 0.1, pi, pi + 0.1):
                expected = span / 2 * ((pi / span) ** 2 - 1.0)
                for height in (1e-3, 5e-4):
                    length = path_length(
                        sine_path(height, span, epsilon), 1.0, epsilon, b_squared
                    )
                    estimated = 2 * (length - span) / height**2
                    assert abs(estimated - expected) < 2e-5


def check_flat_excursions() -> None:
    print("a ell height arc gap start_bearing_rad")
    for a in (0.0, 1.0, 3.0):
        b_squared = 1.0 + a**2
        for span in (3.2, 4.0, 6.0):
            excess = span - pi
            coarse = excursion_height(excess, b_squared, QUADRATURE_PANELS)
            height = excursion_height(excess, b_squared, 2 * QUADRATURE_PANELS)
            arc = excursion_arc(height, b_squared, 2 * QUADRATURE_PANELS)
            gap = shortcut_gap(height, b_squared, 2 * QUADRATURE_PANELS)
            quadrature_delta = abs(arc - excursion_arc(coarse, b_squared, QUADRATURE_PANELS))
            assert gap > max(1e-10, 100 * quadrature_delta)
            assert isclose(span - gap, arc, abs_tol=2e-12)
            assert isclose(span_excess(height, b_squared, 512), excess, abs_tol=2e-14)
            for epsilon in (0.007, 1.0, 250.0):
                for sign in (-1, 1):
                    path = excursion_path(sign * height, b_squared, epsilon)
                    integrated = path_length(path, pi, epsilon, b_squared, panels=512)
                    assert isclose(integrated, arc, rel_tol=2e-12)
                    displacement = integrate(lambda u: path(u)[2], pi, panels=512)
                    assert isclose(displacement, epsilon * span, rel_tol=2e-12)
            print(f"{a:.0f} {span:.1f} {height:.9f} {arc:.12f} {gap:.9f} +/-{atan(height):.9f}")


def check_near_threshold() -> None:
    print("a span_excess height stable_gap raw_span_minus_arc scaled_gap asymptotic")
    for a in (0.0, 1.0, 3.0):
        b_squared = 1.0 + a**2
        asymptotic = 1.0 / (pi * b_squared)
        for excess in (1e-4, 1e-8, 1e-12):
            height = excursion_height(excess, b_squared, 512)
            stable_gap = shortcut_gap(height, b_squared, 512)
            scaled = stable_gap / excess**2
            target_amplitude_scale = 2 / sqrt(pi * b_squared)
            assert isclose(scaled, asymptotic, rel_tol=3e-4)
            assert isclose(height / sqrt(excess), target_amplitude_scale, rel_tol=3e-4)
            raw_gap = (pi + excess) - excursion_arc(height, b_squared, 512)
            print(f"{a:.0f} {excess:.0e} {height:.9e} {stable_gap:.9e} "
                  f"{raw_gap:.9e} {scaled:.9f} {asymptotic:.9f}")


def check_circular_patch() -> None:
    epsilon, span, b_squared = 1.0, 4.0, 2.0
    height = excursion_height(span - pi, b_squared, 512)
    flat = excursion_arc(height, b_squared, 512)
    print("epsilon_times_curvature inward_arc flat_arc outward_arc")
    for curvature in (0.001, 0.01):
        # Radius 1/curvature, embedded patch |r| < R/10 and angle < pi.
        assert curvature * height < 0.1 and curvature * span < pi
        inward = path_length(excursion_path(-height, b_squared, epsilon), pi,
                             epsilon, b_squared, curvature, 512)
        outward = path_length(excursion_path(height, b_squared, epsilon), pi,
                              epsilon, b_squared, curvature, 512)
        assert inward + 1e-6 < flat < outward - 1e-6
        assert max(flat - inward, outward - flat) <= curvature * height * span
        print(f"{curvature:.3f} {inward:.12f} {flat:.12f} {outward:.12f}")


def main() -> None:
    if not __debug__:
        raise RuntimeError("This check requires Python assertions")
    check_below_threshold()
    check_rational_sine_margin()
    check_sine_second_variation()
    check_flat_excursions()
    check_near_threshold()
    check_circular_patch()


if __name__ == "__main__":
    main()
