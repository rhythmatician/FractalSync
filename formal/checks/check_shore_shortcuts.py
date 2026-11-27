"""Evaluate flat-Shore shortcut references and solve local circular geodesics."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, permutations
from math import atan, copysign, cos, fsum, hypot, isclose, log1p, pi, sin, sqrt

QUADRATURE_PANELS = 256
MetricPath = Callable[[float], tuple[float, float, float]]


def integrate(
    integrand: Callable[[float], float], stop: float, panels: int = QUADRATURE_PANELS
) -> float:
    """Composite Simpson quadrature on an even number of panels."""
    assert panels > 0 and panels % 2 == 0
    step = stop / panels
    return simpson_samples([integrand(step * index) for index in range(panels + 1)], step)


def simpson_samples(samples: Sequence[float], step: float) -> float:
    assert len(samples) >= 3 and len(samples) % 2 == 1
    interior = fsum(
        (4 if index % 2 else 2) * value for index, value in enumerate(samples[1:-1], start=1)
    )
    return step * (samples[0] + interior + samples[-1]) / 3


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
        tangent = (1.0 - curvature * r) / rho * dy
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


def sine_gap_upper(amplitude: Fraction, span: Fraction, b_squared: int) -> Fraction:
    # Concavity of sqrt and pi < 22/7 yield a rational upper bound on the
    # sine excursion's length minus its span.
    amplitude_squared = amplitude**2
    return amplitude_squared * span / 4 * (
        (1 + b_squared * amplitude_squared) * (Fraction(22, 7) / span) ** 2
        - 1 / (1 + amplitude_squared)
    )


def check_rational_sine_margin() -> None:
    gap_upper = sine_gap_upper(Fraction(1, 4), Fraction(4), 2)
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


def check_restricted_flat_paths() -> None:
    span, b_squared, strip_width = 4.0, 2.0, 0.05
    height = excursion_height(span - pi, b_squared, 512)
    unrestricted = excursion_arc(height, b_squared, 512)
    assert height > strip_width
    # The positive branch stays in the upper half-plane. The reflected
    # branch is inadmissible there, so the first remains minimizing alone.
    assert isclose(
        path_length(excursion_path(height, b_squared, 1.0), pi, 1.0, b_squared),
        unrestricted,
        rel_tol=2e-12,
    )

    # In the strip, every path pays at least this much tangential length.
    strip_lower = span / hypot(1.0, strip_width)
    strip_upper = path_length(sine_path(strip_width, span, 1.0), 1.0, 1.0, b_squared)
    rational_width, rational_span = Fraction(1, 20), Fraction(4)
    unrestricted_upper = rational_span + sine_gap_upper(Fraction(1, 4), rational_span, 2)
    assert unrestricted_upper**2 * (1 + rational_width**2) < rational_span**2
    assert sine_gap_upper(rational_width, rational_span, 2) < 0
    assert unrestricted < strip_lower < strip_upper < span
    print(f"Flat strip: unrestricted {unrestricted:.12f}, strip bounds "
          f"{strip_lower:.12f} to {strip_upper:.12f}")

    # The same straight unit-speed geodesic is minimizing up to arclength
    # pi. Its continuation to arclength 4 has strictly shorter competitors.
    start, local_end, extended_end = 0.0, 3.0, span
    assert shore_travel((start, local_end), 1.0) < pi
    assert shore_travel((start, extended_end), 1.0) > unrestricted


def check_flat_pairwise_bounds() -> None:
    b_squared = 2.0
    points = (0.0, 1.2, 3.6, 5.4)
    distances: dict[tuple[float, float], float] = {}
    for left, right in combinations(points, 2):
        span = right - left
        forward = reverse = span
        if span > pi:
            height = excursion_height(span - pi, b_squared, 512)
            path = excursion_path(height, b_squared, 1.0)

            def reversed_path(
                u: float, forward_path: MetricPath = path
            ) -> tuple[float, float, float]:
                r, dr, dy = forward_path(pi - u)
                return r, -dr, -dy

            forward = path_length(path, pi, 1.0, b_squared, panels=512)
            reverse = path_length(reversed_path, pi, 1.0, b_squared, panels=1024)
        assert 0 < 2 * log1p(span / 2) < min(forward, reverse)
        assert isclose(forward, reverse, rel_tol=2e-12)
        distances[left, right], distances[right, left] = forward, reverse
    for start, via, end in permutations(points, 3):
        assert distances[start, end] <= distances[start, via] + distances[via, end] + 2e-12


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
        inward = path_length(excursion_path(height, b_squared, epsilon), pi,
                             epsilon, b_squared, curvature, 512)
        outward = path_length(excursion_path(-height, b_squared, epsilon), pi,
                              epsilon, b_squared, curvature, 512)
        assert inward + 1e-6 < flat < outward - 1e-6
        assert max(flat - inward, outward - flat) <= curvature * height * span
        print(f"{curvature:.3f} {inward:.12f} {flat:.12f} {outward:.12f}")


@dataclass(frozen=True)
class CircularGeodesic:
    midpoint: float
    sine_coefficient: float
    length: float


def circular_half_shot(
    midpoint: float, span: float, b_squared: float, delta: float, panels: int
) -> list[tuple[float, float]]:
    """Integrate (q, dq/dx) from the symmetric midpoint to the right endpoint."""
    assert 0.0 < abs(delta) * span < pi
    step = span / (2 * panels)

    def acceleration(q: float, slope: float) -> float:
        assert abs(delta * q) < 0.1
        rho_squared = 1.0 + q * q
        normal_numerator = 1.0 + b_squared * q * q
        h = 1.0 - delta * q
        normal_metric = normal_numerator / rho_squared**2
        tangent_metric = h * h / rho_squared
        log_normal_derivative = 2 * b_squared * q / normal_numerator - 4 * q / rho_squared
        log_tangent_derivative = -2 * delta / h - 2 * q / rho_squared
        return (
            tangent_metric * log_tangent_derivative / (2 * normal_metric)
            + (log_tangent_derivative - log_normal_derivative / 2) * slope**2
        )

    q, slope = midpoint, 0.0
    samples = [(q, slope)]
    for _ in range(panels):
        k1 = acceleration(q, slope)
        slope2 = slope + step * k1 / 2
        k2 = acceleration(q + step * slope / 2, slope2)
        slope3 = slope + step * k2 / 2
        k3 = acceleration(q + step * slope2 / 2, slope3)
        slope4 = slope + step * k3
        k4 = acceleration(q + step * slope3, slope4)
        q += step * (slope + 2 * slope2 + 2 * slope3 + slope4) / 6
        slope += step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        samples.append((q, slope))
    return samples


def circular_geodesic(
    span: float, b_squared: float, delta: float, bracket: tuple[float, float], panels: int
) -> CircularGeodesic:
    """Shoot for q(span)=0, then measure the reflected graph in ambient coordinates."""
    lo, hi = sorted(bracket)
    lo_residual = circular_half_shot(lo, span, b_squared, delta, panels)[-1][0]
    hi_residual = circular_half_shot(hi, span, b_squared, delta, panels)[-1][0]
    assert lo_residual * hi_residual < 0.0
    for _ in range(48):
        midpoint = (lo + hi) / 2
        samples = circular_half_shot(midpoint, span, b_squared, delta, panels)
        residual = samples[-1][0]
        if residual == 0.0:
            break
        if (residual > 0.0) == (lo_residual > 0.0):
            lo = midpoint
        else:
            hi = midpoint
    assert abs(residual) < 3e-14
    step = span / (2 * panels)
    momentum = (1.0 - delta * midpoint) / hypot(1.0, midpoint)
    orientation = copysign(1.0, delta)
    graph_weight = sqrt(b_squared - 1.0)
    momentum_drift = 0.0
    speeds = []
    for index, (q, slope) in enumerate(samples):
        angle = abs(delta) * (span / 2 + index * step)
        radial_velocity = -orientation * slope
        tangent_velocity = 1.0 - delta * q
        ambient_x = radial_velocity * cos(angle) - tangent_velocity * sin(angle)
        ambient_y = radial_velocity * sin(angle) + tangent_velocity * cos(angle)
        rho = hypot(1.0, q)
        graph_velocity = graph_weight * q * slope / rho
        speed = hypot(ambient_x, ambient_y, graph_velocity) / rho
        speeds.append(speed)
        momentum_drift = max(momentum_drift, abs((tangent_velocity / rho) ** 2 / speed - momentum))
    assert momentum_drift < 4e-8
    length = 2 * simpson_samples(speeds, step)
    sine_coefficient = 4 / span * simpson_samples(
        [q * cos(pi * index * step / span) for index, (q, _) in enumerate(samples)], step
    )
    return CircularGeodesic(midpoint, sine_coefficient, length)


def refined_circular_geodesic(
    span: float, b_squared: float, delta: float, bracket: tuple[float, float]
) -> CircularGeodesic:
    coarse, medium, fine = (
        circular_geodesic(span, b_squared, delta, bracket, panels) for panels in (128, 256, 512)
    )
    midpoint_change = abs(fine.midpoint - medium.midpoint)
    assert midpoint_change < max(5e-11, abs(medium.midpoint - coarse.midpoint) / 6)
    assert midpoint_change < 2e-7
    length_change = abs(fine.length - medium.length)
    assert length_change < min(3e-9, max(1e-13, abs(span - fine.length) / 200))
    return fine


def check_circular_geodesics() -> None:
    print("Circular threshold: a delta midpoint sine_coefficient A/cubic_prediction saving")
    for a in (0.0, 1.0, 3.0):
        b_squared = 1.0 + a**2
        previous_error = 1.0
        for magnitude in (1e-3, 1e-5, 1e-7):
            inward = refined_circular_geodesic(pi, b_squared, magnitude, (0.0, 0.25))
            reversed_normal = refined_circular_geodesic(pi, b_squared, -magnitude, (-0.25, 0.0))
            assert isclose(inward.midpoint, -reversed_normal.midpoint, abs_tol=1e-13)
            assert isclose(
                inward.sine_coefficient, -reversed_normal.sine_coefficient, abs_tol=1e-13
            )
            assert isclose(inward.length, reversed_normal.length, abs_tol=1e-13)
            prediction = (8 * magnitude / (pi * b_squared)) ** (1 / 3)
            ratio = inward.sine_coefficient / prediction
            error = abs(ratio - 1.0)
            assert error < previous_error / 5
            previous_error = error
            saving = pi - inward.length
            assert saving > 0.0
            assert isclose(saving, 1.5 * magnitude * prediction, rel_tol=0.02)
            print(f"{a:.0f} {magnitude:.0e} {inward.midpoint:.9f} "
                  f"{inward.sine_coefficient:.9f} {ratio:.9f} {saving:.9e}")
        assert previous_error < 1e-4

    span, b_squared = 2.5, 2.0
    print("Circular subcritical: delta midpoint midpoint/linear_prediction")
    for delta in (1e-3, 1e-4):
        shot = refined_circular_geodesic(span, b_squared, delta, (0.0, 0.25))
        prediction = delta * (1 / cos(span / 2) - 1)
        assert isclose(shot.midpoint, prediction, rel_tol=2e-4)
        assert shot.length < span
        print(f"{delta:.0e} {shot.midpoint:.9f} {shot.midpoint / prediction:.9f}")

    span, b_squared = 4.0, 2.0
    flat_height = excursion_height(span - pi, b_squared, 512)
    flat_length = excursion_arc(flat_height, b_squared, 512)
    print("Circular supercritical: delta inward_length flat_length outward_length")
    for delta in (0.001, 0.01):
        inward = refined_circular_geodesic(
            span, b_squared, delta, (0.75 * flat_height, 1.25 * flat_height)
        )
        outward = refined_circular_geodesic(
            span, b_squared, delta, (-1.25 * flat_height, -0.75 * flat_height)
        )
        assert inward.length < flat_length < outward.length < span
        for shot, height in ((inward, flat_height), (outward, -flat_height)):
            trial_length = path_length(
                excursion_path(height, b_squared, 1.0), pi, 1.0, b_squared, delta, 512
            )
            assert shot.length < trial_length
        print(f"{delta:.3f} {inward.length:.12f} {flat_length:.12f} {outward.length:.12f}")


def main() -> None:
    if not __debug__:
        raise RuntimeError("This check requires Python assertions")
    check_below_threshold()
    check_rational_sine_margin()
    check_sine_second_variation()
    check_flat_excursions()
    check_restricted_flat_paths()
    check_flat_pairwise_bounds()
    check_near_threshold()
    check_circular_patch()
    check_circular_geodesics()


if __name__ == "__main__":
    main()
