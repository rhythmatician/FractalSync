"""Compare steering demand on fixed trap arcs using the project's Rust baseline."""
from __future__ import annotations

from cmath import sqrt as complex_sqrt
from dataclasses import dataclass
from math import fsum, sin, sqrt

from runtime_core import (
    ManifoldConfig,
    geometry_provider_version,
    manifold_geodesic_acceleration,
    manifold_induced_metric,
)


FLOOR = 1e-4
HALF_WINDOW = 4
NORMAL_ACCELERATION = 1.0


@dataclass(frozen=True)
class Sample:
    point: complex
    velocity: complex
    acceleration: complex


def trap_sample(delta: float, side: int, xi: float) -> Sample:
    """Return the positive phase lift and its first two angle derivatives."""
    phase_offset = delta * xi
    rotation_increment = complex(-2 * sin(phase_offset / 2) ** 2, sin(phase_offset))
    radius = (1 - side * delta) / 4
    trap_value = -radius * (1 + rotation_increment)
    discriminant = side * delta - 4 * radius * rotation_increment
    slope = complex_sqrt(side * delta) * complex_sqrt(discriminant / (side * delta))
    velocity = 1j * trap_value / slope
    acceleration = 1j * velocity * (1 - 2 * trap_value / discriminant)
    return Sample((slope - 1) / 2, velocity, acceleration)


def distance_metric(sample: Sample, config: ManifoldConfig) -> tuple[float, float]:
    """Return arc length per phase and signed curvature using the Rust metric."""
    g = manifold_induced_metric(sample.point, config)
    velocity = sample.velocity
    x, y = velocity.real, velocity.imag
    speed_squared = g[0][0] * x * x + 2 * g[0][1] * x * y + g[1][1] * y * y
    volume = sqrt(g[0][0] * g[1][1] - g[0][1] ** 2)
    connection = complex(*manifold_geodesic_acceleration(x, y, sample.point, config))
    normal_numerator = (velocity.conjugate() * (sample.acceleration + connection)).imag
    speed = sqrt(speed_squared)
    return speed, volume * normal_numerator / speed**3


def cardioid_metric(sample: Sample) -> tuple[float, float]:
    """Evaluate the analytic quartic conformal metric inside the main cardioid."""
    u = 1 - complex_sqrt(1 - 4 * sample.point)
    a = 1 - abs(u) ** 2
    if a <= 0:
        raise ValueError(f"Point outside the main cardioid: {sample.point}")
    s = abs(1 - u)
    ruler = a * s / 4
    gradient = (-u + a * (u - 1) / (2 * s**2)) * (1 - u) / s
    floored = (ruler**4 + FLOOR**4)**0.25
    floored_gradient = (ruler / floored)**3 * gradient
    velocity = sample.velocity
    normal = 1j * velocity / abs(velocity)
    euclidean = (velocity.conjugate() * sample.acceleration).imag / abs(velocity)**3
    curvature = floored * euclidean + (normal.conjugate() * floored_gradient).real
    return abs(velocity) / floored, curvature


def summarize(samples: list[tuple[float, float]], phase_step: float) -> tuple[float, ...]:
    arc_length = phase_step * fsum(
        speed / 2 if i in (0, len(samples) - 1) else speed
        for i, (speed, _) in enumerate(samples)
    )
    peak = max(abs(curvature) for _, curvature in samples)
    center = samples[len(samples) // 2][1]
    speed = sqrt(NORMAL_ACCELERATION / peak)
    return center, peak, arc_length, speed, arc_length / speed


def main() -> None:
    points = 513
    graph = ManifoldConfig(epsilon=FLOOR)
    conformal = ManifoldConfig(epsilon=FLOOR, lambda_sq=0)
    print(f"provider={geometry_provider_version()}, samples={points}, a_cap={NORMAL_ACCELERATION}")
    print("delta side metric center_curvature peak_abs_curvature arc_length speed sampled_time")
    for delta in (1e-2, 1e-4, 1e-6):
        for side in (1, -1):
            step = 2 * HALF_WINDOW / (points - 1)
            trace = [trap_sample(delta, side, -HALF_WINDOW + i * step) for i in range(points)]
            for label, measurements in (
                ("distance_graph", [distance_metric(sample, graph) for sample in trace]),
                ("distance_conformal", [distance_metric(sample, conformal) for sample in trace]),
                ("cardioid_quartic", [cardioid_metric(sample) for sample in trace]),
            ):
                values = summarize(measurements, delta * step)
                print(f"{delta:g} {side:+d} {label}", *(f"{value:.8g}" for value in values))


if __name__ == "__main__":
    main()
