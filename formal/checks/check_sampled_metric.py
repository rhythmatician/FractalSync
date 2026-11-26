"""Measure the installed geometry-provider/6 binding against local references for #120."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from math import cos, hypot, isfinite, log, log2, sin, sqrt, tau
from statistics import median
import sys
from time import perf_counter
from typing import TypedDict

import runtime_core as core

EPSILON = 1e-4
D_REF = 0.1
LAMBDA_SQ = 1.0
ANISOTROPY = LAMBDA_SQ / log(2.0) ** 2
BULB_CENTER = -1 + 0j
BULB_RADIUS = 0.25
DEPTHS = (0.24, 0.15, 0.05, 0.01, 4 * EPSILON, EPSILON, EPSILON / 4)
ANGLES = 12


class Jet(TypedDict):
    D: float | None
    grad_D: list[float | None]
    hessian_D: list[list[float | None]]
    resolved_scale: float | None
    requested_scale: float | None
    validity: str
    singularity: str
    provider_version: str
    tile_id: str
    is_bridge: bool


@dataclass(frozen=True)
class Sample:
    point: complex
    distance: float | None = None
    normal: complex | None = None
    tangent_hessian: float | None = None


@dataclass(frozen=True)
class Measurement:
    point: tuple[float, float]
    status: str
    query_ms: float
    metric_ms: float
    sampled_d: float | None
    exact_d: float | None
    resolved_scale: float | None
    tile_id: str | None
    metrics: dict[str, float]
    error: str | None = None


def query(point: complex) -> Jet:
    jet: Jet = json.loads(core.geometry_provider_query(point, EPSILON))
    return jet


def finite(value: float | None) -> float:
    if value is None or not isfinite(value):
        raise ValueError("regular jet contained a nonfinite value")
    return value


def ideal_condition(distance: float) -> float:
    return 1.0 + ANISOTROPY * (distance / hypot(distance, EPSILON)) ** 2


def bulb_sample(depth: float, index: int) -> Sample:
    angle = tau * index / ANGLES
    normal = complex(cos(angle), sin(angle))
    point = BULB_CENTER + (BULB_RADIUS - depth) * normal
    # Use the represented point as the input to the exact reference formula.
    radial = point - BULB_CENTER
    return Sample(
        point, abs(radial) - BULB_RADIUS, radial / abs(radial), 1.0 / abs(radial)
    )


def measure(sample: Sample, config: core.ManifoldConfig) -> Measurement:
    point = (sample.point.real, sample.point.imag)
    start = perf_counter()
    try:
        jet = query(sample.point)
    except RuntimeError as error:
        return Measurement(
            point,
            "query_exception",
            (perf_counter() - start) * 1e3,
            0.0,
            None,
            sample.distance,
            None,
            None,
            {},
            str(error),
        )
    query_ms = (perf_counter() - start) * 1e3
    start = perf_counter()
    try:
        metric = core.manifold_induced_metric(sample.point, config)
    except RuntimeError as error:
        status = f"{jet['validity']}/{jet['singularity']}"
        if jet["validity"] == "regular":
            status = f"metric_refused/{status}"
        return Measurement(
            point,
            status,
            query_ms,
            (perf_counter() - start) * 1e3,
            jet["D"],
            sample.distance,
            jet["resolved_scale"],
            jet["tile_id"],
            {},
            str(error),
        )
    metric_ms = (perf_counter() - start) * 1e3
    if (
        jet["validity"] != "regular"
        or jet["is_bridge"]
        or jet["provider_version"] != "geometry-provider/6"
    ):
        raise AssertionError(
            f"binding supplied a metric for an inadmissible jet at {point}"
        )

    distance = finite(jet["D"])
    gx, gy = map(finite, jet["grad_D"])
    gradient = complex(gx, gy)
    rho = hypot(distance, EPSILON)
    (xx, xy), (yx, yy) = metric
    # Solve the characteristic quadratic of the actual native 2x2 matrix.
    half_trace = (xx + yy) / 2
    half_spread = hypot((xx - yy) / 2, (xy + yx) / 2)
    small, large = half_trace - half_spread, half_trace + half_spread
    if not (0 < small <= large and isfinite(large) and abs(xy - yx) <= 1e-12 * large):
        raise AssertionError(
            f"native metric is not finite, symmetric and positive definite at {point}"
        )
    condition = large / small
    grad_rho = gradient * (distance / rho)
    jet_condition = 1.0 + ANISOTROPY * abs(grad_rho) ** 2
    result = {
        "rho": rho,
        "sigma": log2(D_REF / rho),
        "eigenvalue_min": small,
        "eigenvalue_max": large,
        "condition": condition,
        "condition_ideal_sampled_D": ideal_condition(distance),
        "condition_error_ideal_sampled_D": abs(condition - ideal_condition(distance)),
        "condition_error_sampled_jet": abs(condition - jet_condition),
        "eigenvalue_min_relative_error_sampled_jet": abs(small * rho**2 - 1.0),
        "principal_tangent_speed": 1.0 / sqrt(small),
        "principal_normal_speed": 1.0 / sqrt(large),
    }
    if sample.distance is not None:
        result["distance_error"] = abs(distance - sample.distance)
    if (
        sample.distance is not None
        and sample.normal is not None
        and sample.tangent_hessian is not None
    ):
        normal, tangent = sample.normal, 1j * sample.normal
        exact_rho = hypot(sample.distance, EPSILON)
        exact_condition = ideal_condition(sample.distance)
        hessian = tuple(finite(value) for row in jet["hessian_D"] for value in row)
        exact_hessian = tuple(
            sample.tangent_hessian * a * b
            for a in (tangent.real, tangent.imag)
            for b in (tangent.real, tangent.imag)
        )

        def directional_speed(direction: complex) -> float:
            x, y = direction.real, direction.imag
            return 1.0 / sqrt(x * (xx * x + xy * y) + y * (yx * x + yy * y))

        result.update(
            {
                "gradient_error": abs(gradient - normal),
                "hessian_frobenius_error": hypot(
                    *(a - b for a, b in zip(hessian, exact_hessian, strict=True))
                ),
                "grad_rho_error": abs(
                    grad_rho - normal * (sample.distance / exact_rho)
                ),
                "condition_ideal_exact_D": exact_condition,
                "condition_error_ideal_exact_D": abs(condition - exact_condition),
                "analytic_tangent_speed": directional_speed(tangent),
                "analytic_normal_speed": directional_speed(normal),
                "ideal_tangent_speed": exact_rho,
                "ideal_normal_speed": exact_rho / sqrt(exact_condition),
            }
        )
    return Measurement(
        point,
        "accepted",
        query_ms,
        metric_ms,
        distance,
        sample.distance,
        jet["resolved_scale"],
        jet["tile_id"],
        result,
    )


def band_summary(measurements: list[Measurement]) -> dict[str, object]:
    accepted = [entry for entry in measurements if entry.status == "accepted"]
    keys = sorted({key for entry in accepted for key in entry.metrics})
    ranges = {
        key: spread([entry.metrics[key] for entry in accepted if key in entry.metrics])
        for key in keys
    }
    return {
        "accepted": len(accepted),
        "refused": len(measurements) - len(accepted),
        "status_counts": dict(Counter(entry.status for entry in measurements)),
        "accepted_positive_D": sum(
            entry.sampled_d is not None and entry.sampled_d > 0 for entry in accepted
        ),
        "accepted_resolved_scale": sorted(
            {
                entry.resolved_scale
                for entry in accepted
                if entry.resolved_scale is not None
            }
        ),
        "query_ms_median": median(entry.query_ms for entry in measurements),
        "metric_ms_median": median(entry.metric_ms for entry in accepted)
        if accepted
        else None,
        "metrics_min_median_max": ranges,
    }


def spread(values: list[float]) -> tuple[float, float, float]:
    return min(values), median(values), max(values)


def initial_velocity(
    point: complex, direction: complex, config: core.ManifoldConfig
) -> complex:
    (xx, xy), (yx, yy) = core.manifold_induced_metric(point, config)
    x, y = direction.real, direction.imag
    return direction / sqrt(x * (xx * x + xy * y) + y * (yx * x + yy * y))


def regularized_jet(jet: Jet) -> tuple[float, complex, list[list[float]]]:
    distance = finite(jet["D"])
    gradient = [finite(value) for value in jet["grad_D"]]
    rho = hypot(distance, EPSILON)
    hessian = [
        [
            distance / rho * finite(jet["hessian_D"][i][j])
            + EPSILON**2 / rho**3 * gradient[i] * gradient[j]
            for j in range(2)
        ]
        for i in range(2)
    ]
    return rho, complex(*gradient) * (distance / rho), hessian


def stencil_check(
    point: complex, fraction: float, config: core.ManifoldConfig
) -> dict[str, object]:
    center = query(point)
    center_status = f"{center['validity']}/{center['singularity']}"
    report: dict[str, object] = {
        "point": (point.real, point.imag),
        "center_status": center_status,
    }
    if center["validity"] != "regular":
        return report
    h = fraction * finite(center["resolved_scale"])
    offsets = (h, -h, 1j * h, -1j * h)
    neighbors = [query(point + offset) for offset in offsets]
    report.update(
        {
            "step": h,
            "neighbor_statuses": [
                f"{jet['validity']}/{jet['singularity']}" for jet in neighbors
            ],
            "same_resolved_scale": all(
                jet["resolved_scale"] == center["resolved_scale"] for jet in neighbors
            ),
            "same_tile_id": all(
                jet["tile_id"] == center["tile_id"] for jet in neighbors
            ),
        }
    )
    if any(jet["validity"] != "regular" for jet in neighbors):
        return report

    try:
        g = core.manifold_induced_metric(point, config)
        g_sides = [
            core.manifold_induced_metric(point + offset, config) for offset in offsets
        ]
        gamma = core.manifold_christoffel_symbols(point, config)
    except RuntimeError as error:
        report["metric_refusal"] = str(error)
        return report

    def difference(plus: float, minus: float) -> float:
        return (plus - minus) / (2 * h)

    _, gradient, hessian = regularized_jet(center)
    rho_sides = [regularized_jet(jet) for jet in neighbors]
    fd_gradient = complex(
        *(difference(rho_sides[2 * j][0], rho_sides[2 * j + 1][0]) for j in range(2))
    )
    fd_hessian = [
        (rho_sides[2 * j][1] - rho_sides[2 * j + 1][1]) / (2 * h) for j in range(2)
    ]
    hessian_error = hypot(
        *(abs(fd_hessian[j] - complex(hessian[0][j], hessian[1][j])) for j in range(2))
    )
    derivatives = [
        [
            [
                difference(g_sides[2 * k][i][j], g_sides[2 * k + 1][i][j])
                for j in range(2)
            ]
            for i in range(2)
        ]
        for k in range(2)
    ]
    determinant = g[0][0] * g[1][1] - g[0][1] * g[1][0]
    inverse = (
        (g[1][1] / determinant, -g[0][1] / determinant),
        (-g[1][0] / determinant, g[0][0] / determinant),
    )
    gamma_errors = []
    for upper in range(2):
        for i in range(2):
            for j in range(2):
                fd = 0.5 * sum(
                    inverse[upper][k]
                    * (
                        derivatives[i][j][k]
                        + derivatives[j][i][k]
                        - derivatives[k][i][j]
                    )
                    for k in range(2)
                )
                gamma_errors.append(fd - gamma[upper][i][j])
    report.update(
        {
            "grad_rho_abs_error": abs(fd_gradient - gradient),
            "hess_rho_frobenius_error": hessian_error,
            "christoffel_frobenius_error": hypot(*gamma_errors),
        }
    )
    return report


def conservative_run(
    point: complex, velocity: complex, steps: int, config: core.ManifoldConfig
) -> dict[str, object]:
    horizon = 0.125
    dt = horizon / steps
    initial = point
    initial_energy: float | None = None
    drifts: list[float] = []
    max_connection, max_force, max_speed = 0.0, 0.0, abs(velocity)
    refusal = None
    try:
        initial_energy = core.manifold_total_energy(
            velocity.real, velocity.imag, point, config
        )
        for _ in range(steps):
            gamma = core.manifold_geodesic_acceleration(
                velocity.real, velocity.imag, point, config
            )
            shore_x, shore_y = core.manifold_potential_force(point, config)
            wall_x, wall_y = core.manifold_wall_force(point, config)
            accel = core.manifold_apply_generalized_force(
                shore_x + wall_x, shore_y + wall_y, point, config
            )
            max_connection = max(max_connection, hypot(*gamma))
            max_force = max(max_force, hypot(*accel))
            x, y, vx, vy, energy = core.manifold_integrate_step(
                point.real,
                point.imag,
                velocity.real,
                velocity.imag,
                0.0,
                0.0,
                0.0,
                dt,
                config,
            )
            if not all(isfinite(value) for value in (x, y, vx, vy, energy.total)):
                refusal = "nonfinite accepted state"
                break
            point, velocity = complex(x, y), complex(vx, vy)
            max_speed = max(max_speed, abs(velocity))
            drifts.append(energy.total - initial_energy)
    except RuntimeError as error:
        refusal = str(error)
    return {
        "dt": dt,
        "attempted_steps": steps,
        "completed_steps": len(drifts),
        "refusal": refusal,
        "initial_point": (initial.real, initial.imag),
        "initial_energy": initial_energy,
        "final_point": (point.real, point.imag),
        "final_velocity": (velocity.real, velocity.imag),
        "end_energy_drift": drifts[-1] if drifts else None,
        "max_abs_energy_drift": max(map(abs, drifts), default=None),
        "max_connection_acceleration": max_connection,
        "max_potential_acceleration": max_force,
        "max_coordinate_speed": max_speed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--details",
        action="store_true",
        help="Include each sample and refusal in the report",
    )
    parser.add_argument(
        "--no-rollout", action="store_true", help="Skip the conservative rollouts"
    )
    parser.add_argument(
        "--no-stencil",
        action="store_true",
        help="Skip local finite-difference comparisons",
    )
    args = parser.parse_args()
    version = core.geometry_provider_version()
    if version != "geometry-provider/6":
        parser.error(f"expected geometry-provider/6, found {version}")
    config = core.ManifoldConfig(
        d_ref=D_REF, epsilon=EPSILON, lambda_sq=LAMBDA_SQ, kappa=1.0
    )
    bands = {
        f"bulb_depth_{depth:g}": [bulb_sample(depth, index) for index in range(ANGLES)]
        for depth in DEPTHS
    }
    bands["bulb_exact_Shore"] = [
        Sample(bulb_sample(0.0, index).point, 0.0) for index in range(ANGLES)
    ]
    bands["cut_controls"] = [Sample(-1 + 0j, -0.25), Sample(-1.001 + 0j, -0.249)]
    bands["offshore_left_axis"] = [
        Sample(complex(-2 - offset), offset) for offset in (0.0001, 0.01, 0.1, 0.5)
    ]
    bands["wall_tip"] = [Sample(-2 + 0j, 0.0)]
    bands["generic_exterior"] = [
        Sample(complex(x, y))
        for x, y in (
            (0.3, 0.0),
            (0.5, 0),
            (0.75, 0),
            (1, 0),
            (1.5, 0),
            (1.9, 0),
            (0.3, 1.1),
        )
    ]
    measurements = {}
    for name, samples in bands.items():
        print(f"Sampling {name} ({len(samples)} points)", file=sys.stderr, flush=True)
        measurements[name] = [measure(sample, config) for sample in samples]
    report: dict[str, object] = {
        "provider_version": version,
        "geometry_scale_alpha": core.geometry_provider_scale_alpha(),
        "epsilon": EPSILON,
        "d_ref": D_REF,
        "lambda_sq": LAMBDA_SQ,
        "kappa": config.kappa,
        "mu": config.mu,
        "bands": {
            name: band_summary(entries) for name, entries in measurements.items()
        },
    }
    if args.details:
        report["samples"] = {
            name: [asdict(entry) for entry in entries]
            for name, entries in measurements.items()
        }
    if not args.no_stencil:
        report["finite_difference_checks"] = [
            stencil_check(bulb_sample(depth, 2).point, fraction, config)
            for depth in (0.05, 0.01, EPSILON)
            for fraction in (0.5, 0.25)
        ]
    if not args.no_rollout:
        start = -1.15 + 0.085j
        try:
            velocity = initial_velocity(start, 0.4 + 0.9j, config)
        except RuntimeError as error:
            report["rollout_initialization_refusal"] = str(error)
        else:
            report["conservative_rollouts"] = {
                "coarse": conservative_run(start, velocity, 15, config),
                "fine": conservative_run(start, velocity, 30, config),
                "perturbed_fine": conservative_run(start + 1e-6j, velocity, 30, config),
            }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
