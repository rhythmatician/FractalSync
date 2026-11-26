"""Research comparison of fixed-length paths in the exterior conformal metric."""

from __future__ import annotations

from collections.abc import Callable
import json
from math import cosh, exp, isclose, log, tanh

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from runtime_core import geometry_provider_query, geometry_provider_version


def green_and_slope(c: complex) -> tuple[float, complex]:
    z, derivative, weight = c, 1 + 0j, 1.0
    for _ in range(512):
        if abs(z) > 1e6 and 4 * abs(c) < abs(z) ** 2:
            return weight * log(abs(z)), weight * (derivative / z)
        derivative, z = 2 * z * derivative + 1, z**2 + c
        weight /= 2
    raise ArithmeticError(f"Exterior escape unresolved at {c}")


def map_jet(c: complex) -> tuple[float, complex]:
    jet = json.loads(geometry_provider_query(c, 1e-4))
    if jet["validity"] != "regular":
        raise ArithmeticError(f"Map status {jet['validity']} at {c}")
    gradient = complex(*jet["grad_D"])
    if not abs(gradient) > 0:
        raise ArithmeticError(f"Map direction undefined at {c}")
    return float(jet["D"]), gradient


def trace(start: complex, length: float, field: Callable[[float, complex], complex]) -> complex:
    def derivative(time: float, position: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return np.array([field(time, complex(position[0]))], dtype=np.complex128)

    solution = solve_ivp(
        derivative, (0, length), [start], rtol=1e-10, atol=1e-12, max_step=length / 64
    )
    if not solution.success:
        raise ArithmeticError(solution.message)
    return complex(solution.y[0, -1])


def main() -> None:
    start = 0.5 + 0.5j
    length = 0.75
    g0, _ = green_and_slope(start)
    distance0, _ = map_jet(start)
    trap0 = abs(start * (start + 1)) ** 2

    def ray(_: float, c: complex) -> complex:
        g, u = green_and_slope(c)
        return g / u

    def equipotential(time: float, c: complex) -> complex:
        return 1j * ray(time, c)

    def tangent_geodesic(time: float, c: complex) -> complex:
        _, u = green_and_slope(c)
        sech = 1 / cosh(time)
        return g0 * (-sech * tanh(time) + 1j * sech**2) / u

    def map_contour(time: float, c: complex) -> complex:
        _, gradient = map_jet(c)
        return 1j * gradient / abs(gradient) * abs(ray(time, c))

    def trap_contour(time: float, c: complex) -> complex:
        gradient = 2 * (2 * c + 1).conjugate() * c * (c + 1)
        return 1j * gradient / abs(gradient) * abs(ray(time, c))

    print("provider", geometry_provider_version(), "g0", g0)
    print("path end_c g_over_g0 change_map_D change_trap_F")
    for field, expected in (
        (ray, exp(length)), (equipotential, 1.0),
        (tangent_geodesic, 1 / cosh(length)), (map_contour, None), (trap_contour, None),
    ):
        end = trace(start, length, field)
        g, _ = green_and_slope(end)
        d, _ = map_jet(end)
        trap_change = abs(end * (end + 1)) ** 2 - trap0
        if expected is not None:
            assert isclose(g / g0, expected, rel_tol=1e-8)
        if field is trap_contour:
            assert abs(trap_change) <= 1e-8
        print(field.__name__, end, f"{g / g0:.8g}", f"{d - distance0:.8g}", f"{trap_change:.8g}")


if __name__ == "__main__":
    main()
