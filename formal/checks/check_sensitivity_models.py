#!/usr/bin/env python3
"""Independent analytic-model checks for research issue #161.

This is an offline research experiment, not runtime authority or a proof.
Dependencies: numpy and scipy. It does not import FractalSync application code.
Run: python check_sensitivity_models.py
"""

from __future__ import annotations

import json
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad


RealMatrix = NDArray[np.float64]
ComplexFunction = Callable[[float], complex]


def real_inner(left: ComplexFunction, right: ComplexFunction) -> float:
    value, error = quad(
        lambda x: float((left(x).conjugate() * right(x)).real),
        -np.inf,
        np.inf,
        epsabs=2.0e-11,
        epsrel=2.0e-11,
        limit=200,
    )
    if error > 1.0e-8:
        raise AssertionError(f"quadrature error estimate too large: {error}")
    return float(value)


def raw_template(y: float) -> complex:
    polynomial = 1.0 + 0.2 * y + 0.1j * y * y
    return complex(np.exp(-0.5 * (y - 0.8) ** 2) * polynomial)


def raw_derivative(y: float) -> complex:
    polynomial = 1.0 + 0.2 * y + 0.1j * y * y
    polynomial_derivative = 0.2 + 0.2j * y
    return complex(
        np.exp(-0.5 * (y - 0.8) ** 2)
        * (polynomial_derivative - (y - 0.8) * polynomial)
    )


NORMALIZER = float(np.sqrt(real_inner(raw_template, raw_template)))


def template(y: float) -> complex:
    return raw_template(y) / NORMALIZER


def translation_generator(y: float) -> complex:
    return raw_derivative(y) / NORMALIZER


def dilation_generator(y: float) -> complex:
    return 0.5 * template(y) + y * translation_generator(y)


def rendered_template(x: float, mu: float, scale: float) -> complex:
    return template((x - mu) / scale) / float(np.sqrt(scale))


def finite_difference_gram(mu: float, scale: float) -> RealMatrix:
    step = 1.0e-5 * scale

    def d_mu(x: float) -> complex:
        return (
            rendered_template(x, mu + step, scale)
            - rendered_template(x, mu - step, scale)
        ) / (2.0 * step)

    def d_scale(x: float) -> complex:
        return (
            rendered_template(x, mu, scale + step)
            - rendered_template(x, mu, scale - step)
        ) / (2.0 * step)

    cross = real_inner(d_mu, d_scale)
    return np.array(
        [[real_inner(d_mu, d_mu), cross], [cross, real_inner(d_scale, d_scale)]],
        dtype=np.float64,
    )


def main() -> None:
    cross = real_inner(translation_generator, dilation_generator)
    gram = np.array(
        [
            [real_inner(translation_generator, translation_generator), cross],
            [cross, real_inner(dilation_generator, dilation_generator)],
        ],
        dtype=np.float64,
    )
    eigenvalues = np.linalg.eigvalsh(gram)
    if not eigenvalues[0] > 0.0:
        raise AssertionError("analytic template Gram matrix is not SPD")
    if not abs(cross) > 0.1:
        raise AssertionError("chosen asymmetric template lost its cross term")

    a0, b, c0 = float(gram[0, 0]), cross, float(gram[1, 1])
    schur = c0 - b * b / a0
    if not schur > 0.0:
        raise AssertionError("nonpositive Schur complement")

    finite_difference_errors: list[float] = []
    for mu, scale in [(0.3, 0.7), (-0.8, 1.4), (0.0, 0.2)]:
        measured = finite_difference_gram(mu, scale)
        predicted = gram / scale**2
        relative_error = float(
            np.linalg.norm(measured - predicted, ord=2)
            / np.linalg.norm(predicted, ord=2)
        )
        if relative_error >= 2.0e-7:
            raise AssertionError(f"finite-difference metric error {relative_error}")
        finite_difference_errors.append(relative_error)

    # Verify the coordinate shear with nonzero translation-scale coupling.
    shear_errors: list[float] = []
    for d_mu, d_scale, scale in [(0.3, -0.6, 0.2), (4.0, 0.5, 3.0)]:
        original = (
            a0 * d_mu**2 + 2.0 * b * d_mu * d_scale + c0 * d_scale**2
        ) / scale**2
        d_xi = np.sqrt(a0) * (d_mu + b / a0 * d_scale)
        d_z = np.sqrt(schur) * d_scale
        z = np.sqrt(schur) * scale
        hyperbolic = schur * (d_xi**2 + d_z**2) / z**2
        error = float(abs(original - hyperbolic) / max(1.0, abs(original)))
        if error > 1.0e-13:
            raise AssertionError(f"coordinate change error {error}")
        shear_errors.append(error)

    # Exact fold response plus an independently observed tangential coordinate.
    # The scalar distance exponent changes magnitude but cancels from condition.
    a = 1.0
    fold_condition_numbers: dict[str, list[float]] = {}
    distances = [1.0e-2, 1.0e-4, 1.0e-6]
    for p in [-0.5, 0.0, 0.5, 1.0, 2.0]:
        conditions: list[float] = []
        for distance in distances:
            jacobian = np.diag([1.0 / (2.0 * np.sqrt(distance)), 1.0])
            h_inv_sqrt = np.diag([1.0 / np.sqrt(1.0 + a * a), 1.0])
            normalized = distance**p * jacobian @ h_inv_sqrt
            measured = float(np.linalg.cond(normalized) ** 2)
            expected = 1.0 / (4.0 * (1.0 + a * a) * distance)
            if not np.isclose(measured, expected, rtol=1.0e-12, atol=0.0):
                raise AssertionError("fold condition number disagrees with formula")
            conditions.append(measured)
        fold_condition_numbers[str(p)] = conditions

    epsilon = 1.0e-4
    floor_distances = [1.0e-1, 1.0e-4, 1.0e-7]
    floor_responses = [
        float(np.hypot(distance, epsilon) / (4.0 * distance))
        for distance in floor_distances
    ]

    print(
        json.dumps(
            {
                "status": "all analytic-model checks passed; numerical evidence only",
                "template": "normalized shifted Gaussian times a complex polynomial",
                "gram": gram.tolist(),
                "gram_eigenvalues": eigenvalues.tolist(),
                "schur_complement": schur,
                "hyperbolic_curvature": -1.0 / schur,
                "finite_difference_relative_errors": finite_difference_errors,
                "coordinate_shear_relative_errors": shear_errors,
                "fold_distances": distances,
                "fold_condition_numbers_by_exponent": fold_condition_numbers,
                "fixed_epsilon": epsilon,
                "floor_distances": floor_distances,
                "floor_normalized_responses": floor_responses,
                "numpy_version": np.__version__,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
