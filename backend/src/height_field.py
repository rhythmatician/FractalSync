"""
Height-field utilities for Mandelbrot control.

Defines f(c) = log|z_N(c)| along with its gradient and a controller
projection step.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class HeightFieldSample:
    height: float
    gradient: np.ndarray


def iterate_with_derivative(c: complex, iterations: int) -> tuple[complex, complex]:
    z = 0.0 + 0.0j
    w = 0.0 + 0.0j
    for _ in range(iterations):
        w = 2.0 * z * w + 1.0
        z = z * z + c
    return z, w


def height_field(
    c: complex,
    iterations: int = 32,
    epsilon: float = 1e-8,
) -> HeightFieldSample:
    z, w = iterate_with_derivative(c, iterations)
    z_mag_sq = max((z.real * z.real + z.imag * z.imag), epsilon)
    z_mag = np.sqrt(z_mag_sq)
    height = np.log(z_mag + epsilon)

    denom = z_mag_sq + epsilon
    a = (np.conjugate(z) / denom) * w
    gradient = np.array([a.real, -a.imag], dtype=np.float32)

    return HeightFieldSample(height=height, gradient=gradient)


def controller_step(
    c: complex,
    delta_model: np.ndarray,
    target_height: float,
    normal_risk: float,
    height_gain: float = 0.15,
    iterations: int = 32,
    epsilon: float = 1e-8,
) -> tuple[complex, np.ndarray, HeightFieldSample]:
    sample = height_field(c, iterations=iterations, epsilon=epsilon)
    g = sample.gradient.astype(np.float64)
    g2 = max(np.dot(g, g), epsilon)

    normal_component = float(np.dot(g, delta_model)) / g2
    projection_scale = (1.0 - np.clip(normal_risk, 0.0, 1.0)) * normal_component
    projected = delta_model - projection_scale * g

    height_error = sample.height - target_height
    servo_scale = -height_gain * height_error / g2
    servo = servo_scale * g

    delta = projected + servo
    new_c = complex(c.real + float(delta[0]), c.imag + float(delta[1]))

    return new_c, delta.astype(np.float32), sample
