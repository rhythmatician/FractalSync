"""
Contour-following controller and height-field sampling.

This mirrors runtime-core's height_controller module for backend parity.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class HeightFieldParams:
    iterations: int = 64
    min_magnitude: float = 1.0e-6


@dataclass
class HeightFieldSample:
    height: float
    gradient: complex
    z: complex
    w: complex
    magnitude: float


@dataclass
class ContourControllerParams:
    correction_gain: float = 0.8
    projection_epsilon: float = 1.0e-6


@dataclass
class ContourStep:
    c: complex
    height: float
    height_error: float
    gradient: complex
    corrected_delta: complex


@dataclass
class ContourState:
    c: complex
    target_height: float
    last_delta: complex

    @classmethod
    def create(cls, c: complex, field_params: HeightFieldParams | None = None) -> "ContourState":
        params = field_params or HeightFieldParams()
        sample = sample_height_field(c, params)
        return cls(c=c, target_height=sample.height, last_delta=0j)

    def set_target_height(self, target_height: float) -> None:
        self.target_height = target_height

    def step(
        self,
        model_delta: complex,
        field_params: HeightFieldParams | None = None,
        controller_params: ContourControllerParams | None = None,
    ) -> ContourStep:
        params = field_params or HeightFieldParams()
        controller = controller_params or ContourControllerParams()
        sample = sample_height_field(self.c, params)
        height_error = sample.height - self.target_height
        corrected_delta = contour_correct_delta(
            model_delta,
            sample.gradient,
            height_error,
            controller,
        )
        self.c += corrected_delta
        self.last_delta = corrected_delta
        return ContourStep(
            c=self.c,
            height=sample.height,
            height_error=height_error,
            gradient=sample.gradient,
            corrected_delta=corrected_delta,
        )


def _dot(a: complex, b: complex) -> float:
    return a.real * b.real + a.imag * b.imag


def sample_height_field(c: complex, params: HeightFieldParams | None = None) -> HeightFieldSample:
    params = params or HeightFieldParams()
    z = 0j
    w = 0j
    for _ in range(params.iterations):
        w = 2.0 * z * w + 1.0
        z = z * z + c
    magnitude = abs(z)
    safe_mag = max(magnitude, params.min_magnitude)
    height = math.log(safe_mag)
    denom = max(safe_mag * safe_mag, params.min_magnitude * params.min_magnitude)
    conj_z = z.conjugate()
    g_complex = (conj_z * w) / denom
    gradient = complex(g_complex.real, -g_complex.imag)
    return HeightFieldSample(
        height=height,
        gradient=gradient,
        z=z,
        w=w,
        magnitude=magnitude,
    )


def contour_correct_delta(
    model_delta: complex,
    gradient: complex,
    height_error: float,
    params: ContourControllerParams | None = None,
) -> complex:
    params = params or ContourControllerParams()
    grad_norm_sq = gradient.real * gradient.real + gradient.imag * gradient.imag
    denom = grad_norm_sq + params.projection_epsilon
    normal_scale = _dot(gradient, model_delta) / denom
    delta_tangent = model_delta - gradient * normal_scale
    correction_scale = params.correction_gain * height_error / denom
    correction = gradient * correction_scale
    return delta_tangent - correction
