"""
Orbit synthesizer for generating c(t) = carrier + residual.

Implements deterministic orbit-first synthesis with lobe-based carrier
and epicyclic residuals for live Julia-set visualization.

This module is the authoritative source for converting control signals
(lobe, sub_lobe, θ, ω, s, α, residual phases) into Julia parameter c(t).
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

from .mandelbrot_orbits import MandelbrotGeometry


@dataclass
class OrbitState:
    """Complete state for orbit synthesis."""

    lobe: int  # Period number (1=cardioid, 2=period-2, etc.)
    sub_lobe: int  # Sub-lobe index
    theta: float  # Carrier angle (radians)
    omega: float  # Angular velocity (rad/s)
    s: float  # Radius scaling factor
    alpha: float  # Residual amplitude (relative to lobe radius)
    residual_phases: np.ndarray  # Phases for k residual circles (radians)
    residual_omegas: np.ndarray  # Angular velocities for residuals (rad/s)


class OrbitSynthesizer:
    """
    Synthesizes c(t) from orbit parameters using deterministic geometry.

    Formula: c(t) = c_carrier(θ) + c_residual(residual_phases)

    where:
        - c_carrier = s * LobePoint(lobe, sub_lobe, θ)
        - c_residual = α * R * Σ(g_k / k² * exp(i*ϕ_k))
        - R = lobe radius (for scaling residuals relative to carrier)
        - g_k = band gate (1.0 for full strength, 0.0 for muted)

    All geometric functions come from MandelbrotGeometry (authoritative source).
    """

    def __init__(
        self,
        k_residuals: int = 6,
        residual_cap: float = 0.5,
    ):
        """
        Initialize orbit synthesizer.

        Args:
            k_residuals: Number of residual epicycles
            residual_cap: Maximum residual magnitude (relative to lobe radius)
        """
        self.k_residuals = k_residuals
        self.residual_cap = residual_cap

    def synthesize(
        self,
        state: OrbitState,
        band_gates: Optional[np.ndarray] = None,
    ) -> complex:
        """
        Synthesize Julia parameter c from orbit state.

        Args:
            state: Current orbit state
            band_gates: Optional per-band gates (k_residuals,) in [0, 1]

        Returns:
            Complex Julia parameter c
        """
        # Carrier: deterministic orbit point
        c_carrier = MandelbrotGeometry.lobe_point_at_angle(
            state.lobe, state.theta, state.s, state.sub_lobe
        )

        # Residual: epicyclic texture
        if state.alpha == 0.0:
            return c_carrier

        # Get lobe radius for scaling
        radius = self._get_lobe_radius(state.lobe, state.sub_lobe)

        # Sum residual circles
        c_residual = 0j
        for k in range(self.k_residuals):
            # Amplitude decreases as 1/k²
            amplitude = state.alpha * (state.s * radius) / ((k + 1) ** 2)

            # Band gate (default to 1.0 if not provided)
            g_k = 1.0 if band_gates is None else float(band_gates[k])

            # Phasor
            c_residual += amplitude * g_k * np.exp(1j * state.residual_phases[k])

        # Cap residual magnitude
        mag = abs(c_residual)
        cap = self.residual_cap * radius
        if mag > cap:
            c_residual *= cap / mag

        return c_carrier + c_residual

    def _get_lobe_radius(self, lobe: int, sub_lobe: int) -> float:
        """Get lobe radius for residual scaling."""
        if lobe == 1:
            # Cardioid: use reference scale
            return 0.25
        else:
            return MandelbrotGeometry.period_n_bulb_radius(lobe, sub_lobe)

    def step(
        self,
        state: OrbitState,
        dt: float,
        band_gates: Optional[np.ndarray] = None,
    ) -> Tuple[complex, OrbitState]:
        """
        Step forward in time and synthesize c(t).

        Args:
            state: Current orbit state
            dt: Time step (seconds)
            band_gates: Optional per-band gates

        Returns:
            Tuple of (c, new_state)
        """
        # Synthesize current c
        c = self.synthesize(state, band_gates)

        # Advance state
        new_theta = (state.theta + state.omega * dt) % (2 * np.pi)
        new_residual_phases = (state.residual_phases + state.residual_omegas * dt) % (
            2 * np.pi
        )

        new_state = OrbitState(
            lobe=state.lobe,
            sub_lobe=state.sub_lobe,
            theta=new_theta,
            omega=state.omega,
            s=state.s,
            alpha=state.alpha,
            residual_phases=new_residual_phases,
            residual_omegas=state.residual_omegas,
        )

        return c, new_state

    def generate_sequence(
        self,
        state: OrbitState,
        n_samples: int,
        dt: float,
        band_gates: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, OrbitState]:
        """
        Generate a sequence of c(t) values.

        Args:
            state: Initial orbit state
            n_samples: Number of samples to generate
            dt: Time step between samples
            band_gates: Optional per-band gates (k_residuals,) or (n_samples, k_residuals)

        Returns:
            Tuple of (c_sequence, final_state)
            - c_sequence: Complex array (n_samples,)
            - final_state: OrbitState after n_samples steps
        """
        c_sequence = np.zeros(n_samples, dtype=np.complex64)
        current_state = state

        for i in range(n_samples):
            # Get band gates for this sample (if varying)
            gates_i = None
            if band_gates is not None:
                if band_gates.ndim == 2:
                    gates_i = band_gates[i]
                else:
                    gates_i = band_gates

            c, current_state = self.step(current_state, dt, gates_i)
            c_sequence[i] = c

        return c_sequence, current_state

    def compute_velocity(
        self,
        state: OrbitState,
        band_gates: Optional[np.ndarray] = None,
    ) -> complex:
        """
        Compute velocity dc/dt at current state (analytic derivative).

        Args:
            state: Current orbit state
            band_gates: Optional per-band gates

        Returns:
            Complex velocity dc/dt
        """
        # Carrier velocity: ω * d/dθ[lobe_point_at_angle]
        tangent = MandelbrotGeometry.lobe_tangent_at_angle(
            state.lobe, state.theta, state.s, state.sub_lobe
        )
        v_carrier = state.omega * tangent

        # Residual velocity: Σ(ω_k * amplitude * i * exp(i*ϕ_k))
        if state.alpha == 0.0:
            return v_carrier

        radius = self._get_lobe_radius(state.lobe, state.sub_lobe)
        v_residual = 0j

        for k in range(self.k_residuals):
            amplitude = state.alpha * (state.s * radius) / ((k + 1) ** 2)
            g_k = 1.0 if band_gates is None else float(band_gates[k])

            # Velocity: derivative of exp(i*ϕ) is i*ω*exp(i*ϕ)
            v_residual += (
                amplitude
                * g_k
                * state.residual_omegas[k]
                * 1j
                * np.exp(1j * state.residual_phases[k])
            )

        # Cap velocity magnitude proportionally
        v_total = v_carrier + v_residual
        return v_total


def create_initial_state(
    lobe: int = 1,
    sub_lobe: int = 0,
    theta: float = 0.0,
    omega: float = 1.0,
    s: float = 1.02,
    alpha: float = 0.3,
    k_residuals: int = 6,
    residual_omega_scale: float = 2.0,
) -> OrbitState:
    """
    Create initial orbit state with sensible defaults.

    Args:
        lobe: Lobe number (1=cardioid)
        sub_lobe: Sub-lobe index
        theta: Initial carrier angle (radians)
        omega: Carrier angular velocity (rad/s)
        s: Radius scaling factor
        alpha: Residual amplitude
        k_residuals: Number of residual circles
        residual_omega_scale: Scale factor for residual angular velocities

    Returns:
        OrbitState with random residual phases
    """
    # Random initial phases
    residual_phases = np.random.uniform(0, 2 * np.pi, k_residuals)

    # Residual omegas: scale with k (higher harmonics rotate faster)
    residual_omegas = residual_omega_scale * omega * np.arange(1, k_residuals + 1)

    return OrbitState(
        lobe=lobe,
        sub_lobe=sub_lobe,
        theta=theta,
        omega=omega,
        s=s,
        alpha=alpha,
        residual_phases=residual_phases,
        residual_omegas=residual_omegas,
    )
