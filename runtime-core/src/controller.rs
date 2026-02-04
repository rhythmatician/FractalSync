//! Orbit state machine and synthesis
//!
//! This module contains the deterministic controller used by the
//! audio‑driven orbit synthesiser.  The controller maintains a
//! carrier orbit and a set of epicycles (residuals).  At each step
//! the carrier phase `theta` and residual phases advance according
//! to their angular velocities, then the complex Julia parameter
//! `c(t)` is synthesised as the sum of the carrier point on the
//! Mandelbrot lobe and the residual epicycles.  The amplitude of
//! each residual decays exponentially as 1/2^(k+1) and is modulated
//! by the controller's `alpha` parameter and the band gate vector.

use crate::geometry::{lobe_point_at_angle, period_n_bulb_radius};
use rand::{rngs::StdRng, Rng, SeedableRng};

/// Shared runtime constants to keep backend and frontend in lockstep.
/// Exposed through bindings so both sides can assert parity at startup.
pub const SAMPLE_RATE: usize = 48_000;
pub const HOP_LENGTH: usize = 1_024;
pub const N_FFT: usize = 4_096;
pub const WINDOW_FRAMES: usize = 10;
pub const DEFAULT_K_RESIDUALS: usize = 6;
pub const DEFAULT_RESIDUAL_CAP: f64 = 0.5;
pub const DEFAULT_RESIDUAL_OMEGA_SCALE: f64 = 2.0;
pub const DEFAULT_BASE_OMEGA: f64 = 1.0;
pub const DEFAULT_ORBIT_SEED: u64 = 1337;

/// Parameters controlling the residual epicycle sums.  These values
/// determine the number of residuals and the cap on their combined
/// magnitude.  The same parameters are used for both Python and
/// WebAssembly bindings.
#[derive(Clone, Copy, Debug)]
pub struct ResidualParams {
    /// Number of residual epicycles (k)
    pub k_residuals: usize,
    /// Maximum allowed residual magnitude as a multiple of the lobe radius
    pub residual_cap: f64,
    /// Scaling factor applied to the carrier radius when computing the
    /// amplitude of the first residual.  In practice this is 1.0.
    pub radius_scale: f64,
}

impl Default for ResidualParams {
    fn default() -> Self {
        Self {
            k_residuals: DEFAULT_K_RESIDUALS,
            residual_cap: DEFAULT_RESIDUAL_CAP,
            radius_scale: 1.0,
        }
    }
}

/// Orbit state: carrier and residual phases.
#[derive(Clone, Debug)]
pub struct OrbitState {
    pub lobe: u32,
    pub sub_lobe: u32,
    pub theta: f64,
    pub omega: f64,
    pub s: f64,
    pub alpha: f64,
    pub residual_phases: Vec<f64>,
    pub residual_omegas: Vec<f64>,
}

impl OrbitState {
    /// Create a new random OrbitState with arbitrary initial phases.
    /// Residual frequencies are multiples of the base frequency
    /// (`omega`) scaled by `residual_omega_scale`.  This mirrors the
    /// behaviour of the existing Python and WASM implementations.
    pub fn new(
        lobe: u32,
        sub_lobe: u32,
        theta: f64,
        omega: f64,
        s: f64,
        alpha: f64,
        k_residuals: usize,
        residual_omega_scale: f64,
    ) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let seed: u64 = rng.gen();
        Self::new_with_seed(
            lobe,
            sub_lobe,
            theta,
            omega,
            s,
            alpha,
            k_residuals,
            residual_omega_scale,
            seed,
        )
    }

    /// Create a new OrbitState with deterministic residual phases.
    ///
    /// This is the constructor you want if you need bit-for-bit
    /// repeatability between runs.
    pub fn new_with_seed(
        lobe: u32,
        sub_lobe: u32,
        theta: f64,
        omega: f64,
        s: f64,
        alpha: f64,
        k_residuals: usize,
        residual_omega_scale: f64,
        seed: u64,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);

        let residual_phases: Vec<f64> = (0..k_residuals)
            .map(|_| rng.gen::<f64>() * 2.0 * std::f64::consts::PI)
            .collect();
        let residual_omegas: Vec<f64> = (0..k_residuals)
            .map(|k| residual_omega_scale * omega * (k as f64 + 1.0))
            .collect();
        Self {
            lobe,
            sub_lobe,
            theta,
            omega,
            s,
            alpha,
            residual_phases,
            residual_omegas,
        }
    }

    /// Advance the internal phases by dt.  This mutates `theta` and
    /// the residual phases but does not perform synthesis.  Call
    /// [`synthesize`] afterwards to compute the next complex value.
    pub fn advance(&mut self, dt: f64) {
        // Wrap the angle into [0, 2π) to avoid unbounded growth.
        self.theta = (self.theta + self.omega * dt) % (2.0 * std::f64::consts::PI);
        for (phase, omega) in self
            .residual_phases
            .iter_mut()
            .zip(self.residual_omegas.iter())
        {
            *phase = (*phase + omega * dt) % (2.0 * std::f64::consts::PI);
        }
    }
}

/// Synthesize the complex parameter c(t) from the given state and
/// residual parameters.  This function is pure and does not mutate
/// the state.
pub fn synthesize(
    state: &OrbitState,
    residual_params: ResidualParams,
    band_gates: Option<&[f64]>,
) -> num_complex::Complex64 {
    // Carrier: deterministic point on the lobe
    let carrier = lobe_point_at_angle(state.lobe, state.sub_lobe, state.theta, state.s);

    // No residuals or zero depth: return carrier early
    if residual_params.k_residuals == 0 || state.alpha == 0.0 {
        return carrier;
    }

    // Determine lobe radius for scaling.  We use the same radius
    // definition as geometry::period_n_bulb_radius for non‑cardioid
    // lobes and a fixed 0.25 for the cardioid.
    let radius = if state.lobe == 1 {
        0.25
    } else {
        period_n_bulb_radius(state.lobe, state.sub_lobe)
    } * residual_params.radius_scale;

    let mut residual_real = 0.0;
    let mut residual_imag = 0.0;

    for k in 0..residual_params.k_residuals {
        // Amplitude decays exponentially as 1/2^(k+1) for tighter jitter
        let amplitude = (state.alpha * (state.s * radius)) / 2.0_f64.powi(k as i32 + 1);
        // Optional gating for each residual band
        let gate = band_gates.map(|g| g.get(k).copied().unwrap_or(1.0)).unwrap_or(1.0);
        let phase = state.residual_phases.get(k).copied().unwrap_or(0.0);
        residual_real += amplitude * gate * phase.cos();
        residual_imag += amplitude * gate * phase.sin();
    }

    // Cap residual magnitude to prevent runaway orbits
    let mag = (residual_real * residual_real + residual_imag * residual_imag).sqrt();
    let cap = residual_params.residual_cap * radius;
    if mag > cap && mag > 0.0 {
        let scale = cap / mag;
        residual_real *= scale;
        residual_imag *= scale;
    }

    // Sum carrier and residual
    num_complex::Complex64::new(carrier.re + residual_real, carrier.im + residual_imag)
}

/// Advance the state by dt and compute the new c(t).  Returns the
/// updated complex value.  This convenience function calls
/// `advance()` on the state then `synthesize()`.  The state is
/// mutated in place.
pub fn step(
    state: &mut OrbitState,
    dt: f64,
    residual_params: ResidualParams,
    band_gates: Option<&[f64]>,
) -> num_complex::Complex64 {
    state.advance(dt);
    synthesize(state, residual_params, band_gates)
}
