//! Orbit state machine and synthesis
//!
//! This module contains the deterministic controller used by the
//! audio‑driven orbit synthesiser.  The controller maintains a
//! carrier orbit and a set of epicycles (residuals).  At each step
//! the carrier phase `theta` and residual phases advance according
//! to their angular velocities, then the complex Julia parameter
//! `c(t)` is synthesised as the sum of the carrier point on the
//! Mandelbrot lobe and the residual epicycles.  The amplitude of
//! each residual decays as 1/(k+1)² and is modulated by the
//! controller’s `alpha` parameter and the band gate vector.

use crate::geometry::{lobe_point_at_angle, period_n_bulb_radius, Complex};

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
            k_residuals: 6,
            residual_cap: 2.0,
            radius_scale: 1.0,
        }
    }
}

/// State for the deterministic orbit synthesiser.
///
/// Both the backend and frontend share this struct (via pyo3 or
/// wasm‑bindgen).  It contains the current lobe index, sub‑lobe,
/// angular position (`theta`), angular velocity (`omega`), radial
/// scaling factor (`s`), modulation depth (`alpha`) and the phase
/// and angular velocity of each residual epicycle.
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
    /// Create a new OrbitState with random residual phases.  The
    /// residual frequencies are multiples of the base frequency
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
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

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
) -> Complex {
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
        // Amplitude decays as 1/(k+1)^2
        let amplitude = (state.alpha * (state.s * radius)) / ((k as f64 + 1.0).powi(2));
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
    Complex::new(carrier.real + residual_real, carrier.imag + residual_imag)
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
) -> Complex {
    state.advance(dt);
    synthesize(state, residual_params, band_gates)
}
