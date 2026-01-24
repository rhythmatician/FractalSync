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

use crate::distance_field::DistanceField;
use crate::geometry::{lobe_point_at_angle, period_n_bulb_radius, Complex};
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
    Complex::new(carrier.real + residual_real, carrier.imag + residual_imag)
}

/// Advance the state by dt and compute the new c(t).  Returns the
/// updated complex value.  This convenience function calls
/// `advance()` on the state then `synthesize()`.  The state is
/// mutated in place.
///
/// If a distance field is provided, dt is scaled based on proximity
/// to the Mandelbrot boundary, creating a "potential well" that slows
/// down the orbit near the boundary.
pub fn step(
    state: &mut OrbitState,
    dt: f64,
    residual_params: ResidualParams,
    band_gates: Option<&[f64]>,
    distance_field: Option<&DistanceField>,
    h: f64,                         // transient strength in [0,1]
    d_star: Option<f64>,            // optional target distance band
    max_step: Option<f64>,          // optional max step magnitude
) -> Complex {
    // Scale dt if distance field is available
    let effective_dt = if let Some(field) = distance_field {
        // Get current c position (before advance)
        let c_current = synthesize(state, residual_params, band_gates);
        let velocity_scale = field.get_velocity_scale(c_current) as f64;
        dt * velocity_scale
    } else {
        dt
    };

    // Synthesize current c before advancing time
    let c_current = synthesize(state, residual_params, band_gates);

    // Advance phases by effective dt
    state.advance(effective_dt);

    // Proposed next c based on normal synthesis
    let c_proposed = synthesize(state, residual_params, band_gates);

    // If distance field is available, apply contour-biased integrator to the
    // proposed delta to favor tangential motion and reduce chaotic jumps.
    if let Some(field) = distance_field {
        let u_real = c_proposed.real - c_current.real;
        let u_imag = c_proposed.imag - c_current.imag;

        // transient strength (h) provided by caller
        let h_val = h;

        // Use provided d_star or fallback to the field's slowdown_threshold
        let d_s = d_star.unwrap_or(field.slowdown_threshold as f64);

        // Use provided max_step or fallback default
        let max_s = max_step.unwrap_or(0.5_f64);

        return contour_biased_step(c_current, u_real, u_imag, h_val, Some(field), d_s, max_s);
    }

    c_proposed
}

/// Contour-biased integrator: turn a proposed delta `u` into an actual
/// complex step Δc that favors tangential motion along iso-distance
/// contours and only permits normal motion when `h` (transient) is high.
///
/// Arguments:
///  - `c`: current complex point
///  - `u_real`, `u_imag`: proposed delta in complex-plane coordinates
///  - `h`: transient strength in [0,1]
///  - `distance_field`: optional distance field for sampling/gradients
///  - `d_star`: target distance band (normalized [0,1]) to softly servo toward
///  - `max_step`: maximum allowed step magnitude
pub fn contour_biased_step(
    c: Complex,
    u_real: f64,
    u_imag: f64,
    h: f64,
    distance_field: Option<&DistanceField>,
    d_star: f64,
    max_step: f64,
) -> Complex {
    // Default behaviour if no DF available: clamp proposed delta magnitude
    let u_mag = (u_real * u_real + u_imag * u_imag).sqrt();
    if distance_field.is_none() {
        let scale = if u_mag > max_step { max_step / u_mag } else { 1.0 };
        return Complex::new(c.real + u_real * scale, c.imag + u_imag * scale);
    }

    let df = distance_field.unwrap();
    let d = df.sample_bilinear(c) as f64; // current distance

    // Gradient (∂d/∂x, ∂d/∂y)
    let (gx, gy) = df.gradient(c);
    let grad_norm = (gx * gx + gy * gy).sqrt();

    // If gradient is tiny, fallback to clamped u
    if grad_norm <= 1e-12 {
        let scale = if u_mag > max_step { max_step / u_mag } else { 1.0 };
        return Complex::new(c.real + u_real * scale, c.imag + u_imag * scale);
    }

    // Normal (points toward increasing distance)
    let nx = gx / grad_norm;
    let ny = gy / grad_norm;

    // Tangent = normalize([-gy, gx])
    let tx = -gy / grad_norm;
    let ty = gx / grad_norm;

    // Project proposed u into tangent and normal components
    let proj_t = u_real * tx + u_imag * ty;
    let proj_n = u_real * nx + u_imag * ny;

    // Heuristics: small normal allowed between hits, more during hits
    let normal_scale_no_hit = 0.05_f64; // suppress normal when no transient
    let normal_scale_hit = 1.0_f64; // allow near full normal during hits
    let tangential_scale = 1.0_f64; // always allow tangential motion

    // Interpolate normal scale by transient strength h
    let normal_scale = normal_scale_no_hit + (normal_scale_hit - normal_scale_no_hit) * (h.clamp(0.0, 1.0) as f64);

    // Soft servo toward d_star: add a small normal offset toward (d_star - d)
    let servo_gain = 0.2_f64;
    let servo = servo_gain * (d_star - d);

    // Compose delta in tangent+normal basis
    let mut dx = tx * (proj_t * tangential_scale) + nx * (proj_n * normal_scale + servo);
    let mut dy = ty * (proj_t * tangential_scale) + ny * (proj_n * normal_scale + servo);

    // Clamp magnitude
    let mag = (dx * dx + dy * dy).sqrt();
    if mag > max_step && mag > 0.0 {
        let s = max_step / mag;
        dx *= s;
        dy *= s;
    }

    Complex::new(c.real + dx, c.imag + dy)
}
