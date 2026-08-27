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
        let residual_phases = residual_phases_for_seed(seed, k_residuals);
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

/// Generate the deterministic residual phases for a given seed.
///
/// This is the single source of truth for residual phase generation. Both
/// [`OrbitState::new_with_seed`] (used by the runtime) and the training-time
/// differentiable mirror call this so training and runtime share identical
/// phase statistics — eliminating the historical golden-angle vs seeded-RNG
/// parity gap.
pub fn residual_phases_for_seed(seed: u64, k_residuals: usize) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..k_residuals)
        .map(|_| rng.gen::<f64>() * 2.0 * std::f64::consts::PI)
        .collect()
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

/// A c-space integrator for the Player (issue #88, Q2).
///
/// Unlike [`OrbitState`], which traces a fixed closed carrier loop, the
/// Player holds `c` as persistent state in the complex plane and moves it
/// toward a model-driven *target* point on the Mandelbrot boundary, biased
/// to follow the Shore's contours via [`crate::minimap::contour_biased_step`].
///
/// This restores the audio-driven wandering behaviour that the pre-wasm
/// TypeScript port had (`cBase = mandelbrotBoundary(s, alpha)`), while
/// keeping the canonical Rust math and actually exercising the minimap.
#[derive(Clone, Debug)]
pub struct PlayerState {
    /// Current Julia parameter in the complex plane.
    pub c: num_complex::Complex64,
    /// Persistent c-space velocity (Momentum). The pull toward the model's
    /// target is an *acceleration*; friction bleeds it off. This prevents the
    /// servo from parking at a fixed point when the target barely moves.
    pub velocity: num_complex::Complex64,
    /// Per-frame velocity retention (friction). 1.0 = no friction.
    pub drag: f64,
    /// Active Mandelbrot lobe (1 = main cardioid).
    pub lobe: u32,
    /// Sub-lobe within the period.
    pub sub_lobe: u32,
    /// Radial scale target from the model (how far from the Shore).
    pub s: f64,
    /// Angular position target from the model (where along the boundary).
    pub alpha: f64,
    /// Speed multiplier from the model.
    pub omega_scale: f64,
    /// Mip level used for the contour step (0 = finest).
    pub level: usize,
    /// Target shore-proximity distance (0..1) the servo pulls toward.
    pub d_star: f64,
    /// Maximum world-space step per frame.
    pub max_step: f64,
}

impl Default for PlayerState {
    fn default() -> Self {
        Self {
            c: num_complex::Complex64::new(0.0, 0.0),
            velocity: num_complex::Complex64::new(0.0, 0.0),
            drag: 0.90,
            lobe: 1,
            sub_lobe: 0,
            s: 0.5,
            alpha: 0.5,
            omega_scale: 1.0,
            level: 0,
            d_star: 0.5,
            max_step: 0.05,
        }
    }
}

impl PlayerState {
    /// Create a PlayerState starting at the boundary point for the given
    /// `(s, alpha)` so the first frame is already on the Shore.
    pub fn new(lobe: u32, sub_lobe: u32, s: f64, alpha: f64) -> Self {
        let c = crate::geometry::lobe_point_at_angle(
            lobe,
            sub_lobe,
            alpha * 2.0 * std::f64::consts::PI,
            s,
        );
        Self {
            c,
            lobe,
            sub_lobe,
            s,
            alpha,
            ..Self::default()
        }
    }

    /// Apply model-predicted control signals.
    pub fn apply_controls(&mut self, s: f64, alpha: f64, omega_scale: f64) {
        self.s = s;
        self.alpha = alpha;
        self.omega_scale = omega_scale;
    }

    /// Advance the Player by dt, moving `c` toward the model-driven target
    /// point on the boundary, biased along the Shore's contours.
    ///
    /// * `dt` – frame time in seconds.
    /// * `h` – transient/hit signal in [0, 1]; near 1 allows crossing contours.
    /// * `band_gates` – optional per-band jitter modulation.
    ///
    /// Returns the new `c`. When no mip pyramid is loaded, falls back to a
    /// plain clamped move toward the target (still audio-driven, no loop).
    pub fn step(
        &mut self,
        dt: f64,
        h: f64,
        band_gates: Option<&[f64]>,
    ) -> num_complex::Complex64 {
        // Target point on the boundary the model wants to reach.
        let target = crate::geometry::lobe_point_at_angle(
            self.lobe,
            self.sub_lobe,
            self.alpha * 2.0 * std::f64::consts::PI,
            self.s,
        );

        // The pull toward the target is an ACCELERATION, not a velocity.
        // Velocity persists (Momentum) and friction bleeds it off each frame,
        // so c coasts when audio goes quiet and never parks at a fixed point
        // while the model's target keeps drifting. The gain includes dt so
        // `a` is a per-second acceleration; steady-state speed under a fixed
        // offset is a/(1-drag), giving responsive but bounded motion.
        let accel_gain = self.omega_scale.clamp(0.1, 10.0) * 2.0 * dt;
        let mut a_re = (target.re - self.c.re) * accel_gain;
        let mut a_im = (target.im - self.c.im) * accel_gain;

        // Optional residual jitter from the band gates (impulse per frame).
        if let Some(gates) = band_gates {
            for (k, &g) in gates.iter().enumerate() {
                let amp = 0.004 * g.clamp(0.0, 1.0) / (k as f64 + 1.0);
                let phase = self.alpha * 2.0 * std::f64::consts::PI * (k as f64 + 2.0);
                a_re += amp * phase.cos();
                a_im += amp * phase.sin();
            }
        }

        // Integrate: v = drag*v + a; c += v*dt.
        self.velocity = self.velocity.scale(self.drag)
            + num_complex::Complex64::new(a_re, a_im);
        let proposed_re = self.velocity.re * dt;
        let proposed_im = self.velocity.im * dt;

        // Bias the proposed motion along the Shore's contours.
        let (nr, ni) = crate::minimap::contour_biased_step(
            self.c.re,
            self.c.im,
            proposed_re,
            proposed_im,
            h.clamp(0.0, 1.0),
            self.d_star,
            self.max_step,
            self.level,
        )
        .unwrap_or((self.c.re + proposed_re, self.c.im + proposed_im));

        self.c = num_complex::Complex64::new(nr, ni);
        self.c
    }
}
