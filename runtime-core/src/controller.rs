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

/// Contract version of the runtime controller.
///
/// Every exported model records this in its ONNX metadata
/// (`controller_version`); the browser refuses to load an orbit_control
/// model whose version differs from its own. Bump whenever the flags-off
/// semantics of OrbitController::step change (constants, formula, order of
/// operations) - in the SAME commit as regenerating shared/golden_vectors.json
/// and updating the trainer mirror. Version history:
///   1 - May baseline restored (mandelbrotBoundary + harmonic epicycles),
///       with momentum/shore_bias as opt-in flags (default off).
///   2 - Shore-wall physics: contour_biased_step gains the Skyrim wall
///       (transient-gated boundary crossing), fractal tilt (gradient-damped
///       normal motion near the Shore), and the energy servo (loud audio
///       pulls c toward the Shore). step() signatures gain an `energy`
///       argument; shore-bias paths now receive the real transient signal
///       `h` instead of a hardcoded 0.0.
///   3 - Gravity valley: the momentum integrators gain a constant restoring
///       acceleration toward the valley floor at the origin (Physics runs
///       always; quiet audio lets c settle into the valley). The energy
///       servo becomes a pure uphill PUSH along the shore normal — the
///       resting height now emerges from the gravity/push force balance
///       (∝ energy) instead of a fixed d_star target.
///   4 - Manifold physics (issue #106): when `manifold_physics` is on, step()
///       routes through a LEGACY ADAPTER (`step_manifold`) that translates the
///       old (s, alpha, energy) target servo into a generalized force covector
///       and hands it to the musically-ignorant manifold kernel in
///       `crate::manifold`. The kernel owns the induced metric G(c), the
///       analytic Christoffel connection Γ, the native potential U=κσ(c)
///       (Shore = high-energy ridge), and metric-consistent forces. Equations
///       of motion: ṙ + Γ(ṙ,ṙ) = -G⁻¹∇U + G⁻¹Q. The adapter fails closed on
///       manifold error (no silent flat-physics fallback). This is a
///       transitional seam, not destination Controls v2 (issue #107).
pub const CONTROLLER_VERSION: &str = "orbit-controller/4";

/// Gravity: restoring acceleration toward the valley floor at the origin,
/// per frame at |c| = 1. The Map is a landscape — the Shore ridges are
/// HIGH, the interior valley at the lobe center is LOW. Without external
/// forces c settles into the valley (domain contract: Physics runs always;
/// Inside → settle toward lobe center). Audio energy provides the uphill
/// push (see minimap::MUSIC_PUSH_GAIN); gravity is its constant counterforce.
pub const GRAVITY_ACCEL: f64 = 0.01;

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
    /// Audio energy in [0, 1] (loudness). Raises the effective target
    /// shore-proximity: loud audio pulls c toward the Shore (domain
    /// contract: Energy governs distance from The Shore).
    pub energy: f64,
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
            energy: 0.0,
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

    /// Apply model-predicted control signals (LEGACY Controls v1 — RETIRED from destination path, issue #107).
    ///
    /// This `(s, alpha, omega_scale) -> target c` servo is the pre-#107 control surface that
    /// indirectly specified a target Mandelbrot location. The destination physics path (`manifold`
    /// + `controls::MotionControls` generalized forces) does NOT use this method. It is retained
    /// only for backward compatibility with `golden_vectors.json` player_step_cases and the
    /// `orbit_controller` legacy adapter. New code must use `crate::controls::ControlsV2` and
    /// `crate::controls::integrate_motion_controls`.
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

        // Gravity: the valley. Constant restoring pull toward the origin —
        // the physics default that c falls back into when the music goes
        // quiet and the player stops pushing uphill.
        a_re -= GRAVITY_ACCEL * self.c.re;
        a_im -= GRAVITY_ACCEL * self.c.im;

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

        // Bias the proposed motion along the Shore's contours. `h` gates
        // the wall (transients make boundary crossing easy) and `energy`
        // raises the servo's target proximity (loud → near Shore).
        let (nr, ni) = crate::minimap::contour_biased_step(
            self.c.re,
            self.c.im,
            proposed_re,
            proposed_im,
            h.clamp(0.0, 1.0),
            self.d_star,
            self.max_step,
            self.level,
            self.energy,
        )
        .unwrap_or((self.c.re + proposed_re, self.c.im + proposed_im));

        self.c = num_complex::Complex64::new(nr, ni);
        self.c
    }
}

/// The May-proven orbit controller, ported verbatim from the TypeScript
/// implementation that produced the only visually-acceptable results
/// (pre-commit fe1087b). This is the restored baseline.
///
/// Semantics (exact port of the old TS `step()`):
///   theta advances by omega*dt each frame (wobble phase only)
///   cBase = mandelbrotBoundary(s, alpha)  <- model DIRECTLY positions c
///   c = cBase + sum_k gate_k * 0.05 * e^{i * freq_k * theta}
///   where freq_k = (k+2) and boundary is:
///     theta_b = 2*pi*alpha
///     r = 0.25 * (1 - cos(theta_b))
///     c = r * e^{i*theta_b/2} * min(s, 1.5), with s clamped [0.01, 3]
///
/// Unlike OrbitState (closed loop ignoring audio) this is audio-driven:
/// s and alpha move c around the Map every frame. Unlike PlayerState
/// (unproven momentum) this is the empirically validated baseline.
#[derive(Clone, Debug)]
pub struct OrbitController {
    /// Wobble phase in radians (advances by omega*dt each step).
    pub theta: f64,
    /// Base angular velocity of the wobble phase.
    pub omega: f64,
    /// Radial scale from the model (clamped [0.01, 3.0] internally).
    pub s: f64,
    /// Angular position on the cardioid from the model ([0, 1] internally).
    pub alpha: f64,

    // ---- Opt-in refinements (PlayerState ideas, one at a time) ----
    // All default OFF so step() is bit-identical to the May TS controller.
    // Enable ONE at a time and evaluate visually before enabling another.
    //
    /// Refinement 1 — MOMENTUM: c becomes persistent state with velocity
    /// and drag; the boundary point becomes an attractor rather than a
    /// hard position. Smooths control jitter, adds coasting.
    pub momentum: bool,
    /// Persistent c position (used only when momentum or shore_bias is on).
    pub c: num_complex::Complex64,
    /// Velocity state for refinement 1.
    pub velocity: num_complex::Complex64,
    /// Per-frame velocity retention when momentum is on (0.90 = May+10%).
    pub drag: f64,
    /// Audio thrust magnitude for momentum (0 = off). When > 0, sustained
    /// audio energy applies a tangential thrust each frame so loud audio
    /// literally builds inertia: c can never be stationary under sustained
    /// volume. Set via set_thrust() from the runtime's energy signal.
    pub thrust: f64,
    //
    /// Refinement 2 — SHORE BIAS: route motion through the minimap's
    /// contour_biased_step so c hugs the Shore. Requires a loaded pyramid;
    /// silently no-ops without one.
    pub shore_bias: bool,
    /// Target shore-proximity for refinement 2's servo.
    pub d_star: f64,
    /// Max world-space step per frame for refinement 2.
    pub max_step: f64,
    /// Mip level for refinement 2.
    pub level: usize,
    /// Audio energy in [0, 1] (loudness). Raises the effective target
    /// shore-proximity: loud audio pulls c toward the Shore (domain
    /// contract: Energy governs distance from The Shore).
    pub energy: f64,
    //
    /// Refinement 3 — MANIFOLD PHYSICS: Player moves on the Mandelbrot
    /// configuration manifold with proper differential geometry. When enabled,
    /// replaces planar momentum with manifold-aware integration using the
    /// induced metric G(c), Christoffel symbols, and native potential U=κσ(c).
    pub manifold_physics: bool,
    /// Manifold configuration (used only when manifold_physics is on).
    pub manifold_config: crate::manifold::ManifoldConfig,
    /// Planar velocity (vx, vy) for manifold integration.
    pub planar_velocity: (f64, f64),
    /// Drag coefficient for manifold physics (beta in Q_drag = -beta*G*v).
    pub manifold_drag: f64,
    /// Diagnostic: the most recent manifold-physics failure, if any.
    ///
    /// When manifold mode is selected and the integrator fails, the controller
    /// FAILS CLOSED: it holds the last valid (c, v) and records the error here
    /// rather than silently substituting flat Euclidean dynamics. A non-None
    /// value means the last manifold step did not advance.
    pub manifold_error: Option<String>,
}

impl Default for OrbitController {
    fn default() -> Self {
        Self {
            theta: 0.0,
            omega: 1.0,
            s: 1.0,
            alpha: 0.0,
            momentum: false,
            c: num_complex::Complex64::new(0.0, 0.0),
            velocity: num_complex::Complex64::new(0.0, 0.0),
            drag: 0.90,
            thrust: 0.0,
            shore_bias: false,
            d_star: 0.5,
            max_step: 0.05,
            level: 0,
            energy: 0.0,
            manifold_physics: false,
            manifold_config: crate::manifold::ManifoldConfig::default(),
            planar_velocity: (0.0, 0.0),
            manifold_drag: 0.1,
            manifold_error: None,
        }
    }
}

impl OrbitController {
    pub fn new(s: f64, alpha: f64, omega: f64) -> Self {
        Self {
            theta: 0.0,
            omega,
            s: s.clamp(0.01, 3.0),
            alpha: alpha.clamp(0.0, 1.0),
            ..Self::default()
        }
    }

    /// Apply model-predicted control signals (LEGACY Controls v1 — RETIRED from destination path, issue #107).
    ///
    /// See `PlayerState::apply_controls` — same retirement. Destination uses `ControlsV2`.
    pub fn apply_controls(&mut self, s: f64, alpha: f64) {
        self.s = s.clamp(0.01, 3.0);
        self.alpha = alpha.clamp(0.0, 1.0);
    }

    /// May's exact `mandelbrotBoundary(s, alpha)` formula.
    ///
    /// Main cardioid: theta = 2*pi*alpha; r = 0.25*(1-cos(theta));
    /// c = r*e^{i*theta/2} scaled by min(s, 1.5).
    pub fn mandelbrot_boundary(&self) -> num_complex::Complex64 {
        let theta = 2.0 * std::f64::consts::PI * self.alpha;
        let r = 0.25 * (1.0 - theta.cos());
        let scale = self.s.min(1.5); // Cap at 1.5 to avoid escaping too far
        num_complex::Complex64::new(
            r * (theta / 2.0).cos() * scale,
            r * (theta / 2.0).sin() * scale,
        )
    }

    /// Advance one frame exactly as the May TS controller did.
    ///
    /// * `dt` – frame time in seconds.
    /// * `band_gates` – per-band residual gates in [0, 1].
    /// * `h` – transient/hit signal in [0, 1]; near 1 opens the Shore wall
    ///   (boundary crossing becomes easy). Only used on refinement paths.
    ///
    /// Returns the new c. Residuals are harmonic epicycles at freq (k+2)
    /// times the wobble phase, amplitude 0.05*gate — identical to the TS.
    ///
    /// With all refinement flags off (the default), this is bit-identical
    /// to the May TS controller. Each flag layers ONE PlayerState idea:
    ///   momentum         -> c is persistent state; boundary point attracts
    ///   shore_bias       -> motion routed through minimap contour biasing
    ///   manifold_physics -> LEGACY ADAPTER to the musically-ignorant manifold
    ///                       kernel (issue #106); transitional, not Controls v2
    pub fn step(
        &mut self,
        dt: f64,
        band_gates: Option<&[f64]>,
        h: f64,
    ) -> num_complex::Complex64 {
        // Update wobble phase (unchanged by refinements).
        self.theta = (self.theta + self.omega * dt) % (2.0 * std::f64::consts::PI);

        // Base position from the model's (s, alpha).
        let base = self.mandelbrot_boundary();

        // Residual modulation: harmonics of the wobble phase.
        let mut res_re = 0.0;
        let mut res_im = 0.0;
        if let Some(gates) = band_gates {
            for (k, &g) in gates.iter().enumerate() {
                let gate = g.clamp(0.0, 1.0);
                let freq = (k as f64 + 2.0) * 1.0;
                let phase = freq * self.theta;
                res_re += gate * 0.05 * phase.cos();
                res_im += gate * 0.05 * phase.sin();
            }
        }
        let target = num_complex::Complex64::new(base.re + res_re, base.im + res_im);

        if !self.momentum && !self.shore_bias && !self.manifold_physics {
            // Baseline path: c IS the target. Bit-identical to May.
            return target;
        }

        if self.manifold_physics {
            // Manifold physics path: LEGACY ADAPTER translating the old
            // (s, alpha, energy) servo into a generalized force covector for
            // the musically-ignorant manifold kernel (issue #106).
            return self.step_manifold(dt, band_gates, h);
        }

        if !self.momentum {
            // Shore bias only: move from current c toward the target,
            // biased along contours. No velocity state.
            let proposed = target - self.c;
            return self.apply_shore_bias(proposed.re, proposed.im, h);
        }

        // Momentum path: pull toward the target is an acceleration.
        let accel_gain = 2.0 * dt;
        let mut a_re = (target.re - self.c.re) * accel_gain;
        let mut a_im = (target.im - self.c.im) * accel_gain;

        // Gravity: the valley. Constant restoring pull toward the origin —
        // the physics default that c falls back into when the music goes
        // quiet and the player stops pushing uphill.
        a_re -= GRAVITY_ACCEL * self.c.re;
        a_im -= GRAVITY_ACCEL * self.c.im;

        // Audio thrust: sustained volume builds inertia. The thrust is
        // TANGENTIAL to the vector from c to the target — perpendicular
        // to the pull — so it never fights the attraction (which would
        // prevent convergence) but keeps c orbiting its target instead
        // of parking on it. Magnitude scales with the runtime's energy
        // signal: loud audio = strong thrust, silence = pure attraction
        // (c settles). This makes sustained volume literally impossible
        // to be stationary, per the domain contract's Momentum concept.
        if self.thrust > 0.0 {
            let dx = target.re - self.c.re;
            let dy = target.im - self.c.im;
            let d = (dx * dx + dy * dy).sqrt();
            if d > 1e-9 {
                // Tangent unit vector (rotate pull direction by 90°).
                let tx = -dy / d;
                let ty = dx / d;
                a_re += self.thrust * tx;
                a_im += self.thrust * ty;
            }
        }

        self.velocity = self.velocity.scale(self.drag)
            + num_complex::Complex64::new(a_re, a_im);
        let v_dt = self.velocity.scale(dt);

        if self.shore_bias {
            self.apply_shore_bias(v_dt.re, v_dt.im, h)
        } else {
            self.c = self.c + v_dt;
            self.c
        }
    }

    /// Route a proposed delta through the minimap contour bias (refinement 2).
    /// Falls back to plain clamped motion when no pyramid is loaded.
    fn apply_shore_bias(
        &mut self,
        du_re: f64,
        du_im: f64,
        h: f64,
    ) -> num_complex::Complex64 {
        let (nr, ni) = crate::minimap::contour_biased_step(
            self.c.re,
            self.c.im,
            du_re,
            du_im,
            h.clamp(0.0, 1.0),
            self.d_star,
            self.max_step,
            self.level,
            self.energy,
        )
        .unwrap_or((self.c.re + du_re, self.c.im + du_im));
        self.c = num_complex::Complex64::new(nr, ni);
        self.c
    }

    /// LEGACY ADAPTER — manifold physics step (issue #106).
    ///
    /// This method is a TRANSITIONAL compatibility seam between the legacy
    /// `(s, alpha, energy)` controller surface and the destination manifold
    /// Physics kernel in `crate::manifold`. It is NOT destination Controls v2
    /// (issue #107) and must not be described or tested as such.
    ///
    /// The manifold kernel itself (`crate::manifold::integrate_step`) is
    /// musically ignorant: it accepts an explicit generalized force covector
    /// `Q_control = (Qx, Qy)` and knows nothing about `s`, `alpha`, audio
    /// Energy, band gates, onset/transient `h`, transition readiness, target
    /// `c`, or Shore target distance. All of that legacy state lives HERE, in
    /// the adapter, which translates the old target servo into a generalized
    /// force covector before calling the kernel.
    ///
    /// Force units: `q_control` is a generalized force COVECTOR (units of
    /// force), NOT an already-integrated impulse. It is NOT multiplied by `dt`
    /// here; the kernel integrates continuous force exactly once (v += a*dt).
    ///
    /// Fail-closed: if the manifold integrator errors, this method does NOT
    /// silently substitute flat Euclidean dynamics. It holds the last valid
    /// (c, v) and records the failure in `self.manifold_error`.
    fn step_manifold(
        &mut self,
        dt: f64,
        band_gates: Option<&[f64]>,
        _h: f64,
    ) -> num_complex::Complex64 {
        // ---- Legacy target synthesis (adapter-only; not manifold authority) ----
        let target = self.mandelbrot_boundary();
        let mut res_re = 0.0;
        let mut res_im = 0.0;
        if let Some(gates) = band_gates {
            for (k, &g) in gates.iter().enumerate() {
                let gate = g.clamp(0.0, 1.0);
                let freq = (k as f64 + 2.0) * 1.0;
                let phase = freq * self.theta;
                res_re += gate * 0.05 * phase.cos();
                res_im += gate * 0.05 * phase.sin();
            }
        }
        let target = num_complex::Complex64::new(target.re + res_re, target.im + res_im);

        // ---- Legacy target servo -> generalized force covector ----
        // Direction: from current c toward the legacy (s, alpha) target.
        let dx = target.re - self.c.re;
        let dy = target.im - self.c.im;
        let dist = (dx * dx + dy * dy).sqrt();

        // Magnitude scales with omega and legacy audio energy. This is a
        // generalized force (units of force), NOT an impulse: no *dt here.
        let force_mag = self.omega.clamp(0.1, 10.0) * self.energy.clamp(0.0, 1.0) * 2.0;

        let q_control = if dist > 1e-9 {
            (dx / dist * force_mag, dy / dist * force_mag)
        } else {
            (0.0, 0.0)
        };

        // ---- Manifold kernel (musically ignorant) ----
        match crate::manifold::integrate_step(
            self.c,
            self.planar_velocity,
            q_control,
            self.manifold_drag,
            dt,
            &self.manifold_config,
        ) {
            Ok((c_new, v_new, _info)) => {
                self.manifold_error = None;
                self.c = c_new;
                self.planar_velocity = v_new;
                self.c
            }
            Err(e) => {
                // FAIL CLOSED: hold the last valid state; do not substitute
                // flat dynamics. Surface the diagnostic for the caller.
                self.manifold_error = Some(e);
                self.c
            }
        }
    }
}
