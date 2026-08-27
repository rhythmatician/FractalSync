//! Golden-vector generator: the mechanical parity contract.
//!
//! Run from the repo root:
//!
//! ```text
//! cargo run --release -p runtime_core --bin generate_golden_vectors
//! ```
//!
//! Writes `shared/golden_vectors.json` — a deterministic record of the
//! canonical Rust math (controller, geometry, proxies). Every mirror of that
//! math (PyTorch trainer, TypeScript mock, any future port) must reproduce
//! these vectors bit-for-bit within tolerance, enforced by parity tests in
//! each language. If you change the canonical math, regenerate this file and
//! update every mirror in the same commit — otherwise the tests fail.
//!
//! Authority: docs/adr/0001-rust-first-parity.md

use num_complex::Complex64;
use runtime_core::controller::{OrbitState, PlayerState, ResidualParams};
use runtime_core::geometry::lobe_point_at_angle;
use runtime_core::proxies::mandelbrot_cardioid_proximity;
use serde::Serialize;

#[derive(Serialize)]
struct OrbitCase {
    lobe: u32,
    sub_lobe: u32,
    theta: f64,
    omega: f64,
    s: f64,
    alpha: f64,
    seed: u64,
    band_gates: Vec<f64>,
    /// c = synthesize(state, params, gates)
    c_re: f64,
    c_im: f64,
}

#[derive(Serialize)]
struct PlayerStepCase {
    s0: f64,
    alpha0: f64,
    dt: f64,
    h: f64,
    d_star: f64,
    max_step: f64,
    level: usize,
    /// Per-step controls applied before each step.
    controls: Vec<[f64; 3]>, // [s, alpha, omega_scale]
    band_gates: Option<Vec<f64>>,
    /// c after N steps.
    c_re: f64,
    c_im: f64,
}

#[derive(Serialize)]
struct GoldenVectors {
    generator: String,
    authority: String,
    carrier_cases: Vec<CarrierCase>,
    orbit_cases: Vec<OrbitCase>,
    player_step_cases: Vec<PlayerStepCase>,
    proximity_cases: Vec<ProximityCase>,
}

#[derive(Serialize)]
struct CarrierCase {
    lobe: u32,
    sub_lobe: u32,
    theta: f64,
    s: f64,
    c_re: f64,
    c_im: f64,
}

#[derive(Serialize)]
struct ProximityCase {
    c_re: f64,
    c_im: f64,
    proximity: f64,
}

fn main() {
    let mut g = GoldenVectors {
        generator: format!("runtime_core {}", env!("CARGO_PKG_VERSION")),
        authority: "docs/adr/0001-rust-first-parity.md".to_string(),
        carrier_cases: Vec::new(),
        orbit_cases: Vec::new(),
        player_step_cases: Vec::new(),
        proximity_cases: Vec::new(),
    };

    // ---- Carrier: lobe_point_at_angle over a deterministic grid ----
    for i in 0..8 {
        for j in 0..8 {
            let theta = 2.0 * std::f64::consts::PI * (i as f64) / 8.0;
            let s = 0.5 + 1.5 * (j as f64) / 7.0;
            let c = lobe_point_at_angle(1, 0, theta, s);
            g.carrier_cases.push(CarrierCase {
                lobe: 1,
                sub_lobe: 0,
                theta,
                s,
                c_re: c.re,
                c_im: c.im,
            });
        }
    }

    // ---- Full synthesis: OrbitState with residuals + gates ----
    let rp = ResidualParams {
        k_residuals: 6,
        residual_cap: 0.5,
        radius_scale: 1.0,
    };
    let mut idx = 0u64;
    for &theta_i in &[0.0f64, 0.7, 1.9, 3.1, 4.4, 5.5] {
        for &s in &[0.6f64, 1.02, 1.7, 2.4] {
            for &alpha in &[0.0f64, 0.35, 0.95] {
                let seed = 1337 + idx;
                idx += 1;
                let state = OrbitState::new_with_seed(1, 0, theta_i, 1.0, s, alpha, 6, 2.0, seed);
                // Vary gates deterministically per case.
                let gates: Vec<f64> = (0..6)
                    .map(|k| ((seed as f64) * 0.37 + k as f64 * 0.61).fract().abs())
                    .collect();
                let c = runtime_core::controller::synthesize(&state, rp, Some(&gates));
                g.orbit_cases.push(OrbitCase {
                    lobe: 1,
                    sub_lobe: 0,
                    theta: theta_i,
                    omega: 1.0,
                    s,
                    alpha,
                    seed,
                    band_gates: gates,
                    c_re: c.re,
                    c_im: c.im,
                });
            }
        }
    }

    // ---- PlayerState momentum integrator: multi-step trajectories ----
    // Case A: saturated constant controls (the frozen-c scenario) — must
    // settle at an equilibrium offset, NOT park exactly at start.
    // Case B: slowly varying controls (real model output) — must wander.
    // Case C: no pyramid fallback path is identical to A/B here because the
    // golden generation runs without a pyramid loaded.
    let scenarios: Vec<(&str, fn(f64) -> [f64; 3])> = vec![
        (
            "saturated_constant",
            |_i| [2.69, 0.951, 4.008],
        ),
        (
            "varying",
            |i| {
                [
                    2.7 + 0.03 * (i * 0.05).sin(),
                    (0.95 + 0.002 * (i * 0.03).cos()).clamp(0.0, 1.0),
                    4.0,
                ]
            },
        ),
    ];
    for (name, ctrl) in scenarios {
        let mut p = PlayerState::new(1, 0, 2.7, 0.95);
        p.d_star = 0.5;
        p.max_step = 0.05;
        p.level = 0;
        let gates = vec![0.95f64; 6];
        let mut controls = Vec::with_capacity(120);
        for i in 0..120u32 {
            let [s, a, w] = ctrl(i as f64);
            p.apply_controls(s, a, w);
            controls.push([s, a, w]);
            p.step(1.0 / 60.0, 0.0, Some(&gates)); // advance with gates applied
        }
        let final_c = p.c;
        g.player_step_cases.push(PlayerStepCase {
            s0: 2.7,
            alpha0: 0.95,
            dt: 1.0 / 60.0,
            h: 0.0,
            d_star: 0.5,
            max_step: 0.05,
            level: 0,
            controls,
            band_gates: Some(vec![0.95; 6]),
            c_re: final_c.re,
            c_im: final_c.im,
        });
        let _ = name; // names are implicit by order; keep JSON lean
    }

    // ---- Cardioid proximity ----
    for i in 0..16 {
        let re = -1.5 + 2.5 * (i as f64) / 15.0;
        let im = -0.75 + 1.5 * ((i * 7) % 16) as f64 / 15.0;
        let prox = mandelbrot_cardioid_proximity(Complex64::new(re, im));
        g.proximity_cases.push(ProximityCase {
            c_re: re,
            c_im: im,
            proximity: prox,
        });
    }

    let json = serde_json::to_string_pretty(&g).expect("serialize golden vectors");
    std::fs::create_dir_all("../shared").expect("create shared dir");
    std::fs::write("../shared/golden_vectors.json", json).expect("write golden vectors");
    println!(
        "Wrote ../shared/golden_vectors.json: {} carrier, {} orbit, {} player, {} proximity cases",
        g.carrier_cases.len(),
        g.orbit_cases.len(),
        g.player_step_cases.len(),
        g.proximity_cases.len()
    );
}
