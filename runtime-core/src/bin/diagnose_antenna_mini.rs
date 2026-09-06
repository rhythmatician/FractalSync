//! Diagnostic harness for the tour_antenna_mini numerical-instability event
//! (issue #82 follow-up / shore-crossing instability investigation).
//!
//! DIAGNOSTIC ONLY — changes nothing in production Physics. This binary
//! replays the exact `tour_antenna_mini` Controls v2 sequence through the
//! authoritative `controls::integrate_motion_controls` seam (the same Rust
//! functions the browser wasm and the PyO3 trainer call) and dumps a
//! per-frame decomposition of the equations of motion:
//!
//!   a_total = -Gamma(v,v) + G^-1 (Q_potential + Q_drive + Q_drag)
//!
//! plus Hessian/Christoffel FD-convergence sweeps, substep experiments,
//! eikonal checks, and energy accounting at the ignition frames.
//!
//! Run: cargo run --release -p runtime_core --bin diagnose_antenna_mini
//! (from runtime-core/ so relative paths resolve).

use num_complex::Complex64;
use runtime_core::controls::MotionControls;
use runtime_core::manifold::{
    apply_generalized_force, derivative_step, drag_force, geodesic_acceleration,
    kinetic_energy, mandelbrot_scale,
    potential_energy, potential_force, scale_gradient, scale_hessian, signed_distance,
    wall_force, wall_potential, ManifoldConfig,
};

const DT: f64 = 1024.0 / 48000.0;

/// The tour_antenna_mini action script (mirrors
/// frontend/src/lib/shoreCrossingVariants.ts — tour west shore x150
/// (t0.4 g1.0) then brake-settle east x400 (b0.5)).
const ACTIONS: [([f64; 2], f64, f64, f64, usize); 2] = [
    ([-1.0, -0.06], 0.4, 0.0, 1.0, 150), // tour west shore
    ([1.0, 0.06], 0.0, 0.5, 1.0, 400),   // brake-settle east
];

#[derive(Clone, Copy)]
struct Frame {
    tick: usize,
    c: Complex64,
    v: (f64, f64),
    d: f64,
    rho: f64,
    sigma: f64,
    sigma_dot: f64,
    grad: (f64, f64),
    hess: [[f64; 2]; 2],
    gamma_vv: (f64, f64),
    q_pot: (f64, f64),
    q_drive: (f64, f64),
    q_drag: (f64, f64),
    q_wall: (f64, f64),
    a_force: (f64, f64),
    a_total: (f64, f64),
    k: f64,
    u_sigma: f64,
    u_wall: f64,
    e: f64,
    rejected: bool,
}

fn evaluate(c: Complex64, v: (f64, f64), cfg: &ManifoldConfig) -> Frame {
    let d = signed_distance(c).unwrap_or(f64::NAN);
    let rho = (d * d + cfg.epsilon * cfg.epsilon).sqrt();
    let sigma = mandelbrot_scale(c, cfg).unwrap_or(f64::NAN);
    let grad = scale_gradient(c, cfg).unwrap_or((f64::NAN, f64::NAN));
    let hess = scale_hessian(c, cfg).unwrap_or([[f64::NAN; 2]; 2]);
    let gamma_vv = geodesic_acceleration(v, c, cfg).unwrap_or((f64::NAN, f64::NAN));
    let q_pot = potential_force(c, cfg).unwrap_or((f64::NAN, f64::NAN));
    let q_wall = wall_force(c, cfg).unwrap_or((f64::NAN, f64::NAN));
    let k = kinetic_energy(v, c, cfg).unwrap_or(f64::NAN);
    let u_sigma = potential_energy(c, cfg).unwrap_or(f64::NAN);
    let u_wall = wall_potential(c, cfg).unwrap_or(f64::NAN);
    Frame {
        tick: 0,
        c,
        v,
        d,
        rho,
        sigma,
        sigma_dot: grad.0 * v.0 + grad.1 * v.1,
        grad,
        hess,
        gamma_vv,
        q_pot,
        q_drive: (0.0, 0.0),
        q_drag: (0.0, 0.0),
        q_wall,
        a_force: (0.0, 0.0),
        a_total: (0.0, 0.0),
        k,
        u_sigma,
        u_wall,
        e: k + u_sigma + u_wall,
        rejected: false,
    }
}

fn hess_norm(h: &[[f64; 2]; 2]) -> f64 {
    // Frobenius norm
    (h[0][0] * h[0][0] + h[0][1] * h[0][1] + h[1][0] * h[1][0] + h[1][1] * h[1][1]).sqrt()
}

fn gamma_norm(g: (f64, f64)) -> f64 {
    (g.0 * g.0 + g.1 * g.1).sqrt()
}

fn fmt_frame(f: &Frame) -> String {
    format!(
        "tick={} c=({:.6},{:.6}) v=({:.4},{:.4}) |v|={:.4} D={:.3e} rho={:.3e} sigma={:.4} sigma_dot={:.3} \
grad=({:.3},{:.3}) |grad|={:.3} Hxx={:.4e} Hxy={:.4e} Hyy={:.4e} |H|={:.4e} \
Gamma(v,v)=({:.4},{:.4}) |Gvv|={:.4} Qpot=({:.3},{:.3}) Qdrive=({:.3},{:.3}) Qdrag=({:.3},{:.3}) Qwall=({:.3},{:.3}) \
a_force=({:.4},{:.4}) a_total=({:.4},{:.4}) |a|={:.4} K={:.5} Usig={:.4} Uwall={:.4} E={:.4}{}",
        f.tick,
        f.c.re, f.c.im,
        f.v.0, f.v.1, (f.v.0 * f.v.0 + f.v.1 * f.v.1).sqrt(),
        f.d, f.rho, f.sigma, f.sigma_dot,
        f.grad.0, f.grad.1, (f.grad.0 * f.grad.0 + f.grad.1 * f.grad.1).sqrt(),
        f.hess[0][0], f.hess[0][1], f.hess[1][1], hess_norm(&f.hess),
        f.gamma_vv.0, f.gamma_vv.1, gamma_norm(f.gamma_vv),
        f.q_pot.0, f.q_pot.1, f.q_drive.0, f.q_drive.1, f.q_drag.0, f.q_drag.1, f.q_wall.0, f.q_wall.1,
        f.a_force.0, f.a_force.1, f.a_total.0, f.a_total.1,
        (f.a_total.0 * f.a_total.0 + f.a_total.1 * f.a_total.1).sqrt(),
        f.k, f.u_sigma, f.u_wall, f.e,
        if f.rejected { " REJECTED" } else { "" }
    )
}

fn main() {
    let cfg = ManifoldConfig::default();
    let h_prod = derivative_step();
    println!("=== tour_antenna_mini native replay (authoritative Rust seam) ===");
    println!("dt = {:.9}  derivative_step h = {:.6e}", DT, h_prod);
    println!("config: {:?}", cfg);
    println!();

    // ---- Phase 1: full replay, find ignition frame ----
    let mut c = Complex64::new(-1.7549, 0.0);
    let mut v = (0.0f64, 0.0f64);
    let mut frames: Vec<Frame> = Vec::new();
    let mut tick = 0usize;
    let mut first_reject: Option<usize> = None;

    for (ai, (dir, throttle, brake, grip, count)) in ACTIONS.iter().enumerate() {
        for _ in 0..*count {
            let motion = MotionControls {
                direction: *dir,
                throttle: *throttle,
                brake: *brake,
                grip: *grip,
                impulse: 0.0,
            };
            let mut f = evaluate(c, v, &cfg);
            f.tick = tick;
            // Recompute the exact covectors the kernel will use this step.
            let clamped = motion.clamped();
            let q_drive = clamped
                .drive_covector(c, &cfg)
                .unwrap_or((0.0, 0.0));
            let beta = clamped.friction_beta();
            let q_drag = drag_force(v, c, beta, &cfg).unwrap_or((0.0, 0.0));
            f.q_drive = q_drive;
            f.q_drag = q_drag;
            let q_total = (
                f.q_pot.0 + f.q_wall.0 + q_drive.0 + q_drag.0,
                f.q_pot.1 + f.q_wall.1 + q_drive.1 + q_drag.1,
            );
            f.a_force = apply_generalized_force(q_total, c, &cfg).unwrap_or((f64::NAN, f64::NAN));
            f.a_total = (-f.gamma_vv.0 + f.a_force.0, -f.gamma_vv.1 + f.a_force.1);
            frames.push(f);

            match runtime_core::controls::integrate_motion_controls(c, v, &motion, DT, &cfg) {
                Ok((cn, vn, _info)) => {
                    c = cn;
                    v = vn;
                }
                Err(e) => {
                    if first_reject.is_none() {
                        first_reject = Some(tick);
                        println!("HARD-GUARD REJECTION at tick {} (action {}): {}", tick, ai, e);
                        println!("  state at rejection: c=({:.9},{:.9}) v=({:.9},{:.9})", c.re, c.im, v.0, v.1);
                    }
                    // Controller fails closed: hold last valid state.
                    break;
                }
            }
            tick += 1;
        }
        if first_reject.is_some() {
            break;
        }
    }

    // Identify ignition: last ordinary frame, first anomalous-accel frame,
    // largest-accel frame.
    let anoms: Vec<usize> = frames
        .iter()
        .filter(|f| (f.a_total.0 * f.a_total.0 + f.a_total.1 * f.a_total.1).sqrt() > 10.0)
        .map(|f| f.tick)
        .collect();
    let first_anom = anoms.first().copied();
    let max_a = frames
        .iter()
        .enumerate()
        .max_by(|a, b| {
            let aa = a.1.a_total.0 * a.1.a_total.0 + a.1.a_total.1 * a.1.a_total.1;
            let bb = b.1.a_total.0 * b.1.a_total.0 + b.1.a_total.1 * b.1.a_total.1;
            aa.partial_cmp(&bb).unwrap()
        })
        .map(|(i, _)| i);

    println!("--- ignition summary ---");
    println!("total frames recorded: {}", frames.len());
    match first_anom {
        Some(t) => println!("first anomalous |a_total| frame: tick {}", t),
        None => println!("no anomalous acceleration frame found"),
    }
    if let Some(i) = max_a {
        println!("largest |a_total| frame: tick {} |a|={:.4}", frames[i].tick,
            (frames[i].a_total.0 * frames[i].a_total.0 + frames[i].a_total.1 * frames[i].a_total.1).sqrt());
    }
    println!("first hard-guard rejection: {:?}", first_reject);
    println!();

    // Full per-frame trace for parity comparison against the wasm replay.
    // Machine-parseable full-precision line so cross-runtime comparison is
    // not limited by print rounding.
    println!("--- per-frame trace (every frame, full precision) ---");
    for f in &frames {
        println!(
            "P tick={} c_re={:.17e} c_im={:.17e} v_re={:.17e} v_im={:.17e}",
            f.tick, f.c.re, f.c.im, f.v.0, f.v.1
        );
    }
    println!();

    // ---- Phase 2: dump critical window (±5 frames around first anomaly) ----
    if let Some(t0) = first_anom {
        let lo = t0.saturating_sub(5);
        let hi = (t0 + 5).min(frames.len() - 1);
        println!("--- critical window ticks {}..{} ---", lo, hi);
        for f in &frames[lo..=hi] {
            println!("{}", fmt_frame(f));
        }
        println!();

        // ---- Phase 4: FD convergence at last-ordinary and first-anom frames ----
        println!("--- FD convergence sweep (h multiples of production h={:.6e}) ---", h_prod);
        for t in [t0.saturating_sub(1), t0] {
            let f = &frames[t];
            println!("@ tick {} c=({:.9},{:.9})", f.tick, f.c.re, f.c.im);
            for mult in [0.25, 0.5, 1.0, 2.0, 4.0, 8.0] {
                let h = h_prod * mult;
                let g = fd_grad(f.c, h, &cfg);
                let hs = fd_hess(f.c, h, &cfg);
                let gv = geodesic_acceleration(f.v, f.c, &cfg).unwrap_or((f64::NAN, f64::NAN));
                println!(
                    "  h×{:<5}: grad=({:>12.4},{:>12.4}) |grad|={:>12.4}  Hxx={:>12.4e} Hxy={:>12.4e} Hyy={:>12.4e} |H|={:>12.4e}  Gamma(v,v)=({:>10.4},{:>10.4})",
                    mult, g.0, g.1, (g.0*g.0+g.1*g.1).sqrt(),
                    hs[0][0], hs[0][1], hs[1][1], hess_norm(&hs), gv.0, gv.1
                );
            }
        }
        println!();

        // ---- Phase 5: substep experiment from the state immediately before ignition ----
        let pre = &frames[t0.saturating_sub(1)];
        println!("--- substep experiment from tick {} (pre-ignition state) ---", pre.tick);
        println!("start c=({:.9},{:.9}) v=({:.9},{:.9})", pre.c.re, pre.c.im, pre.v.0, pre.v.1);
        let motion = MotionControls {
            direction: ACTIONS.iter().find(|_| pre.tick < 150).map(|a| a.0).unwrap_or([1.0, 0.06]),
            throttle: if pre.tick < 150 { 0.4 } else { 0.0 },
            brake: if pre.tick < 150 { 0.0 } else { 0.5 },
            grip: 1.0,
            impulse: 0.0,
        };
        for (label, nsub) in [("dt", 1usize), ("dt/2", 2), ("dt/4", 4), ("dt/8", 8), ("dt/16", 16)] {
            let sub = DT / nsub as f64;
            let mut sc = pre.c;
            let mut sv = pre.v;
            let mut max_a_mid = 0.0f64;
            let mut exploded = false;
            for _ in 0..nsub {
                match runtime_core::controls::integrate_motion_controls(sc, sv, &motion, sub, &cfg) {
                    Ok((cn, vn, _)) => {
                        // intermediate acceleration at the pre-step state
                        let g = geodesic_acceleration(sv, sc, &cfg).unwrap_or((0.0, 0.0));
                let qp = potential_force(sc, &cfg).unwrap_or((0.0, 0.0));
                let qw = wall_force(sc, &cfg).unwrap_or((0.0, 0.0));
                        let qd = drag_force(sv, sc, motion.clamped().friction_beta(), &cfg).unwrap_or((0.0, 0.0));
                        let qdr = motion.clamped().drive_covector(sc, &cfg).unwrap_or((0.0, 0.0));
                let af = apply_generalized_force(
                    (qp.0 + qw.0 + qdr.0 + qd.0, qp.1 + qw.1 + qdr.1 + qd.1),
                    sc,
                    &cfg,
                )
                .unwrap_or((0.0, 0.0));
                        let am = ((-g.0 + af.0).powi(2) + (-g.1 + af.1).powi(2)).sqrt();
                        if am > max_a_mid { max_a_mid = am; }
                        sc = cn;
                        sv = vn;
                    }
                    Err(_) => {
                        exploded = true;
                        break;
                    }
                }
            }
            let k = kinetic_energy(sv, sc, &cfg).unwrap_or(f64::NAN);
            let u = potential_energy(sc, &cfg).unwrap_or(f64::NAN) + wall_potential(sc, &cfg).unwrap_or(f64::NAN);
            println!(
                "  {:>6}: c=({:.9},{:.9}) v=({:.6},{:.6}) |v|={:.6} K={:.6} U={:.6} E={:.6} max|a_mid|={:.4}{}",
                label, sc.re, sc.im, sv.0, sv.1, (sv.0*sv.0+sv.1*sv.1).sqrt(), k, u, k + u, max_a_mid,
                if exploded { "  [REJECTED mid-substep]" } else { "" }
            );
        }
        println!();

        // ---- Phase 7: eikonal check at critical frames ----
        println!("--- eikonal check |grad D| (production h and h/2, 2h) ---");
        for t in [t0.saturating_sub(1), t0, t0 + 1].into_iter().filter(|t| *t < frames.len()) {
            let f = &frames[t];
            for mult in [0.5, 1.0, 2.0] {
                let h = h_prod * mult;
                let gd = fd_grad_d(f.c, h);
                println!("  tick {} h×{:<4}: |grad D| = {:.6}  (|grad D| - 1 = {:+.6})", f.tick, mult, gd, gd - 1.0);
            }
        }
        println!();

        // ---- Phase 8: energy ledger across the window ----
        println!("--- energy ledger (window) ---");
        println!("tick     K          U_sigma    U_wall     E          dE       dK        dU_sig    dU_wall");
        for i in lo..=hi {
            let f = &frames[i];
            let prev = if i > 0 { &frames[i - 1] } else { f };
            println!(
                "{:>4}  {:>10.5} {:>10.4} {:>10.4} {:>10.5} {:>+9.5} {:>+9.5} {:>+9.5} {:>+9.5}",
                f.tick, f.k, f.u_sigma, f.u_wall, f.e,
                f.e - prev.e, f.k - prev.k, f.u_sigma - prev.u_sigma, f.u_wall - prev.u_wall
            );
        }
    }

    // ---- Phase 3: wall-force wiring evidence (static, from source) ----
    println!();
    println!("--- wall-force wiring (traced from runtime-core/src/manifold.rs integrate_step) ---");
    println!("Q_total = Q_sigma + Q_wall + Q_control + Q_drag");
    println!("wall_force() participates in both integration and energy accounting.");
}

/// Central-difference gradient of sigma at arbitrary step h (diagnostic only).
fn fd_grad(c: Complex64, h: f64, cfg: &ManifoldConfig) -> (f64, f64) {
    let sp = mandelbrot_scale(Complex64::new(c.re + h, c.im), cfg).unwrap_or(f64::NAN);
    let smx = mandelbrot_scale(Complex64::new(c.re - h, c.im), cfg).unwrap_or(f64::NAN);
    let spy = mandelbrot_scale(Complex64::new(c.re, c.im + h), cfg).unwrap_or(f64::NAN);
    let smy = mandelbrot_scale(Complex64::new(c.re, c.im - h), cfg).unwrap_or(f64::NAN);
    ((sp - smx) / (2.0 * h), (spy - smy) / (2.0 * h))
}

/// Central-difference Hessian of sigma at arbitrary step h (diagnostic only).
fn fd_hess(c: Complex64, h: f64, cfg: &ManifoldConfig) -> [[f64; 2]; 2] {
    let (gpx, _) = grad_at(Complex64::new(c.re + h, c.im), h, cfg);
    let (gmx, _) = grad_at(Complex64::new(c.re - h, c.im), h, cfg);
    let (gpy, gpyy) = grad_at(Complex64::new(c.re, c.im + h), h, cfg);
    let (gmy, gmyy) = grad_at(Complex64::new(c.re, c.im - h), h, cfg);
    [
        [(gpx - gmx) / (2.0 * h), (gpy - gmy) / (2.0 * h)],
        [(gpy - gmy) / (2.0 * h), (gpyy - gmyy) / (2.0 * h)],
    ]
}

fn grad_at(c: Complex64, h: f64, cfg: &ManifoldConfig) -> (f64, f64) {
    let sp = mandelbrot_scale(Complex64::new(c.re + h, c.im), cfg).unwrap_or(f64::NAN);
    let smx = mandelbrot_scale(Complex64::new(c.re - h, c.im), cfg).unwrap_or(f64::NAN);
    let spy = mandelbrot_scale(Complex64::new(c.re, c.im + h), cfg).unwrap_or(f64::NAN);
    let smy = mandelbrot_scale(Complex64::new(c.re, c.im - h), cfg).unwrap_or(f64::NAN);
    ((sp - smx) / (2.0 * h), (spy - smy) / (2.0 * h))
}

/// |grad D| via central differences of the signed distance at step h.
fn fd_grad_d(c: Complex64, h: f64) -> f64 {
    let dp = signed_distance(Complex64::new(c.re + h, c.im)).unwrap_or(f64::NAN);
    let dm = signed_distance(Complex64::new(c.re - h, c.im)).unwrap_or(f64::NAN);
    let dy = signed_distance(Complex64::new(c.re, c.im + h)).unwrap_or(f64::NAN);
    let dn = signed_distance(Complex64::new(c.re, c.im - h)).unwrap_or(f64::NAN);
    let gx = (dp - dm) / (2.0 * h);
    let gy = (dy - dn) / (2.0 * h);
    (gx * gx + gy * gy).sqrt()
}
