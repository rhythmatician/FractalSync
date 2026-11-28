//! Throwaway #137 reference experiment in a flat metric.
const STRENGTH: f64 = 16.0;

fn bowl([x, y]: [f64; 2]) -> (f64, [f64; 2]) {
    let r = x.hypot(y);
    if r <= 2.0 {
        return (0.0, [0.0; 2]);
    }
    let q = ((r - 2.0) / 2.0).ln_1p();
    let radial_force = -STRENGTH * q.powi(3) / r;
    (
        STRENGTH * q.powi(4) / 4.0,
        [radial_force * (x / r), radial_force * (y / r)],
    )
}

fn run(name: &str, initial_velocity: [f64; 2], dt: f64) {
    let mut p = [1.9, 0.0];
    let mut v = initial_velocity;
    let energy = |p, v: [f64; 2]| bowl(p).0 + (v[0] * v[0] + v[1] * v[1]) / 2.0;
    let initial = energy(p, v);
    let mut drift: f64 = 0.0;
    let mut peak: f64 = 0.0;
    let mut exit = None;
    let mut returned = None;
    let mut angle: f64 = 0.0;
    let mut angular_momentum_error: f64 = 0.0;
    for i in 1..=(32.0 / dt) as usize {
        let old = p;
        let (_, a) = bowl(p);
        for j in 0..2 {
            v[j] += 0.5 * dt * a[j];
            p[j] += dt * v[j];
        }
        let (_, a) = bowl(p);
        for j in 0..2 {
            v[j] += 0.5 * dt * a[j];
        }
        let r = p[0].hypot(p[1]);
        peak = peak.max(r);
        if returned.is_none() {
            if r > 2.0 {
                exit.get_or_insert(i as f64 * dt);
            } else if exit.is_some() {
                returned = Some(i as f64 * dt);
            }
            if exit.is_some() {
                angle += (old[0] * p[1] - old[1] * p[0]).atan2(old[0] * p[0] + old[1] * p[1]);
            }
        }
        drift = drift.max((energy(p, v) - initial).abs());
        angular_momentum_error = angular_momentum_error
            .max((p[0] * v[1] - p[1] * v[0] - 1.9 * initial_velocity[1]).abs());
    }
    println!(
        "{name} dt={dt:.5} peak={peak:.6} exit={exit:?} return={returned:?} exterior_angle={angle:.6} energy_error={drift:.3e} momentum_error={angular_momentum_error:.3e}"
    );
}

fn main() {
    for dt in [1.0 / 256.0, 1.0 / 512.0] {
        run("radial", [0.5, 0.0], dt);
        run("tangential", [0.3, 0.5], dt);
    }
}
