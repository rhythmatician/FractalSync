//! Offline conservative-energy diagnostic through canonical Rust Physics.
use num_complex::Complex64;
use runtime_core::geometry_provider::query_geometry;
use runtime_core::manifold::{integrate_step, total_energy, ManifoldConfig};

const ENERGY_LIMIT: f64 = 0.05;

fn main() -> Result<(), String> {
    let mut arguments = std::env::args().skip(1);
    let mut case_filter = None;
    let mut check = false;
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--check" => check = true,
            "--case" => case_filter = Some(arguments.next().ok_or("--case requires a name")?),
            _ => return Err(format!("unknown argument: {argument}")),
        }
    }
    let config = ManifoldConfig::default();
    let mut rows = Vec::new();
    let mut failures = 0;
    for (case, start, velocity, horizon, base_dt) in [
        (
            "low_curvature",
            Complex64::new(-0.5, 0.1),
            (0.01, 0.0),
            0.3_f64,
            0.002_f64,
        ),
        (
            "high_curvature",
            Complex64::new(0.4, 0.05),
            (0.0, 0.02),
            0.3,
            0.002,
        ),
        (
            "near_shore",
            Complex64::new(0.3, 0.05),
            (0.0, 0.02),
            0.2,
            0.002,
        ),
        (
            "inward",
            Complex64::new(0.4, 0.05),
            (-0.03, 0.0),
            0.5,
            0.005,
        ),
    ] {
        if case_filter.as_ref().is_some_and(|name| name != case) {
            continue;
        }
        for dt in [base_dt, base_dt * 0.5, base_dt * 0.25] {
            let mut c = start;
            let mut v = velocity;
            let initial = total_energy(v, c, &config)?;
            let mut previous = initial;
            let mut excursion = 0.0_f64;
            let mut max_step = 0.0_f64;
            let mut peak_transition = None;
            let mut completed = 0;
            let steps = (horizon / dt).round() as usize;
            let mut error = None;
            for _ in 0..steps {
                match integrate_step(c, v, (0.0, 0.0), 0.0, dt, &config) {
                    Ok((next_c, next_v, energy)) => {
                        excursion = excursion.max((energy.total - initial).abs());
                        let change = (energy.total - previous).abs();
                        if change > max_step {
                            max_step = change;
                            peak_transition = Some(serde_json::json!({
                                "before": {"c":[c.re,c.im],"v":[v.0,v.1],"jet":query_geometry(c, config.epsilon)?},
                                "after": {"c":[next_c.re,next_c.im],"v":[next_v.0,next_v.1],"jet":query_geometry(next_c, config.epsilon)?}
                            }));
                        }
                        c = next_c;
                        v = next_v;
                        previous = energy.total;
                        completed += 1;
                    }
                    Err(reason) => {
                        error = Some(reason);
                        break;
                    }
                }
            }
            let passed = error.is_none() && excursion < ENERGY_LIMIT && max_step < ENERGY_LIMIT;
            failures += usize::from(!passed);
            eprintln!("{case}, dt={dt}: {completed}/{steps} steps, excursion={excursion}, largest change={max_step}, passed={passed}");
            rows.push(serde_json::json!({"case":case,"dt":dt,"steps":steps,"completed":completed,"max_excursion":excursion,"max_step_change":max_step,"peak_transition":peak_transition,"passed":passed,"error":error}));
        }
    }
    if rows.is_empty() {
        return Err("no matching energy case".to_string());
    }
    println!("{}", serde_json::to_string_pretty(&serde_json::json!({"diagnostic_only":true,"limit":ENERGY_LIMIT,"provider_version":runtime_core::geometry_provider::GEOMETRY_PROVIDER_VERSION,"controller_version":runtime_core::controller::CONTROLLER_VERSION,"rows":rows})).map_err(|error| error.to_string())?);
    if check && failures != 0 {
        return Err(format!("{failures} energy cases failed"));
    }
    Ok(())
}
