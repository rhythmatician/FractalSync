//! Preparation and timing are deliberately outside the Physics evaluation path.
use num_complex::Complex64;
use runtime_core::{manifold, operational_field::prepare_raster_baseline};
use serde_json::json;
use std::{hint::black_box, sync::Arc, time::Instant};

fn stats(mut times: Vec<f64>) -> serde_json::Value {
    let first = times[0];
    times.sort_by(f64::total_cmp);
    json!({"samples":times.len(),"first_us":first,"p50_us":times[times.len()/2],"p99_us":times[(times.len()*99/100).min(times.len()-1)],"max_us":times.last().unwrap()})
}

fn main() -> Result<(), String> {
    let start = Instant::now();
    let field = Arc::new(prepare_raster_baseline()?);
    let preparation = start.elapsed().as_secs_f64();
    let first = Instant::now();
    black_box(
        field
            .query([0.4, 0.05])
            .map_err(|_| "missing prepared point")?,
    );
    let first_us = first.elapsed().as_secs_f64() * 1e6;
    let mut times = Vec::new();
    let mut misses = Vec::new();
    let mut max_scalar_difference = 0.0f64;
    let mut max_gradient_absolute_error = 0.0f64;
    let mut max_hessian_relative_error = 0.0f64;
    for i in 0..20000 {
        // Deterministic distinct points, including grid joins and both corridors.
        let p = [
            0.2 + (i % 997) as f64 / 997.0 * 0.4,
            0.025 + (i % 293) as f64 / 293.0 * 0.15,
        ];
        let now = Instant::now();
        let j = black_box(field.query(black_box(p))).map_err(|_| "missing prepared point")?;
        times.push(now.elapsed().as_secs_f64() * 1e6);
        let now = Instant::now();
        black_box(field.query(black_box([2.0, p[1]]))).unwrap_err();
        misses.push(now.elapsed().as_secs_f64() * 1e6);
        if i < 1000 {
            let reference =
                runtime_core::distance_field::sample_signed_distance_field(&[Complex64::new(
                    p[0], p[1],
                )])?[0] as f64;
            max_scalar_difference = max_scalar_difference.max((j.value - reference).abs());
            for delta in [1e-6, 1e-7] {
                for a in 0..2 {
                    let mut plus = p;
                    let mut minus = p;
                    plus[a] += delta;
                    minus[a] -= delta;
                    let plus = field.query(plus).unwrap();
                    let minus = field.query(minus).unwrap();
                    let fd = (plus.value - minus.value) / (2.0 * delta);
                    max_gradient_absolute_error = max_gradient_absolute_error
                        .max((fd - j.gradient[a]).abs());
                    for b in 0..2 {
                        let fd = (plus.gradient[b] - minus.gradient[b]) / (2.0 * delta);
                        max_hessian_relative_error = max_hessian_relative_error
                            .max((fd - j.hessian[b][a]).abs() / j.hessian[b][a].abs().max(1.0));
                    }
                }
            }
        }
    }
    // Eviction occurs outside the timer. This is a CPU-cache stress measurement,
    // not a promise that every hardware cache is cold on every host.
    let mut eviction = vec![0u64; 8 * 1024 * 1024];
    let mut cold = Vec::new();
    for i in 0..1000 {
        for item in eviction.iter_mut().step_by(8) {
            *item = item.wrapping_add(i);
        }
        black_box(&eviction);
        let p = [0.25 + i as f64 * 0.0003, 0.05];
        let now = Instant::now();
        black_box(field.query(black_box(p))).unwrap();
        cold.push(now.elapsed().as_secs_f64() * 1e6);
    }
    runtime_core::operational_field::activate(Some(field.clone()));
    let config = manifold::ManifoldConfig::default();
    let mut steps = Vec::new();
    let mut failures = 0;
    for i in 0..1000 {
        let now = Instant::now();
        let result = black_box(manifold::integrate_step(
            Complex64::new(0.4 + i as f64 * 1e-6, 0.05),
            (0.001, 0.0),
            (0.0, 0.0),
            0.0,
            0.001,
            &config,
        ));
        steps.push(now.elapsed().as_secs_f64() * 1e6);
        if result.is_err() {
            failures += 1;
        }
    }
    let mut energy_rollouts = Vec::new();
    for (name, x, count) in [("high_curvature", 0.4, 150), ("near_shore", 0.3, 100)] {
        let mut c = Complex64::new(x, 0.05);
        let mut v = (0.0, 0.02);
        let initial = manifold::total_energy(v, c, &config)?;
        let mut excursion = 0.0f64;
        for _ in 0..count {
            let (next_c, next_v, _) =
                manifold::integrate_step(c, v, (0.0, 0.0), 0.0, 0.002, &config)?;
            c = next_c;
            v = next_v;
            excursion = excursion.max((manifold::total_energy(v, c, &config)? - initial).abs());
        }
        let drift = (manifold::total_energy(v, c, &config)? - initial).abs();
        energy_rollouts.push(json!({"name":name,"steps":count,"dt":0.002,"absolute_drift":drift,"maximum_excursion":excursion,"unchanged_limit":0.05,"passed":drift<0.05 && excursion<0.05}));
    }
    runtime_core::operational_field::activate(None);
    println!("{}",serde_json::to_string_pretty(&json!({
        "profile":"experimental-operational-field/1", "identity":format!("{:016x}",field.identity()),
        "reference":"existing 1024 raster; measured comparison only; true-Shore fidelity and cut information unknown",
        "coverage":{"origin":[-0.1,-0.2],"shape":[1605,1005],"spacing":field.spacing(),"epsilon_min":1e-4,"lower_halo_nodes":2,"upper_halo_nodes":3},
        "coefficient_bytes":field.coefficient_bytes(),"preparation_seconds":preparation,
        "first_prepared_query_us":first_us,"distinct_queries":stats(times),"unavailable_queries":stats(misses),
        "cache_stress_queries":stats(cold),"cache_stress_eviction_bytes":eviction.len()*8,
        "shipping_euler_step_on_operational_profile":stats(steps),"step_failures":failures,
        "energy_rollouts":energy_rollouts,
        "measured_raster_max_scalar_difference":max_scalar_difference,
        "neighbor_gradient_max_absolute_error":max_gradient_absolute_error,
        "neighbor_hessian_max_relative_error":max_hessian_relative_error,
        "derivative_tolerance_unchanged":1e-4,
        "derivative_checks_pass":max_gradient_absolute_error<1e-4 && max_hessian_relative_error<1e-4,
        "notes":"No canonical provider change. Adapter allocates legacy GeometryJet strings; direct field query does not allocate. Timing excludes preparation/loading."
    })).unwrap());
    Ok(())
}
