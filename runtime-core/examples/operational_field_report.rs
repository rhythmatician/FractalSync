//! Preparation and timing are deliberately outside the Physics evaluation path.
use num_complex::Complex64;
use runtime_core::{
    manifold,
    operational_field::{
        acceptance::{
            frozen_experiment_report_with_integrator, reference_policy_for_field, FrozenRollout,
            Integrator, PhysicsProfile, EXPERIMENT_MANIFEST,
        },
        prepare_raster_baseline, PreparedField,
    },
};
use serde_json::json;
use std::{hint::black_box, sync::Arc, time::Instant};

fn stats(mut times: Vec<f64>) -> serde_json::Value {
    let first = times[0];
    times.sort_by(f64::total_cmp);
    json!({"samples":times.len(),"first_us":first,"p50_us":times[times.len()/2],"p99_us":times[(times.len()*99/100).min(times.len()-1)],"max_us":times.last().unwrap()})
}

fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 3 || args.len() % 2 != 1 {
        return Err(
            "use --prepare asset.bin once, then --asset asset.bin for frozen replay".into(),
        );
    }
    if args[1] == "--prepare" {
        if args.len() != 3 {
            return Err("prepare takes exactly one output path".into());
        }
        let start = Instant::now();
        let field = prepare_raster_baseline()?;
        let preparation = start.elapsed().as_secs_f64();
        let bytes = field.asset_bytes();
        std::fs::write(&args[2], &bytes).map_err(|e| e.to_string())?;
        println!(
            "{}",
            json!({"field_id":format!("{:016x}",field.identity()),"preparation_compute_seconds":preparation,"preparation_serialization_write_seconds":start.elapsed().as_secs_f64()-preparation,"preparation_total_seconds":start.elapsed().as_secs_f64(),"asset_bytes":bytes.len(),"coefficient_bytes":field.coefficient_bytes(),"frozen_experiment":serde_json::from_str::<serde_json::Value>(runtime_core::operational_field::acceptance::EXPERIMENT_MANIFEST).unwrap()})
        );
        return Ok(());
    }
    if args[1] != "--asset" {
        return Err("unknown operational experiment command".into());
    }
    let start = Instant::now();
    let bytes = std::fs::read(&args[2]).map_err(|e| e.to_string())?;
    let field = Arc::new(PreparedField::from_asset_bytes(&bytes).map_err(str::to_string)?);
    let loading = start.elapsed().as_secs_f64();
    let policy_start = Instant::now();
    let mut manifest = EXPERIMENT_MANIFEST.to_string();
    let mut policy = None;
    let mut method = Integrator::Euler;
    let mut seen = std::collections::HashSet::new();
    for pair in args[3..].chunks_exact(2) {
        if !seen.insert(&pair[0]) {
            return Err("duplicate replay option".into());
        }
        match pair[0].as_str() {
            "--manifest" => {
                manifest = std::fs::read_to_string(&pair[1]).map_err(|e| e.to_string())?
            }
            "--policy" => {
                policy = Some(std::fs::read_to_string(&pair[1]).map_err(|e| e.to_string())?)
            }
            "--integrator" => method = Integrator::parse_name(&pair[1])?,
            _ => return Err("unknown replay option".into()),
        }
    }
    let cuts = policy
        .as_deref()
        .map(|p| reference_policy_for_field(&field, p))
        .transpose()?
        .unwrap_or_default();
    let policy_loading = policy_start.elapsed().as_secs_f64();
    let first = Instant::now();
    black_box(
        field
            .query([0.4, 0.05])
            .map_err(|_| "missing prepared point")?,
    );
    let first_us = first.elapsed().as_secs_f64() * 1e6;
    // This measured reference comparison is offline diagnostic work.
    runtime_core::distance_field::load_builtin_distance_field("default")?;
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
                    max_gradient_absolute_error =
                        max_gradient_absolute_error.max((fd - j.gradient[a]).abs());
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
    let frozen = FrozenRollout::new(
        field.clone(),
        PhysicsProfile::new(manifold::ManifoldConfig::default(), 0.001, 0.0, method)?,
        cuts.clone(),
    )?;
    let mut transactional_times = Vec::new();
    let mut transactional_counts = None;
    for i in 0..1000 {
        let mut state = frozen.initial_state([0.4 + i as f64 * 1e-6, 0.05], [0.001, 0.0])?;
        let now = Instant::now();
        let counts = frozen.attempt_step(&mut state, [0.0; 2])?;
        transactional_times.push(now.elapsed().as_secs_f64() * 1e6);
        transactional_counts = Some(counts);
    }
    let experiment =
        frozen_experiment_report_with_integrator(field.clone(), &manifest, &cuts, method)?;
    println!("{}",serde_json::to_string_pretty(&json!({
        "profile":"experimental-operational-field/1", "identity":format!("{:016x}",field.identity()),
        "reference":"existing 1024 raster; measured comparison only; true-Shore fidelity and cut information unknown",
        "coverage":experiment["experiment_manifest"]["preparation"],
        "policy_loading_seconds":policy_loading,
        "coefficient_bytes":field.coefficient_bytes(),"asset_loading_seconds":loading,
        "first_prepared_query_us":first_us,"distinct_queries":stats(times),"unavailable_queries":stats(misses),
        "cache_stress_queries":stats(cold),"cache_stress_eviction_bytes":eviction.len()*8,
        "transactional_step_profile":frozen.profile().manifest(),"transactional_step":stats(transactional_times),"transactional_step_work":transactional_counts,
        "frozen_experiment":experiment,
        "measured_raster_max_scalar_difference":max_scalar_difference,
        "derivative_verifier":experiment["neighbor_differential_self_consistency"]["verifier"],
        "neighbor_gradient_max_absolute_error":experiment["neighbor_differential_self_consistency"]["evidence"]["gradient_max_absolute_error"],
        "neighbor_hessian_max_relative_error":experiment["neighbor_differential_self_consistency"]["evidence"]["hessian_max_normalized_error"],
        "legacy_central_differences":{"gradient_max_absolute_error":max_gradient_absolute_error,"hessian_max_normalized_error":max_hessian_relative_error,"tolerance":1e-4,"passed":max_gradient_absolute_error<1e-4 && max_hessian_relative_error<1e-4,"role":"preserved finite-increment diagnostic; not a truncation-corrected derivative estimate"},
        "derivative_tolerance_unchanged":1e-4,
        "derivative_checks_pass":experiment["neighbor_differential_self_consistency"]["passed"],
        "notes":"No canonical provider change. Adapter allocates legacy GeometryJet strings; direct field query does not allocate. Timing excludes preparation/loading."
    })).unwrap());
    Ok(())
}
