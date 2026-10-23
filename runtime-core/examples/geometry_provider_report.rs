//! Reproducible #120/#145 geometry classification and synchronous-cost sample.
//! cargo run --release -p runtime_core --example geometry_provider_report
use num_complex::Complex64;
use runtime_core::geometry_provider::{query_geometry, GeometryValidity, DYADIC_FIT_ERROR_FACTOR};
use std::time::Instant;
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut rows = Vec::new();
    for (name, c) in [
        ("regular_interior", Complex64::new(-0.5, 0.1)),
        ("regular_shore_approach", Complex64::new(0.3, 0.05)),
        ("crossing_corridor", Complex64::new(0.35, 0.05)),
        ("unique_cusp_projection", Complex64::new(0.0, 0.0)),
        ("cut_locus", Complex64::new(-1.0, 0.0)),
        ("cardioid_cusp", Complex64::new(0.25, 0.0)),
        (
            "rejected_rollout_endpoint",
            Complex64::new(0.29835945330580743, 0.0515177858286498),
        ),
        (
            "high_curvature_rollout",
            Complex64::new(0.4005984941807052, 0.054142245287925383),
        ),
        (
            "remaining_crossing_refusal",
            Complex64::new(0.21656074663216787, 0.12610803467485909),
        ),
    ] {
        for epsilon in [1e-2, 1e-4, 1e-6, 1e-8] {
            // Each child owns an initially empty cache. Timings exclude process
            // startup and distinguish real cold work from immediate repetition.
            if args.len() == 1 {
                let result = std::process::Command::new(std::env::current_exe().unwrap())
                    .args(["--sample", name, &epsilon.to_string()])
                    .output()
                    .unwrap();
                assert!(
                    result.status.success(),
                    "{}",
                    String::from_utf8_lossy(&result.stderr)
                );
                rows.push(serde_json::from_slice::<serde_json::Value>(&result.stdout).unwrap());
                continue;
            }
            assert_eq!(args.len(), 4, "expected --sample CASE EPSILON");
            assert_eq!(args[1], "--sample");
            if args[2] != name || args[3].parse::<f64>().unwrap() != epsilon {
                continue;
            }
            let start = Instant::now();
            let jet = query_geometry(c, epsilon).unwrap();
            let first_us = start.elapsed().as_micros();
            let start = Instant::now();
            let repeat = query_geometry(c, epsilon).unwrap();
            let warm_us = start.elapsed().as_micros();
            assert_eq!(
                serde_json::to_value(&jet).unwrap(),
                serde_json::to_value(&repeat).unwrap()
            );
            if jet.validity == GeometryValidity::Regular {
                assert!(jet.resolved_scale <= jet.requested_scale);
                assert!(jet.estimated_error <= DYADIC_FIT_ERROR_FACTOR * jet.requested_scale);
            }
            println!(
                "{}",
                serde_json::json!({"case":name,"c":[c.re,c.im],"epsilon":epsilon,
                "cache_protocol":"isolated_process_cold_then_immediate_repeat",
                "cold_us":first_us,"warm_us":warm_us,"jet":jet})
            );
            return;
        }
    }
    println!("{}", serde_json::to_string_pretty(&rows).unwrap());
}
