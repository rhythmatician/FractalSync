//! Reproducible #120/#145 geometry classification and synchronous-cost sample.
//! cargo run --release -p runtime_core --example geometry_provider_report
use num_complex::Complex64;
use runtime_core::geometry_provider::{query_geometry, GeometryValidity, DYADIC_FIT_ERROR_FACTOR};
use std::time::Instant;
fn main() {
    let mut rows = Vec::new();
    for (name, c) in [
        ("regular_interior", Complex64::new(-0.5, 0.1)),
        ("regular_shore_approach", Complex64::new(0.3, 0.05)),
        ("crossing_corridor", Complex64::new(0.35, 0.05)),
        ("cut_locus", Complex64::new(0.0, 0.0)),
        ("cardioid_cusp", Complex64::new(0.25, 0.0)),
    ] {
        for epsilon in [1e-2, 1e-4, 1e-6, 1e-8] {
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
            rows.push(
                serde_json::json!({"case":name,"c":[c.re,c.im],"epsilon":epsilon,
                "first_us":first_us,"warm_us":warm_us,"jet":jet}),
            );
        }
    }
    println!("{}", serde_json::to_string_pretty(&rows).unwrap());
}
