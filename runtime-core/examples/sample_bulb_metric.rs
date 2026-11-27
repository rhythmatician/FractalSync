//! Compare the current distance bridge with the analytic period-two circle.

use std::f64::consts::{LN_2, TAU};

use num_complex::Complex64;
use runtime_core::geometry_provider::{query_geometry, GEOMETRY_PROVIDER_VERSION};
use runtime_core::manifold::{induced_metric_from_jet, ManifoldConfig};

fn main() -> Result<(), String> {
    let config = ManifoldConfig::default();
    println!("provider={GEOMETRY_PROVIDER_VERSION}");
    println!("depth accepted refused max_distance_error max_condition_error");
    for depth in [0.24_f64, 0.15, 0.06, 0.01, 0.001, 0.0001, 0.000025] {
        let ideal = 1.0
            + config.lambda_sq / LN_2.powi(2) * depth.powi(2)
                / (depth.powi(2) + config.epsilon.powi(2));
        let mut accepted = 0;
        let mut refused = 0;
        let mut worst_distance = 0.0_f64;
        let mut worst_condition = 0.0_f64;
        for i in 0..12 {
            let angle = TAU * f64::from(i) / 12.0;
            let c = Complex64::from_polar(0.25 - depth, angle) - 1.0;
            let jet = query_geometry(c, config.epsilon)?;
            match induced_metric_from_jet(&jet, &config) {
                Ok(g) => {
                    let spread = (g[0][0] - g[1][1]).hypot(2.0 * g[0][1]);
                    let trace = g[0][0] + g[1][1];
                    assert!(trace > spread, "metric failed the eigenvalue check at {c}");
                    let condition = (trace + spread) / (trace - spread);
                    accepted += 1;
                    worst_distance = worst_distance.max((jet.d + depth).abs());
                    worst_condition = worst_condition.max((condition - ideal).abs());
                }
                Err(_) => refused += 1,
            }
        }
        println!("{depth:.6} {accepted} {refused} {worst_distance:.6e} {worst_condition:.6e}");
    }
    for x in [-2.001, -1.0, -0.75] {
        let c = Complex64::new(x, 0.0);
        let jet = query_geometry(c, config.epsilon)?;
        println!(
            "{c}: {:?} {:?} bridge={} metric_accepted={}",
            jet.validity,
            jet.singularity,
            jet.is_bridge,
            induced_metric_from_jet(&jet, &config).is_ok()
        );
    }
    Ok(())
}
