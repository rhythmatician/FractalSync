//! Offline comparison of two frozen source constructions. Never a query certificate.
use runtime_core::{
    geometry::lobe_point_at_angle,
    geometry_provider::GEOMETRY_SCALE_ALPHA,
    operational_field::{
        verification::{coefficient_magnitude_evidence, source_refinement_evidence},
        PreparedField,
    },
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Plan {
    schema: String,
    coarse_field_id: String,
    fine_field_id: String,
    coarse_source_spacing: f64,
    fine_source_spacing: f64,
    angle_divisions: usize,
    multiplier_insets: Vec<f64>,
    fixed_controls: [[f64; 2]; 3],
}
fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("expected coarse asset, fine asset and frozen comparison-plan paths".into());
    }
    let text = std::fs::read_to_string(&args[3]).map_err(|e| e.to_string())?;
    let plan: Plan = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    if plan.schema != "operational-source-refinement-plan/1"
        || plan.fixed_controls != [[0.3, 0.05], [0.4, 0.05], [0.6, 0.15]]
        || plan.angle_divisions != 256
        || plan.multiplier_insets != [1e-6, 1e-5, 1e-4]
        || plan.coarse_source_spacing != 0.002
        || plan.fine_source_spacing != 0.001
    {
        return Err("changed fixed comparison plan".into());
    }
    let load = |path: &str, id: &str| -> Result<PreparedField, String> {
        let f = PreparedField::from_asset_bytes(&std::fs::read(path).map_err(|e| e.to_string())?)
            .map_err(str::to_string)?;
        if format!("{:016x}", f.identity()) != id {
            return Err("asset identity mismatch".into());
        }
        Ok(f)
    };
    let coarse = load(&args[1], &plan.coarse_field_id)?;
    let fine = load(&args[2], &plan.fine_field_id)?;
    let mut points: Vec<_> = [
        "quality_probe",
        "high_curvature_and_crossing_start",
        "ridge_start",
    ]
    .into_iter()
    .zip(plan.fixed_controls)
    .collect();
    for i in 0..1000 {
        points.push((
            "neighbor_grid",
            [
                0.2 + (i % 997) as f64 / 997.0 * 0.4,
                0.025 + (i % 293) as f64 / 293.0 * 0.15,
            ],
        ));
    }
    for i in 1..plan.angle_divisions {
        for inset in &plan.multiplier_insets {
            let c = lobe_point_at_angle(
                1,
                0,
                std::f64::consts::PI * i as f64 / plan.angle_divisions as f64,
                1.0 - inset,
            );
            points.push(("scalar_audit", [c.re, c.im]));
        }
    }
    let mut samples = Vec::new();
    let mut unavailable = 0;
    let mut failures = 0;
    for (group, point) in points {
        let (Ok(a), Ok(b)) = (coarse.query(point), fine.query(point)) else {
            unavailable += 1;
            samples.push(serde_json::json!({"group":group,"point":point,"available":false}));
            continue;
        };
        let evidence = source_refinement_evidence(a, b, plan.coarse_source_spacing)
            .map_err(|e| format!("invalid comparison: {e:?}"))?;
        let requested_scale = GEOMETRY_SCALE_ALPHA * b.value.hypot(1e-4);
        let budget = 0.25 * requested_scale;
        let passes = evidence.maximum_difference <= budget;
        failures += usize::from(!passes);
        samples.push(
            serde_json::json!({"group":group,"point":point,"available":true,
            "coarse_scalar":a.value,"fine_scalar":b.value,"source_refinement":evidence,
            "requested_scale":requested_scale,"unchanged_quality_budget":budget,
            "within_pointwise_comparison_budget":passes}),
        );
    }
    let coefficient_retention = coefficient_magnitude_evidence(&coarse, &fine, 1e-12)
        .map_err(|e| format!("invalid coefficient comparison: {e:?}"))?;
    println!(
        "{}",
        serde_json::json!({"schema":"operational-source-refinement/1",
        "plan":serde_json::from_str::<serde_json::Value>(&text).unwrap(),
        "claim":"measured pointwise sensitivity to source coarse-grid refinement; not a distance bound, fit residual, consecutive coefficient-resolution convergence or regional certificate",
        "formula":"max(abs(F_f-F_c), h_source*norm(grad_f-grad_c), h_source^2*Frobenius(H_f-H_c))",
        "source_spacing_differs_from_coefficient_spacing":true,
        "samples":samples,"unavailable":unavailable,"pointwise_comparison_failures":failures,
        "all_pointwise_comparisons_within_budget":failures == 0 && unavailable == 0,
        "coefficient_magnitude_comparison":coefficient_retention,
        "coefficient_comparison_claim":"stored nodal magnitudes and signs only; no reconstructed-field or true-distance bound",
        "runtime_quality_metadata_changed":false})
    );
    Ok(())
}
