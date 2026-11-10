//! Offline, fixed-sample audit. No new membership or distance proof algorithm.
use runtime_core::{
    geometry::lobe_point_at_angle,
    geometry_provider::GEOMETRY_SCALE_ALPHA,
    operational_field::{
        acceptance::{certified_realm_scalar_error_floor, reference_realm},
        PreparedField,
    },
};
use serde::Deserialize;
#[derive(Deserialize)]
struct Plan {
    field_id: String,
    angle_divisions: usize,
    multiplier_insets: Vec<f64>,
    epsilon: f64,
    quality_fraction: f64,
}
fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("expected asset and fixed audit-plan paths".into());
    }
    let text = std::fs::read_to_string(&args[2]).map_err(|e| e.to_string())?;
    let plan: Plan = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    if plan.angle_divisions < 2
        || plan.angle_divisions > 4096
        || plan.multiplier_insets.is_empty()
        || plan.multiplier_insets.len() > 16
        || plan
            .multiplier_insets
            .iter()
            .any(|s| !s.is_finite() || *s <= 0.0 || *s >= 1.0)
        || plan.epsilon != 1e-4
        || plan.quality_fraction != 0.25
    {
        return Err("invalid or changed audit contract".into());
    }
    let field =
        PreparedField::from_asset_bytes(&std::fs::read(&args[1]).map_err(|e| e.to_string())?)
            .map_err(str::to_string)?;
    if format!("{:016x}", field.identity()) != plan.field_id {
        return Err("audit plan belongs to another field".into());
    }
    let mut samples = 0;
    let mut unavailable = 0;
    let mut conclusive = 0;
    let mut mismatches = 0;
    let mut violations = Vec::new();
    for i in 1..plan.angle_divisions {
        let theta = std::f64::consts::PI * i as f64 / plan.angle_divisions as f64;
        for inset in &plan.multiplier_insets {
            samples += 1;
            let c = lobe_point_at_angle(1, 0, theta, 1.0 - inset);
            let Ok(jet) = field.query([c.re, c.im]) else {
                unavailable += 1;
                continue;
            };
            let realm = reference_realm([c.re, c.im]);
            if realm != runtime_core::operational_field::acceptance::Realm::Unknown {
                conclusive += 1;
            }
            if let Some(lower) = certified_realm_scalar_error_floor(jet.value, realm) {
                mismatches += 1;
                let requested_scale = GEOMETRY_SCALE_ALPHA * jet.value.hypot(plan.epsilon);
                let budget = plan.quality_fraction * requested_scale;
                if lower > budget {
                    violations.push(serde_json::json!({"point":[c.re,c.im],"theta":theta,"multiplier_inset":inset,"returned_scalar":jet.value,"certified_realm":realm,"returned_scalar_error_lower_bound":lower,"unchanged_quality_budget":budget,"lower_bound_over_budget":lower/budget}));
                }
            }
        }
    }
    println!(
        "{}",
        serde_json::json!({"plan":serde_json::from_str::<serde_json::Value>(&text).unwrap(),"field_id":plan.field_id,"samples":samples,"unavailable":unavailable,"conclusive_reference_realms":conclusive,"certified_sign_mismatches":mismatches,"quality_violations":violations,
        "claim":"pointwise error floor of returned scalar versus true signed distance, using conclusive reference realm; no distance upper bound or derivative-fidelity claim"})
    );
    Ok(())
}
