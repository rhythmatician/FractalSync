//! Offline source/reconstruction isolation. Not an operational preparation.
use num_complex::Complex64;
use runtime_core::{
    distance_field::{load_builtin_distance_field, sample_signed_distance_field},
    geometry::lobe_point_at_angle,
    geometry_provider::GEOMETRY_SCALE_ALPHA,
    operational_field::{
        acceptance::{certified_realm_scalar_error_floor, reference_realm, Realm},
        PreparedField,
    },
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Plan {
    field_id: String,
    origin: [f64; 2],
    coefficient_spacings: Vec<f64>,
    angle_divisions: usize,
    multiplier_insets: Vec<f64>,
    epsilon: f64,
    quality_fraction: f64,
}

/// A complete finite stencil of the same fixed raster reconstruction. This is
/// constructed only offline to isolate spacing, never fitted in a live query.
fn probe(c: Complex64, origin: [f64; 2], h: f64) -> Result<f64, String> {
    let local_origin = [
        origin[0] + (((c.re - origin[0]) / h).floor() - 3.0) * h,
        origin[1] + (((c.im - origin[1]) / h).floor() - 3.0) * h,
    ];
    let points: Vec<_> = (0..8)
        .flat_map(|y| {
            (0..8).map(move |x| {
                Complex64::new(
                    local_origin[0] + x as f64 * h,
                    local_origin[1] + y as f64 * h,
                )
            })
        })
        .collect();
    let coefficients = sample_signed_distance_field(&points)?
        .into_iter()
        .map(f64::from)
        .collect();
    PreparedField::new(local_origin, h, [8, 8], coefficients)
        .map_err(str::to_string)?
        .query([c.re, c.im])
        .map(|j| j.value)
        .map_err(|_| "offline probe support unavailable".into())
}

fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("expected existing asset and fixed source-audit plan".into());
    }
    let text = std::fs::read_to_string(&args[2]).map_err(|e| e.to_string())?;
    let p: Plan = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    if p.angle_divisions < 2
        || p.angle_divisions > 4096
        || p.multiplier_insets.is_empty()
        || p.multiplier_insets.len() > 16
        || p.multiplier_insets
            .iter()
            .any(|x| !x.is_finite() || *x <= 0.0 || *x >= 1.0)
        || p.coefficient_spacings.is_empty()
        || p.coefficient_spacings.len() > 16
        || p.coefficient_spacings
            .iter()
            .any(|x| !x.is_finite() || *x <= 0.0)
        || p.origin.iter().any(|x| !x.is_finite())
        || p.epsilon != 1e-4
        || p.quality_fraction != 0.25
    {
        return Err("invalid source-audit contract".into());
    }
    let field =
        PreparedField::from_asset_bytes(&std::fs::read(&args[1]).map_err(|e| e.to_string())?)
            .map_err(str::to_string)?;
    if format!("{:016x}", field.identity()) != p.field_id {
        return Err("source audit belongs to another field".into());
    }
    if !p.coefficient_spacings.contains(&field.spacing()) {
        return Err("source audit must include the existing asset's spacing".into());
    }
    load_builtin_distance_field("default")?;
    let metadata = runtime_core::distance_field::distance_field_metadata();
    let mut rows = Vec::new();
    let mut stage_violations = vec![0usize; p.coefficient_spacings.len() + 2];
    let mut baseline_reproduction_error = 0.0f64;
    let mut unknown = 0;
    for i in 1..p.angle_divisions {
        let theta = std::f64::consts::PI * i as f64 / p.angle_divisions as f64;
        for inset in &p.multiplier_insets {
            let c = lobe_point_at_angle(1, 0, theta, 1.0 - inset);
            let realm = reference_realm([c.re, c.im]);
            if realm == Realm::Unknown {
                unknown += 1;
            }
            let raw = f64::from(sample_signed_distance_field(&[c])?[0]);
            let prepared = field
                .query([c.re, c.im])
                .map_err(|_| "audit sample outside coverage")?
                .value;
            let reconstructed: Vec<_> = p
                .coefficient_spacings
                .iter()
                .map(|h| probe(c, p.origin, *h))
                .collect::<Result<_, _>>()?;
            for (h, value) in p.coefficient_spacings.iter().zip(&reconstructed) {
                if *h == field.spacing() {
                    baseline_reproduction_error =
                        baseline_reproduction_error.max((value - prepared).abs());
                }
            }
            let values: Vec<_> = [raw, prepared]
                .into_iter()
                .chain(reconstructed.iter().copied())
                .collect();
            let flags: Vec<_> = values
                .iter()
                .enumerate()
                .map(|(stage, value)| {
                    let requested = GEOMETRY_SCALE_ALPHA * value.hypot(p.epsilon);
                    let fails = certified_realm_scalar_error_floor(*value, realm)
                        .is_some_and(|lower| lower > p.quality_fraction * requested);
                    stage_violations[stage] += usize::from(fails);
                    fails
                })
                .collect();
            rows.push(serde_json::json!({"point":[c.re,c.im],"theta":theta,"multiplier_inset":inset,"realm":realm,"raw_raster":raw,"prepared":prepared,"reconstructed":reconstructed,"certified_budget_violations":flags}));
        }
    }
    if baseline_reproduction_error > 1e-12 {
        return Err(format!(
            "offline probes do not reproduce the current evaluator: {baseline_reproduction_error}"
        ));
    }
    // This diagnoses the source, not the sign convention: standard deep interior
    // and exterior controls must agree before attributing near-Shore errors.
    let controls =
        sample_signed_distance_field(&[Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.5)])?;
    if controls[0] >= 0.0 || controls[1] <= 0.0 {
        return Err("source-raster sign controls disagree".into());
    }
    println!(
        "{}",
        serde_json::json!({"plan":serde_json::from_str::<serde_json::Value>(&text).unwrap(),"field_id":p.field_id,"samples":rows.len(),"unknown_realms":unknown,"raster_metadata":metadata,"sign_controls":controls,"baseline_reproduction_max_absolute_error":baseline_reproduction_error,"stage_order":"raw raster, existing asset, then coefficient_spacings in declared order","certified_budget_violation_counts":stage_violations,"samples_by_stage":rows,"claim":"Pointwise certified error floors only; no regional fidelity bound; offline diagnostic probes are not rollout assets"})
    );
    Ok(())
}
