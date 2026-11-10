//! Offline construction for one declared coverage experiment; no live fallback.
use runtime_core::operational_field::{acceptance::Preparation, prepare_raster_region};
use serde::Deserialize;
use std::{io::Write, time::Instant};

#[derive(Deserialize)]
struct CoverageExperiment {
    preparation: Preparation,
    maximum_coefficient_bytes: usize,
    anchors: Vec<[f64; 2]>,
}

fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() == 4 && args[1] == "--policy" {
        #[derive(Deserialize)]
        struct Spec {
            field_id: String,
            points: Vec<[f64; 2]>,
        }
        let spec: Spec =
            serde_json::from_str(&std::fs::read_to_string(&args[3]).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        if spec.points.len() > 64 || spec.points.iter().flatten().any(|v| !v.is_finite()) {
            return Err("invalid reference query plan".into());
        }
        let field = runtime_core::operational_field::PreparedField::from_asset_bytes(
            &std::fs::read(&args[2]).map_err(|e| e.to_string())?,
        )
        .map_err(str::to_string)?;
        if format!("{:016x}", field.identity()) != spec.field_id {
            return Err("reference query plan belongs to another field".into());
        }
        let mut cuts = Vec::new();
        for p in spec.points {
            field
                .query(p)
                .map_err(|_| "reference query lies outside field coverage")?;
            let jet = runtime_core::geometry_provider::query_geometry(
                num_complex::Complex64::new(p[0], p[1]),
                1e-4,
            )?;
            if jet.validity != runtime_core::geometry_provider::GeometryValidity::Singular
                || jet.singularity != runtime_core::geometry_provider::SingularityKind::CutLocus
            {
                return Err("planned authoritative cut refusal unavailable".into());
            }
            cuts.push(runtime_core::operational_field::acceptance::ReferenceCut {
                bounds: [p[0], p[1], p[0], p[1]],
                authority: format!(
                    "current canonical {} {} {:?}; exact point only",
                    jet.provider_version, jet.tile_id, jet.singularity
                ),
            });
        }
        runtime_core::operational_field::acceptance::validate_reference_cuts(&cuts)?;
        println!(
            "{}",
            serde_json::json!({"field_id":spec.field_id,"refusals":cuts,"unknown_elsewhere":true,"distance_fidelity":"unknown","source":"existing canonical provider, offline snapshot before rollout"})
        );
        return Ok(());
    }
    if args.len() != 3 {
        return Err("expected frozen manifest path and NEW asset output path".into());
    }
    let text = std::fs::read_to_string(&args[1]).map_err(|e| e.to_string())?;
    let spec: CoverageExperiment = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let p = spec.preparation;
    let bytes = p.shape[0]
        .checked_mul(p.shape[1])
        .and_then(|n| n.checked_mul(8))
        .ok_or("preparation byte count overflow")?;
    if bytes > spec.maximum_coefficient_bytes {
        return Err("declared preparation memory budget exceeded".into());
    }
    // Refuse overwrite before spending preparation work. No existing asset is replaced.
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[2])
        .map_err(|e| e.to_string())?;
    let start = Instant::now();
    let field = prepare_raster_region(p.origin, p.spacing, p.shape)?;
    let compute = start.elapsed().as_secs_f64();
    for anchor in &spec.anchors {
        field
            .query(*anchor)
            .map_err(|_| format!("declared anchor unavailable: {anchor:?}"))?;
    }
    let asset = field.asset_bytes();
    output.write_all(&asset).map_err(|e| e.to_string())?;
    output.sync_all().map_err(|e| e.to_string())?;
    println!(
        "{}",
        serde_json::json!({"field_id":format!("{:016x}",field.identity()),"manifest":serde_json::from_str::<serde_json::Value>(&text).unwrap(),"coefficient_bytes":field.coefficient_bytes(),"asset_bytes":asset.len(),"preparation_compute_seconds":compute,"preparation_total_seconds":start.elapsed().as_secs_f64(),"anchors_available":spec.anchors.len(),"reference_fidelity":"unknown"})
    );
    Ok(())
}
