//! Offline finite-contour source, with no changes to operational evaluation.
use runtime_core::operational_field::{
    acceptance::Preparation,
    preparation::{prepare_contour_region, ContourPreparation},
};
use serde::Deserialize;
use std::{io::Write, time::Instant};
#[derive(Deserialize)]
struct Experiment {
    preparation: Preparation,
    source_preparation: ContourPreparation,
}
fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("expected frozen manifest and NEW asset output paths".into());
    }
    let text = std::fs::read_to_string(&args[1]).map_err(|e| e.to_string())?;
    let spec: Experiment = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    if spec.preparation.origin != spec.source_preparation.origin
        || spec.preparation.spacing != spec.source_preparation.spacing
        || spec.preparation.shape != spec.source_preparation.shape
    {
        return Err("source and rollout coverage declarations differ".into());
    }
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[2])
        .map_err(|e| e.to_string())?;
    let timer = Instant::now();
    let (field, stats) = prepare_contour_region(&spec.source_preparation)?;
    let compute = timer.elapsed().as_secs_f64();
    let bytes = field.asset_bytes();
    output.write_all(&bytes).map_err(|e| e.to_string())?;
    output.sync_all().map_err(|e| e.to_string())?;
    println!(
        "{}",
        serde_json::json!({"field_id":format!("{:016x}",field.identity()),"manifest":serde_json::from_str::<serde_json::Value>(&text).unwrap(),"statistics":stats,"preparation_compute_seconds":compute,"preparation_total_seconds":timer.elapsed().as_secs_f64(),"asset_bytes":bytes.len(),"reference_fidelity":"unknown; finite-mask contour is measured data only"})
    );
    Ok(())
}
