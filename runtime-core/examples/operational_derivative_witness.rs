//! Preserve the original diagnostic and compare the corrected consistency check.
use runtime_core::operational_field::PreparedField;
use serde_json::json;
fn main() -> Result<(), String> {
    let path = std::env::args().nth(1).ok_or("asset path required")?;
    let field = PreparedField::from_asset_bytes(&std::fs::read(path).map_err(|e| e.to_string())?)
        .map_err(str::to_string)?;
    let mut records = Vec::new();
    let mut worst = (0.0, [0.0; 2], 0, 0);
    // This is exactly the original report's first 1,000 points and two deltas.
    // Additional smaller deltas are diagnostic only, not replacements for them.
    for delta in [1e-6, 1e-7, 5e-8, 2.5e-8] {
        let mut hessian_error = 0.0f64;
        let mut gradient_error = 0.0f64;
        for i in 0..1000 {
            let p = [
                0.2 + (i % 997) as f64 / 997.0 * 0.4,
                0.025 + (i % 293) as f64 / 293.0 * 0.15,
            ];
            let jet = field.query(p).map_err(|_| "witness point unavailable")?;
            for a in 0..2 {
                let (mut plus, mut minus) = (p, p);
                plus[a] += delta;
                minus[a] -= delta;
                let plus = field.query(plus).unwrap();
                let minus = field.query(minus).unwrap();
                gradient_error = gradient_error
                    .max(((plus.value - minus.value) / (2.0 * delta) - jet.gradient[a]).abs());
                for b in 0..2 {
                    let fd = (plus.gradient[b] - minus.gradient[b]) / (2.0 * delta);
                    let error = (fd - jet.hessian[b][a]).abs() / jet.hessian[b][a].abs().max(1.0);
                    hessian_error = hessian_error.max(error);
                    if delta == 1e-6 && error > worst.0 {
                        worst = (error, p, a, b);
                    }
                }
            }
        }
        records.push(json!({"delta":delta,"gradient_max_absolute_error":gradient_error,"hessian_max_normalized_error":hessian_error,"original_increment":delta==1e-6 || delta==1e-7}));
    }
    let (_, p, a, b) = worst;
    let jet = field.query(p).unwrap();
    let mut witness = Vec::new();
    for delta in [1e-6, 5e-7, 2.5e-7, 1e-7, 5e-8, 2.5e-8] {
        let sample = |d: f64| {
            let (mut plus, mut minus) = (p, p);
            plus[a] += d;
            minus[a] -= d;
            (field.query(plus).unwrap().gradient[b] - field.query(minus).unwrap().gradient[b])
                / (2.0 * d)
        };
        let fd = sample(delta);
        let richardson = (4.0 * sample(delta / 2.0) - fd) / 3.0;
        witness.push(json!({"delta":delta,"central_difference":fd,"normalized_error":(fd-jet.hessian[b][a]).abs()/jet.hessian[b][a].abs().max(1.0),"richardson":richardson,"richardson_normalized_error":(richardson-jet.hessian[b][a]).abs()/jet.hessian[b][a].abs().max(1.0)}));
    }
    let corrected = runtime_core::operational_field::verification::fixed_neighbor_report(&field)
        .map_err(|e| format!("{e:?}"))?;
    println!(
        "{}",
        json!({"field_id":format!("{:016x}",field.identity()),"original_tolerance":1e-4,"original_gate_passed":records.iter().filter(|r|r["original_increment"]==true).all(|r|r["gradient_max_absolute_error"].as_f64().unwrap()<1e-4 && r["hessian_max_normalized_error"].as_f64().unwrap()<1e-4),"grid_by_delta":records,"corrected_fixed_grid":corrected,"worst_original_witness":{"point":p,"gradient_component":b,"difference_axis":a,"returned_hessian":jet.hessian[b][a],"increments":witness},"claim":"Original central measurements preserved; corrected consistency verifier retains the same points, base increments and tolerance; no reference differential fidelity claim"})
    );
    Ok(())
}
