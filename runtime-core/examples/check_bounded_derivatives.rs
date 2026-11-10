//! Check neighboring operational derivatives on the fixed 1,000-point corpus.
use runtime_core::operational_field::{verification::fixed_neighbor_report, PreparedField};

fn main() -> Result<(), String> {
    let path = std::env::args().nth(1).ok_or("missing prepared asset")?;
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    let field = PreparedField::from_asset_bytes(&bytes)?;
    let report = fixed_neighbor_report(&field).map_err(|e| format!("{e:?}"))?;
    let mut join_gradient = 0.0_f64;
    let mut join_hessian = 0.0_f64;
    for x in [
        0.3205 - 1e-9,
        0.3205,
        0.3205 + 1e-9,
        0.4 - 1e-9,
        0.4,
        0.4 + 1e-9,
    ] {
        let point = [x, 0.0475];
        let center = field.query(point).map_err(|_| "join unavailable")?;
        for delta in [1e-6, 1e-7] {
            for axis in 0..2 {
                let (mut plus, mut minus) = (point, point);
                plus[axis] += delta;
                minus[axis] -= delta;
                let a = field.query(plus).map_err(|_| "join stencil unavailable")?;
                let b = field.query(minus).map_err(|_| "join stencil unavailable")?;
                join_gradient = join_gradient
                    .max(((a.value - b.value) / (2.0 * delta) - center.gradient[axis]).abs());
                for component in 0..2 {
                    let expected = center.hessian[component][axis];
                    join_hessian = join_hessian.max(
                        ((a.gradient[component] - b.gradient[component]) / (2.0 * delta)
                            - expected)
                            .abs()
                            / expected.abs().max(1.0),
                    );
                }
            }
        }
    }
    println!(
        "{}",
        serde_json::json!({"field_id":format!("{:016x}",field.identity()),"neighbors":report,"joins":{"gradient_absolute_error":join_gradient,"hessian_normalized_error":join_hessian,"tolerance":1e-4,"passed":join_gradient<1e-4 && join_hessian<1e-4}})
    );
    if report["passed"] != true || join_gradient >= 1e-4 || join_hessian >= 1e-4 {
        return Err("neighbor differential check failed".into());
    }
    Ok(())
}
