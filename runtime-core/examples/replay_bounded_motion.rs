//! Replay the original #148 driven and undriven Controls crossing scenario.
use runtime_core::controls::MotionControls;
use runtime_core::manifold::ManifoldConfig;
use runtime_core::operational_engine::{
    reference_policy_for_field, FrozenRollout, Integrator, PhysicsProfile,
};
use runtime_core::operational_field::PreparedField;
use std::sync::Arc;

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let asset = std::fs::read(args.next().ok_or("missing asset")?).map_err(|e| e.to_string())?;
    let policy =
        std::fs::read_to_string(args.next().ok_or("missing policy")?).map_err(|e| e.to_string())?;
    let field = Arc::new(PreparedField::from_asset_bytes(&asset)?);
    let cuts = reference_policy_for_field(&field, &policy)?;
    let config = ManifoldConfig {
        d_ref: 0.1,
        epsilon: 1e-4,
        lambda_sq: 1.0,
        kappa: 0.5,
        mu: std::f64::consts::FRAC_1_PI,
    };
    let controls = |throttle| MotionControls {
        direction: [1.0, 0.0],
        throttle,
        brake: 0.0,
        grip: 0.0,
        impulse: 0.0,
    };
    let profile = PhysicsProfile::new(
        config,
        0.02,
        controls(0.0).friction_beta(),
        Integrator::Rk4FourSubsteps,
    )?;
    let rollout = FrozenRollout::new(field, profile, cuts)?;
    let run = |throttle| -> Result<(bool, [f64; 2]), String> {
        let mut state = rollout.initial_state([0.3, 0.05], [0.0; 2])?;
        for _ in 0..400 {
            let q = rollout.drive_covector(state.position, &controls(throttle))?;
            rollout.attempt_step(&mut state, q)?;
            if rollout.probe(state.position)?["operational_value"]
                .as_f64()
                .ok_or("missing scalar")?
                > 0.0
            {
                return Ok((true, state.position));
            }
        }
        Ok((false, state.position))
    };
    let undriven = run(0.0)?;
    let driven = run(1.0)?;
    if undriven.0 || !driven.0 {
        return Err(format!(
            "crossing failed: undriven={undriven:?}, driven={driven:?}"
        ));
    }
    println!(
        "{}",
        serde_json::json!({"operational_crossing":"passed","reference_crossing":"unverified","field_id":format!("{:016x}",rollout.field_id()),"physics_profile_id":format!("{:016x}",rollout.profile().id()),"undriven":undriven,"driven":driven})
    );
    Ok(())
}
