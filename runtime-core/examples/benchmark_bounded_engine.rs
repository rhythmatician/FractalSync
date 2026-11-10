//! Cold prepared-query and complete-step timings, excluding asset loading.
use runtime_core::manifold::ManifoldConfig;
use runtime_core::operational_engine::{
    reference_policy_for_field, FrozenRollout, Integrator, PhysicsProfile,
};
use runtime_core::operational_field::PreparedField;
use std::sync::Arc;
use std::time::Instant;

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let asset = args.next().ok_or("missing asset")?;
    let policy = args.next().ok_or("missing reference policy")?;
    let loading = Instant::now();
    let bytes = std::fs::read(asset).map_err(|e| e.to_string())?;
    let policy = std::fs::read_to_string(policy).map_err(|e| e.to_string())?;
    let field = Arc::new(PreparedField::from_asset_bytes(&bytes)?);
    let cuts = reference_policy_for_field(&field, &policy)?;
    let load_ns = loading.elapsed().as_nanos();
    let point = [0.4, 0.05];
    let first = Instant::now();
    std::hint::black_box(
        field
            .query(std::hint::black_box(point))
            .map_err(|_| "GeometryUnavailable")?,
    );
    let first_ns = first.elapsed().as_nanos();
    let mut samples = Vec::with_capacity(10_000);
    for i in 0..10_000 {
        let t = Instant::now();
        std::hint::black_box(
            field
                .query(std::hint::black_box([
                    point[0] + ((i % 101) as f64 - 50.0) * 1e-6,
                    point[1],
                ]))
                .map_err(|_| "GeometryUnavailable")?,
        );
        samples.push(t.elapsed().as_nanos());
    }
    let profile = PhysicsProfile::new(
        ManifoldConfig::default(),
        0.001,
        0.0,
        Integrator::Rk4FourSubsteps,
    )?;
    let rollout = FrozenRollout::new(field, profile, cuts)?;
    let mut state = rollout.initial_state(point, [0.0, 0.0])?;
    let mut steps = Vec::with_capacity(1000);
    for _ in 0..1000 {
        let t = Instant::now();
        rollout.attempt_step(&mut state, [0.0, 0.0])?;
        steps.push(t.elapsed().as_nanos());
    }
    let first_step = steps[0];
    samples.sort_unstable();
    steps.sort_unstable();
    println!(
        "{}",
        serde_json::json!({"field_id":format!("{:016x}",rollout.field_id()),"physics_profile_id":format!("{:016x}",rollout.profile().id()),"load_ns":load_ns,"first_query_ns":first_ns,"query_p99_ns":samples[9899],"query_max_ns":samples[9999],"first_step_ns":first_step,"step_p99_ns":steps[989],"step_max_ns":steps[999],"work":rollout.last_evaluation_work(),"query_count":samples.len(),"step_count":steps.len(),"scope":"native process, fixed local region, no cache stress; preparation excluded"})
    );
    Ok(())
}
