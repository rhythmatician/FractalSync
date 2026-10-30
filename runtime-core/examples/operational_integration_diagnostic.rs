//! Same field, separate frozen mechanical profiles. Does not replace original tests.
use num_complex::Complex64;
use runtime_core::{
    manifold::{self, ManifoldConfig},
    operational_field::{
        acceptance::{reference_policy_for_field, FrozenRollout, Integrator, PhysicsProfile},
        PreparedField,
    },
};
use serde::Deserialize;
use std::sync::Arc;
#[derive(Deserialize)]
struct Plan {
    field_id: String,
    start: [f64; 2],
    epsilon: f64,
    kappa: f64,
    launch_energy_factor: f64,
    drag: f64,
    horizon_seconds: f64,
    timesteps: Vec<f64>,
    methods: Vec<String>,
}
fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("expected asset, reference policy and fixed integration plan".into());
    }
    let text = std::fs::read_to_string(&args[3]).map_err(|e| e.to_string())?;
    let p: Plan = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    if p.start != [0.4, 0.05]
        || p.epsilon != 1e-4
        || p.kappa != 0.1
        || p.launch_energy_factor != 25.0
        || p.drag != 0.0
        || p.horizon_seconds != 5.0
        || p.timesteps != [0.002, 0.001, 0.0005, 0.00025, 0.000125]
        || (p.methods != ["Euler", "Rk4"] && p.methods != ["Rk4FourSubsteps"])
    {
        return Err("changed integration diagnostic contract".into());
    }
    let field = Arc::new(
        PreparedField::from_asset_bytes(&std::fs::read(&args[1]).map_err(|e| e.to_string())?)
            .map_err(str::to_string)?,
    );
    if format!("{:016x}", field.identity()) != p.field_id {
        return Err("integration plan belongs to another field".into());
    }
    let cuts = reference_policy_for_field(
        &field,
        &std::fs::read_to_string(&args[2]).map_err(|e| e.to_string())?,
    )?;
    let config = ManifoldConfig {
        kappa: p.kappa,
        epsilon: p.epsilon,
        ..ManifoldConfig::default()
    };
    let locate = FrozenRollout::new(
        field.clone(),
        PhysicsProfile::new(config.clone(), 0.001, p.drag, Integrator::Euler)?,
        cuts.clone(),
    )?;
    let ((barrier, launch), _) = locate.evaluate(|config| {
        let (mut lo, mut hi) = (0.2, 0.5);
        for _ in 0..80 {
            let mid = (lo + hi) * 0.5;
            if manifold::signed_distance(Complex64::new(mid, 0.05))? < 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let start = Complex64::new(p.start[0], p.start[1]);
        let barrier = manifold::potential_energy(Complex64::new((lo + hi) * 0.5, 0.05), config)?
            - manifold::potential_energy(start, config)?;
        Ok((
            barrier,
            (2.0 * p.launch_energy_factor * barrier
                / manifold::induced_metric(start, config)?[0][0])
                .sqrt(),
        ))
    })?;
    let mut runs = Vec::new();
    for name in &p.methods {
        let method = Integrator::parse_name(name)?;
        for dt in &p.timesteps {
            let frame = FrozenRollout::new(
                field.clone(),
                PhysicsProfile::new(config.clone(), *dt, p.drag, method)?,
                cuts.clone(),
            )?;
            let mut state = frame.initial_state(p.start, [-launch, 0.0])?;
            let initial = state;
            let mut maximum_excursion = 0.0f64;
            let mut crossing = None;
            let mut segment_crossing = None;
            let mut failure = None;
            let mut work = 0usize;
            let mut monitor_queries = 0usize;
            let mut committed = 0;
            for step in 0..(p.horizon_seconds / dt).round() as usize {
                let previous = state;
                let result = frame.attempt_step(&mut state, [0.0; 2]);
                work += frame.last_evaluation_work().geometry_evaluations;
                if let Err(e) = result {
                    failure = Some(e);
                    break;
                }
                committed += 1;
                maximum_excursion = maximum_excursion.max((state.energy - initial.energy).abs());
                monitor_queries += 1;
                if field
                    .query(state.position)
                    .map_err(|_| "accepted endpoint unavailable")?
                    .value
                    < 0.0
                {
                    crossing = Some((step + 1) as f64 * dt);
                    let (mut lo, mut hi) = (0.0, 1.0);
                    for _ in 0..60 {
                        let mid = (lo + hi) * 0.5;
                        let c = [
                            previous.position[0] + mid * (state.position[0] - previous.position[0]),
                            previous.position[1] + mid * (state.position[1] - previous.position[1]),
                        ];
                        monitor_queries += 1;
                        if field
                            .query(c)
                            .map_err(|_| "accepted segment unavailable")?
                            .value
                            < 0.0
                        {
                            hi = mid;
                        } else {
                            lo = mid;
                        }
                    }
                    segment_crossing = Some(previous.mechanical_time + (lo + hi) * 0.5 * dt);
                    break;
                }
            }
            runs.push(serde_json::json!({"physics_profile":frame.profile().manifest(),"initial":initial,"final":state,"committed_steps":committed,"first_negative_sample_time":crossing,"bracketed_segment_zero_time":segment_crossing,"maximum_energy_excursion":maximum_excursion,"geometry_evaluations_in_steps":work,"additional_prepared_monitor_queries":monitor_queries,"failure":failure}));
        }
    }
    let mut drag_controls = Vec::new();
    for pure_drag in [false, true] {
        for substeps in [1, 2, 4, 8, 16, 32, 64] {
            let config = if pure_drag {
                ManifoldConfig {
                    kappa: 0.0,
                    mu: 0.0,
                    ..ManifoldConfig::default()
                }
            } else {
                ManifoldConfig::default()
            };
            let frame = FrozenRollout::new(
                field.clone(),
                PhysicsProfile::new(config, 0.01 / substeps as f64, 0.5, Integrator::Rk4)?,
                cuts.clone(),
            )?;
            // Match the existing Torch control's float32 inputs exactly.
            let position = [-0.5, f64::from(0.1f32)];
            let velocity = [f64::from(0.2f32); 2];
            let mut state = frame.initial_state(position, velocity)?;
            let initial = state;
            let ((kinetic, potential_power, drag_power), _) = frame.evaluate(|config| {
                let c = Complex64::new(position[0], position[1]);
                let v = (velocity[0], velocity[1]);
                let potential = manifold::potential_force(c, config)?;
                let wall = manifold::wall_force(c, config)?;
                let drag = manifold::drag_force(v, c, 0.5, config)?;
                Ok((
                    manifold::kinetic_energy(v, c, config)?,
                    v.0 * (potential.0 + wall.0) + v.1 * (potential.1 + wall.1),
                    v.0 * drag.0 + v.1 * drag.1,
                ))
            })?;
            for _ in 0..substeps {
                frame.attempt_step(&mut state, [0.0; 2])?;
            }
            let (final_kinetic, _) = frame.evaluate(|config| {
                manifold::kinetic_energy(
                    (state.velocity[0], state.velocity[1]),
                    Complex64::new(state.position[0], state.position[1]),
                    config,
                )
            })?;
            drag_controls.push(serde_json::json!({"pure_drag":pure_drag,"diagnostic_substeps":substeps,"requested_interval":0.01,"physics_profile":frame.profile().manifest(),"initial":initial,"final":state,"initial_kinetic":kinetic,"final_kinetic":final_kinetic,"delta_kinetic":final_kinetic-kinetic,"delta_total":state.energy-initial.energy,"drag_work":state.integrated_work,"energy_minus_work_residual":state.energy-initial.energy-state.integrated_work,"initial_potential_power":potential_power,"initial_drag_power":drag_power,"initial_kinetic_power":potential_power+drag_power,"pure_drag_exact_kinetic_if_no_potential":if pure_drag {Some(kinetic*(-2.0_f64*0.5*0.01).exp())}else{None}}));
        }
    }
    println!(
        "{}",
        serde_json::json!({"plan":serde_json::from_str::<serde_json::Value>(&text).unwrap(),"field_id":p.field_id,"barrier":barrier,"launch_speed":launch,"runs":runs,"drag_controls":drag_controls,"claim":"same operational scalar field; distinct immutable integrator profiles; original crossing-time and threshold tests unchanged; interpolated segment root is not an exact ODE event or reference crossing"})
    );
    Ok(())
}
