//! Research-only period-two Physics comparison using the existing frozen engine.
#[cfg(not(feature = "experimental-operational-field"))]
fn main() {
    eprintln!("Enable experimental-operational-field to run this example.");
    std::process::exit(2);
}

#[cfg(feature = "experimental-operational-field")]
fn main() -> Result<(), String> {
    pilot::run()
}

#[cfg(feature = "experimental-operational-field")]
mod pilot {
    use runtime_core::manifold::ManifoldConfig;
    use runtime_core::operational_engine::{
        reference_policy_for_field, FrozenRollout, Integrator, MechanicalState, PhysicsProfile,
        ReferenceCut,
    };
    use runtime_core::operational_field::{FieldJet, PreparedField};
    use serde::Serialize;
    use serde_json::{json, Value};
    use std::{sync::Arc, time::Instant};

    const ORIGIN: [f64; 2] = [-0.94, 0.06];
    const SPACING: f64 = 1.0 / 512.0;
    const SIDE: usize = 49;
    const START: [f64; 2] = [-0.9, 0.1];
    const VELOCITY: [f64; 2] = [0.02, -0.01];
    const STEP: f64 = 0.01;
    const STEPS: usize = 40;
    const BETA: f64 = 0.2;
    const ENERGY_RESIDUAL_LIMIT: f64 = 0.05;
    const REFINEMENT_LIMIT: f64 = 1e-6;
    const BASELINE_DISTANCE_LIMIT: f64 = 0.001;
    const RETAINED_FIELD_ID: u64 = 0x9721bb7647a0b8e6;

    #[test]
    fn prepared_quadratic_matches_analytic_jet_on_both_sides_of_joins() -> Result<(), String> {
        let validation = validate_dynamical_jet(&dynamical_field()?)?;
        assert_eq!(validation.points, 26);
        assert!(validation.max_value_error < 1e-13);
        assert!(validation.max_gradient_error < 1e-11);
        assert!(validation.max_hessian_error < 1e-8);
        Ok(())
    }

    fn dynamical_value([x, y]: [f64; 2]) -> f64 {
        2.0 * ((x + 1.0).powi(2) + y * y) - 0.125
    }

    fn circle_distance([x, y]: [f64; 2]) -> f64 {
        (x + 1.0).hypot(y) - 0.25
    }

    fn dynamical_jet(point: [f64; 2]) -> FieldJet {
        FieldJet {
            value: dynamical_value(point),
            gradient: [4.0 * (point[0] + 1.0), 4.0 * point[1]],
            hessian: [[4.0, 0.0], [0.0, 4.0]],
        }
    }

    fn circle_jet([x, y]: [f64; 2]) -> FieldJet {
        let radius = (x + 1.0).hypot(y);
        let [nx, ny] = [(x + 1.0) / radius, y / radius];
        FieldJet {
            value: radius - 0.25,
            gradient: [nx, ny],
            hessian: [
                [ny * ny / radius, -nx * ny / radius],
                [-nx * ny / radius, nx * nx / radius],
            ],
        }
    }

    fn sample_field(
        value: impl Fn([f64; 2]) -> f64,
        correction: f64,
    ) -> Result<PreparedField, String> {
        let mut coefficients = Vec::with_capacity(SIDE * SIDE);
        for y in 0..SIDE {
            for x in 0..SIDE {
                let point = [
                    ORIGIN[0] + SPACING * x as f64,
                    ORIGIN[1] + SPACING * y as f64,
                ];
                if circle_distance(point) >= 0.0 {
                    return Err("coefficient halo leaves the period-two interior".into());
                }
                coefficients.push(value(point) - correction);
            }
        }
        PreparedField::new(ORIGIN, SPACING, [SIDE; 2], coefficients).map_err(str::to_owned)
    }

    fn dynamical_field() -> Result<PreparedField, String> {
        // Quintic B-spline sampling adds h² ΔD / 4 to a quadratic; ΔD = 8.
        sample_field(dynamical_value, 2.0 * SPACING * SPACING)
    }

    #[derive(Default, Serialize)]
    struct JetError {
        points: usize,
        max_value_error: f64,
        max_gradient_error: f64,
        max_hessian_error: f64,
    }

    fn sampled_jet_error(
        field: &PreparedField,
        exact_jet: impl Fn([f64; 2]) -> FieldJet,
    ) -> Result<JetError, String> {
        let mut max = JetError::default();
        let offsets = [9.0 - 1e-5, 9.0, 9.0 + 1e-5, 19.25, 41.75];
        let points = offsets.into_iter().flat_map(|y| {
            offsets
                .into_iter()
                .map(move |x| [ORIGIN[0] + x * SPACING, ORIGIN[1] + y * SPACING])
        });
        for point in points.chain([START]) {
            let jet = field
                .query(point)
                .map_err(|_| format!("validation jet unavailable at {point:?}"))?;
            let exact = exact_jet(point);
            max.points += 1;
            max.max_value_error = max.max_value_error.max((jet.value - exact.value).abs());
            for axis in 0..2 {
                max.max_gradient_error = max
                    .max_gradient_error
                    .max((jet.gradient[axis] - exact.gradient[axis]).abs());
                for other in 0..2 {
                    max.max_hessian_error = max
                        .max_hessian_error
                        .max((jet.hessian[axis][other] - exact.hessian[axis][other]).abs());
                }
            }
        }
        Ok(max)
    }

    fn validate_dynamical_jet(field: &PreparedField) -> Result<JetError, String> {
        let max = sampled_jet_error(field, dynamical_jet)?;
        if max.max_value_error >= 1e-13
            || max.max_gradient_error >= 1e-11
            || max.max_hessian_error >= 1e-8
        {
            return Err(format!("quadratic jet check failed: {}", json!(max)));
        }
        Ok(max)
    }

    fn load_baseline() -> Result<(Arc<PreparedField>, Vec<ReferenceCut>, &'static str), String> {
        match std::env::args().skip(1).collect::<Vec<_>>().as_slice() {
            [] => circle_baseline(),
            [flag] if flag == "--circle-distance" => circle_baseline(),
            [flag, asset, policy] if flag == "--retained" => {
                let bytes = std::fs::read(asset).map_err(|error| error.to_string())?;
                let field = PreparedField::from_asset_bytes(&bytes)?;
                if field.identity() != RETAINED_FIELD_ID {
                    return Err("asset does not match the retained baseline field ID".into());
                }
                let policy = std::fs::read_to_string(policy).map_err(|error| error.to_string())?;
                let cuts = reference_policy_for_field(&field, &policy)?;
                Ok((Arc::new(field), cuts, "retained prepared distance"))
            }
            _ => Err(
                "usage: dynamical_ruler_ablation [--circle-distance | --retained ASSET POLICY]"
                    .into(),
            ),
        }
    }

    fn circle_baseline() -> Result<(Arc<PreparedField>, Vec<ReferenceCut>, &'static str), String> {
        let field = sample_field(circle_distance, 0.0)?;
        Ok((
            Arc::new(field),
            Vec::new(),
            "sampled circle-distance surrogate",
        ))
    }

    fn make_rollout(
        field: Arc<PreparedField>,
        cuts: Vec<ReferenceCut>,
        lambda_sq: f64,
        dt: f64,
    ) -> Result<FrozenRollout, String> {
        let config = ManifoldConfig {
            lambda_sq,
            kappa: 0.0,
            ..ManifoldConfig::default()
        };
        let profile = PhysicsProfile::new(config, dt, BETA, Integrator::Rk4FourSubsteps)?;
        FrozenRollout::new(field, profile, cuts)
    }

    #[derive(Serialize)]
    struct Sample {
        state: MechanicalState,
        coordinate_speed: f64,
        intrinsic_speed: f64,
        energy_work_residual: f64,
    }

    fn sample(
        rollout: &FrozenRollout,
        state: MechanicalState,
        initial_energy: f64,
    ) -> Result<Sample, String> {
        let probe = rollout.probe(state.position)?;
        let metric: [[f64; 2]; 2] =
            serde_json::from_value(probe["metric"].clone()).map_err(|error| error.to_string())?;
        let [vx, vy] = state.velocity;
        let speed_squared = vx * (metric[0][0] * vx + metric[0][1] * vy)
            + vy * (metric[1][0] * vx + metric[1][1] * vy);
        if !speed_squared.is_finite() || speed_squared < 0.0 {
            return Err("invalid metric speed".into());
        }
        Ok(Sample {
            state,
            coordinate_speed: vx.hypot(vy),
            intrinsic_speed: speed_squared.sqrt(),
            energy_work_residual: state.energy - initial_energy - state.integrated_work,
        })
    }

    #[derive(Serialize)]
    struct Replay {
        profile: Value,
        max_energy_work_residual: f64,
        geometry_evaluations: usize,
        coefficient_reads_upper_bound: usize,
        step_elapsed_us: u128,
        samples: Vec<Sample>,
    }

    fn replay(rollout: &FrozenRollout, substeps: usize) -> Result<Replay, String> {
        let mut state = rollout.initial_state(START, VELOCITY)?;
        let initial_energy = state.energy;
        let mut replay = Replay {
            profile: rollout.profile().manifest(),
            max_energy_work_residual: 0.0,
            geometry_evaluations: 0,
            coefficient_reads_upper_bound: 0,
            step_elapsed_us: 0,
            samples: vec![sample(rollout, state, initial_energy)?],
        };
        let mut elapsed = std::time::Duration::ZERO;
        for step in 0..STEPS {
            // Construct the coordinate covector once per outer input interval.
            // Refinement repeats it unchanged during both half steps.
            let drive = if step < STEPS / 2 {
                [2.0, -1.0]
            } else {
                [-0.5, 1.0]
            };
            for _ in 0..substeps {
                let begin = Instant::now();
                let work = rollout.attempt_step(&mut state, drive)?;
                elapsed += begin.elapsed();
                replay.geometry_evaluations += work.geometry_evaluations;
                replay.coefficient_reads_upper_bound += work.coefficient_reads_upper_bound;
                replay.max_energy_work_residual = replay
                    .max_energy_work_residual
                    .max((state.energy - initial_energy - state.integrated_work).abs());
            }
            if circle_distance(state.position) >= 0.0 {
                return Err("trajectory left the certified period-two interior".into());
            }
            replay.samples.push(sample(rollout, state, initial_energy)?);
        }
        replay.step_elapsed_us = elapsed.as_micros();
        if replay.max_energy_work_residual >= ENERGY_RESIDUAL_LIMIT {
            return Err("energy-work residual exceeds the existing replay limit".into());
        }
        Ok(replay)
    }

    fn compare(
        field: Arc<PreparedField>,
        cuts: Vec<ReferenceCut>,
        name: &str,
        lambda_sq: f64,
    ) -> Result<Value, String> {
        let coarse = replay(
            &make_rollout(field.clone(), cuts.clone(), lambda_sq, STEP)?,
            1,
        )
        .map_err(|error| format!("{name}, coarse: {error}"))?;
        let fine = replay(&make_rollout(field, cuts, lambda_sq, STEP / 2.0)?, 2)
            .map_err(|error| format!("{name}, fine: {error}"))?;
        let mut position_error: f64 = 0.0;
        let mut velocity_error: f64 = 0.0;
        for (coarse, fine) in coarse.samples.iter().zip(&fine.samples) {
            let [cx, cy] = coarse.state.position;
            let [fx, fy] = fine.state.position;
            position_error = position_error.max((cx - fx).hypot(cy - fy));
            let [cx, cy] = coarse.state.velocity;
            let [fx, fy] = fine.state.velocity;
            velocity_error = velocity_error.max((cx - fx).hypot(cy - fy));
        }
        if position_error >= REFINEMENT_LIMIT || velocity_error >= REFINEMENT_LIMIT {
            return Err(format!(
                "{name}: refinement differences {position_error:e} in position and {velocity_error:e} in velocity"
            ));
        }
        Ok(json!({
            "name": name,
            "max_step_halving_position_difference": position_error,
            "max_step_halving_velocity_difference": velocity_error,
            "coarse": coarse,
            "fine": fine,
        }))
    }

    pub fn run() -> Result<(), String> {
        let (distance, cuts, baseline) = load_baseline()?;
        let dynamical = Arc::new(dynamical_field()?);
        let jet_error = validate_dynamical_jet(&dynamical)?;
        let distance_error = sampled_jet_error(&distance, circle_jet)?;
        if distance_error.max_value_error >= BASELINE_DISTANCE_LIMIT {
            return Err(format!(
                "baseline sampled distance discrepancy: {}",
                json!(distance_error)
            ));
        }
        let profiles = [
            compare(distance.clone(), cuts.clone(), "distance graph", 1.0)?,
            compare(distance, cuts.clone(), "distance conformal", 0.0)?,
            compare(dynamical, cuts, "dynamical conformal", 0.0)?,
        ];
        println!(
            "{}",
            json!({
                "experiment": "period-two interior ablation",
                "baseline": baseline,
                "baseline_sampled_jet_discrepancy": distance_error,
                "baseline_sampled_distance_limit": BASELINE_DISTANCE_LIMIT,
                "dynamical_jet_validation": jet_error,
                "position_velocity_step_halving_limit": REFINEMENT_LIMIT,
                "energy_work_residual_limit": ENERGY_RESIDUAL_LIMIT,
                "start": START,
                "initial_velocity": VELOCITY,
                "outer_input_dt": STEP,
                "outer_input_steps": STEPS,
                "drive_first_half": [2.0, -1.0],
                "drive_second_half": [-0.5, 1.0],
                "profiles": profiles,
            })
        );
        Ok(())
    }
}
