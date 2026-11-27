//! Research-only two-component attachment crossing through the frozen Physics engine.
#[cfg(not(feature = "experimental-operational-field"))]
fn main() {
    eprintln!("Enable experimental-operational-field to run this example.");
    std::process::exit(2);
}

#[cfg(feature = "experimental-operational-field")]
fn main() -> Result<(), String> {
    fixture::run()
}

#[cfg(feature = "experimental-operational-field")]
mod fixture {
    use num_complex::Complex64;
    use runtime_core::manifold::ManifoldConfig;
    use runtime_core::operational_engine::{
        reference_policy_for_field, FrozenRollout, Integrator, MechanicalState, PhysicsProfile,
    };
    use runtime_core::operational_field::PreparedField;
    use serde::Serialize;
    use serde_json::{json, Value};
    use std::sync::Arc;

    const ATTACHMENT: f64 = -0.75;
    const FLOOR: f64 = 1.0 / 128.0;
    const GRID: f64 = 1.0 / 8192.0;
    const ORIGIN: [f64; 2] = [ATTACHMENT - 320.0 * GRID, -32.0 * GRID];
    const SHAPE: [usize; 2] = [641, 65];
    const START: [f64; 2] = [ATTACHMENT - 2.0 * FLOOR, 0.0];
    const VELOCITY: [f64; 2] = [0.05, 0.0];
    const HOLD: [f64; 2] = [0.0; 2];
    const STEP: f64 = 0.005;
    const STEPS: u32 = 200;
    const BETA: f64 = 0.2;
    const RETAINED_FIELD_ID: u64 = 0x9721bb7647a0b8e6;

    #[derive(Clone, Copy)]
    enum Profile {
        DistanceGraph,
        DistanceConformal,
        DynamicalConformal,
    }

    impl Profile {
        fn name(self) -> &'static str {
            match self {
                Self::DistanceGraph => "tangent-line distance graph",
                Self::DistanceConformal => "tangent-line distance conformal",
                Self::DynamicalConformal => "quartic dynamical conformal",
            }
        }

        fn lambda_sq(self) -> f64 {
            match self {
                Self::DistanceGraph => 1.0,
                Self::DistanceConformal | Self::DynamicalConformal => 0.0,
            }
        }

        fn axis_metric_sqrt(self, x: f64) -> f64 {
            match self {
                Self::DynamicalConformal => 1.0 / axis_quartic_ruler(x).value,
                Self::DistanceGraph | Self::DistanceConformal => {
                    let rho = x.hypot(FLOOR);
                    let graph = self.lambda_sq() * (x / (rho * std::f64::consts::LN_2)).powi(2);
                    (1.0 + graph).sqrt() / rho
                }
            }
        }
    }

    fn tangent_distance([x, _]: [f64; 2]) -> f64 {
        x - ATTACHMENT
    }

    fn effective_scalar([x, y]: [f64; 2]) -> f64 {
        let c = Complex64::new(x, y);
        let s = (1.0 - 4.0 * c).sqrt();
        let cardioid = (1.0 - (1.0 - s).norm_sqr()) * s.norm() / 4.0;
        let bulb = 0.125 - 2.0 * (c + 1.0).norm_sqr();
        // In this local chart the two positive sets are disjoint.
        let r = cardioid.max(bulb).max(0.0);
        let r2 = r * r;
        -r2 / (r2.hypot(FLOOR * FLOOR) + FLOOR * FLOOR).sqrt()
    }

    fn prepared_field(scalar: fn([f64; 2]) -> f64) -> Result<PreparedField, String> {
        let [nx, ny] = SHAPE;
        let mut coefficients = Vec::with_capacity(nx * ny);
        for j in 0..ny {
            for i in 0..nx {
                coefficients.push(scalar([
                    ORIGIN[0] + GRID * i as f64,
                    ORIGIN[1] + GRID * j as f64,
                ]));
            }
        }
        PreparedField::new(ORIGIN, GRID, SHAPE, coefficients).map_err(str::to_owned)
    }

    #[derive(Default, Serialize)]
    struct AxisJet {
        value: f64,
        first: f64,
        second: f64,
    }

    fn axis_quartic_ruler(x: f64) -> AxisJet {
        let (r, first, second) = if x < 0.0 {
            (-x - 2.0 * x * x, -1.0 - 4.0 * x, -4.0)
        } else {
            let s = (1.0 - x).sqrt();
            (2.0 * s * s * x / (1.0 + s), -2.0 + 3.0 * s, -1.5 / s)
        };
        let value = (r.powi(4) + FLOOR.powi(4)).sqrt().sqrt();
        let d = r.powi(3) * first / value.powi(3);
        AxisJet {
            value,
            first: d,
            second: (3.0 * r * r * first * first + r.powi(3) * second) / value.powi(3)
                - 3.0 * d * d / value,
        }
    }

    fn ruler_check(field: &PreparedField) -> Result<Value, String> {
        let mut error = AxisJet::default();
        let mut points = 0;
        for i in -576..=576 {
            let x = f64::from(i) * GRID / 3.0;
            let sampled = field
                .query([ATTACHMENT + x, 0.0])
                .map_err(|_| "candidate axis jet unavailable")?;
            let expected = axis_quartic_ruler(x);
            let d = sampled.value;
            let rho = d.hypot(FLOOR);
            let first = d * sampled.gradient[0] / rho;
            let second = d * sampled.hessian[0][0] / rho
                + FLOOR.powi(2) * sampled.gradient[0].powi(2) / rho.powi(3);
            error.value = error.value.max((rho - expected.value).abs());
            error.first = error.first.max((first - expected.first).abs());
            error.second = error.second.max((second - expected.second).abs());
            points += 1;
        }
        if error.value >= 2e-6 || error.first >= 1e-3 || error.second >= 0.5 {
            return Err(format!(
                "prepared physical ruler discrepancy: {}",
                json!(error)
            ));
        }
        Ok(json!({"points": points, "max_errors": error}))
    }

    #[test]
    fn effective_encoding_preserves_exact_floor_and_axis_reference() -> Result<(), String> {
        assert_eq!(effective_scalar([ATTACHMENT, 0.0]), 0.0);
        let attached = axis_quartic_ruler(0.0);
        assert_eq!(
            (attached.value, attached.first, attached.second),
            (FLOOR, 0.0, 0.0)
        );
        for i in -128..=128 {
            let x = f64::from(i) * GRID;
            let encoded = effective_scalar([ATTACHMENT + x, 0.0]).hypot(FLOOR);
            assert!((encoded - axis_quartic_ruler(x).value).abs() <= 2e-15);
        }
        ruler_check(&prepared_field(effective_scalar)?)?;
        Ok(())
    }

    fn axis_arc(profile: Profile, from: f64, to: f64, panels: u32) -> f64 {
        if from < 0.0 && to > 0.0 {
            return axis_arc(profile, from, 0.0, panels) + axis_arc(profile, 0.0, to, panels);
        }
        let step = (to - from) / f64::from(panels);
        let interior: f64 = (1..panels)
            .map(|i| {
                let weight = if i % 2 == 0 { 2.0 } else { 4.0 };
                weight * profile.axis_metric_sqrt(from + f64::from(i) * step)
            })
            .sum();
        step * (profile.axis_metric_sqrt(from) + interior + profile.axis_metric_sqrt(to)) / 3.0
    }

    #[derive(Serialize)]
    struct AxisPrediction {
        initial_intrinsic_speed: f64,
        crossing_intrinsic_speed: f64,
        crossing_coordinate_speed: f64,
        crossing_time: f64,
        reference_end: f64,
    }

    fn predicted_axis(profile: Profile, panels: u32) -> Result<AxisPrediction, String> {
        let start = START[0] - ATTACHMENT;
        let v = VELOCITY[0] * profile.axis_metric_sqrt(start);
        let crossing_length = axis_arc(profile, start, 0.0, panels);
        let crossing_intrinsic_speed = v - BETA * crossing_length;
        let duration = STEP * f64::from(STEPS);
        let traveled = -v * (-BETA * duration).exp_m1() / BETA;
        if !(crossing_intrinsic_speed > 0.0 && traveled > crossing_length) {
            return Err("analytic prediction does not reach the attachment".into());
        }
        let mut lo = 0.0;
        let mut hi = 4.0 * FLOOR;
        if axis_arc(profile, start, hi, panels) < traveled {
            return Err("analytic endpoint has no local bracket".into());
        }
        for _ in 0..48 {
            let mid = (lo + hi) / 2.0;
            if axis_arc(profile, start, mid, panels) < traveled {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        Ok(AxisPrediction {
            initial_intrinsic_speed: v,
            crossing_intrinsic_speed,
            crossing_coordinate_speed: crossing_intrinsic_speed * FLOOR,
            crossing_time: -(-BETA * crossing_length / v).ln_1p() / BETA,
            reference_end: ATTACHMENT + (lo + hi) / 2.0,
        })
    }

    fn physics(profile: Profile, dt: f64) -> Result<PhysicsProfile, String> {
        let config = ManifoldConfig {
            epsilon: FLOOR,
            lambda_sq: profile.lambda_sq(),
            kappa: 0.0,
            mu: 0.0,
            ..ManifoldConfig::default()
        };
        PhysicsProfile::new(config, dt, BETA, Integrator::Rk4FourSubsteps)
    }

    #[derive(Serialize)]
    struct Crossing {
        time: f64,
        coordinate_speed: f64,
        intrinsic_speed: f64,
    }

    #[derive(Serialize)]
    struct Replay {
        profile: Value,
        crossing: Crossing,
        max_abs_transverse_position: f64,
        max_energy_work_residual: f64,
        geometry_evaluations: usize,
        samples: Vec<MechanicalState>,
    }

    fn replay(
        field: Arc<PreparedField>,
        profile: Profile,
        refinement: u32,
    ) -> Result<Replay, String> {
        let dt = STEP / f64::from(refinement);
        // The empty supplied cut list is only for the analytic fixture. The
        // engine still applies its canonical refusals and hard disk guard.
        let rollout = FrozenRollout::new(field, physics(profile, dt)?, Vec::new())?;
        let attachment_probe = rollout.probe([ATTACHMENT, 0.0])?;
        let attachment_metric_sqrt = attachment_probe["metric"][0][0]
            .as_f64()
            .ok_or("missing attachment metric")?
            .sqrt();
        let mut state = rollout.initial_state(START, VELOCITY)?;
        let initial_energy = state.energy;
        let mut samples = vec![state];
        let mut crossing = None;
        let mut max_abs_transverse_position: f64 = 0.0;
        let mut max_energy_work_residual: f64 = 0.0;
        let mut geometry_evaluations = 0;
        for _ in 0..STEPS {
            for _ in 0..refinement {
                let previous = state;
                let work = rollout.attempt_step(&mut state, HOLD)?;
                geometry_evaluations += work.geometry_evaluations;
                max_energy_work_residual = max_energy_work_residual
                    .max((state.energy - initial_energy - state.integrated_work).abs());
                max_abs_transverse_position =
                    max_abs_transverse_position.max(state.position[1].abs());
                if previous.position[0] < ATTACHMENT && state.position[0] >= ATTACHMENT {
                    if crossing.is_some() {
                        return Err("more than one attachment crossing".into());
                    }
                    let t = (ATTACHMENT - previous.position[0])
                        / (state.position[0] - previous.position[0]);
                    let speed =
                        previous.velocity[0] + t * (state.velocity[0] - previous.velocity[0]);
                    crossing = Some(Crossing {
                        time: previous.mechanical_time + t * dt,
                        coordinate_speed: speed,
                        intrinsic_speed: speed * attachment_metric_sqrt,
                    });
                }
            }
            samples.push(state);
        }
        if max_abs_transverse_position >= 1e-8 || max_energy_work_residual >= 1e-5 {
            return Err(format!("axis or ledger discrepancy: {max_abs_transverse_position:e}, {max_energy_work_residual:e}"));
        }
        Ok(Replay {
            profile: rollout.profile().manifest(),
            crossing: crossing.ok_or("no attachment crossing")?,
            max_abs_transverse_position,
            max_energy_work_residual,
            geometry_evaluations,
            samples,
        })
    }

    fn compare(field: Arc<PreparedField>, profile: Profile) -> Result<Value, String> {
        let reference_coarse = predicted_axis(profile, 256)?;
        let prediction = predicted_axis(profile, 512)?;
        let reference_time_change =
            (prediction.crossing_time - reference_coarse.crossing_time).abs();
        let reference_end_change =
            (prediction.reference_end - reference_coarse.reference_end).abs();
        if reference_time_change >= 1e-9 || reference_end_change >= 1e-9 {
            return Err("analytic reference quadrature did not converge on the fixture".into());
        }
        let coarse = replay(Arc::clone(&field), profile, 1)?;
        let fine = replay(field, profile, 2)?;
        let mut position_error: f64 = 0.0;
        let mut velocity_error: f64 = 0.0;
        for (a, b) in coarse.samples.iter().zip(&fine.samples) {
            position_error = position_error
                .max((a.position[0] - b.position[0]).hypot(a.position[1] - b.position[1]));
            velocity_error = velocity_error
                .max((a.velocity[0] - b.velocity[0]).hypot(a.velocity[1] - b.velocity[1]));
        }
        let end = fine.samples.last().ok_or("missing final state")?;
        let endpoint_error = (end.position[0] - prediction.reference_end).abs();
        let crossing_difference = (fine.crossing.time - coarse.crossing.time).abs();
        let predicted_time_error = (fine.crossing.time - prediction.crossing_time).abs();
        let predicted_speed_error =
            (fine.crossing.coordinate_speed - prediction.crossing_coordinate_speed).abs();
        if position_error >= 1e-6
            || velocity_error >= 1e-6
            || crossing_difference >= 1e-5
            || endpoint_error >= 1e-5
            || predicted_time_error >= 1e-4
            || predicted_speed_error >= 1e-5
        {
            return Err(format!(
                "refinement/analytic disagreement: {position_error:e}, {velocity_error:e}, {crossing_difference:e}, {endpoint_error:e}, {predicted_time_error:e}, {predicted_speed_error:e}"
            ));
        }
        Ok(json!({
            "profile": profile.name(),
            "prediction": prediction,
            "reference_quadrature_crossing_time_change": reference_time_change,
            "reference_quadrature_endpoint_change": reference_end_change,
            "max_step_halving_position_difference": position_error,
            "max_step_halving_velocity_difference": velocity_error,
            "crossing_time_step_halving_difference": crossing_difference,
            "analytic_endpoint_discrepancy": endpoint_error,
            "analytic_crossing_time_discrepancy": predicted_time_error,
            "analytic_crossing_speed_discrepancy": predicted_speed_error,
            "coarse": coarse,
            "fine": fine
        }))
    }

    fn archive_probe(args: &[String]) -> Result<Value, String> {
        match args {
            [] => Ok(Value::Null),
            [flag, asset, policy] if flag == "--probe-retained" => {
                let bytes = std::fs::read(asset).map_err(|e| e.to_string())?;
                let field = Arc::new(PreparedField::from_asset_bytes(&bytes)?);
                if field.identity() != RETAINED_FIELD_ID {
                    return Err("asset does not match the retained field ID".into());
                }
                let policy = std::fs::read_to_string(policy).map_err(|e| e.to_string())?;
                let cuts = reference_policy_for_field(&field, &policy)?;
                let frame =
                    FrozenRollout::new(field, physics(Profile::DistanceConformal, STEP)?, cuts)?;
                Ok(match frame.probe([ATTACHMENT, 0.0]) {
                    Ok(probe) => json!({"status": "sampled", "probe": probe}),
                    Err(reason) => json!({"status": "refused", "reason": reason}),
                })
            }
            _ => Err("usage: attachment_ruler_crossing [--probe-retained ASSET POLICY]".into()),
        }
    }

    pub fn run() -> Result<(), String> {
        let args = std::env::args().skip(1).collect::<Vec<_>>();
        let archive = archive_probe(&args)?;
        let distance = Arc::new(prepared_field(tangent_distance)?);
        let dynamical = Arc::new(prepared_field(effective_scalar)?);
        let ruler = ruler_check(&dynamical)?;
        let profiles = [
            compare(Arc::clone(&distance), Profile::DistanceGraph)
                .map_err(|e| format!("distance graph: {e}"))?,
            compare(distance, Profile::DistanceConformal)
                .map_err(|e| format!("distance conformal: {e}"))?,
            compare(dynamical, Profile::DynamicalConformal)
                .map_err(|e| format!("dynamical conformal: {e}"))?,
        ];
        println!(
            "{}",
            json!({
                "fixture": "analytic attachment; synthetic field policy",
                "retained_attachment_probe": archive,
                "attachment": [ATTACHMENT, 0.0],
                "floor": FLOOR,
                "grid_spacing": GRID,
                "initial_position": START,
                "initial_velocity": VELOCITY,
                "dt": STEP,
                "outer_steps": STEPS,
                "held_covector": HOLD,
                "axis_ruler_validation": ruler,
                "profiles": profiles
            })
        );
        Ok(())
    }
}
