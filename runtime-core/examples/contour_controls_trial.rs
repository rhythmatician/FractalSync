//! Short matched contour trial using FrozenRollout and Controls v2 drive.
#[cfg(not(feature = "experimental-operational-field"))]
fn main() {
    eprintln!("Enable experimental-operational-field to run this example.");
    std::process::exit(2);
}

#[cfg(feature = "experimental-operational-field")]
fn main() -> Result<(), String> {
    trial::run()
}

#[cfg(feature = "experimental-operational-field")]
mod trial {
    use num_complex::Complex64;
    use runtime_core::controls::{MotionControls, MAX_DRIVE_FORCE};
    use runtime_core::manifold::ManifoldConfig;
    use runtime_core::operational_engine::{
        reference_policy_for_field, FrozenRollout, Integrator, PhysicsProfile,
    };
    use runtime_core::operational_field::PreparedField;
    use serde_json::{json, Value};
    use std::sync::Arc;

    const FIELD_ID: u64 = 0x9721bb7647a0b8e6;
    const SPEED: f64 = 0.3;
    const DT: f64 = 0.01;
    const STEPS: usize = 60;
    const RESPONSE: f64 = 8.0;
    type Metric = [[f64; 2]; 2];

    #[derive(Clone, Copy)]
    enum Contour {
        Multiplier,
        MapScale,
    }

    #[derive(Clone, Copy)]
    enum Chart {
        Cardioid,
        PeriodTwo,
    }

    impl Chart {
        fn start(self) -> [f64; 2] {
            match self {
                Self::Cardioid => [-0.45, 0.1],
                Self::PeriodTwo => [-0.9, 0.1],
            }
        }

        fn multiplier(self, point: [f64; 2]) -> Result<(f64, [f64; 2]), String> {
            match self {
                Self::Cardioid => cardioid_multiplier(point),
                Self::PeriodTwo => {
                    let [x, y] = point;
                    let dx = x + 1.0;
                    let level = 16.0 * (dx * dx + y * y);
                    if level.is_finite() && level < 1.0 {
                        Ok((level, [32.0 * dx, 32.0 * y]))
                    } else {
                        Err("outside period-two multiplier chart".into())
                    }
                }
            }
        }
    }

    fn cardioid_multiplier([x, y]: [f64; 2]) -> Result<(f64, [f64; 2]), String> {
        let s = (1.0 - 4.0 * Complex64::new(x, y)).sqrt();
        let u = 1.0 - s;
        let square = u.norm_sqr();
        if !(square < 1.0 && s.re > 0.0) {
            return Err("outside main-cardioid multiplier chart".into());
        }
        let p = 2.0 * u.conj() / s;
        Ok((square, [2.0 * p.re, -2.0 * p.im]))
    }

    fn map_jet(field: &PreparedField, position: [f64; 2]) -> Result<(f64, [f64; 2]), String> {
        let jet = field
            .query(position)
            .map_err(|_| "no complete Map stencil")?;
        if jet.value >= 0.0 {
            return Err("outside negative Map comparison region".into());
        }
        // On D<0, sigma=log2(d_ref/sqrt(D²+eps²)) increases with D.
        // The D and sigma level sets have the same oriented tangent there.
        Ok((jet.value, jet.gradient))
    }

    fn norm(metric: Metric, [x, y]: [f64; 2]) -> f64 {
        (x * (metric[0][0] * x + metric[0][1] * y) + y * (metric[1][0] * x + metric[1][1] * y))
            .sqrt()
    }

    fn tangent(gradient: [f64; 2], metric: Metric) -> Result<[f64; 2], String> {
        let vector = [gradient[1], -gradient[0]];
        let magnitude = norm(metric, vector);
        if !magnitude.is_finite() || magnitude <= 1e-8 {
            return Err("unresolved contour tangent".into());
        }
        // Both fixture gradients give an upward tangent after this choice.
        let direction = if vector[1] > 0.0 { 1.0 } else { -1.0 };
        Ok(vector.map(|x| direction * x / magnitude))
    }

    fn metric_at(rollout: &FrozenRollout, p: [f64; 2]) -> Result<Metric, String> {
        serde_json::from_value(rollout.probe(p)?["metric"].clone()).map_err(|e| e.to_string())
    }

    fn feedback(metric: Metric, tangent: [f64; 2], velocity: [f64; 2]) -> MotionControls {
        let delta = [
            SPEED * tangent[0] - velocity[0],
            SPEED * tangent[1] - velocity[1],
        ];
        MotionControls {
            direction: delta,
            throttle: (RESPONSE * norm(metric, delta) / MAX_DRIVE_FORCE).clamp(0.0, 1.0),
            brake: 0.0,
            grip: 0.0,
            impulse: 0.0,
        }
    }

    fn compare(
        rollout: &FrozenRollout,
        field: &PreparedField,
        chart: Chart,
        contour: Contour,
    ) -> Result<Value, String> {
        let start = chart.start();
        let (multiplier_level, _) = chart.multiplier(start)?;
        let (map_level, _) = map_jet(field, start)?;
        let target_tangent = |position, metric| {
            let (_, gradient) = match contour {
                Contour::Multiplier => chart.multiplier(position)?,
                Contour::MapScale => map_jet(field, position)?,
            };
            tangent(gradient, metric)
        };
        let start_metric = metric_at(rollout, start)?;
        let start_velocity = target_tangent(start, start_metric)?.map(|x| SPEED * x);
        let equal_start_speed = norm(start_metric, start_velocity);
        assert!((equal_start_speed - SPEED).abs() < 1e-12);
        let mut state = rollout.initial_state(start, start_velocity)?;
        let mut peak_throttle: f64 = 0.0;
        let mut peak_speed_error: f64 = 0.0;
        let mut peak_multiplier_error: f64 = 0.0;
        let mut peak_map_error: f64 = 0.0;
        let mut frames = vec![state];
        for _ in 0..STEPS {
            let metric = metric_at(rollout, state.position)?;
            let direction = target_tangent(state.position, metric)?;
            let controls = feedback(metric, direction, state.velocity);
            peak_throttle = peak_throttle.max(controls.throttle);
            let covector = rollout.drive_covector(state.position, &controls)?;
            rollout.attempt_step(&mut state, covector)?;
            let (mul_square, _) = chart.multiplier(state.position)?;
            let (map, _) = map_jet(field, state.position)?;
            peak_multiplier_error =
                peak_multiplier_error.max((mul_square - multiplier_level).abs());
            peak_map_error = peak_map_error.max((map - map_level).abs());
            let speed = norm(metric_at(rollout, state.position)?, state.velocity);
            peak_speed_error = peak_speed_error.max((speed - SPEED).abs());
            frames.push(state);
        }
        assert!(state.position[1] > start[1] + 0.005);
        assert!(peak_speed_error < 0.1 && peak_throttle < 1.0);
        Ok(json!({
            "initial_intrinsic_speed": equal_start_speed,
            "initial_velocity": start_velocity,
            "peak_throttle": peak_throttle,
            "peak_intrinsic_speed_error": peak_speed_error,
            "peak_multiplier_magnitude_squared_error": peak_multiplier_error,
            "peak_map_scalar_error": peak_map_error,
            "frames": frames,
        }))
    }

    pub fn run() -> Result<(), String> {
        let args = std::env::args().skip(1).collect::<Vec<_>>();
        let [asset, policy] = args.as_slice() else {
            return Err("usage: contour_controls_trial ASSET POLICY".into());
        };
        let bytes = std::fs::read(asset).map_err(|e| e.to_string())?;
        let field = Arc::new(PreparedField::from_asset_bytes(&bytes)?);
        if field.identity() != FIELD_ID {
            return Err("asset does not match the retained field ID".into());
        }
        let policy = std::fs::read_to_string(policy).map_err(|e| e.to_string())?;
        let cuts = reference_policy_for_field(&field, &policy)?;
        let config = ManifoldConfig {
            kappa: 0.0,
            mu: 0.0,
            ..ManifoldConfig::default()
        };
        let neutral = feedback([[1.0, 0.0], [0.0, 1.0]], [0.0, 1.0], [0.0, SPEED]);
        let profile = PhysicsProfile::new(
            config,
            DT,
            neutral.friction_beta(),
            Integrator::Rk4FourSubsteps,
        )?;
        let rollout = FrozenRollout::new(Arc::clone(&field), profile, cuts)?;
        let cardioid = Chart::Cardioid;
        let bulb = Chart::PeriodTwo;
        let natural = compare(&rollout, &field, cardioid, Contour::Multiplier)?;
        let map = compare(&rollout, &field, cardioid, Contour::MapScale)?;
        let bulb_natural = compare(&rollout, &field, bulb, Contour::Multiplier)?;
        let bulb_map = compare(&rollout, &field, bulb, Contour::MapScale)?;
        let identity = [[1.0, 0.0], [0.0, 1.0]];
        assert!(cardioid.multiplier([0.5, 0.5]).is_err());
        assert!(bulb.multiplier([-0.75, 0.0]).is_err());
        for (chart, center) in [(cardioid, [0.0, 0.0]), (bulb, [-1.0, 0.0])] {
            let (_, critical) = chart.multiplier(center)?;
            assert!(tangent(critical, identity).is_err());
        }
        assert!(rollout
            .probe([-1.0, 0.0])
            .expect_err("archive cut must refuse")
            .contains("ReferenceSingularity"));
        println!(
            "{}",
            json!({
                "profile": rollout.profile().manifest(),
                "natural_multiplier": natural,
                "map_iso_scale": map,
            "period_two": {
                "natural_multiplier": bulb_natural,
                "map_iso_scale": bulb_map,
            },
            })
        );
        Ok(())
    }
}
