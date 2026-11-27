//! Research-only local ruler comparison using the existing frozen Physics engine.
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
    use num_complex::Complex64;
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
    const EXTERIOR_STAGE: usize = 8;
    const EXTERIOR_SCALAR_LIMIT: f64 = 1e-7;
    const REFERENCE_DELTA: f64 = 0.0002;

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Region {
        PeriodTwo,
        Cardioid,
        Exterior,
    }

    impl Region {
        fn name(self) -> &'static str {
            match self {
                Self::PeriodTwo => "period-two",
                Self::Cardioid => "cardioid",
                Self::Exterior => "exterior",
            }
        }

        fn start(self) -> [f64; 2] {
            match self {
                Self::PeriodTwo => START,
                Self::Cardioid => [-0.45, 0.1],
                Self::Exterior => [0.5, 0.5],
            }
        }

        fn spacing(self) -> f64 {
            match self {
                Self::PeriodTwo => SPACING,
                Self::Cardioid | Self::Exterior => 1.0 / 1024.0,
            }
        }

        fn origin(self) -> [f64; 2] {
            match self {
                Self::PeriodTwo => ORIGIN,
                Self::Cardioid | Self::Exterior => self.start().map(|x| x - 24.0 * self.spacing()),
            }
        }

        fn point(self, x: f64, y: f64) -> [f64; 2] {
            let origin = self.origin();
            [
                origin[0] + self.spacing() * x,
                origin[1] + self.spacing() * y,
            ]
        }

        fn signed_ruler(self, point: [f64; 2]) -> Result<f64, String> {
            let [x, y] = point;
            let c = Complex64::new(x, y);
            if !c.norm_sqr().is_finite() || c.norm_sqr() >= 4.0 {
                return Err("sample outside the Physics disk".into());
            }
            match self {
                Self::PeriodTwo if circle_distance(point) < 0.0 => Ok(dynamical_value(point)),
                Self::PeriodTwo => Err("sample outside the period-two interior".into()),
                Self::Cardioid => cardioid_ruler(c),
                Self::Exterior => {
                    let sample = exterior_sample(c, EXTERIOR_STAGE)?;
                    if sample.tail_budget >= EXTERIOR_SCALAR_LIMIT {
                        return Err(
                            "exterior scalar tail diagnostic exceeds the local budget".into()
                        );
                    }
                    Ok(sample.value)
                }
            }
        }

        fn probe_points(self) -> impl Iterator<Item = [f64; 2]> {
            let offsets = [9.0 - 1e-5, 9.0, 9.0 + 1e-5, 19.25, 41.75];
            offsets
                .into_iter()
                .flat_map(move |y| offsets.into_iter().map(move |x| self.point(x, y)))
                .chain([self.start()])
        }
    }

    fn cardioid_ruler(c: Complex64) -> Result<f64, String> {
        let s = (1.0 - 4.0 * c).sqrt();
        let multiplier = 1.0 - s;
        let margin = 1.0 - multiplier.norm_sqr();
        if !(margin > 0.0 && s.re > 0.0) {
            return Err("sample outside the attracting main-cardioid chart".into());
        }
        Ok(-margin * s.norm() / 4.0)
    }

    struct ExteriorSample {
        value: f64,
        tail_budget: f64,
    }

    fn exterior_sample(c: Complex64, stage: usize) -> Result<ExteriorSample, String> {
        // Keep a fixed stage across each grid. Choosing a stopping stage per
        // point could introduce joins into the function being differentiated.
        let (mut z, mut derivative) = (Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0));
        for _ in 0..stage {
            derivative = 2.0 * z * derivative + 1.0;
            z = z * z + c;
            if !z.norm_sqr().is_finite() || !derivative.norm_sqr().is_finite() {
                return Err("exterior orbit or derivative exceeded the finite range".into());
            }
        }
        let radius = z.norm();
        let q = c.norm() / radius.powi(2);
        let derivative_ratio = derivative.norm() / radius;
        if !(radius >= 4.0 && q <= 0.25 && derivative_ratio.is_finite()) {
            return Err("sample did not pass the exterior escape guard at the fixed stage".into());
        }
        let log_radius = radius.ln();
        // Exact-input tail formula with its common dyadic scaling removed.
        // Evaluations here are binary64 diagnostics, not outward enclosures.
        let accumulation = 1.5 * q;
        let green_tail = (12.0 / 17.0) * q;
        let derivative_tail = (accumulation * derivative_ratio + (12.0 / 17.0) / radius.powi(2))
            / (1.0 - accumulation);
        if !(derivative_ratio > derivative_tail) {
            return Err("exterior tail diagnostic cannot resolve a nonzero denominator".into());
        }
        let value = log_radius / derivative_ratio;
        let lower = (log_radius - green_tail).max(0.0) / (derivative_ratio + derivative_tail);
        let upper = (log_radius + green_tail) / (derivative_ratio - derivative_tail);
        if !upper.is_finite() {
            return Err("exterior tail diagnostic is nonfinite".into());
        }
        Ok(ExteriorSample {
            value,
            tail_budget: (value - lower).max(upper - value),
        })
    }

    #[test]
    fn prepared_quadratic_matches_analytic_jet_on_both_sides_of_joins() -> Result<(), String> {
        let region = Region::PeriodTwo;
        let validation = validate_quadratic(&dynamical_field(region)?)?;
        assert_eq!(validation.points, 26);
        assert!(validation.max_value_error < 1e-13);
        assert!(validation.max_gradient_error < 1e-11);
        assert!(validation.max_hessian_error < 1e-8);
        Ok(())
    }

    #[test]
    fn guarded_chart_values_match_reference_cases() -> Result<(), String> {
        assert_eq!(cardioid_ruler(Complex64::new(0.0, 0.0))?, -0.25);
        assert_eq!(cardioid_ruler(Complex64::new(-0.3125, 0.0))?, -0.28125);
        assert!(cardioid_ruler(Complex64::new(0.25, 0.0)).is_err());
        let exterior = exterior_sample(Complex64::new(0.5, 0.5), EXTERIOR_STAGE)?;
        assert!((exterior.value - 0.096156915).abs() < 1e-9);
        assert!(exterior.tail_budget < 2e-10);
        assert!(exterior_sample(Complex64::new(0.0, 0.0), EXTERIOR_STAGE).is_err());
        for region in [Region::Cardioid, Region::Exterior] {
            validate_regional_jet(&dynamical_field(region)?, region)?;
        }
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
        region: Region,
        value: impl Fn([f64; 2]) -> Result<f64, String>,
        correction: f64,
    ) -> Result<PreparedField, String> {
        let mut coefficients = Vec::with_capacity(SIDE * SIDE);
        for y in 0..SIDE {
            for x in 0..SIDE {
                let point = region.point(x as f64, y as f64);
                coefficients.push(
                    value(point).map_err(|error| format!("at {point:?}: {error}"))? - correction,
                );
            }
        }
        PreparedField::new(region.origin(), region.spacing(), [SIDE; 2], coefficients)
            .map_err(str::to_owned)
    }

    fn dynamical_field(region: Region) -> Result<PreparedField, String> {
        // Quintic B-spline sampling adds h² ΔD / 4 to a quadratic; ΔD = 8.
        let correction = if region == Region::PeriodTwo {
            2.0 * SPACING * SPACING
        } else {
            0.0
        };
        sample_field(region, |point| region.signed_ruler(point), correction)
    }

    #[derive(Default, Serialize)]
    struct JetError {
        points: usize,
        max_value_error: f64,
        max_gradient_error: f64,
        max_hessian_error: f64,
    }

    fn query(field: &PreparedField, point: [f64; 2]) -> Result<FieldJet, String> {
        field
            .query(point)
            .map_err(|_| format!("prepared jet unavailable at {point:?}"))
    }

    fn sampled_jet_error(
        region: Region,
        evaluated: impl Fn([f64; 2]) -> Result<FieldJet, String>,
        reference: impl Fn([f64; 2]) -> Result<FieldJet, String>,
    ) -> Result<JetError, String> {
        let mut max = JetError::default();
        for point in region.probe_points() {
            let jet = evaluated(point)?;
            let exact = reference(point)?;
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

    fn validate_quadratic(field: &PreparedField) -> Result<JetError, String> {
        let max = sampled_jet_error(
            Region::PeriodTwo,
            |point| query(field, point),
            |point| Ok(dynamical_jet(point)),
        )?;
        if max.max_value_error >= 1e-13
            || max.max_gradient_error >= 1e-11
            || max.max_hessian_error >= 1e-8
        {
            return Err(format!("quadratic jet check failed: {}", json!(max)));
        }
        Ok(max)
    }

    fn stencil_jet(
        [x, y]: [f64; 2],
        delta: f64,
        value: impl Fn([f64; 2]) -> Result<f64, String>,
    ) -> Result<FieldJet, String> {
        let at = |dx, dy| value([x + dx * delta, y + dy * delta]);
        let center = at(0.0, 0.0)?;
        let (xm, xp, ym, yp) = (at(-1.0, 0.0)?, at(1.0, 0.0)?, at(0.0, -1.0)?, at(0.0, 1.0)?);
        let mixed = (at(1.0, 1.0)? - at(1.0, -1.0)? - at(-1.0, 1.0)? + at(-1.0, -1.0)?)
            / (4.0 * delta * delta);
        Ok(FieldJet {
            value: center,
            gradient: [(xp - xm) / (2.0 * delta), (yp - ym) / (2.0 * delta)],
            hessian: [
                [(xp - 2.0 * center + xm) / delta.powi(2), mixed],
                [mixed, (yp - 2.0 * center + ym) / delta.powi(2)],
            ],
        })
    }

    fn validate_regional_jet(field: &PreparedField, region: Region) -> Result<Value, String> {
        if region == Region::PeriodTwo {
            return Ok(json!(validate_quadratic(field)?));
        }
        let scalar = |point| region.signed_ruler(point);
        let fine = |point| stencil_jet(point, REFERENCE_DELTA / 2.0, scalar);
        let coarse = |point| stencil_jet(point, REFERENCE_DELTA, scalar);
        let prepared_error = sampled_jet_error(region, |point| query(field, point), fine)?;
        let stencil_change = sampled_jet_error(region, coarse, fine)?;
        if prepared_error.max_value_error >= 1e-5
            || prepared_error.max_gradient_error >= 1e-4
            || prepared_error.max_hessian_error >= 1e-2
            || stencil_change.max_gradient_error >= 1e-5
            || stencil_change.max_hessian_error >= 1e-3
        {
            return Err(format!(
                "sampled jet tolerance exceeded: prepared={}, stencil={}",
                json!(prepared_error),
                json!(stencil_change)
            ));
        }
        let mut report = json!({
            "prepared_vs_scalar_stencil": prepared_error,
            "reference_stencil_step_change": stencil_change,
            "reference_delta": REFERENCE_DELTA / 2.0,
        });
        if region == Region::Exterior {
            let later_value = |[x, y]: [f64; 2]| {
                Ok(exterior_sample(Complex64::new(x, y), EXTERIOR_STAGE + 1)?.value)
            };
            let stage_change = sampled_jet_error(region, fine, |point| {
                stencil_jet(point, REFERENCE_DELTA / 2.0, later_value)
            })?;
            let mut tail_budget: f64 = 0.0;
            for [x, y] in region.probe_points() {
                let sample = exterior_sample(Complex64::new(x, y), EXTERIOR_STAGE)?;
                tail_budget = tail_budget.max(sample.tail_budget);
            }
            if stage_change.max_value_error >= EXTERIOR_SCALAR_LIMIT {
                return Err("exterior next-stage discrepancy exceeds the local budget".into());
            }
            report["exterior_stage"] = json!(EXTERIOR_STAGE);
            report["exterior_next_stage_change"] = json!(stage_change);
            report["exterior_scalar_tail_budget_diagnostic"] = json!(tail_budget);
        }
        Ok(report)
    }

    fn load_baseline(
        args: &[String],
    ) -> Result<(Arc<PreparedField>, Vec<ReferenceCut>, &'static str), String> {
        match args {
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
                "usage: dynamical_ruler_ablation [--circle-distance | --retained ASSET POLICY [--region period-two|cardioid|exterior]]"
                    .into(),
            ),
        }
    }

    fn circle_baseline() -> Result<(Arc<PreparedField>, Vec<ReferenceCut>, &'static str), String> {
        let region = Region::PeriodTwo;
        let field = sample_field(
            region,
            |point| {
                region.signed_ruler(point)?;
                Ok(circle_distance(point))
            },
            0.0,
        )?;
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

    fn replay(
        rollout: &FrozenRollout,
        substeps: usize,
        region: Region,
        local_field: &PreparedField,
    ) -> Result<Replay, String> {
        let mut state = rollout.initial_state(region.start(), VELOCITY)?;
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
                region.signed_ruler(state.position)?;
                query(local_field, state.position)?;
                replay.geometry_evaluations += work.geometry_evaluations;
                replay.coefficient_reads_upper_bound += work.coefficient_reads_upper_bound;
                replay.max_energy_work_residual = replay
                    .max_energy_work_residual
                    .max((state.energy - initial_energy - state.integrated_work).abs());
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
        region: Region,
        local_field: &PreparedField,
    ) -> Result<Value, String> {
        let coarse = replay(
            &make_rollout(field.clone(), cuts.clone(), lambda_sq, STEP)?,
            1,
            region,
            local_field,
        )
        .map_err(|error| format!("{name}, coarse: {error}"))?;
        let fine = replay(
            &make_rollout(field, cuts, lambda_sq, STEP / 2.0)?,
            2,
            region,
            local_field,
        )
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
        let args = std::env::args().skip(1).collect::<Vec<_>>();
        let (region, baseline_args) = match args.as_slice() {
            [base @ .., flag, name] if flag == "--region" => {
                let region = match name.as_str() {
                    "period-two" => Region::PeriodTwo,
                    "cardioid" => Region::Cardioid,
                    "exterior" => Region::Exterior,
                    _ => return Err(format!("unknown region: {name}")),
                };
                (region, base)
            }
            _ => (Region::PeriodTwo, args.as_slice()),
        };
        if region != Region::PeriodTwo
            && baseline_args.first().map(String::as_str) != Some("--retained")
        {
            return Err(
                "the cardioid and exterior cases require the retained distance asset".into(),
            );
        }
        let (distance, cuts, baseline) = load_baseline(baseline_args)?;
        let dynamical = Arc::new(dynamical_field(region)?);
        let jet_error = validate_regional_jet(&dynamical, region)?;
        let distance_error = if region == Region::PeriodTwo {
            let error = sampled_jet_error(
                region,
                |point| query(&distance, point),
                |point| Ok(circle_jet(point)),
            )?;
            if error.max_value_error >= BASELINE_DISTANCE_LIMIT {
                return Err(format!(
                    "baseline sampled distance discrepancy: {}",
                    json!(error)
                ));
            }
            Some(error)
        } else {
            for point in region.probe_points() {
                let value = query(&distance, point)?.value;
                if value * region.signed_ruler(point)? <= 0.0 {
                    return Err(format!(
                        "retained field sign disagrees with chart at {point:?}"
                    ));
                }
            }
            None
        };
        let profiles = [
            compare(
                distance.clone(),
                cuts.clone(),
                "distance graph",
                1.0,
                region,
                &dynamical,
            )?,
            compare(
                distance,
                cuts.clone(),
                "distance conformal",
                0.0,
                region,
                &dynamical,
            )?,
            compare(
                dynamical.clone(),
                cuts,
                "dynamical conformal",
                0.0,
                region,
                &dynamical,
            )?,
        ];
        println!(
            "{}",
            json!({
                "experiment": if region == Region::PeriodTwo {
                    "period-two interior ablation"
                } else {
                    "regional ruler ablation"
                },
                "region": region.name(),
                "baseline": baseline,
                "baseline_sampled_jet_discrepancy": distance_error,
                "baseline_sampled_distance_limit": BASELINE_DISTANCE_LIMIT,
                "dynamical_jet_validation": jet_error,
                "position_velocity_step_halving_limit": REFINEMENT_LIMIT,
                "energy_work_residual_limit": ENERGY_RESIDUAL_LIMIT,
                "start": region.start(),
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
