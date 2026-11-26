//! Compare the frozen RK work accumulator with an affine analytic fixture.
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
    use runtime_core::manifold::ManifoldConfig;
    use runtime_core::operational_engine::{
        FrozenRollout, Integrator, MechanicalState, PhysicsProfile, ReferenceCut,
    };
    use runtime_core::operational_field::PreparedField;
    use serde_json::{json, Value};
    use std::sync::Arc;

    const FLOOR: f64 = 0.5;
    const METRIC: f64 = 4.0;
    const DRAG: f64 = 1.5;
    const DRIVE: f64 = 1.0;
    const START: f64 = 0.3;
    const SPEED: f64 = 0.3;
    const DURATION: f64 = 0.8;

    #[derive(Default)]
    struct Increment {
        x: f64,
        v: f64,
        drive: f64,
        drag: f64,
    }

    fn weighted([first, second, third, fourth]: [f64; 4], h: f64) -> f64 {
        h * (first + 2.0 * (second + third) + fourth) / 6.0
    }

    fn stage_sum(v: f64, drive: f64, h: f64) -> Increment {
        // For this affine ODE the four stage velocities are polynomials in
        // h and the initial acceleration. Evaluate them without the engine.
        let a = drive / METRIC - DRAG * v;
        let b = DRAG * h / 2.0;
        let half = h * a / 2.0;
        let stages = [
            v,
            v + half,
            v + half * (1.0 - b),
            v + h * a * (1.0 - b + b * b),
        ];
        Increment {
            x: weighted(stages, h),
            v: weighted(stages.map(|stage| drive / METRIC - DRAG * stage), h),
            drive: drive * weighted(stages, h),
            drag: -DRAG * METRIC * weighted(stages.map(|stage| stage * stage), h),
        }
    }

    fn exact_at(t: f64) -> (f64, f64, f64, f64) {
        let terminal = DRIVE / (METRIC * DRAG);
        let offset = SPEED - terminal;
        let e1 = -(-DRAG * t).exp_m1() / DRAG;
        let e2 = -(-2.0 * DRAG * t).exp_m1() / (2.0 * DRAG);
        let displacement = terminal * t + offset * e1;
        let drive = DRIVE * displacement;
        let drag = -DRAG
            * METRIC
            * (terminal * terminal * t + 2.0 * terminal * offset * e1 + offset * offset * e2);
        (
            START + displacement,
            terminal + offset * (-DRAG * t).exp(),
            drive,
            drag,
        )
    }

    fn rollout(
        method: Integrator,
        dt: f64,
        cuts: Vec<ReferenceCut>,
    ) -> Result<FrozenRollout, String> {
        // Zero stored coefficients give exactly zero jet and constant G=4I.
        let field = PreparedField::new([0.0, -0.25], 1.0 / 64.0, [65, 33], vec![0.0; 65 * 33])?;
        let config = ManifoldConfig {
            epsilon: FLOOR,
            lambda_sq: 0.0,
            kappa: 0.0,
            mu: 0.0,
            ..ManifoldConfig::default()
        };
        FrozenRollout::new(
            Arc::new(field),
            PhysicsProfile::new(config, dt, DRAG, method)?,
            cuts,
        )
    }

    fn initial(rollout: &FrozenRollout) -> Result<MechanicalState, String> {
        rollout.initial_state([START, 0.0], [SPEED, 0.0])
    }

    fn run_mesh(method: Integrator, substeps: u32, steps: u32) -> Result<Value, String> {
        let dt = DURATION / f64::from(steps);
        let h = dt / f64::from(substeps);
        let frame = rollout(method, dt, vec![])?;
        let mut state = initial(&frame)?;
        let initial_energy = state.energy;
        let (mut x, mut v) = (START, SPEED);
        let (mut drive, mut drag) = (0.0, 0.0);
        let mut max_work_discrepancy: f64 = 0.0;
        let mut max_energy_discrepancy: f64 = 0.0;
        let mut max_residual: f64 = 0.0;
        let mut max_integral_discrepancy: f64 = 0.0;
        let mut max_reference_state_discrepancy: f64 = 0.0;
        for step in 1..=steps {
            for _ in 0..substeps {
                let inc = stage_sum(v, DRIVE, h);
                assert!(inc.drive > 0.0 && inc.drag < 0.0);
                x += inc.x;
                v += inc.v;
                drive += inc.drive;
                drag += inc.drag;
            }
            frame.attempt_step(&mut state, [DRIVE, 0.0])?;
            assert!((state.position[0] - x).abs() < 2e-14);
            assert!((state.velocity[0] - v).abs() < 2e-14);
            assert_eq!(state.position[1], 0.0);
            max_work_discrepancy =
                max_work_discrepancy.max((state.integrated_work - (drive + drag)).abs());
            max_energy_discrepancy =
                max_energy_discrepancy.max((state.energy - METRIC * v * v / 2.0).abs());
            max_residual =
                max_residual.max((state.energy - initial_energy - state.integrated_work).abs());
            let (xr, vr, wr_drive, wr_drag) = exact_at(f64::from(step) * dt);
            max_integral_discrepancy =
                max_integral_discrepancy.max((state.integrated_work - (wr_drive + wr_drag)).abs());
            max_reference_state_discrepancy = max_reference_state_discrepancy
                .max((state.position[0] - xr).hypot(state.velocity[0] - vr));
        }
        assert!(max_work_discrepancy < 2e-14 && max_energy_discrepancy < 2e-14);
        let (_, _, exact_drive, exact_drag) = exact_at(DURATION);
        Ok(json!({
            "outer_steps": steps,
            "substeps": substeps,
            "stage_drive": drive,
            "stage_drag": drag,
            "exact_drive": exact_drive,
            "exact_drag": exact_drag,
            "integrated_work": state.integrated_work,
            "max_stage_work_discrepancy": max_work_discrepancy,
            "max_endpoint_energy_discrepancy": max_energy_discrepancy,
            "max_ledger_residual": max_residual,
            "max_exact_integral_discrepancy": max_integral_discrepancy,
            "max_exact_state_discrepancy": max_reference_state_discrepancy,
        }))
    }

    fn refusal(method: Integrator, substeps: u32) -> Result<Value, String> {
        // Place a synthetic cut around the first weighted endpoint, away
        // from the fourth stage. Four-step RK reaches it internally.
        let dt = 0.4;
        let h = dt / f64::from(substeps);
        let weighted_endpoint = START + stage_sum(SPEED, DRIVE, h).x;
        let cuts = vec![ReferenceCut {
            bounds: [
                weighted_endpoint - 1e-7,
                -1e-6,
                weighted_endpoint + 1e-7,
                1e-6,
            ],
            authority: "synthetic weighted-endpoint exclusion".to_owned(),
        }];
        let frame = rollout(method, dt, cuts)?;
        let mut state = initial(&frame)?;
        state.integrated_work = 0.125;
        state.mechanical_time = 0.75;
        let before = serde_json::to_vec(&state).map_err(|e| e.to_string())?;
        let reason = frame
            .attempt_step(&mut state, [DRIVE, 0.0])
            .expect_err("weighted endpoint must refuse");
        let phase = if substeps == 1 {
            "Final endpoint"
        } else {
            "RK substep 1 endpoint"
        };
        assert!(
            reason.contains(phase) && reason.contains("ReferenceSingularity"),
            "{reason}"
        );
        assert_eq!(
            serde_json::to_vec(&state).map_err(|e| e.to_string())?,
            before
        );
        frame.attempt_step(&mut state, [-DRIVE, 0.0])?;
        let (mut velocity, mut work) = (SPEED, 0.125);
        for _ in 0..substeps {
            let inc = stage_sum(velocity, -DRIVE, h);
            velocity += inc.v;
            work += inc.drive + inc.drag;
        }
        assert!((state.integrated_work - work).abs() < 2e-14);
        Ok(json!({"reason": reason, "retry_integrated_work": state.integrated_work}))
    }

    pub fn run() -> Result<(), String> {
        let mut report = Vec::new();
        for (method, substeps) in [(Integrator::Rk4, 1), (Integrator::Rk4FourSubsteps, 4)] {
            let coarse = run_mesh(method, substeps, 4)?;
            let fine = run_mesh(method, substeps, 8)?;
            let error = |run: &Value| run["max_exact_integral_discrepancy"].as_f64().unwrap();
            assert!(error(&fine) < 0.2 * error(&coarse));
            report.push(json!({
                "coarse": coarse,
                "fine": fine,
                "refusal": refusal(method, substeps)?,
            }));
        }
        println!("{}", json!(report));
        Ok(())
    }
}
