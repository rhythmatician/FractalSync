//! Offline #185 waypoint trial. Uses the retained provider and ordinary Controls inputs.
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
        reference_policy_for_field, FrozenRollout, Integrator, MechanicalState, PhysicsProfile,
    };
    use runtime_core::operational_field::PreparedField;
    use serde::Serialize;
    use serde_json::json;
    use std::{collections::BTreeMap, sync::Arc};

    const FIELD_ID: u64 = 0x9721bb7647a0b8e6;
    const SPEED: f64 = 0.3;
    const RESPONSE: f64 = 8.0;
    const DT: f64 = 0.01;
    const STEPS: usize = 120;
    const HOLD: usize = 6;
    const TRAP: Complex64 = Complex64::new(0.25, 0.125);
    type Point = [f64; 2];
    type Metric = [[f64; 2]; 2];

    #[derive(Clone, Copy, Serialize)]
    enum Chart {
        Cardioid,
        Bulb,
    }

    impl Chart {
        fn parameter(self, [x, y]: Point) -> Point {
            let u = Complex64::new(x, y);
            let c = match self {
                Self::Cardioid => u * (2.0 - u) / 4.0,
                Self::Bulb => -1.0 + u / 4.0,
            };
            [c.re, c.im]
        }

        fn multiplier_covector(self, [x, y]: Point) -> Option<Point> {
            let c = Complex64::new(x, y);
            let (u, derivative) = match self {
                Self::Cardioid => {
                    let root = (1.0 - 4.0 * c).sqrt();
                    if root.re <= 0.0 {
                        return None;
                    }
                    (1.0 - root, 2.0 / root)
                }
                Self::Bulb => (4.0 * (c + 1.0), Complex64::new(4.0, 0.0)),
            };
            (u.norm_sqr() < 1.0).then(|| squared_level_covector(u, derivative))
        }
    }

    fn squared_level_covector(value: Complex64, derivative: Complex64) -> Point {
        let gradient = 2.0 * value * derivative.conj();
        [gradient.re, gradient.im]
    }

    #[derive(Clone, Copy, Serialize)]
    enum Arm {
        Map,
        Multiplier,
        PointTrap,
    }

    #[derive(Clone, Copy, Serialize)]
    struct Settings {
        forecast: usize,
        throttle: f64,
    }

    #[derive(Clone, Copy, Serialize)]
    struct Route {
        id: &'static str,
        start: Point,
        goal: Point,
        launch: Point,
    }

    impl Route {
        fn in_chart(self, chart: Chart) -> Self {
            Self {
                start: chart.parameter(self.start),
                goal: chart.parameter(self.goal),
                ..self
            }
        }
    }

    // Declared in multiplier coordinates to offer similar route layouts in each chart.
    // The controller only sees the resulting Cartesian goal and its absolute miss.
    const ROUTES: [Route; 6] = [
        Route {
            id: "a",
            start: [-0.56, 0.24],
            goal: [-0.47, 0.40],
            launch: [-1.0, 0.0],
        },
        Route {
            id: "b",
            start: [0.27, 0.51],
            goal: [0.44, 0.45],
            launch: [0.0, 1.0],
        },
        Route {
            id: "c",
            start: [-0.48, -0.30],
            goal: [-0.29, -0.46],
            launch: [0.0, 1.0],
        },
        Route {
            id: "d",
            start: [0.54, -0.38],
            goal: [0.69, -0.20],
            launch: [0.0, -1.0],
        },
        Route {
            id: "e",
            start: [-0.36, 0.13],
            goal: [-0.34, -0.13],
            launch: [0.0, -1.0],
        },
        Route {
            id: "f",
            start: [0.18, 0.18],
            goal: [-0.18, -0.18],
            launch: [-1.0, -1.0],
        },
    ];

    #[derive(Default, Serialize)]
    struct Trace {
        frames: Vec<MechanicalState>,
        peak_throttle: f64,
        best_miss: Option<f64>,
        final_miss: Option<f64>,
        forecast_refusals: BTreeMap<String, usize>,
        executed_refusal: Option<String>,
        unavailable_extra_pair: usize,
        selected_extra_pair: usize,
    }

    fn difference(a: Point, b: Point) -> Point {
        [a[0] - b[0], a[1] - b[1]]
    }

    fn miss(a: Point, b: Point) -> f64 {
        let [x, y] = difference(a, b);
        x.hypot(y)
    }

    fn metric_at(rollout: &FrozenRollout, p: Point) -> Result<Metric, String> {
        serde_json::from_value(rollout.probe(p)?["metric"].clone()).map_err(|e| e.to_string())
    }

    fn norm(g: Metric, [x, y]: Point) -> f64 {
        (x * (g[0][0] * x + g[0][1] * y) + y * (g[1][0] * x + g[1][1] * y)).sqrt()
    }

    fn unit(g: Metric, v: Point) -> Option<Point> {
        let length = norm(g, v);
        (length.is_finite() && length > 1e-8).then(|| v.map(|x| x / length))
    }

    fn tangent(g: Metric, [x, y]: Point) -> Option<Point> {
        unit(g, [-y, x])
    }

    fn headings(
        g: Metric,
        map: Point,
        goal: Point,
        velocity: Point,
        alternative: Option<Point>,
        arm: Arm,
    ) -> (Vec<(Point, bool)>, bool) {
        let mut choices = Vec::new();
        for vector in [goal, velocity] {
            if let Some(direction) = unit(g, vector) {
                choices.push((direction, false));
            }
        }
        let mut add_pair = |direction: Option<Point>, extra: bool| {
            if let Some(direction) = direction {
                choices.push((direction, extra));
                choices.push((direction.map(|x| -x), extra));
            }
        };
        let map_tangent = tangent(g, map);
        // Apply G^-1 to the signed Map covector. The determinant cancels on normalization.
        let [x, y] = map;
        let map_normal = unit(g, [g[1][1] * x - g[0][1] * y, g[0][0] * y - g[1][0] * x]);
        add_pair(map_tangent, false);
        add_pair(map_normal, false);
        let control_pair = map_tangent
            .zip(map_normal)
            .and_then(|(t, n)| unit(g, [t[0] + n[0], t[1] + n[1]]));
        let alternative_pair = alternative.and_then(|covector| tangent(g, covector));
        let unavailable = !matches!(arm, Arm::Map) && alternative_pair.is_none();
        add_pair(alternative_pair.or(control_pair), true);
        (choices, unavailable)
    }

    fn step(
        rollout: &FrozenRollout,
        state: &mut MechanicalState,
        heading: Point,
        goal: Point,
        cap: f64,
    ) -> Result<f64, String> {
        let g = metric_at(rollout, state.position)?;
        let speed = SPEED.min(4.0 * norm(g, difference(goal, state.position)));
        let desired = unit(g, heading).unwrap_or([0.0, 0.0]).map(|x| x * speed);
        let delta = difference(desired, state.velocity);
        let controls = MotionControls {
            direction: delta,
            throttle: (RESPONSE * norm(g, delta) / MAX_DRIVE_FORCE).min(cap),
            brake: 0.0,
            grip: 0.0,
            impulse: 0.0,
        };
        let drive = rollout.drive_covector(state.position, &controls)?;
        rollout.attempt_step(state, drive)?;
        Ok(controls.throttle)
    }

    fn navigate(
        trace: &mut Trace,
        rollout: &FrozenRollout,
        field: &PreparedField,
        chart: Chart,
        route: Route,
        settings: Settings,
        arm: Arm,
    ) -> Result<(), String> {
        let g = metric_at(rollout, route.start)?;
        let velocity = unit(g, route.launch)
            .ok_or("invalid launch direction")?
            .map(|x| x * SPEED);
        let mut state = rollout.initial_state(route.start, velocity)?;
        trace.frames.push(state);
        for steps_left in (HOLD..=STEPS).rev().step_by(HOLD) {
            let g = metric_at(rollout, state.position)?;
            let map = field
                .query(state.position)
                .map_err(|err| format!("{err:?}"))?;
            let alternative = match arm {
                Arm::Map => None,
                Arm::Multiplier => chart.multiplier_covector(state.position),
                Arm::PointTrap => {
                    let c = Complex64::new(state.position[0], state.position[1]);
                    // P2 is exact in this finite fixture. Refuse the trap after a first-step
                    // |P1| >= 2 bailout, before evaluating any later critical iterate.
                    (c.norm_sqr() < 4.0)
                        .then(|| squared_level_covector(c * (c + 1.0) - TRAP, 2.0 * c + 1.0))
                }
            };
            let (choices, unavailable) = headings(
                g,
                map.gradient,
                difference(route.goal, state.position),
                state.velocity,
                alternative,
                arm,
            );
            trace.unavailable_extra_pair += usize::from(unavailable);
            let mut best = None;
            for (heading, extra) in choices {
                let mut forecast = state;
                let result = (0..settings.forecast.min(steps_left)).try_for_each(|_| {
                    step(
                        rollout,
                        &mut forecast,
                        heading,
                        route.goal,
                        settings.throttle,
                    )
                    .map(|_| ())
                });
                match result {
                    Ok(()) => {
                        let score = miss(forecast.position, route.goal);
                        if best.is_none_or(|(old, _, _)| score < old) {
                            best = Some((score, heading, extra));
                        }
                    }
                    Err(err) => *trace.forecast_refusals.entry(err).or_default() += 1,
                }
            }
            let (_, heading, extra) = best.ok_or("all forecasts refused")?;
            trace.selected_extra_pair += usize::from(extra);
            for _ in 0..HOLD {
                let throttle = step(rollout, &mut state, heading, route.goal, settings.throttle)?;
                trace.peak_throttle = trace.peak_throttle.max(throttle);
                trace.frames.push(state);
            }
        }
        Ok(())
    }

    pub fn run() -> Result<(), String> {
        let args = std::env::args().skip(1).collect::<Vec<_>>();
        let [asset, policy] = args.as_slice() else {
            return Err("usage: waypoint_guidance_ablation ASSET POLICY".into());
        };
        let field = Arc::new(PreparedField::from_asset_bytes(
            &std::fs::read(asset).map_err(|err| err.to_string())?,
        )?);
        if field.identity() != FIELD_ID {
            return Err("asset does not match the retained field ID".into());
        }
        let policy = std::fs::read_to_string(policy).map_err(|err| err.to_string())?;
        let cuts = reference_policy_for_field(&field, &policy)?;
        let config = ManifoldConfig {
            kappa: 0.0,
            mu: 0.0,
            ..ManifoldConfig::default()
        };
        let profile = PhysicsProfile::new(config, DT, 0.05, Integrator::Rk4FourSubsteps)?;
        let rollout = FrozenRollout::new(Arc::clone(&field), profile, cuts)?;
        let mut rows = Vec::new();
        for settings in [
            Settings {
                forecast: 12,
                throttle: 0.25,
            },
            Settings {
                forecast: 24,
                throttle: 0.25,
            },
            Settings {
                forecast: 12,
                throttle: 0.5,
            },
        ] {
            for chart in [Chart::Cardioid, Chart::Bulb] {
                for route in ROUTES.into_iter().map(|route| route.in_chart(chart)) {
                    for arm in [Arm::Map, Arm::Multiplier, Arm::PointTrap] {
                        let mut trace = Trace::default();
                        trace.executed_refusal =
                            navigate(&mut trace, &rollout, &field, chart, route, settings, arm)
                                .err();
                        trace.best_miss = trace
                            .frames
                            .iter()
                            .map(|state| miss(state.position, route.goal))
                            .reduce(f64::min);
                        if trace.executed_refusal.is_none() {
                            trace.final_miss = trace
                                .frames
                                .last()
                                .map(|state| miss(state.position, route.goal));
                        }
                        rows.push(json!({"settings": settings, "chart": chart, "route": route, "arm": arm, "trace": trace}));
                    }
                }
            }
        }
        println!(
            "{}",
            json!({
                "profile": rollout.profile().manifest(), "field": format!("{FIELD_ID:016x}"),
                "steps": STEPS, "hold": HOLD, "speed": SPEED, "response": RESPONSE,
                "trap": [TRAP.re, TRAP.im], "rows": rows,
            })
        );
        Ok(())
    }
}
