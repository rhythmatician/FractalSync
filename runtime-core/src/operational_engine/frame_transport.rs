use super::{AcceptedStepTrace, FrozenRollout};
use crate::{geometry_provider, manifold};
use num_complex::Complex64;
use std::sync::Arc;

/// Oriented metric-unit frame at the final endpoint of a diagnostic transport.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransportedFrame {
    pub axis: [f64; 2],
    pub companion: [f64; 2],
}

impl FrozenRollout {
    /// Apply endpoint-normalized covariant Euler to affine chords between
    /// the committed endpoints. The chord displacement supplies the base
    /// velocity integral. The recorded endpoint velocities are unused.
    ///
    /// The trace must come from this rollout instance. The signed turn rate,
    /// in radians per mechanical second, is held for
    /// the whole trace. The initial direction is normalized at its start.
    /// This diagnostic concerns the chord reconstruction. It supplies no
    /// angle bound for the continuous Physics path.
    pub fn transport_endpoint_chords(
        &self,
        trace: &AcceptedStepTrace,
        initial_direction: [f64; 2],
        turn_rate: f64,
    ) -> Result<TransportedFrame, String> {
        self.transport_chords(
            trace,
            initial_direction,
            std::iter::repeat_n(turn_rate, trace.endpoints().len() - 1),
        )
    }

    /// Apply the same diagnostic transport with a signed turn rate in radians
    /// per mechanical second held separately on each committed endpoint chord.
    /// Rates are ordered by the trace's endpoints and must cover every chord.
    pub fn transport_endpoint_chords_with_rates(
        &self,
        trace: &AcceptedStepTrace,
        initial_direction: [f64; 2],
        turn_rates: &[f64],
    ) -> Result<TransportedFrame, String> {
        self.transport_chords(trace, initial_direction, turn_rates.iter().copied())
    }

    fn transport_chords(
        &self,
        trace: &AcceptedStepTrace,
        initial_direction: [f64; 2],
        turn_rates: impl ExactSizeIterator<Item = f64> + Clone,
    ) -> Result<TransportedFrame, String> {
        // The policy allocation is owned by this rollout and retained by
        // the trace. Fingerprints alone do not establish its provenance.
        if !Arc::ptr_eq(&trace.rollout_policy, &self.cuts)
            || trace.field_id != self.field_id()
            || trace.physics_profile_id != self.profile.id()
        {
            return Err("FrozenIdentity: trace belongs to another rollout".into());
        }
        let endpoints = trace.endpoints();
        let chord_count = endpoints.len() - 1;
        if turn_rates.len() != chord_count {
            return Err("expected one turn rate per accepted chord".into());
        }
        // Use the integration duration even when adding it to the absolute
        // mechanical clock rounded away part or all of the interval.
        let h = self.profile.dt / chord_count as f64;
        let turns = turn_rates.map(|rate| h * rate);
        if h <= 0.0 || turns.clone().any(|turn| !turn.is_finite()) {
            return Err("unsupported chord duration or turn rate".into());
        }

        let (frame, _) = self.evaluate(|config| {
            let query = |position: [f64; 2]| {
                geometry_provider::query_geometry(
                    Complex64::new(position[0], position[1]),
                    config.epsilon,
                )
            };
            let mut start_jet = query(endpoints[0].position)?;
            let mut metric =
                MetricFactor::new(manifold::induced_metric_from_jet(&start_jet, config)?)?;
            let mut axis = metric.unit(initial_direction)?;
            for (pair, turn) in endpoints.windows(2).zip(turns) {
                let [dx, dy] = std::array::from_fn(|i| pair[1].position[i] - pair[0].position[i]);
                let gamma = manifold::christoffel_symbols_from_jet(&start_jet, config)?;
                let connection = gamma.map(|[[g00, g01], [g10, g11]]| {
                    dx * (g00 * axis[0] + g01 * axis[1]) + dy * (g10 * axis[0] + g11 * axis[1])
                });
                let quarter = metric.quarter_turn(axis)?;
                let increment: [f64; 2] =
                    std::array::from_fn(|i| turn * quarter[i] - connection[i]);
                let increment_norm = increment[0].hypot(increment[1]);
                if !increment_norm.is_finite()
                    || increment_norm >= axis[0].hypot(axis[1]) * (1.0 - 32.0 * f64::EPSILON)
                {
                    return Err("covariant Euler increment lacks nonvanishing margin".into());
                }
                let trial = std::array::from_fn(|i| axis[i] + increment[i]);
                start_jet = query(pair[1].position)?;
                metric = MetricFactor::new(manifold::induced_metric_from_jet(&start_jet, config)?)?;
                axis = metric.unit(trial)?;
            }
            Ok(TransportedFrame {
                axis,
                companion: metric.quarter_turn(axis)?,
            })
        })?;
        Ok(frame)
    }
}

/// Orientation-preserving upper Cholesky factor, so |L v|² = g(v, v).
pub(super) struct MetricFactor {
    x_scale: f64,
    shear: f64,
    y_scale: f64,
}

impl MetricFactor {
    pub(super) fn new(metric: [[f64; 2]; 2]) -> Result<Self, String> {
        let x_scale = metric[0][0].sqrt();
        let shear = metric[0][1] / x_scale;
        let y_squared = metric[1][1] - shear * shear;
        if !x_scale.is_finite()
            || x_scale <= 0.0
            || !y_squared.is_finite()
            || y_squared <= 32.0 * f64::EPSILON * metric[1][1]
        {
            return Err("metric factorization lacks positive margin".into());
        }
        Ok(Self {
            x_scale,
            shear,
            y_scale: y_squared.sqrt(),
        })
    }

    fn coordinates(&self, [x, y]: [f64; 2]) -> [f64; 2] {
        [self.x_scale * x + self.shear * y, self.y_scale * y]
    }

    pub(super) fn unit(&self, v: [f64; 2]) -> Result<[f64; 2], String> {
        let [x, y] = self.coordinates(v);
        let length = x.hypot(y);
        if !length.is_finite() || length <= f64::MIN_POSITIVE {
            return Err("unsupported metric norm for transported axis".into());
        }
        checked_direction(v.map(|coordinate| coordinate / length))
    }

    fn quarter_turn(&self, v: [f64; 2]) -> Result<[f64; 2], String> {
        let [x, y] = self.coordinates(v);
        let rotated_y = x / self.y_scale;
        let rotated_x = (-y - self.shear * rotated_y) / self.x_scale;
        checked_direction([rotated_x, rotated_y])
    }
}

fn checked_direction(direction: [f64; 2]) -> Result<[f64; 2], String> {
    if direction.iter().all(|coordinate| coordinate.is_finite()) && direction != [0.0; 2] {
        Ok(direction)
    } else {
        Err("invalid transported direction".into())
    }
}

#[cfg(test)]
mod tests {
    use super::super::{FrozenRollout, Integrator, PhysicsProfile, ReferenceCut};
    use crate::{manifold::ManifoldConfig, operational_field::PreparedField};
    use std::sync::Arc;

    const EPSILON: f64 = 0.08;

    fn flat_shore() -> Arc<PreparedField> {
        let coefficients = (0..101).flat_map(|_| (0..101).map(|x| -0.5 + f64::from(x) * 0.01));
        Arc::new(PreparedField::new([-0.5; 2], 0.01, [101; 2], coefficients.collect()).unwrap())
    }

    fn sheared_shore() -> Arc<PreparedField> {
        let coefficients = (0..101)
            .flat_map(|y| (0..101).map(move |x| -1.5 + f64::from(x) * 0.01 + f64::from(y) * 0.02));
        Arc::new(PreparedField::new([-0.5; 2], 0.01, [101; 2], coefficients.collect()).unwrap())
    }

    fn rollout(field: Arc<PreparedField>, dt: f64, lambda_sq: f64) -> FrozenRollout {
        let config = ManifoldConfig {
            epsilon: EPSILON,
            lambda_sq,
            kappa: 0.0,
            mu: 0.0,
            ..ManifoldConfig::default()
        };
        let profile = PhysicsProfile::new(config, dt, 0.0, Integrator::Rk4FourSubsteps).unwrap();
        FrozenRollout::new(field, profile, vec![]).unwrap()
    }

    fn chord_angle(start: [f64; 2], end: [f64; 2]) -> f64 {
        let dx = end[0] - start[0];
        let dy = end[1] - start[1];
        let ruler_squared = EPSILON * EPSILON + start[0] * start[0];
        if dx == 0.0 {
            dy * start[0] / ruler_squared
        } else {
            dy * (dx * (start[0] + end[0]) / ruler_squared).ln_1p() / (2.0 * dx)
        }
    }

    // Independent coframe in the oriented normal/tangent basis of F(x, y) = x + 2y.
    struct AffineGraphMetric {
        normal_weight: f64,
    }

    impl AffineGraphMetric {
        fn new(lambda_sq: f64) -> Self {
            Self {
                normal_weight: 1.0 + 5.0 * lambda_sq / std::f64::consts::LN_2.powi(2),
            }
        }

        fn scales(&self, field_value: f64) -> (f64, f64) {
            (
                EPSILON.hypot(field_value),
                EPSILON.hypot(self.normal_weight.sqrt() * field_value),
            )
        }

        fn coordinates(&self, [x, y]: [f64; 2], [dx, dy]: [f64; 2]) -> [f64; 2] {
            let (rho, sigma) = self.scales(x + 2.0 * y);
            let gradient_length = 5.0_f64.sqrt();
            [
                sigma * (dx + 2.0 * dy) / (rho * rho * gradient_length),
                (-2.0 * dx + dy) / (rho * gradient_length),
            ]
        }

        fn chord_angle(&self, [x0, y0]: [f64; 2], [x1, y1]: [f64; 2]) -> f64 {
            let f0 = x0 + 2.0 * y0;
            let f1 = x1 + 2.0 * y1;
            let across = f1 - f0;
            let along = -2.0 * (x1 - x0) + (y1 - y0);
            let (rho0, sigma0) = self.scales(f0);
            if across == 0.0 {
                return along * f0 / (rho0 * sigma0);
            }

            // Evaluate the log of the endpoint ratio after rationalizing the
            // difference of square roots, so close endpoints keep precision.
            let (rho1, sigma1) = self.scales(f1);
            let root_weight = self.normal_weight.sqrt();
            let relative_change = across
                * (f1 + f0)
                * (root_weight / (rho1 + rho0) + self.normal_weight / (sigma1 + sigma0))
                / (root_weight * rho0 + sigma0);
            along * relative_change.ln_1p() / (root_weight * across)
        }
    }

    fn unsigned_angle_difference(left: f64, right: f64) -> f64 {
        let delta = left - right;
        delta.sin().atan2(delta.cos()).abs()
    }

    fn simpson_integral(f: impl Fn(f64) -> f64, start: f64, end: f64, panels: u32) -> f64 {
        let step = (end - start) / f64::from(panels);
        let interior: f64 = (1..panels)
            .map(|i| {
                let weight = if i % 2 == 0 { 2.0 } else { 4.0 };
                weight * f(start + f64::from(i) * step)
            })
            .sum();
        step * (f(start) + interior + f(end)) / 3.0
    }

    // Recover the positive-normal free geodesic from speed and the conserved
    // tangential momentum. Return its position, connection angle, and flight time.
    fn positive_free_geodesic(
        graph: &AffineGraphMetric,
        start: [f64; 2],
        velocity: [f64; 2],
        duration: f64,
        panels: u32,
    ) -> ([f64; 2], f64, f64) {
        let gradient_length = 5.0_f64.sqrt();
        let [x, y] = start;
        let field_start = x + 2.0 * y;
        let tangent_start = (-2.0 * x + y) / gradient_length;
        let (rho_start, _) = graph.scales(field_start);
        let [normal_speed, tangent_speed] = graph.coordinates(start, velocity);
        assert!(normal_speed > 0.0);
        let speed = normal_speed.hypot(tangent_speed);
        let momentum = tangent_speed / rho_start;

        let time_per_field = |field_value| {
            let (rho, sigma) = graph.scales(field_value);
            let normal_speed = (speed.powi(2) - (momentum * rho).powi(2)).sqrt();
            sigma / (gradient_length * rho * rho * normal_speed)
        };
        let flight_time =
            |field_value| simpson_integral(time_per_field, field_start, field_value, panels);

        let mut lo = field_start;
        let mut hi = 0.32;
        // On this positive-field interval, rho increases. The far endpoint
        // therefore bounds the normalized tangential speed before the turn.
        assert!(momentum.abs() * graph.scales(hi).0 < 0.7 * speed);
        assert!(flight_time(hi) > duration);
        for _ in 0..52 {
            let mid = (lo + hi) / 2.0;
            if flight_time(mid) < duration {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let field_end = (lo + hi) / 2.0;
        let (rho_end, _) = graph.scales(field_end);
        let normal_end = field_end / gradient_length;
        let tangent_end = tangent_start
            + simpson_integral(
                |f| momentum * graph.scales(f).0.powi(2) * time_per_field(f),
                field_start,
                field_end,
                panels,
            );
        let end = [
            (normal_end - 2.0 * tangent_end) / gradient_length,
            (2.0 * normal_end + tangent_end) / gradient_length,
        ];
        let angle = (momentum * rho_end / speed).asin() - (momentum * rho_start / speed).asin();
        (end, angle, flight_time(field_end))
    }

    #[test]
    fn endpoint_chord_transport_refines_toward_the_conformal_connection_angle() {
        let field = flat_shore();
        let initial = [0.04, 0.05];
        let theta: f64 = 0.37;
        let mut previous_error = f64::INFINITY;
        for count in [8, 16, 32] {
            let rollout = rollout(field.clone(), 0.32 / f64::from(count), 0.0);
            let mut state = rollout.initial_state(initial, [0.12, 0.4]).unwrap();
            let mut axis = [theta.cos(), theta.sin()];
            let mut angle = theta;
            for _ in 0..count {
                let (_, trace) = rollout
                    .attempt_step_with_trace(&mut state, [0.0; 2])
                    .unwrap();
                let frame = rollout
                    .transport_endpoint_chords(&trace, axis, 0.0)
                    .unwrap();
                axis = frame.axis;
                for pair in trace.endpoints().windows(2) {
                    angle += chord_angle(pair[0].position, pair[1].position);
                }
                let rho = EPSILON.hypot(state.position[0]);
                let [ax, ay] = axis.map(|value| value / rho);
                let [bx, by] = frame.companion.map(|value| value / rho);
                assert!((ax.hypot(ay) - 1.0).abs() < 2e-13);
                assert!((bx.hypot(by) - 1.0).abs() < 2e-13);
                assert!((ax * bx + ay * by).abs() < 2e-13);
                assert!(ax * by - ay * bx > 0.0);
            }
            let error = (axis[1].atan2(axis[0]) - angle).abs();
            assert!(error < 0.65 * previous_error, "{count}: {error:e}");
            assert!(error < 0.02, "{count}: {error:e}");
            // Renormalizing the original coordinate ray at every endpoint
            // would keep an exact metric-unit norm and leave this angle error.
            assert!((theta - angle).abs() > 0.1);
            previous_error = error;
        }
    }

    #[test]
    fn moving_sheared_transport_refines_toward_the_analytic_chord_angles() {
        let field = sheared_shore();
        let lambda_sq = 0.7;
        let graph = AffineGraphMetric::new(lambda_sq);
        let conformal = AffineGraphMetric::new(0.0);
        let initial_axis = [0.9, -0.1];
        let duration = 0.32;
        for (route, initial, velocity, turn_rate) in [
            ("positive", [0.04, 0.05], [-0.12, 0.4], 0.9),
            ("crossing", [-0.025, -0.01], [0.42, 0.25], -0.65),
        ] {
            let mut previous_error = f64::INFINITY;
            for count in [8, 16, 32] {
                let rollout = rollout(field.clone(), duration / f64::from(count), lambda_sq);
                let mut state = rollout.initial_state(initial, velocity).unwrap();
                let mut axis = initial_axis;
                let [ax, ay] = graph.coordinates(initial, axis);
                let initial_angle = ay.atan2(ax);
                let mut minimum_field = initial[0] + 2.0 * initial[1];
                let mut graph_turn = 0.0;
                let mut conformal_turn = 0.0;
                for _ in 0..count {
                    let (_, trace) = rollout
                        .attempt_step_with_trace(&mut state, [0.0; 2])
                        .unwrap();
                    assert_eq!(trace.endpoints().len(), 5);
                    let frame = rollout
                        .transport_endpoint_chords(&trace, axis, turn_rate)
                        .unwrap();
                    axis = frame.axis;
                    for pair in trace.endpoints().windows(2) {
                        graph_turn += graph.chord_angle(pair[0].position, pair[1].position);
                        conformal_turn += conformal.chord_angle(pair[0].position, pair[1].position);
                        let [x, y] = pair[1].position;
                        minimum_field = minimum_field.min(x + 2.0 * y);
                    }
                    let [ax, ay] = graph.coordinates(state.position, frame.axis);
                    let [bx, by] = graph.coordinates(state.position, frame.companion);
                    assert!((ax.hypot(ay) - 1.0).abs() < 2e-13);
                    assert!((bx.hypot(by) - 1.0).abs() < 2e-13);
                    assert!((ax * bx + ay * by).abs() < 2e-13);
                    assert!(ax * by - ay * bx > 0.0);
                }

                let [ax, ay] = graph.coordinates(state.position, axis);
                let expected = initial_angle + duration * turn_rate + graph_turn;
                let error = unsigned_angle_difference(ay.atan2(ax), expected);
                let conformal_gap = unsigned_angle_difference(graph_turn, conformal_turn);
                eprintln!(
                    "{route}, {count}: angle error {error:e}, conformal gap {conformal_gap:e}"
                );
                assert!(error < 0.65 * previous_error, "{route}, {count}: {error:e}");
                assert!(
                    conformal_gap > 0.5,
                    "{route}, {count}: {graph_turn:e} vs {conformal_turn:e}"
                );
                assert_ne!(minimum_field, 0.0);
                assert_eq!(minimum_field < 0.0, route == "crossing");
                assert!(state.position[0] + 2.0 * state.position[1] > 0.0);
                previous_error = error;
            }
            assert!(previous_error < 0.02, "{route}: {previous_error:e}");
        }
    }

    #[test]
    fn moving_sheared_transport_refines_toward_the_continuous_free_geodesic() {
        let field = sheared_shore();
        let lambda_sq = ManifoldConfig::default().lambda_sq;
        let graph = AffineGraphMetric::new(lambda_sq);
        let start = [0.04, 0.05];
        let velocity = [-0.12, 0.4];
        let initial_axis = [0.9, -0.1];
        let duration = 0.16;
        let turn_rate = 0.9;
        let (coarse_end, coarse_angle, _) =
            positive_free_geodesic(&graph, start, velocity, duration, 512);
        let (reference_end, reference_connection, reference_clock) =
            positive_free_geodesic(&graph, start, velocity, duration, 1024);
        let reference_position_change =
            (reference_end[0] - coarse_end[0]).hypot(reference_end[1] - coarse_end[1]);
        let reference_angle_change = (reference_connection - coarse_angle).abs();
        eprintln!(
            "reference panels doubled: position {reference_position_change:e}, angle {reference_angle_change:e}"
        );
        assert!(reference_position_change < 1e-11);
        assert!(reference_angle_change < 1e-11);
        assert!((reference_clock - duration).abs() < 1e-14);

        let [ax, ay] = graph.coordinates(start, initial_axis);
        let reference_angle = ay.atan2(ax) + reference_connection + duration * turn_rate;
        let mut previous_position_error = f64::INFINITY;
        let mut previous_angle_error = f64::INFINITY;
        for count in [4, 8, 16] {
            let rollout = rollout(field.clone(), duration / f64::from(count), lambda_sq);
            let mut state = rollout.initial_state(start, velocity).unwrap();
            let mut axis = initial_axis;
            let mut chord_connection = 0.0;
            for _ in 0..count {
                let (_, trace) = rollout
                    .attempt_step_with_trace(&mut state, [0.0; 2])
                    .unwrap();
                axis = rollout
                    .transport_endpoint_chords(&trace, axis, turn_rate)
                    .unwrap()
                    .axis;
                for pair in trace.endpoints().windows(2) {
                    chord_connection += graph.chord_angle(pair[0].position, pair[1].position);
                }
            }

            let [x, y] = state.position;
            let position_error = (x - reference_end[0]).hypot(y - reference_end[1]);
            let [ax, ay] = graph.coordinates(state.position, axis);
            let angle_error = unsigned_angle_difference(ay.atan2(ax), reference_angle);
            let chord_error = (chord_connection - reference_connection).abs();
            let clock_error = (state.mechanical_time - reference_clock).abs();
            eprintln!(
                "{count}: position {position_error:e}, angle {angle_error:e}, chord angle {chord_error:e}, clock {clock_error:e}"
            );
            assert!(clock_error < 1e-14, "{count}: {clock_error:e}");
            assert!(
                position_error < 0.2 * previous_position_error,
                "{count}: {position_error:e}"
            );
            assert!(
                angle_error < 0.65 * previous_angle_error,
                "{count}: {angle_error:e}"
            );
            previous_position_error = position_error;
            previous_angle_error = angle_error;
        }
        assert!(
            previous_position_error < 2e-9,
            "{previous_position_error:e}"
        );
        assert!(previous_angle_error < 0.01, "{previous_angle_error:e}");
    }

    #[test]
    fn held_turn_and_refusals_leave_the_trace_reusable() {
        let field = flat_shore();
        let rollout = rollout(field.clone(), 0.2, 0.7);
        let mut state = rollout.initial_state([0.04, 0.05], [0.0; 2]).unwrap();
        state.mechanical_time = 1e16;
        let (_, trace) = rollout
            .attempt_step_with_trace(&mut state, [0.0; 2])
            .unwrap();
        assert_eq!(trace.endpoints()[0].mechanical_time, state.mechanical_time);
        let direction = [1.0, 0.0];
        let zero = rollout
            .transport_endpoint_chords(&trace, direction, 0.0)
            .unwrap();
        assert_eq!(zero.axis[1], 0.0);
        let omega: f64 = 3.0;
        for sign in [-1.0, 1.0] {
            let frame = rollout
                .transport_endpoint_chords(&trace, direction, sign * omega)
                .unwrap();
            let angle = (frame.axis[1] / zero.companion[1]).atan2(frame.axis[0] / zero.axis[0]);
            assert!((angle - 4.0 * (sign * omega * 0.05).atan()).abs() < 2e-13);
        }
        assert!(
            rollout
                .transport_endpoint_chords(&trace, [0.0; 2], 0.0)
                .is_err()
        );
        assert!(
            rollout
                .transport_endpoint_chords(&trace, direction, f64::NAN)
                .is_err()
        );
        assert!(
            rollout
                .transport_endpoint_chords(&trace, direction, 100.0)
                .is_err()
        );
        let different = self::rollout(field.clone(), 0.1, 0.7);
        let same_parameters = self::rollout(field.clone(), 0.2, 0.7);
        assert_eq!(same_parameters.profile().id(), rollout.profile().id());
        let cut_profile = PhysicsProfile::new(
            rollout.profile().config().clone(),
            0.2,
            0.0,
            Integrator::Rk4FourSubsteps,
        )
        .unwrap();
        let cut_rollout = FrozenRollout::new(
            field,
            cut_profile,
            vec![ReferenceCut {
                bounds: [-0.4, -0.4, -0.3, -0.3],
                authority: "unrelated synthetic cut".into(),
            }],
        )
        .unwrap();
        for other in [&different, &cut_rollout, &same_parameters] {
            let error = other
                .transport_endpoint_chords(&trace, direction, 0.0)
                .unwrap_err();
            assert!(error.starts_with("FrozenIdentity:"));
        }
        let retry = rollout
            .transport_endpoint_chords(&trace, direction, 0.0)
            .unwrap();
        assert_eq!(retry.axis, zero.axis);
        assert_eq!(retry.companion, zero.companion);
    }

    #[test]
    fn per_chord_signed_turns_follow_each_accepted_duration() {
        let dt = 0.2;
        let rollout = rollout(flat_shore(), dt, 0.7);
        let mut state = rollout.initial_state([0.04, 0.05], [0.0; 2]).unwrap();
        state.mechanical_time = 1e16;
        let (_, trace) = rollout
            .attempt_step_with_trace(&mut state, [0.0; 2])
            .unwrap();
        let direction = [1.0, 0.0];
        let reference = rollout
            .transport_endpoint_chords(&trace, direction, 0.0)
            .unwrap();
        let rates = [5.0, 0.0, -2.0, 4.0];
        let frame = rollout
            .transport_endpoint_chords_with_rates(&trace, direction, &rates)
            .unwrap();
        let h = dt / rates.len() as f64;
        let expected: f64 = rates.iter().map(|rate| (h * rate).atan()).sum();
        let angle =
            (frame.axis[1] / reference.companion[1]).atan2(frame.axis[0] / reference.axis[0]);
        assert!((angle - expected).abs() < 2e-13);

        let held_rate = 3.0;
        let held = rollout
            .transport_endpoint_chords(&trace, direction, held_rate)
            .unwrap();
        let scheduled = rollout
            .transport_endpoint_chords_with_rates(&trace, direction, &[held_rate; 4])
            .unwrap();
        assert_eq!(held, scheduled);

        for invalid in [&rates[..3], &[0.0; 5][..], &[0.0, 0.0, 0.0, f64::NAN]] {
            assert!(
                rollout
                    .transport_endpoint_chords_with_rates(&trace, direction, invalid)
                    .is_err()
            );
        }
        assert_eq!(
            rollout
                .transport_endpoint_chords_with_rates(&trace, direction, &rates)
                .unwrap(),
            frame
        );
    }

    #[test]
    fn analytic_sheared_metric_gives_an_oriented_unit_frame() {
        let lambda_sq = 0.7;
        let rollout = rollout(sheared_shore(), 0.2, lambda_sq);
        let point = [0.04, 0.05];
        let mut state = rollout.initial_state(point, [0.0; 2]).unwrap();
        let (_, trace) = rollout
            .attempt_step_with_trace(&mut state, [0.0; 2])
            .unwrap();
        let initial = rollout
            .transport_endpoint_chords(&trace, [1.0, 0.0], 0.0)
            .unwrap();
        let frame = rollout
            .transport_endpoint_chords(&trace, [1.0, 0.0], 2.0)
            .unwrap();

        let distance = point[0] + 2.0 * point[1];
        let rho_squared = distance * distance + EPSILON * EPSILON;
        let anisotropy_squared = lambda_sq / std::f64::consts::LN_2.powi(2);
        let dot = |u: [f64; 2], v: [f64; 2]| {
            (u[0] * v[0] + u[1] * v[1]) / rho_squared
                + anisotropy_squared * distance * distance / rho_squared.powi(2)
                    * (u[0] + 2.0 * u[1])
                    * (v[0] + 2.0 * v[1])
        };
        assert!((dot(frame.axis, frame.axis) - 1.0).abs() < 2e-13);
        assert!((dot(frame.companion, frame.companion) - 1.0).abs() < 2e-13);
        assert!(dot(frame.axis, frame.companion).abs() < 2e-13);
        assert!(frame.axis[0] * frame.companion[1] - frame.axis[1] * frame.companion[0] > 0.0);
        let angle = dot(initial.companion, frame.axis).atan2(dot(initial.axis, frame.axis));
        assert!((angle - 4.0 * 0.1_f64.atan()).abs() < 2e-13);
        assert_eq!(
            rollout.last_evaluation_work().geometry_evaluations,
            trace.endpoints().len()
        );
    }
}
