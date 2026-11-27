use super::{AcceptedStepTrace, FrozenRollout};
use crate::{geometry_provider, manifold};
use num_complex::Complex64;

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
    /// The signed turn rate, in radians per mechanical second, is held for
    /// the whole trace. The initial direction is normalized at its start.
    /// This diagnostic concerns the chord reconstruction. It supplies no
    /// angle bound for the continuous Physics path.
    pub fn transport_endpoint_chords(
        &self,
        trace: &AcceptedStepTrace,
        initial_direction: [f64; 2],
        turn_rate: f64,
    ) -> Result<TransportedFrame, String> {
        if trace.field_id != self.field_id() || trace.physics_profile_id != self.profile.id() {
            return Err("FrozenIdentity: trace belongs to another field/profile".into());
        }
        let endpoints = trace.endpoints();
        // Use the integration duration even when adding it to the absolute
        // mechanical clock rounded away part or all of the interval.
        let h = self.profile.dt / (endpoints.len() - 1) as f64;
        let turn = h * turn_rate;
        if h <= 0.0 || !turn.is_finite() {
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
            for pair in endpoints.windows(2) {
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
struct MetricFactor {
    x_scale: f64,
    shear: f64,
    y_scale: f64,
}

impl MetricFactor {
    fn new(metric: [[f64; 2]; 2]) -> Result<Self, String> {
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

    fn unit(&self, v: [f64; 2]) -> Result<[f64; 2], String> {
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
        for other in [&different, &cut_rollout] {
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
    fn analytic_sheared_metric_gives_an_oriented_unit_frame() {
        let coefficients = (0..101)
            .flat_map(|y| (0..101).map(move |x| -1.5 + f64::from(x) * 0.01 + f64::from(y) * 0.02));
        let field = Arc::new(
            PreparedField::new([-0.5; 2], 0.01, [101; 2], coefficients.collect()).unwrap(),
        );
        let lambda_sq = 0.7;
        let rollout = rollout(field, 0.2, lambda_sq);
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
