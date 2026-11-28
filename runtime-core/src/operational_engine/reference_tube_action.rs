//! Conditional travel bounds between accepted RK endpoints.

use super::{AcceptedStepTrace, FrozenRollout};
use crate::crossing_feasibility::{action_floor, LengthLowerBound};
use crate::intrinsic_bounds::{add_up, mul_down, mul_up, sub_down};

/// Independently established bounds for a reference trajectory during one
/// committed RK substep, using that substep's represented duration.
#[derive(Clone, Copy, Debug)]
pub struct ReferenceMotionPanel {
    /// Lower and upper position corners in the frozen nominal chart.
    pub position_box: [[f64; 2]; 2],
    /// Bounds on the absolute ambient velocity components almost everywhere.
    pub velocity_abs_upper: [f64; 2],
}

/// Initial reference-state component errors relative to the first committed state.
#[derive(Clone, Copy, Debug)]
pub struct ReferenceStartUncertainty {
    pub position_abs_upper: [f64; 2],
    pub velocity_abs_upper: [f64; 2],
}

/// A regional acceleration enclosure for one reference substep.
///
/// The caller establishes the acceleration caps throughout the position
/// box and velocity range, for the reference dynamics over this interval.
#[derive(Clone, Copy, Debug)]
pub struct ReferenceAccelerationPanel {
    pub position_box: [[f64; 2]; 2],
    pub velocity_abs_upper: [f64; 2],
    pub acceleration_abs_upper: [f64; 2],
}

impl FrozenRollout {
    /// Construct a reference tube from the frozen nominal spline dynamics.
    /// The initial errors must enclose the reference start. The reference
    /// evolves under the same held control for the represented duration.
    pub fn nominal_reference_motion(
        &self,
        trace: &AcceptedStepTrace,
        initial: ReferenceStartUncertainty,
        position_box: [[f64; 2]; 2],
        velocity_abs_upper: [f64; 2],
    ) -> Option<Vec<ReferenceMotionPanel>> {
        self.check_trace_identity(trace).ok()?;
        if !inside_hard_disk(position_box) || !self.chart_edge_is_clear(position_box) {
            return None;
        }
        let acceleration_abs_upper = self.field.nominal_acceleration_abs_upper(
            position_box,
            velocity_abs_upper,
            trace.held_control.map(f64::abs),
            self.profile.beta,
            self.profile.config(),
        )?;
        let domain = ReferenceAccelerationPanel {
            position_box,
            velocity_abs_upper,
            acceleration_abs_upper,
        };
        self.reference_motion_from_acceleration(
            trace,
            initial,
            &vec![domain; trace.rk_substeps().len()],
        )
    }

    /// Enclose a continuous reference with the supplied initial uncertainty.
    /// Each domain needs an independently justified acceleration enclosure
    /// for the correct dynamics and time interval. The reference must be
    /// continuous in position and velocity through the represented substeps.
    pub fn reference_motion_from_acceleration(
        &self,
        trace: &AcceptedStepTrace,
        initial: ReferenceStartUncertainty,
        domains: &[ReferenceAccelerationPanel],
    ) -> Option<Vec<ReferenceMotionPanel>> {
        self.check_trace_identity(trace).ok()?;
        let steps = trace.rk_substeps();
        let endpoints = trace.endpoints();
        if steps.is_empty() || steps.len() != domains.len() || steps.len() + 1 != endpoints.len() {
            return None;
        }
        let first = endpoints.first()?;
        let mut position = initial_box(first.position, initial.position_abs_upper)?;
        let mut velocity = initial_box(first.velocity, initial.velocity_abs_upper)?;
        let mut result = Vec::with_capacity(steps.len());
        for ((step, endpoints), domain) in steps.iter().zip(endpoints.windows(2)).zip(domains) {
            let h = step.duration;
            if !h.is_finite()
                || h <= 0.0
                || !domain
                    .position_box
                    .into_iter()
                    .flatten()
                    .all(f64::is_finite)
                || !domain
                    .velocity_abs_upper
                    .into_iter()
                    .chain(domain.acceleration_abs_upper)
                    .all(|value| value.is_finite() && value >= 0.0)
            {
                return None;
            }
            let mut panel = ReferenceMotionPanel {
                position_box: position,
                velocity_abs_upper: [0.0; 2],
            };
            for axis in 0..2 {
                let velocity_change = mul_up(h, domain.acceleration_abs_upper[axis]);
                let position_change = mul_up(mul_up(h, 0.5), velocity_change);
                let drift_lo = (h * velocity[0][axis]).next_down();
                let drift_hi = (h * velocity[1][axis]).next_up();
                let next_position = [
                    (position[0][axis] + (drift_lo - position_change).next_down()).next_down(),
                    (position[1][axis] + (drift_hi + position_change).next_up()).next_up(),
                ];
                let next_velocity = [
                    (velocity[0][axis] - velocity_change).next_down(),
                    (velocity[1][axis] + velocity_change).next_up(),
                ];
                let lo = (position[0][axis] + (drift_lo.min(0.0) - position_change).next_down())
                    .next_down();
                let hi =
                    (position[1][axis] + (drift_hi.max(0.0) + position_change).next_up()).next_up();
                // Include the committed endpoints for the action consumer.
                // Only the propagated reference intervals feed the next panel.
                panel.position_box[0][axis] = lo
                    .min(endpoints[0].position[axis])
                    .min(endpoints[1].position[axis]);
                panel.position_box[1][axis] = hi
                    .max(endpoints[0].position[axis])
                    .max(endpoints[1].position[axis]);
                panel.velocity_abs_upper[axis] = next_velocity[0].abs().max(next_velocity[1].abs());
                if !(domain.position_box[0][axis] < panel.position_box[0][axis]
                    && panel.position_box[1][axis] < domain.position_box[1][axis]
                    && panel.velocity_abs_upper[axis] < domain.velocity_abs_upper[axis])
                {
                    return None;
                }
                [position[0][axis], position[1][axis]] = next_position;
                [velocity[0][axis], velocity[1][axis]] = next_velocity;
            }
            result.push(panel);
        }
        Some(result)
    }
}

fn initial_box(center: [f64; 2], error: [f64; 2]) -> Option<[[f64; 2]; 2]> {
    center
        .into_iter()
        .chain(error)
        .all(f64::is_finite)
        .then_some(())?;
    error.into_iter().all(|value| value >= 0.0).then(|| {
        [
            std::array::from_fn(|i| (center[i] - error[i]).next_down()),
            std::array::from_fn(|i| (center[i] + error[i]).next_up()),
        ]
    })
}

fn inside_hard_disk([lo, hi]: [[f64; 2]; 2]) -> bool {
    let corner = std::array::from_fn::<_, 2, _>(|i| lo[i].abs().max(hi[i].abs()));
    add_up(mul_up(corner[0], corner[0]), mul_up(corner[1], corner[1])) < 4.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifold::ManifoldConfig;
    use crate::operational_engine::{Integrator, PhysicsProfile};
    use crate::operational_field::PreparedField;
    use std::sync::Arc;

    #[test]
    fn driven_quadratic_field_builds_tubes_from_nominal_dynamics() {
        let spacing = 1.0 / 16.0;
        let coefficients = (0..33)
            .flat_map(|row| {
                (0..33).map(move |column| {
                    let x = f64::from(column) * spacing - 1.0;
                    let y = f64::from(row) * spacing - 1.0;
                    x + x * y / 8.0 + y * y / 16.0
                })
            })
            .collect();
        let field = PreparedField::new([-1.0; 2], spacing, [33; 2], coefficients).unwrap();
        let profile = PhysicsProfile::new(
            ManifoldConfig {
                epsilon: 0.5,
                mu: 0.0,
                ..ManifoldConfig::default()
            },
            1.0 / 64.0,
            0.125,
            Integrator::Rk4FourSubsteps,
        )
        .unwrap();
        let rollout = FrozenRollout::new(Arc::new(field), profile, vec![]).unwrap();
        let start = [0.125, 0.375];
        let mut state = rollout.initial_state(start, [0.125, 0.0625]).unwrap();
        let control = [0.03125, -0.015625];
        let (_, trace) = rollout
            .attempt_step_with_trace(&mut state, control)
            .unwrap();
        let uncertainty = ReferenceStartUncertainty {
            position_abs_upper: [1.0 / 1024.0; 2],
            velocity_abs_upper: [1.0 / 1024.0; 2],
        };
        let window = [[0.0625, 0.3125], [0.1875, 0.4375]];
        let panels = rollout
            .nominal_reference_motion(&trace, uncertainty, window, [0.25; 2])
            .unwrap();
        assert_eq!(panels.len(), 4);
        let last = panels.last().unwrap();
        assert!(last.position_box[1][0] - last.position_box[0][0] > 2.0 / 1024.0);
        let action = rollout
            .nominal_reference_tube_action(&trace, &panels)
            .unwrap();
        assert!(action.reference_length_upper() > 0.0);
        assert!(action.missed_length_upper() < action.reference_length_upper());

        let first_exit = [[0.0625, 0.3125], [start[0] + 1.0 / 512.0, 0.4375]];
        assert!(rollout
            .nominal_reference_motion(&trace, uncertainty, first_exit, [0.25; 2],)
            .is_none());
    }

    fn affine_rollout(lambda_sq: f64) -> FrozenRollout {
        let spacing = 1.0 / 16.0;
        let coefficients = (0..33)
            .cycle()
            .take(33 * 33)
            .map(|column| f64::from(column) * spacing - 1.0)
            .collect();
        let field = PreparedField::new([-1.0; 2], spacing, [33; 2], coefficients).unwrap();
        let profile = PhysicsProfile::new(
            ManifoldConfig {
                epsilon: 0.5,
                lambda_sq,
                kappa: 0.0,
                mu: 0.0,
                ..ManifoldConfig::default()
            },
            1.0 / 64.0,
            0.0,
            Integrator::Rk4FourSubsteps,
        )
        .unwrap();
        FrozenRollout::new(Arc::new(field), profile, vec![]).unwrap()
    }

    #[test]
    fn affine_field_tube_encloses_a_transverse_geodesic() {
        let rollout = affine_rollout(1.0);
        let mut state = rollout.initial_state([0.0, 0.375], [0.0, 0.25]).unwrap();
        let (_, trace) = rollout
            .attempt_step_with_trace(&mut state, [0.0; 2])
            .unwrap();

        // The nominal field is D=x. G is diagonal and independent of y,
        // and d_x G_yy vanishes at x=0. Thus c(t)=(0, 3/8+t/4) is the
        // geodesic with these initial conditions, with intrinsic speed 1/2.
        // The box includes off-axis positions where the metric varies.
        let exact_duration = 1.0 / 64.0;
        let panel = ReferenceMotionPanel {
            position_box: [[-1.0 / 16.0, 0.375], [1.0 / 16.0, 0.375 + 1.0 / 128.0]],
            velocity_abs_upper: [0.0, 0.25],
        };
        assert_eq!(trace.rk_substeps().len(), 4);
        assert!(trace
            .rk_substeps()
            .iter()
            .all(|step| step.duration == exact_duration / 4.0));
        let bound = rollout
            .nominal_reference_tube_action(&trace, &[panel; 4])
            .unwrap();
        let exact_length = exact_duration / 2.0;
        assert!(bound.reference_length_upper() >= exact_length);
        assert!(bound.reference_length_upper() - exact_length < 1e-12);
        assert!(bound.missed_length_upper() < 1e-12);
        let floor = bound
            .sampled_action_lower(LengthLowerBound::Certified(exact_length))
            .unwrap();
        let compiled = trace.compiled_sampled_action().unwrap();
        assert!(floor <= compiled.lower && compiled.upper - floor < 1e-12);

        // Uncertainty in the initial tangential velocity continues to
        // widen the position box across all four accepted substeps.
        let initial = ReferenceStartUncertainty {
            position_abs_upper: [0.0, 1.0 / 1024.0],
            velocity_abs_upper: [0.0, 1.0 / 64.0],
        };
        let propagated = rollout
            .nominal_reference_motion(
                &trace,
                initial,
                [[-0.0625, 0.3125], [0.0625, 0.4375]],
                [0.5; 2],
            )
            .unwrap();
        let exact_y = 0.375
            + initial.position_abs_upper[1]
            + exact_duration * (0.25 + initial.velocity_abs_upper[1]);
        assert!(propagated[3].position_box[1][1] >= exact_y);
    }

    #[test]
    fn affine_field_tube_encloses_motion_through_a_changing_metric() {
        let rollout = affine_rollout(0.0);
        let mut state = rollout.initial_state([0.0, 0.375], [0.25, 0.0]).unwrap();
        let (_, trace) = rollout
            .attempt_step_with_trace(&mut state, [0.0; 2])
            .unwrap();
        let duration = 1.0 / 64.0;
        assert!(trace
            .rk_substeps()
            .iter()
            .all(|step| step.duration == duration / 4.0));

        // For D=x and lambda_sq=0, x(t)=sinh(t/2)/2 follows the
        // force-free geodesic with intrinsic speed 1/2. For u=t/2
        // in [0,1/128], cosh(u) <= 1+u^2 by the power series. Integrating
        // this velocity cap also gives the box's right boundary.
        let velocity_cap = 0.25 * (1.0 + (1.0_f64 / 128.0).powi(2));
        let panel = ReferenceMotionPanel {
            position_box: [[0.0, 0.375], [duration * velocity_cap, 0.375]],
            velocity_abs_upper: [velocity_cap, 0.0],
        };
        let bound = rollout
            .nominal_reference_tube_action(&trace, &[panel; 4])
            .unwrap();
        let exact_length = duration / 2.0;
        let exact_action = duration / 8.0;
        assert!(bound.reference_length_upper() >= exact_length);
        assert!(bound.reference_length_upper() - exact_length < 5e-7);
        assert!(bound.missed_length_upper() < 5e-7);
        let floor = bound
            .sampled_action_lower(LengthLowerBound::Certified(exact_length))
            .unwrap();
        let compiled = trace.compiled_sampled_action().unwrap();
        assert!(floor <= compiled.lower && compiled.upper - floor < 3e-7);
        assert!((compiled.upper - exact_action).abs() < 1e-12);
    }

    #[test]
    fn constant_metric_reference_tube_encloses_analytic_continuous_motion() {
        let field = PreparedField::new([-1.0; 2], 0.0625, [33; 2], vec![0.375; 33 * 33]).unwrap();
        let profile = PhysicsProfile::new(
            ManifoldConfig {
                epsilon: 0.5,
                kappa: 0.0,
                mu: 0.0,
                ..ManifoldConfig::default()
            },
            0.01,
            0.0,
            Integrator::Rk4FourSubsteps,
        )
        .unwrap();
        let rollout = FrozenRollout::new(Arc::new(field), profile, vec![]).unwrap();
        let mut state = rollout.initial_state([0.0, 0.375], [0.25, 0.0]).unwrap();
        let (_, trace) = rollout
            .attempt_step_with_trace(&mut state, [0.0; 2])
            .unwrap();

        // In the nominal constant field with forces off, c(t)=(t/4, 3/8).
        // The full step is shorter than 1/16 second and thus covered here.
        let panel = ReferenceMotionPanel {
            position_box: [[0.0, 0.375], [1.0 / 64.0, 0.375]],
            velocity_abs_upper: [0.25, 0.0],
        };
        let duration: f64 = trace.rk_substeps().iter().map(|step| step.duration).sum();
        assert!(duration < 1.0 / 16.0);
        let panels = [panel; 4];
        let bound = rollout
            .nominal_reference_tube_action(&trace, &panels)
            .unwrap();
        let expected = duration * 0.25 / 0.625;
        assert!(bound.reference_length_upper >= expected);
        assert!(bound.reference_length_upper - expected < 1e-12);
        assert!(bound.missed_length_upper < 1e-12);
        let floor = bound
            .sampled_action_lower(LengthLowerBound::Certified(expected.next_down()))
            .unwrap();
        let compiled_action = trace.compiled_sampled_action().unwrap();
        assert!(floor <= compiled_action.upper);
        assert!(compiled_action.upper - floor < 1e-12);
        assert_eq!(
            bound.sampled_action_lower(LengthLowerBound::Estimate(expected)),
            None
        );
        assert!(rollout
            .nominal_reference_tube_action(&trace, &panels[..3])
            .is_none());

        let mut invalid = panels;
        invalid[1].velocity_abs_upper[0] = -1.0;
        assert!(rollout
            .nominal_reference_tube_action(&trace, &invalid)
            .is_none());
        invalid[1] = ReferenceMotionPanel {
            position_box: [[0.0, 0.0], [0.25, 0.375]],
            ..panel
        };
        assert!(rollout
            .nominal_reference_tube_action(&trace, &invalid)
            .is_none());
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NominalReferenceTubeAction {
    reference_length_upper: f64,
    /// Sum of positive excesses over substep duration times the speed
    /// implied by each recorded compiled kinetic energy.
    missed_length_upper: f64,
    duration_upper: f64,
}

impl NominalReferenceTubeAction {
    pub fn reference_length_upper(&self) -> f64 {
        self.reference_length_upper
    }

    pub fn missed_length_upper(&self) -> f64 {
        self.missed_length_upper
    }

    /// Apply a certified arrival length for this reference path and interval
    /// to obtain a floor for the represented compiled sampled action.
    pub fn sampled_action_lower(&self, arrival_length: LengthLowerBound) -> Option<f64> {
        match arrival_length {
            LengthLowerBound::Certified(length) if length.is_finite() && length >= 0.0 => {
                Some(action_floor(
                    sub_down(length, self.missed_length_upper),
                    self.duration_upper,
                ))
            }
            _ => None,
        }
    }
}

impl FrozenRollout {
    /// Bound travel continuously within each supplied reference position and
    /// velocity tube, in the nominal exact-real spline metric. The caller
    /// certifies the actual reference motion and synchronization of the panels,
    /// including any event boundaries. The compared sampled speed is the
    /// square root of twice the recorded left kinetic energy. Endpoint
    /// agreement alone does not certify these tube premises.
    pub fn nominal_reference_tube_action(
        &self,
        trace: &AcceptedStepTrace,
        panels: &[ReferenceMotionPanel],
    ) -> Option<NominalReferenceTubeAction> {
        self.check_trace_identity(trace).ok()?;
        let steps = trace.rk_substeps();
        if steps.is_empty()
            || steps.len() != panels.len()
            || steps.len() + 1 != trace.endpoints().len()
        {
            return None;
        }
        let mut reference_length_upper = 0.0;
        let mut missed_length_upper = 0.0;
        let mut duration_upper = 0.0;
        for ((step, pair), panel) in steps.iter().zip(trace.endpoints().windows(2)).zip(panels) {
            let h = step.duration;
            let [lo, hi] = panel.position_box;
            if !h.is_finite()
                || h <= 0.0
                || !inside_hard_disk(panel.position_box)
                || !self.chart_edge_is_clear(panel.position_box)
                || !pair.iter().all(|state| {
                    (0..2).all(|i| lo[i] <= state.position[i] && state.position[i] <= hi[i])
                })
            {
                return None;
            }
            let region = self
                .field
                .navigation_region(panel.position_box, self.profile.config())?;
            let reference_speed =
                region.tube_speed_upper(panel.position_box, panel.velocity_abs_upper)?;
            let reference_length = mul_up(h, reference_speed);
            let kinetic = step.left_kinetic_energy?;
            if !kinetic.is_finite() || kinetic < 0.0 {
                return None;
            }
            let sampled_speed = mul_down(
                kinetic.sqrt().next_down().max(0.0),
                std::f64::consts::SQRT_2.next_down(),
            );
            let sampled_length = mul_down(h, sampled_speed);
            let missed = if reference_length > sampled_length {
                (reference_length - sampled_length).next_up()
            } else {
                0.0
            };
            reference_length_upper = add_up(reference_length_upper, reference_length);
            missed_length_upper = add_up(missed_length_upper, missed);
            duration_upper = add_up(duration_upper, h);
        }
        (reference_length_upper.is_finite()
            && missed_length_upper.is_finite()
            && duration_upper.is_finite())
        .then_some(NominalReferenceTubeAction {
            reference_length_upper,
            missed_length_upper,
            duration_upper,
        })
    }
}
