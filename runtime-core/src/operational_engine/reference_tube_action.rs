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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifold::ManifoldConfig;
    use crate::operational_engine::{Integrator, PhysicsProfile};
    use crate::operational_field::PreparedField;
    use std::sync::Arc;

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
            let corner = std::array::from_fn::<_, 2, _>(|i| lo[i].abs().max(hi[i].abs()));
            let radius_squared_upper =
                add_up(mul_up(corner[0], corner[0]), mul_up(corner[1], corner[1]));
            if !h.is_finite()
                || h <= 0.0
                || radius_squared_upper >= 4.0
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
