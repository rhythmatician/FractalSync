//! Replay the original #148 driven and undriven Controls crossing scenario.
use runtime_core::controls::MotionControls;
use runtime_core::crossing_feasibility::{
    evaluate_feasibility, FeasibilityLimits, LengthLowerBound,
};
use runtime_core::manifold::{kinetic_energy_from_metric, ManifoldConfig};
use runtime_core::operational_engine::{
    reference_policy_for_field, FrozenRollout, Integrator, MechanicalEndpoint, PhysicsProfile,
};
use runtime_core::operational_field::PreparedField;
use runtime_core::timebase::CANONICAL_SAMPLE_RATE;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

const STEP_SAMPLES: u64 = 960;
const STEPS: u64 = 400;
const SUBSTEPS: usize = 4;
const _: () = assert!(STEP_SAMPLES.is_multiple_of(SUBSTEPS as u64));

#[derive(Deserialize)]
struct Probe {
    metric: [[f64; 2]; 2],
    operational_value: f64,
}

#[derive(Clone, Copy, Default, Serialize)]
struct MotionEstimate {
    panel_samples: u64,
    covered_samples: u64,
    intrinsic_length: f64,
    kinetic_action: f64,
    action_floor_from_measured_length: Option<f64>,
}

struct MotionQuadrature {
    elapsed_samples: u64,
    estimates: [MotionEstimate; 3],
}

impl MotionQuadrature {
    fn new() -> Self {
        Self {
            elapsed_samples: 0,
            estimates: [STEP_SAMPLES, STEP_SAMPLES / 2, STEP_SAMPLES / 4].map(|panel_samples| {
                MotionEstimate {
                    panel_samples,
                    ..Default::default()
                }
            }),
        }
    }

    // Each accepted step is a separate continuous interval. Adjacent steps can
    // supply separate pre- and post-impulse energies at the same sample.
    fn record_step(
        &mut self,
        start_sample: u64,
        energies: [Option<f64>; SUBSTEPS + 1],
    ) -> Result<(), &'static str> {
        if start_sample < self.elapsed_samples {
            return Err("replay measurement overlaps a recorded step");
        }
        let end_sample = start_sample
            .checked_add(STEP_SAMPLES)
            .ok_or("sample overflow")?;
        let energies = energies.map(|energy| energy.filter(|k| k.is_finite() && *k >= 0.0));
        let mut estimates = self.estimates;
        for estimate in &mut estimates {
            let stride = (SUBSTEPS as u64 * estimate.panel_samples / STEP_SAMPLES) as usize;
            let half_dt = sample_seconds(estimate.panel_samples) / 2.0;
            for start in (0..SUBSTEPS).step_by(stride) {
                let end = start + stride;
                // A coarser panel must never span an invalid interior sample.
                let (Some(k0), Some(k1)) = (energies[start], energies[end]) else {
                    continue;
                };
                if energies[start + 1..end].iter().any(Option::is_none) {
                    continue;
                }
                let s0 = std::f64::consts::SQRT_2 * k0.sqrt();
                let s1 = std::f64::consts::SQRT_2 * k1.sqrt();
                estimate.covered_samples += estimate.panel_samples;
                estimate.intrinsic_length += half_dt * (s0 + s1);
                estimate.kinetic_action += half_dt * k0 + half_dt * k1;
            }
            if !estimate.intrinsic_length.is_finite() || !estimate.kinetic_action.is_finite() {
                return Err("nonfinite replay motion quadrature");
            }
        }
        self.elapsed_samples = end_sample;
        self.estimates = estimates;
        Ok(())
    }

    fn estimates(&self) -> [MotionEstimate; 3] {
        self.estimates.map(|mut estimate| {
            if self.elapsed_samples > 0 && estimate.covered_samples == self.elapsed_samples {
                let length = estimate.intrinsic_length;
                estimate.action_floor_from_measured_length =
                    Some((0.5 * length / sample_seconds(self.elapsed_samples)) * length);
            }
            estimate
        })
    }
}

fn sample_seconds(samples: u64) -> f64 {
    samples as f64 / CANONICAL_SAMPLE_RATE as f64
}

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let asset = std::fs::read(args.next().ok_or("missing asset")?).map_err(|e| e.to_string())?;
    let policy =
        std::fs::read_to_string(args.next().ok_or("missing policy")?).map_err(|e| e.to_string())?;
    let field = Arc::new(PreparedField::from_asset_bytes(&asset)?);
    let cuts = reference_policy_for_field(&field, &policy)?;
    let config = ManifoldConfig {
        d_ref: 0.1,
        epsilon: 1e-4,
        lambda_sq: 1.0,
        kappa: 0.5,
        mu: std::f64::consts::FRAC_1_PI,
    };
    let controls = |throttle| MotionControls {
        direction: [1.0, 0.0],
        throttle,
        brake: 0.0,
        grip: 0.0,
        impulse: 0.0,
    };
    let profile = PhysicsProfile::new(
        config,
        sample_seconds(STEP_SAMPLES),
        controls(0.0).friction_beta(),
        Integrator::Rk4FourSubsteps,
    )?;
    let rollout = FrozenRollout::new(field, profile, cuts)?;
    let probe = |position| -> Result<Probe, String> {
        serde_json::from_value(rollout.probe(position)?).map_err(|err| err.to_string())
    };
    let run = |throttle| -> Result<_, String> {
        let mut state = rollout.initial_state([0.3, 0.05], [0.0; 2])?;
        let initial_energy = state.energy;
        if probe(state.position)?.operational_value >= 0.0 {
            return Err("crossing replay must start at a negative operational scalar".into());
        }
        let mut motion = MotionQuadrature::new();
        let mut crossing = None;
        for step in 0..STEPS {
            let q = rollout.drive_covector(state.position, &controls(throttle))?;
            let (_, trace) = rollout.attempt_step_with_trace(&mut state, q)?;
            let endpoints: &[MechanicalEndpoint; SUBSTEPS + 1] = trace
                .endpoints()
                .try_into()
                .map_err(|_| "unexpected number of accepted step endpoints")?;
            let probes = endpoints
                .each_ref()
                .map(|endpoint| probe(endpoint.position));
            let scalar = probes[SUBSTEPS]
                .as_ref()
                .map_err(Clone::clone)?
                .operational_value;
            let energies = std::array::from_fn(|i| {
                let [vx, vy] = endpoints[i].velocity;
                probes[i]
                    .as_ref()
                    .ok()
                    .map(|p| kinetic_energy_from_metric((vx, vy), p.metric))
            });
            motion.record_step(step * STEP_SAMPLES, energies)?;
            if scalar > 0.0 {
                crossing = Some(motion.elapsed_samples);
                break;
            }
        }
        Ok((
            (crossing.is_some(), state.position),
            serde_json::json!({
                "first_positive_outer_sample": crossing,
                "duration_samples": motion.elapsed_samples,
                "duration_seconds": sample_seconds(motion.elapsed_samples),
                "quadrature": motion.estimates(),
                "signed_integrated_work": state.integrated_work,
                "mechanical_energy_change": state.energy - initial_energy,
            }),
        ))
    };
    let (undriven, undriven_motion) = run(0.0)?;
    let (driven, driven_motion) = run(1.0)?;
    if undriven.0 || !driven.0 {
        return Err(format!(
            "crossing failed: undriven={undriven:?}, driven={driven:?}"
        ));
    }
    // This task asks for the first true-Shore arrival by the replay deadline.
    // An operational sign change and an observed speed are not certificates.
    let feasibility = evaluate_feasibility(
        LengthLowerBound::Unavailable,
        STEPS * STEP_SAMPLES,
        FeasibilityLimits::default(),
    );
    println!(
        "{}",
        serde_json::json!({
            "operational_crossing": "passed",
            "reference_crossing": "unverified",
            "field_id": format!("{:016x}", rollout.field_id()),
            "physics_profile_id": format!("{:016x}", rollout.profile().id()),
            "undriven": undriven,
            "driven": driven,
            "canonical_sample_rate": CANONICAL_SAMPLE_RATE,
            "undriven_motion": undriven_motion,
            "driven_motion": driven_motion,
            "true_shore_arrival_feasibility": feasibility,
        })
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-12, "{actual} vs {expected}");
    }

    #[test]
    fn measures_each_side_of_an_impulse_at_the_same_sample() {
        let mut motion = MotionQuadrature::new();
        let dt = sample_seconds(STEP_SAMPLES);
        motion.record_step(0, [Some(0.5); SUBSTEPS + 1]).unwrap();
        motion
            .record_step(STEP_SAMPLES, [Some(4.5); SUBSTEPS + 1])
            .unwrap();
        assert_eq!(motion.elapsed_samples, STEP_SAMPLES * 2);
        for result in motion.estimates() {
            assert_eq!(result.covered_samples, motion.elapsed_samples);
            assert_close(result.intrinsic_length, 4.0 * dt);
            assert_close(result.kinetic_action, 5.0 * dt);
            assert_close(result.action_floor_from_measured_length.unwrap(), 4.0 * dt);
        }
    }

    #[test]
    fn refinement_approaches_exact_length_and_action_for_variable_speed() {
        let mut motion = MotionQuadrature::new();
        let dt = sample_seconds(STEP_SAMPLES);
        // Unit metric and speed 1 + (t/dt)^2.
        let energies = std::array::from_fn(|i| {
            let t = i as f64 / SUBSTEPS as f64;
            Some((1.0 + t * t).powi(2) / 2.0)
        });
        motion.record_step(0, energies).unwrap();
        let estimates = motion.estimates();
        let length_errors = estimates.map(|m| (m.intrinsic_length - 4.0 * dt / 3.0).abs());
        let action_errors = estimates.map(|m| (m.kinetic_action - 14.0 * dt / 15.0).abs());
        for errors in [length_errors, action_errors] {
            assert!(errors[1] < 0.27 * errors[0] && errors[2] < 0.27 * errors[1]);
        }
        for estimate in estimates {
            assert!(estimate.kinetic_action >= estimate.action_floor_from_measured_length.unwrap());
        }
    }

    #[test]
    fn missing_endpoints_and_steps_reduce_coverage_without_bridging() {
        let mut motion = MotionQuadrature::new();
        motion
            .record_step(0, [Some(0.5), None, Some(0.5), Some(0.5), Some(0.5)])
            .unwrap();
        motion
            .record_step(2 * STEP_SAMPLES, [Some(0.5); SUBSTEPS + 1])
            .unwrap();
        assert_eq!(motion.elapsed_samples, 3 * STEP_SAMPLES);
        let dt = sample_seconds(STEP_SAMPLES);
        let estimates = motion.estimates();
        assert_eq!(
            estimates.map(|m| m.covered_samples),
            [STEP_SAMPLES, 3 * STEP_SAMPLES / 2, 3 * STEP_SAMPLES / 2]
        );
        for (estimate, covered_steps) in estimates.into_iter().zip([1.0, 1.5, 1.5]) {
            assert_close(estimate.intrinsic_length, covered_steps * dt);
            assert_close(estimate.kinetic_action, covered_steps * dt / 2.0);
            assert!(estimate.action_floor_from_measured_length.is_none());
        }
    }
}
