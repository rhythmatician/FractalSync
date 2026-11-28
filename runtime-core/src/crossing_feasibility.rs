//! Necessary feasibility bounds for navigation deadlines and Shore arrivals.

use crate::intrinsic_bounds::{add_up, div_down, div_up, mul_down, mul_up, sub_down};
use crate::timebase::CANONICAL_SAMPLE_RATE;
use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum LengthLowerBound {
    /// A conservative bound for the actual arrival set and trajectory metric.
    Certified(f64),
    /// An approximate bound for diagnostics, with no authority to exclude a task.
    Estimate(f64),
    Unavailable,
}

/// A positive intrinsic speed cap that applies almost everywhere throughout the
/// evaluated rollout, in intrinsic length units per second.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum SpeedCap {
    Certified(f64),
    Enforced(f64),
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum ImpulseWork {
    /// The approach through the first crest is impulse-free.
    None,
    /// Conservative nonnegative upper bound for all signed impulse work on the
    /// approach. Its certificate must also account for unresolved event times.
    CertifiedUpper(f64),
    Unavailable,
}

/// Evidence for the work needed on the first-arrival prefix of a Shore
/// crossing. The caller must establish the signed work-energy ledger for that
/// prefix and that every accepted trajectory requires the crest event.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CrestWorkConstraint {
    /// Lower bound for the crest work deficit after crediting initial kinetic
    /// energy. Include the potential change and required arrival kinetic energy.
    /// Unknown friction may be omitted if it is known nonnegative.
    pub required_work_lower: f64,
    /// Certified upper bound on the control force in the dual metric,
    /// applicable almost everywhere on the prefix.
    pub control_force_cap: f64,
    pub impulse_work: ImpulseWork,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FeasibilityLimits {
    pub speed_cap: Option<SpeedCap>,
    /// Upper budget for the integral of kinetic energy over the rollout.
    pub kinetic_action_budget: Option<f64>,
    pub crest_work: Option<CrestWorkConstraint>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum FeasibilityStatus {
    RuledOut,
    NotRuledOut,
    InsufficientEvidence,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FeasibilityAssessment {
    pub status: FeasibilityStatus,
    pub length: LengthLowerBound,
    pub duration_samples: u64,
    pub minimum_seconds: Option<f64>,
    pub minimum_kinetic_action: Option<f64>,
    pub maximum_crest_work: Option<f64>,
    pub deadline_status: FeasibilityStatus,
    pub kinetic_action_status: Option<FeasibilityStatus>,
    pub crest_work_status: Option<FeasibilityStatus>,
}

/// Bounds that enclose the speed missed by one left-endpoint kinetic sample.
#[derive(Clone, Copy, Debug)]
pub struct SampledActionPanel {
    pub start_sample: u64,
    pub end_sample: u64,
    pub initial_speed_gap_upper: f64,
    pub speed_growth_upper: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SampledActionBound {
    pub missed_length_upper: f64,
    pub covered_length_lower: f64,
    pub minimum_kinetic_action: f64,
}

/// Bound left-endpoint sampled kinetic action over a complete arrival prefix.
/// The caller supplies a length lower bound in the reference metric and
/// certifies that each panel's reference speed minus its nonnegative sampled
/// speed is at most the initial speed gap plus the speed growth bound times
/// elapsed seconds. Split panels at impulses and other discontinuities.
pub fn sampled_action_lower_bound(
    length: LengthLowerBound,
    start_sample: u64,
    arrival_sample: u64,
    panels: &[SampledActionPanel],
) -> Option<SampledActionBound> {
    let LengthLowerBound::Certified(length) = length else {
        return None;
    };
    let duration = arrival_sample.checked_sub(start_sample)?;
    if duration == 0 || !valid_nonnegative(length) {
        return None;
    }
    let mut next_sample = start_sample;
    let mut missed_length_upper = 0.0;
    for panel in panels {
        if panel.start_sample != next_sample
            || panel.end_sample <= next_sample
            || panel.end_sample > arrival_sample
            || !valid_nonnegative(panel.initial_speed_gap_upper)
            || !valid_nonnegative(panel.speed_growth_upper)
        {
            return None;
        }
        let h = seconds_upper(panel.end_sample - next_sample);
        let excess = add_up(
            mul_up(h, panel.initial_speed_gap_upper),
            mul_up(mul_up(h / 2.0, h), panel.speed_growth_upper),
        );
        missed_length_upper = add_up(missed_length_upper, excess);
        next_sample = panel.end_sample;
    }
    if next_sample != arrival_sample || !missed_length_upper.is_finite() {
        return None;
    }
    let covered_length_lower = sub_down(length, missed_length_upper);
    Some(SampledActionBound {
        missed_length_upper,
        covered_length_lower,
        minimum_kinetic_action: action_floor(covered_length_lower, seconds_upper(duration)),
    })
}

/// Apply necessary bounds to a deadline on the canonical sample clock.
/// With a certified length bound `B`, a supplied positive speed cap `V`
/// gives `T >= B/V`. For positive `T`, Cauchy–Schwarz gives
/// `integral(K dt) >= B²/(2T)` for paths with square-integrable speed.
/// An arrival by the deadline can use at most the reported continuous and
/// impulse work upper bound, provided the supplied crest evidence applies.
/// Equality leaves each condition open. Passing every requested check
/// provides only `NotRuledOut`.
pub fn evaluate_feasibility(
    length: LengthLowerBound,
    duration_samples: u64,
    limits: FeasibilityLimits,
) -> FeasibilityAssessment {
    let duration_upper = seconds_upper(duration_samples);
    let certified_length = match length {
        LengthLowerBound::Certified(value) if valid_nonnegative(value) => Some(value),
        _ => None,
    };
    let speed_cap = limits.speed_cap.and_then(|cap| {
        let (SpeedCap::Certified(value) | SpeedCap::Enforced(value)) = cap;
        (value.is_finite() && value > 0.0).then_some(value)
    });

    let minimum_seconds = certified_length
        .zip(speed_cap)
        .map(|(length, speed)| div_down(length, speed));
    let distance_upper = speed_cap.map(|speed| mul_up(speed, duration_upper));
    let deadline_status = distance_upper.map_or(FeasibilityStatus::InsufficientEvidence, |upper| {
        compare(certified_length, upper)
    });
    let minimum_kinetic_action = certified_length
        .filter(|_| duration_samples > 0)
        .map(|length| action_floor(length, duration_upper));
    let kinetic_action_status = limits.kinetic_action_budget.map(|budget| {
        if valid_nonnegative(budget) {
            compare(minimum_kinetic_action, budget)
        } else {
            FeasibilityStatus::InsufficientEvidence
        }
    });
    let maximum_crest_work = limits
        .crest_work
        .zip(distance_upper)
        .and_then(|(constraint, distance)| crest_work_upper(constraint, distance));
    let crest_work_status = limits.crest_work.map(|constraint| {
        maximum_crest_work.map_or(FeasibilityStatus::InsufficientEvidence, |work| {
            compare(Some(constraint.required_work_lower), work)
        })
    });

    let mut status = FeasibilityStatus::NotRuledOut;
    for check in [
        Some(deadline_status),
        kinetic_action_status,
        crest_work_status,
    ]
    .into_iter()
    .flatten()
    {
        match check {
            FeasibilityStatus::RuledOut => {
                status = FeasibilityStatus::RuledOut;
                break;
            }
            FeasibilityStatus::InsufficientEvidence => status = check,
            FeasibilityStatus::NotRuledOut => {}
        }
    }
    FeasibilityAssessment {
        status,
        length,
        duration_samples,
        minimum_seconds,
        minimum_kinetic_action,
        maximum_crest_work,
        deadline_status,
        kinetic_action_status,
        crest_work_status,
    }
}

fn compare(lower: Option<f64>, upper: f64) -> FeasibilityStatus {
    match lower {
        Some(lower) if lower > upper => FeasibilityStatus::RuledOut,
        Some(_) => FeasibilityStatus::NotRuledOut,
        None => FeasibilityStatus::InsufficientEvidence,
    }
}

fn valid_nonnegative(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

pub(crate) fn action_floor(length_lower: f64, duration_upper: f64) -> f64 {
    let twice_duration = 2.0 * duration_upper;
    let half_mean_speed = if twice_duration.is_finite() {
        div_down(length_lower, twice_duration)
    } else {
        mul_down(0.5, div_down(length_lower, duration_upper))
    };
    mul_down(length_lower, half_mean_speed)
}

fn seconds_upper(samples: u64) -> f64 {
    if samples == 0 {
        return 0.0;
    }
    let nearest = samples as f64;
    let upper_samples = if (nearest as u128) < samples as u128 {
        nearest.next_up()
    } else {
        nearest
    };
    div_up(upper_samples, CANONICAL_SAMPLE_RATE as f64)
}

fn crest_work_upper(constraint: CrestWorkConstraint, distance_upper: f64) -> Option<f64> {
    let impulse_work = match constraint.impulse_work {
        ImpulseWork::None => 0.0,
        ImpulseWork::CertifiedUpper(work) if valid_nonnegative(work) => work,
        ImpulseWork::CertifiedUpper(_) | ImpulseWork::Unavailable => return None,
    };
    if !constraint.required_work_lower.is_finite()
        || !valid_nonnegative(constraint.control_force_cap)
    {
        return None;
    }
    let continuous_work = mul_up(constraint.control_force_cap, distance_upper);
    let total_upper = add_up(continuous_work, impulse_work);
    total_upper.is_finite().then_some(total_upper)
}

#[cfg(test)]
mod tests {
    use super::*;
    use FeasibilityStatus::{InsufficientEvidence, NotRuledOut, RuledOut};

    const SECOND: u64 = CANONICAL_SAMPLE_RATE as u64;

    #[test]
    fn sampled_action_accounts_for_unequal_panels_with_a_speed_jump() {
        // Each panel starts at speed 1.25 and then increases at unit
        // rate. At the boundary an impulse resets reference speed to 1.25.
        // Each recorded left endpoint speed is 1.0.
        let first = SampledActionPanel {
            start_sample: SECOND,
            end_sample: SECOND + SECOND / 4,
            initial_speed_gap_upper: 0.25,
            speed_growth_upper: 1.0,
        };
        let second = SampledActionPanel {
            start_sample: first.end_sample,
            end_sample: 2 * SECOND,
            ..first
        };
        let length = LengthLowerBound::Certified(1.5625);
        let run = |panels: &[SampledActionPanel]| {
            sampled_action_lower_bound(length, SECOND, 2 * SECOND, panels)
        };
        let bound = run(&[first, second]).unwrap();
        assert!(bound.missed_length_upper >= 0.5625);
        assert!(bound.covered_length_lower <= 1.0);
        assert!(bound.minimum_kinetic_action > 0.499999999);
        assert!(bound.minimum_kinetic_action <= 0.5);
        assert_eq!(run(&[first]), None);
        assert_eq!(run(&[second, first]), None);
        assert_eq!(
            run(&[
                first,
                SampledActionPanel {
                    start_sample: first.end_sample + 1,
                    ..second
                }
            ]),
            None
        );
        assert_eq!(
            sampled_action_lower_bound(
                LengthLowerBound::Estimate(1.5625),
                SECOND,
                2 * SECOND,
                &[first, second],
            ),
            None
        );
    }

    fn deadline(speed_cap: SpeedCap) -> FeasibilityLimits {
        FeasibilityLimits {
            speed_cap: Some(speed_cap),
            ..FeasibilityLimits::default()
        }
    }

    #[test]
    fn exact_deadline_stays_open_but_one_sample_earlier_does_not() {
        let length = LengthLowerBound::Certified(2.0);
        for cap in [SpeedCap::Certified(2.0), SpeedCap::Enforced(2.0)] {
            let at_deadline = evaluate_feasibility(length, SECOND, deadline(cap));
            assert_eq!(at_deadline.status, NotRuledOut);
            assert!(at_deadline.minimum_seconds.unwrap() <= 1.0);
            assert_eq!(
                evaluate_feasibility(length, SECOND - 1, deadline(cap)).status,
                RuledOut
            );
            assert_eq!(
                evaluate_feasibility(length, 0, deadline(cap)).status,
                RuledOut
            );
            let diagonal = evaluate_feasibility(LengthLowerBound::Certified(0.0), 0, deadline(cap));
            assert_eq!(diagonal.status, NotRuledOut);
            assert_eq!(diagonal.minimum_kinetic_action, None);
        }
    }

    #[test]
    fn uncertified_length_and_missing_or_invalid_caps_keep_deadline_unresolved() {
        for length in [
            LengthLowerBound::Estimate(10.0),
            LengthLowerBound::Unavailable,
            LengthLowerBound::Certified(f64::INFINITY),
            LengthLowerBound::Certified(-1.0),
        ] {
            let result = evaluate_feasibility(length, SECOND, deadline(SpeedCap::Enforced(1.0)));
            assert_eq!(result.status, InsufficientEvidence);
            assert_eq!(result.length, length);
            assert_eq!(result.minimum_seconds, None);
            assert_eq!(result.minimum_kinetic_action, None);
        }
        for cap in [
            None,
            Some(SpeedCap::Certified(0.0)),
            Some(SpeedCap::Enforced(-1.0)),
            Some(SpeedCap::Certified(f64::NAN)),
            Some(SpeedCap::Certified(f64::INFINITY)),
        ] {
            let result = evaluate_feasibility(
                LengthLowerBound::Certified(2.0),
                SECOND,
                FeasibilityLimits {
                    speed_cap: cap,
                    ..FeasibilityLimits::default()
                },
            );
            assert_eq!(result.status, InsufficientEvidence);
            assert_eq!(result.minimum_seconds, None);
            assert!(result.minimum_kinetic_action.unwrap() > 1.99);
        }
    }

    #[test]
    fn action_budget_can_rule_out_an_arrival_without_a_speed_cap() {
        let length = LengthLowerBound::Certified(2.0);
        let limits = FeasibilityLimits {
            kinetic_action_budget: Some(1.99),
            ..FeasibilityLimits::default()
        };
        let result = evaluate_feasibility(length, SECOND, limits);
        assert_eq!(result.deadline_status, InsufficientEvidence);
        assert_eq!(result.kinetic_action_status, Some(RuledOut));
        assert_eq!(result.status, RuledOut);
        let equality = evaluate_feasibility(
            length,
            SECOND,
            FeasibilityLimits {
                kinetic_action_budget: Some(2.0),
                speed_cap: Some(SpeedCap::Certified(2.0)),
                ..limits
            },
        );
        assert_eq!(equality.status, NotRuledOut);
        assert!(equality.minimum_kinetic_action.unwrap() <= 2.0);
        let missing_duration = evaluate_feasibility(LengthLowerBound::Certified(0.0), 0, limits);
        assert_eq!(
            missing_duration.kinetic_action_status,
            Some(InsufficientEvidence)
        );
    }

    #[test]
    fn crossing_trajectories_agree_with_independent_flat_shore_quadrature() {
        // In the flat a=0 metric, x=eps*sinh(q) makes q an arclength
        // coordinate for a normal trajectory. Integrate using x and x'
        // independently so the test exercises absolute speed on reversals.
        let eps = 0.01;
        let segment_time = 0.5;
        let length = LengthLowerBound::Certified(2.0);
        for (points, expected_length, expected_action, speed_cap) in [
            ([-1.0, 0.0, 1.0], 2.0, 2.0, 2.0),
            ([1.0, 0.0, -1.0], 2.0, 2.0, 2.0),
            ([-1.0, -0.5, 1.0], 2.0, 2.5, 3.0),
            ([-1.0, -1.5, 1.0], 3.0, 6.5, 5.0),
        ] {
            let bounds =
                evaluate_feasibility(length, SECOND, deadline(SpeedCap::Certified(speed_cap)));
            let mut measured_length = 0.0;
            let mut action = 0.0;
            for segment in points.windows(2) {
                let dq_dt = (segment[1] - segment[0]) / segment_time;
                let dt = segment_time / 256.0;
                for sample in 0..256 {
                    let q: f64 = segment[0] + dq_dt * (f64::from(sample) + 0.5) * dt;
                    let x = eps * q.sinh();
                    let dx_dt = eps * q.cosh() * dq_dt;
                    let speed = dx_dt.abs() / x.hypot(eps);
                    measured_length += speed * dt;
                    action += 0.5 * speed * speed * dt;
                }
            }
            assert!((measured_length - expected_length).abs() < 1e-12);
            assert!((action - expected_action).abs() < 1e-12);
            assert_eq!(bounds.deadline_status, NotRuledOut);
            assert!(bounds.minimum_kinetic_action.unwrap() <= action + 1e-12);
        }
    }

    #[test]
    fn crest_budget_uses_upper_travel_and_accounts_for_impulses() {
        // Certified decimal enclosures for the flat-Shore example in
        // docs/research/169-crossing-work-cap.md.
        let length = LengthLowerBound::Certified(1.44363);
        let mut crest = CrestWorkConstraint {
            required_work_lower: 4.64385,
            control_force_cap: 2.0,
            impulse_work: ImpulseWork::None,
        };
        let mut limits = deadline(SpeedCap::Certified(1.0));
        limits.crest_work = Some(crest);
        let result = evaluate_feasibility(length, 2 * SECOND, limits);
        assert_eq!(result.deadline_status, NotRuledOut);
        assert_eq!(result.crest_work_status, Some(RuledOut));
        assert_eq!(result.status, RuledOut);
        assert!(result.maximum_crest_work.unwrap() >= 4.0);
        // Three seconds permits more work than force times the length lower bound.
        assert_eq!(
            evaluate_feasibility(length, 3 * SECOND, limits).status,
            NotRuledOut
        );

        crest.impulse_work = ImpulseWork::CertifiedUpper(1.0);
        limits.crest_work = Some(crest);
        assert_eq!(
            evaluate_feasibility(length, 2 * SECOND, limits).status,
            NotRuledOut
        );
        crest.impulse_work = ImpulseWork::Unavailable;
        limits.crest_work = Some(crest);
        let result = evaluate_feasibility(length, 2 * SECOND, limits);
        assert_eq!(result.crest_work_status, Some(InsufficientEvidence));
        assert_eq!(result.maximum_crest_work, None);
        crest.impulse_work = ImpulseWork::None;
        limits.crest_work = Some(crest);
        limits.speed_cap = None;
        let result = evaluate_feasibility(length, 2 * SECOND, limits);
        assert_eq!(result.crest_work_status, Some(InsufficientEvidence));
        assert_eq!(result.maximum_crest_work, None);
    }

    #[test]
    fn rounding_and_overflow_keep_exact_or_unresolved_comparisons_open() {
        let one_sample = evaluate_feasibility(
            LengthLowerBound::Certified(1.0),
            1,
            deadline(SpeedCap::Certified(CANONICAL_SAMPLE_RATE as f64)),
        );
        assert_eq!(one_sample.status, NotRuledOut);
        assert!(one_sample.minimum_kinetic_action.unwrap() <= 24_000.0);
        let zero_samples = evaluate_feasibility(
            LengthLowerBound::Certified(f64::from_bits(1)),
            0,
            deadline(SpeedCap::Certified(f64::MAX)),
        );
        assert_eq!(zero_samples.minimum_seconds, Some(0.0));
        assert_eq!(zero_samples.deadline_status, RuledOut);
        // One sample above a representable integer rounds down on conversion.
        let samples = (1_u64 << 54) + 1;
        let nearest_seconds = samples as f64 / CANONICAL_SAMPLE_RATE as f64;
        assert!(seconds_upper(samples) > nearest_seconds);
        assert!(seconds_upper(u64::MAX).is_finite());

        let huge = evaluate_feasibility(
            LengthLowerBound::Certified(f64::MAX),
            u64::MAX,
            FeasibilityLimits {
                crest_work: Some(CrestWorkConstraint {
                    required_work_lower: f64::MAX,
                    control_force_cap: f64::MAX,
                    impulse_work: ImpulseWork::None,
                }),
                ..deadline(SpeedCap::Certified(f64::MAX))
            },
        );
        assert_eq!(huge.deadline_status, NotRuledOut);
        assert_eq!(huge.crest_work_status, Some(InsufficientEvidence));
        assert_eq!(huge.maximum_crest_work, None);
        assert_eq!(huge.minimum_kinetic_action, Some(f64::MAX));
    }
}
