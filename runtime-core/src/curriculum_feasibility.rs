//! Training-side checks for explicit timed navigation tasks.

use crate::crossing_feasibility::{
    evaluate_feasibility, FeasibilityAssessment, FeasibilityLimits, LengthLowerBound,
};
use crate::distance_field::RasterNavigationRegion;
use crate::intrinsic_bounds::{
    intrinsic_target_lower_bound, point_lower_bound, IntrinsicMetric, INTRINSIC_BOUNDS_VERSION,
};
use crate::manifold::ManifoldConfig;
use crate::map_certificates::{
    certify_mandelbrot_endpoint, displacement_lower, MandelbrotEndpoint,
};
use crate::timebase::CANONICAL_SAMPLE_RATE;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum CurriculumMetric {
    RuntimeMap,
    IdealDistanceGraph,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CurriculumTarget {
    pub start: [f64; 2],
    pub target: [f64; 2],
    pub metric: CurriculumMetric,
    pub intrinsic_tolerance: f64,
    pub start_sample: u64,
    pub deadline_sample: u64,
    pub kinetic_action_budget: Option<f64>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CurriculumProof {
    bounds_version: &'static str,
    epsilon: f64,
    anisotropy: f64,
    start: MandelbrotEndpoint,
    target: MandelbrotEndpoint,
    displacement_lower: f64,
    point_length_lower: f64,
    target_length_lower: f64,
}

/// Conditional continuous-path diagnostic for a fixed default raster and Physics profile.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RasterCurriculumDiagnostic {
    pub raster_generation: u64,
    pub continuous_assessment: FeasibilityAssessment,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CurriculumFeasibility {
    schema: &'static str,
    sample_rate: usize,
    #[serde(flatten)]
    task: CurriculumTarget,
    proof: Option<CurriculumProof>,
    reference_status: &'static str,
    runtime_geometry_version: Option<&'static str>,
    assessment: FeasibilityAssessment,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_raster_diagnostic: Option<RasterCurriculumDiagnostic>,
}

fn ideal_proof(task: CurriculumTarget) -> Option<CurriculumProof> {
    let profile = ManifoldConfig::default();
    let metric = IntrinsicMetric {
        epsilon: profile.epsilon,
        a: profile.lambda_sq.sqrt() / std::f64::consts::LN_2,
    };
    let start = certify_mandelbrot_endpoint(task.start)?;
    let target = certify_mandelbrot_endpoint(task.target)?;
    let displacement_lower = displacement_lower(task.start, task.target)?;
    let bound = point_lower_bound(
        metric,
        start.bounds(metric)?,
        target.bounds(metric)?,
        displacement_lower,
        start.crossing_to(target),
    )?;
    Some(CurriculumProof {
        bounds_version: INTRINSIC_BOUNDS_VERSION,
        epsilon: metric.epsilon,
        anisotropy: metric.a,
        start,
        target,
        displacement_lower,
        point_length_lower: bound.lower,
        target_length_lower: intrinsic_target_lower_bound(bound.lower, task.intrinsic_tolerance)?,
    })
}

/// Check a deadline and optional action budget for a declared arrival metric.
/// Runtime Map tasks currently lack an ideal-to-operational correspondence
/// certificate. No certified or enforced intrinsic speed cap is declared by
/// this training entry point.
pub fn check_curriculum_target(
    task: CurriculumTarget,
) -> Result<CurriculumFeasibility, &'static str> {
    check_target(task, None)
}

/// Attach a continuous-path diagnostic to a Runtime Map task. The intrinsic
/// arrival ball is taken in the same fixed raster metric as the path. A
/// discrete runtime rollout still needs its own action correspondence.
pub fn check_curriculum_target_with_default_raster(
    task: CurriculumTarget,
    region: &RasterNavigationRegion,
) -> Result<CurriculumFeasibility, &'static str> {
    check_target(task, Some(region))
}

fn check_target(
    task: CurriculumTarget,
    raster: Option<&RasterNavigationRegion>,
) -> Result<CurriculumFeasibility, &'static str> {
    if !task
        .start
        .into_iter()
        .chain(task.target)
        .all(f64::is_finite)
    {
        return Err("curriculum positions must be finite");
    }
    if !task.intrinsic_tolerance.is_finite() || task.intrinsic_tolerance < 0.0 {
        return Err("intrinsic arrival tolerance must be finite and nonnegative");
    }
    if task
        .kinetic_action_budget
        .is_some_and(|b| !b.is_finite() || b < 0.0)
    {
        return Err("kinetic action budget must be finite and nonnegative");
    }
    let duration = task
        .deadline_sample
        .checked_sub(task.start_sample)
        .ok_or("deadline sample precedes start sample")?;
    let proof = match task.metric {
        CurriculumMetric::RuntimeMap => None,
        CurriculumMetric::IdealDistanceGraph => ideal_proof(task),
    };
    let (reference_status, runtime_geometry_version) = match task.metric {
        CurriculumMetric::RuntimeMap => (
            "runtimeMetricCorrespondenceUncertified",
            Some(crate::geometry_provider::GEOMETRY_PROVIDER_VERSION),
        ),
        CurriculumMetric::IdealDistanceGraph if proof.is_some() => ("idealDistanceBounds", None),
        CurriculumMetric::IdealDistanceGraph => ("idealDistanceBoundsUnavailable", None),
    };
    let length = proof.as_ref().map_or(LengthLowerBound::Unavailable, |p| {
        LengthLowerBound::Certified(p.target_length_lower)
    });
    let limits = FeasibilityLimits {
        kinetic_action_budget: task.kinetic_action_budget,
        ..Default::default()
    };
    let assessment = evaluate_feasibility(length, duration, limits);
    let default_raster_diagnostic = match task.metric {
        CurriculumMetric::RuntimeMap => {
            raster
                .filter(|r| r.contains(task.target))
                .and_then(|region| {
                    let point = region.target_lower_bound(task.start, task.target, 0.0)?;
                    let target = intrinsic_target_lower_bound(point, task.intrinsic_tolerance)?;
                    Some(RasterCurriculumDiagnostic {
                        raster_generation: region.raster_generation(),
                        continuous_assessment: evaluate_feasibility(
                            LengthLowerBound::Certified(target),
                            duration,
                            limits,
                        ),
                    })
                })
        }
        CurriculumMetric::IdealDistanceGraph => None,
    };
    Ok(CurriculumFeasibility {
        schema: "curriculum-feasibility/1",
        sample_rate: CANONICAL_SAMPLE_RATE,
        task,
        proof,
        reference_status,
        runtime_geometry_version,
        assessment,
        default_raster_diagnostic,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crossing_feasibility::FeasibilityStatus;

    fn target(metric: CurriculumMetric) -> CurriculumTarget {
        CurriculumTarget {
            start: [0.0, 0.0],
            target: [0.375, 0.0],
            metric,
            intrinsic_tolerance: 0.0,
            start_sample: 48_000,
            deadline_sample: 96_000,
            kinetic_action_budget: Some(0.25),
        }
    }

    #[test]
    fn ideal_target_uses_only_applicable_certified_conditions() {
        let task = target(CurriculumMetric::IdealDistanceGraph);
        let excluded = check_curriculum_target(task).unwrap();
        assert!(excluded.proof.is_some());
        assert_eq!(excluded.assessment.duration_samples, 48_000);
        assert_eq!(excluded.assessment.status, FeasibilityStatus::RuledOut);
        assert_eq!(
            excluded.assessment.deadline_status,
            FeasibilityStatus::InsufficientEvidence
        );
        assert_eq!(
            excluded.assessment.kinetic_action_status,
            Some(FeasibilityStatus::RuledOut)
        );

        let tolerance = excluded.proof.unwrap().point_length_lower.next_up();
        let tolerated = check_curriculum_target(CurriculumTarget {
            intrinsic_tolerance: tolerance,
            ..task
        })
        .unwrap();
        assert_eq!(
            tolerated.assessment.length,
            LengthLowerBound::Certified(0.0)
        );
        assert_eq!(
            tolerated.assessment.status,
            FeasibilityStatus::InsufficientEvidence
        );
        assert_eq!(
            tolerated.assessment.kinetic_action_status,
            Some(FeasibilityStatus::NotRuledOut)
        );
    }

    #[test]
    fn fixed_raster_diagnostics_use_intrinsic_tolerance_without_filtering_runtime() {
        use crate::distance_field::{
            clear_distance_field, global_test_mutex, load_builtin_distance_field,
        };

        let _guard = global_test_mutex()
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        load_builtin_distance_field("mandelbrot_default").unwrap();
        let region =
            RasterNavigationRegion::for_default_profile([[-1.5, -0.45], [-0.4, 0.45]]).unwrap();
        let task = CurriculumTarget {
            start: [-1.0, 0.0],
            target: [-0.7, 0.1],
            kinetic_action_budget: Some(0.02),
            ..target(CurriculumMetric::RuntimeMap)
        };
        let report = check_curriculum_target_with_default_raster(task, &region).unwrap();
        assert_eq!(
            report.assessment.status,
            FeasibilityStatus::InsufficientEvidence
        );
        let diagnostic = report.default_raster_diagnostic.unwrap();
        assert_eq!(diagnostic.raster_generation, region.raster_generation());
        assert_eq!(
            diagnostic.continuous_assessment.kinetic_action_status,
            Some(FeasibilityStatus::RuledOut)
        );
        assert_eq!(
            diagnostic.continuous_assessment.deadline_status,
            FeasibilityStatus::InsufficientEvidence
        );

        let tolerated = check_curriculum_target_with_default_raster(
            CurriculumTarget {
                intrinsic_tolerance: 0.4,
                ..task
            },
            &region,
        )
        .unwrap();
        assert_eq!(
            tolerated
                .default_raster_diagnostic
                .unwrap()
                .continuous_assessment
                .length,
            LengthLowerBound::Certified(0.0)
        );

        clear_distance_field();
        let stale = check_curriculum_target_with_default_raster(task, &region).unwrap();
        assert!(stale.default_raster_diagnostic.is_none());
    }

    #[test]
    fn runtime_tasks_and_invalid_sample_windows_do_not_inherit_ideal_certificates() {
        let task = target(CurriculumMetric::RuntimeMap);
        let report = check_curriculum_target(task).unwrap();
        assert!(report.proof.is_none());
        assert_eq!(
            report.assessment.status,
            FeasibilityStatus::InsufficientEvidence
        );
        assert!(check_curriculum_target(CurriculumTarget {
            deadline_sample: 1,
            ..task
        })
        .is_err());
    }
}
