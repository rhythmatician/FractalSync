//! Offline differential self-consistency evidence, not reference fidelity.
use super::{FieldJet, GeometryUnavailable, PreparedField};
use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerificationError {
    InvalidNeighborhood,
    GeometryUnavailable,
    NonfiniteJet,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct DifferentialEvidence {
    pub gradient_max_absolute_error: f64,
    pub hessian_max_normalized_error: f64,
    pub gradient_resolution_max_absolute_error: f64,
    pub hessian_resolution_max_normalized_error: f64,
    pub legacy_gradient_max_absolute_error: f64,
    pub legacy_hessian_max_normalized_error: f64,
}
impl DifferentialEvidence {
    pub fn passes(&self, tolerance: f64) -> bool {
        tolerance.is_finite()
            && tolerance > 0.0
            && [
                self.gradient_max_absolute_error,
                self.hessian_max_normalized_error,
                self.gradient_resolution_max_absolute_error,
                self.hessian_resolution_max_normalized_error,
            ]
            .iter()
            .all(|e| e.is_finite() && *e < tolerance)
    }
    fn merge(&mut self, other: Self) {
        self.gradient_max_absolute_error = self
            .gradient_max_absolute_error
            .max(other.gradient_max_absolute_error);
        self.hessian_max_normalized_error = self
            .hessian_max_normalized_error
            .max(other.hessian_max_normalized_error);
        self.gradient_resolution_max_absolute_error = self
            .gradient_resolution_max_absolute_error
            .max(other.gradient_resolution_max_absolute_error);
        self.hessian_resolution_max_normalized_error = self
            .hessian_resolution_max_normalized_error
            .max(other.hessian_resolution_max_normalized_error);
        self.legacy_gradient_max_absolute_error = self
            .legacy_gradient_max_absolute_error
            .max(other.legacy_gradient_max_absolute_error);
        self.legacy_hessian_max_normalized_error = self
            .legacy_hessian_max_normalized_error
            .max(other.legacy_hessian_max_normalized_error);
    }
}
fn finite(j: &FieldJet) -> bool {
    j.value.is_finite()
        && j.gradient
            .iter()
            .chain(j.hessian.iter().flatten())
            .all(|v| v.is_finite())
}
pub fn neighboring_evidence(
    query: impl Fn([f64; 2]) -> Result<FieldJet, GeometryUnavailable>,
    point: [f64; 2],
    delta: f64,
) -> Result<DifferentialEvidence, VerificationError> {
    if !point.iter().all(|v| v.is_finite())
        || !delta.is_finite()
        || delta <= 0.0
        || delta / 4.0 == 0.0
    {
        return Err(VerificationError::InvalidNeighborhood);
    }
    let center = query(point).map_err(|_| VerificationError::GeometryUnavailable)?;
    if !finite(&center) {
        return Err(VerificationError::NonfiniteJet);
    }
    let mut evidence = DifferentialEvidence::default();
    for axis in 0..2 {
        let mut estimates = [[0.0; 3]; 3];
        for (level, step) in [delta, delta / 2.0, delta / 4.0].into_iter().enumerate() {
            let (mut plus, mut minus) = (point, point);
            plus[axis] += step;
            minus[axis] -= step;
            if !plus[axis].is_finite()
                || !minus[axis].is_finite()
                || plus[axis] == point[axis]
                || minus[axis] == point[axis]
            {
                return Err(VerificationError::InvalidNeighborhood);
            }
            let plus = query(plus).map_err(|_| VerificationError::GeometryUnavailable)?;
            let minus = query(minus).map_err(|_| VerificationError::GeometryUnavailable)?;
            if !finite(&plus) || !finite(&minus) {
                return Err(VerificationError::NonfiniteJet);
            }
            estimates[level][0] = (plus.value - minus.value) / (2.0 * step);
            for component in 0..2 {
                estimates[level][component + 1] =
                    (plus.gradient[component] - minus.gradient[component]) / (2.0 * step);
            }
        }
        if !estimates.iter().flatten().all(|v| v.is_finite()) {
            return Err(VerificationError::NonfiniteJet);
        }
        for component in 0..3 {
            // Central differences have an O(h^2) truncation term. Cancel it at
            // two successive resolutions, and require both estimates to agree
            // with the jet and with each other. Never select a favorable step.
            let coarse = (4.0 * estimates[1][component] - estimates[0][component]) / 3.0;
            let fine = (4.0 * estimates[2][component] - estimates[1][component]) / 3.0;
            let expected = if component == 0 {
                center.gradient[axis]
            } else {
                center.hessian[component - 1][axis]
            };
            let scale = if component == 0 {
                1.0
            } else {
                expected.abs().max(1.0)
            };
            let error = (coarse - expected).abs().max((fine - expected).abs()) / scale;
            let agreement = (coarse - fine).abs() / scale;
            let legacy = (estimates[0][component] - expected).abs() / scale;
            if ![coarse, fine, error, agreement, legacy]
                .iter()
                .all(|v| v.is_finite())
            {
                return Err(VerificationError::NonfiniteJet);
            }
            if component == 0 {
                evidence.gradient_max_absolute_error =
                    evidence.gradient_max_absolute_error.max(error);
                evidence.gradient_resolution_max_absolute_error = evidence
                    .gradient_resolution_max_absolute_error
                    .max(agreement);
                evidence.legacy_gradient_max_absolute_error =
                    evidence.legacy_gradient_max_absolute_error.max(legacy);
            } else {
                evidence.hessian_max_normalized_error =
                    evidence.hessian_max_normalized_error.max(error);
                evidence.hessian_resolution_max_normalized_error = evidence
                    .hessian_resolution_max_normalized_error
                    .max(agreement);
                evidence.legacy_hessian_max_normalized_error =
                    evidence.legacy_hessian_max_normalized_error.max(legacy);
            }
        }
    }
    Ok(evidence)
}

/// Same fixed 1,000 points, original increments and 1e-4 tolerance as the first
/// report. Extra nested samples improve the estimator, not the acceptance limit.
pub fn fixed_neighbor_report(
    field: &PreparedField,
) -> Result<serde_json::Value, VerificationError> {
    let mut evidence = DifferentialEvidence::default();
    for i in 0..1000 {
        let point = [
            0.2 + (i % 997) as f64 / 997.0 * 0.4,
            0.025 + (i % 293) as f64 / 293.0 * 0.15,
        ];
        for delta in [1e-6, 1e-7] {
            evidence.merge(neighboring_evidence(|p| field.query(p), point, delta)?);
        }
    }
    Ok(
        serde_json::json!({"verifier":"richardson-neighbor/1","points":1000,"base_increments":[1e-6,1e-7],"tolerance":1e-4,"evidence":evidence,"passed":evidence.passes(1e-4),"legacy_central_difference_passed":evidence.legacy_gradient_max_absolute_error<1e-4 && evidence.legacy_hessian_max_normalized_error<1e-4,"scope":"measured differential self-consistency only; no reference fidelity or singularity permission"}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    fn quartic(p: [f64; 2]) -> Result<FieldJet, GeometryUnavailable> {
        let [x, y] = p;
        let a = 1e8;
        Ok(FieldJet {
            value: x * x + 3.0 * x * y + 2.0 * y * y + a * (x.powi(4) + y.powi(4)),
            gradient: [
                2.0 * x + 3.0 * y + 4.0 * a * x.powi(3),
                3.0 * x + 4.0 * y + 4.0 * a * y.powi(3),
            ],
            hessian: [[2.0 + 12.0 * a * x * x, 3.0], [3.0, 4.0 + 12.0 * a * y * y]],
        })
    }
    #[test]
    fn exact_quartic_derivatives_do_not_fail_from_central_truncation() {
        let result = neighboring_evidence(quartic, [0.0; 2], 1e-6).unwrap();
        assert!(result.legacy_hessian_max_normalized_error > 1e-4);
        assert!(result.passes(1e-4), "{result:?}");
    }
    #[test]
    fn corrupted_jets_and_unavailable_support_still_refuse() {
        for hessian in [false, true] {
            let query = |p| {
                let mut j = quartic(p)?;
                if hessian {
                    j.hessian[0][1] += 0.01;
                    j.hessian[1][0] += 0.01;
                } else {
                    j.gradient[0] += 0.001;
                }
                Ok(j)
            };
            assert!(!neighboring_evidence(query, [0.0; 2], 1e-6)
                .unwrap()
                .passes(1e-4));
        }
        assert_eq!(
            neighboring_evidence(
                |p| if p[0] > 0.0 {
                    Err(GeometryUnavailable)
                } else {
                    quartic(p)
                },
                [0.0; 2],
                1e-6
            )
            .unwrap_err(),
            VerificationError::GeometryUnavailable
        );
    }
    #[test]
    fn invalid_neighborhoods_and_nonfinite_jets_cannot_pass() {
        for delta in [0.0, f64::NAN, f64::INFINITY] {
            assert!(neighboring_evidence(quartic, [0.0; 2], delta).is_err());
        }
        assert!(neighboring_evidence(quartic, [f64::NAN, 0.0], 1e-6).is_err());
        assert!(neighboring_evidence(
            |p| {
                let mut j = quartic(p)?;
                j.hessian[1][1] = f64::NAN;
                Ok(j)
            },
            [0.0; 2],
            1e-6
        )
        .is_err());
    }
}
