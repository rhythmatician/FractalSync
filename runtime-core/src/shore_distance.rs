//! Intrinsic distance to the Shore in the unrestricted ideal graph metric.
//!
//! The radial profile is the analytic distance when its input is the true
//! Euclidean distance to a closed Shore. Restricting admissible paths can
//! increase the distance even with the same metric, so the unrestricted
//! profile then gives an analytic lower bound. See the checked results in
//! `formal/FractalSync/StableRadialProfile.lean` and the notes for #160 and #168.

use std::f64::consts::LN_2;

use serde::{Deserialize, Serialize};

use crate::geometry_provider::{GeometryJet, GeometryValidity, GEOMETRY_PROVIDER_VERSION};
use crate::manifold::ManifoldConfig;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShoreDistanceError {
    InvalidEpsilon,
    InvalidAnisotropy,
    InvalidRadius,
    InvalidRadiusError,
    UnavailableProviderDistance,
    Overflow,
}

/// Parameters of the ideal radial profile, with `anisotropy = lambda / ln(2)`.
#[derive(Clone, Copy, Debug)]
pub struct RadialProfile {
    epsilon: f64,
    anisotropy: f64,
}

impl RadialProfile {
    pub fn new(epsilon: f64, anisotropy: f64) -> Result<Self, ShoreDistanceError> {
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(ShoreDistanceError::InvalidEpsilon);
        }
        if !anisotropy.is_finite() || anisotropy < 0.0 {
            return Err(ShoreDistanceError::InvalidAnisotropy);
        }
        Ok(Self {
            epsilon,
            anisotropy,
        })
    }

    /// Binary64 evaluation of the checked closed form. Very small results may
    /// round to zero. The returned float carries no directed rounding guarantee.
    pub fn evaluate(self, radius: f64) -> Result<f64, ShoreDistanceError> {
        if !radius.is_finite() || radius < 0.0 {
            return Err(ShoreDistanceError::InvalidRadius);
        }
        if radius == 0.0 {
            return Ok(0.0);
        }

        let a = self.anisotropy;
        let b = a.hypot(1.0);
        // C = b/(b+a) without overflowing b+a. k = C/b = b-a.
        let scale = 1.0 / (1.0 + a / b);
        let q = radius / self.epsilon;
        if q.is_infinite() {
            // Both asinh arguments are huge. Their reciprocal-square
            // corrections and the second argument's hypot ratio round away.
            let log_q = radius.ln() - self.epsilon.ln();
            let first = (log_q + b.ln() + LN_2) * (scale / b);
            let second = a * (log_q + scale.ln() + LN_2);
            return finite_result(first + second);
        }

        let bq = b * q;
        if bq <= 1e-8 {
            // The normalized density differs from one by at most (bq)²
            // over this interval. Rounding the two summands separately
            // can lose the entire profile when q is the smallest subnormal.
            return Ok(q);
        }
        let large_argument = if bq.is_infinite() {
            b.ln() + q.ln() + LN_2
        } else {
            asinh_nonnegative(bq)
        };
        let first = large_argument * (scale / b);
        if a == 0.0 {
            return Ok(first);
        }

        let ratio = q.hypot(b.recip()) / q.hypot(1.0);
        let second_arg = q * scale * ratio;
        let second = if second_arg <= 1e-8 {
            // asinh(z)/z differs from one by at most z²/6 here. Apply
            // the weight first so a subnormal or underflowed z can contribute.
            q * ((a * scale) * ratio)
        } else {
            a * asinh_nonnegative(second_arg)
        };
        finite_result(first + second)
    }

    /// Evaluate the sharp analytic sensitivity bound M(a) * delta / epsilon
    /// for an independently certified absolute radius error `delta`.
    /// This accounts for radius error alone, with parameters held fixed.
    /// The return value is a floating evaluation of the bound.
    pub fn propagate_radius_error(self, delta: f64) -> Result<f64, ShoreDistanceError> {
        if !delta.is_finite() || delta < 0.0 {
            return Err(ShoreDistanceError::InvalidRadiusError);
        }
        if delta == 0.0 {
            return Ok(0.0);
        }
        let a = self.anisotropy;
        let max_density = if a <= 1.0 { 1.0 } else { 0.5 * a + 0.5 / a };
        let quotient = delta / self.epsilon;
        let bound = if quotient < f64::MIN_POSITIVE && max_density > 1.0 {
            (delta.ln() - self.epsilon.ln() + max_density.ln()).exp()
        } else {
            max_density * quotient
        };
        finite_result(bound)
    }
}

fn asinh_nonnegative(x: f64) -> f64 {
    if x <= 1.0 {
        // Rationalize hypot(1,x) - 1 near zero.
        (x + x * (x / (1.0 + x.hypot(1.0)))).ln_1p()
    } else {
        x.ln() + x.recip().hypot(1.0).ln_1p()
    }
}

fn finite_result(value: f64) -> Result<f64, ShoreDistanceError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(ShoreDistanceError::Overflow)
    }
}

/// Diagnostic estimate from the raster bridge. Keep the source jet with the
/// result to retain the signed radius, provider identity, scale and validity.
/// Raster spacing supplies no certified radius error.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ShoreDistanceEstimate {
    pub value: f64,
    pub epsilon: f64,
    pub anisotropy: f64,
}

impl ShoreDistanceEstimate {
    /// Reuse a coherent query at the supplied configuration's epsilon. A finite
    /// raster radius can support a best-effort estimate at an unresolved or
    /// singular jet. A jet outside the provider or from an operational
    /// scalar with unknown Shore-distance correspondence cannot support it.
    pub fn from_jet(
        jet: &GeometryJet,
        config: &ManifoldConfig,
    ) -> Result<Self, ShoreDistanceError> {
        if !jet.is_bridge
            || jet.provider_version != GEOMETRY_PROVIDER_VERSION
            || !jet.d.is_finite()
            || matches!(
                jet.validity,
                GeometryValidity::OutsideDomain | GeometryValidity::ProviderFailure
            )
        {
            return Err(ShoreDistanceError::UnavailableProviderDistance);
        }
        let anisotropy = config.lambda_sq.sqrt() / LN_2;
        let profile = RadialProfile::new(config.epsilon, anisotropy)?;
        Ok(Self {
            value: profile.evaluate(jet.d.abs())?,
            epsilon: config.epsilon,
            anisotropy,
        })
    }
}
