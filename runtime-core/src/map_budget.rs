//! Error accounting for a normalized, anchored distance-field Map.
//!
//! Bounds supplied here must include the provider's field errors and all
//! coordinate and arithmetic errors not computed by this module. A provider
//! with only convergence evidence must use `Heuristic`, never `Certified`.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Assurance {
    Certified,
    Heuristic,
    Unavailable,
}

#[derive(Clone, Copy, Debug)]
pub struct MapErrorInputs {
    /// Computed positive ruler h-hat.
    pub estimated_scale: f64,
    /// Bound on |h-hat - h|, in physical coordinates.
    pub scale_error: f64,
    /// Pointwise field-error bounds at the anchor and query respectively.
    pub center_error: f64,
    pub query_error: f64,
    pub offset_norm: f64,
    /// Absolute normalized allowance for coordinate and other numerical errors.
    pub numerical_error: f64,
    pub assurance: Assurance,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapStatus {
    Certified,
    Unresolved,
    Uncertified,
}

#[derive(Clone, Copy, Debug)]
pub struct MapErrorAssessment {
    /// Absent when no error evidence is available; heuristic values are estimates.
    pub error_bound: Option<f64>,
    pub status: MapStatus,
    pub assurance: Assurance,
}

// Outward rounding keeps positive arithmetic from understating the budget.
fn up(value: f64) -> f64 {
    if value == f64::INFINITY {
        value
    } else if value == 0.0 {
        f64::from_bits(1)
    } else {
        f64::from_bits(value.to_bits() + 1)
    }
}

/// Evaluate the conservative computed-ruler bound. Returns `None` for
/// invalid or overflowing inputs. The target is absolute and dimensionless.
/// The true field must be 1-Lipschitz for the ruler term to apply.
pub fn assess_map_error(input: MapErrorInputs, tolerance: f64) -> Option<MapErrorAssessment> {
    let nonnegative = [
        input.scale_error,
        input.center_error,
        input.query_error,
        input.offset_norm,
        input.numerical_error,
        tolerance,
    ];
    if !input.estimated_scale.is_finite()
        || input.estimated_scale <= 0.0
        || nonnegative.iter().any(|v| !v.is_finite() || *v < 0.0)
    {
        return None;
    }
    // max(h, h-hat) >= h-hat, so using h-hat is conservative even
    // when the true h is unknown. Pointwise errors are summed, rather
    // than assuming cancellation of correlated errors.
    let field = up(up(input.center_error + input.query_error) / input.estimated_scale);
    let ruler = up(up(up(2.0 * input.offset_norm) * input.scale_error) / input.estimated_scale);
    let bound = up(up(field + ruler) + input.numerical_error);
    if !bound.is_finite() {
        return None;
    }
    let status = match input.assurance {
        Assurance::Certified if bound <= tolerance => MapStatus::Certified,
        Assurance::Certified => MapStatus::Unresolved,
        Assurance::Heuristic | Assurance::Unavailable => MapStatus::Uncertified,
    };
    Some(MapErrorAssessment {
        error_bound: (input.assurance != Assurance::Unavailable).then_some(bound),
        status,
        assurance: input.assurance,
    })
}

/// Equal pointwise error allocation for an exact fixed ruler: delta <= tau*h/2.
pub fn field_error_target(scale: f64, tolerance: f64) -> Option<f64> {
    if !scale.is_finite() || scale <= 0.0 || !tolerance.is_finite() || tolerance < 0.0 {
        return None;
    }
    let product = scale * tolerance;
    if !product.is_finite() {
        return None;
    }
    // Round down to avoid relaxing the requested bound.
    let product = if product > 0.0 {
        f64::from_bits(product.to_bits() - 1)
    } else {
        0.0
    };
    Some(product * 0.5)
}

/// Bound the linearized Map residual for an applicable whole-segment Hessian
/// bound. This does not establish segment validity or authorize query omission.
/// Gradient error is an operator-norm bound in physical field units.
pub fn residual_error_bound(
    hessian_bound: f64,
    scale: f64,
    offset_norm: f64,
    gradient_error: f64,
) -> Option<f64> {
    let values = [hessian_bound, scale, offset_norm, gradient_error];
    if values.iter().any(|v| !v.is_finite() || *v < 0.0) || scale == 0.0 {
        return None;
    }
    let curvature = up(up(up(up(hessian_bound * scale) * offset_norm) * offset_norm) * 0.5);
    let result = up(curvature + up(gradient_error * offset_norm));
    result.is_finite().then_some(result)
}
