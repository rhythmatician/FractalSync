//! ADR 0005 experiment. Immutable C4 quintic B-spline, not canonical Shore distance.
//! Coefficients are samples, not interpolation coefficients. Fidelity must be
//! measured separately; smoothness never certifies Mandelbrot membership.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FieldJet {
    pub value: f64,
    pub gradient: [f64; 2],
    pub hessian: [[f64; 2]; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GeometryUnavailable;

/// Construction is offline. No interior mutability, caches or query-time fitting.
pub struct PreparedField {
    origin: [f64; 2],
    spacing: f64,
    shape: [usize; 2],
    coefficients: Box<[f64]>,
    identity: u64,
}

impl PreparedField {
    pub fn new(
        origin: [f64; 2],
        spacing: f64,
        shape: [usize; 2],
        coefficients: Vec<f64>,
    ) -> Result<Self, &'static str> {
        if !origin.iter().all(|x| x.is_finite())
            || !spacing.is_finite()
            || spacing <= 0.0
            || shape.iter().any(|n| *n < 6)
            || shape[0].checked_mul(shape[1]) != Some(coefficients.len())
            || !coefficients.iter().all(|x| x.is_finite())
            || !(1.0 / spacing.powi(2)).is_finite()
            || origin
                .iter()
                .zip(shape)
                .any(|(x, n)| !(x + spacing * n as f64).is_finite())
        {
            return Err("invalid operational field asset");
        }
        // Diagnostic content fingerprint, not a cryptographic certificate.
        let mut identity = 0xcbf29ce484222325u64;
        for bits in [
            5,
            shape[0] as u64,
            shape[1] as u64,
            origin[0].to_bits(),
            origin[1].to_bits(),
            spacing.to_bits(),
        ]
        .into_iter()
        .chain(coefficients.iter().map(|v| v.to_bits()))
        {
            for byte in bits.to_le_bytes() {
                identity = (identity ^ byte as u64).wrapping_mul(0x100000001b3);
            }
        }
        Ok(Self {
            origin,
            spacing,
            shape,
            coefficients: coefficients.into_boxed_slice(),
            identity,
        })
    }

    pub fn identity(&self) -> u64 {
        self.identity
    }
    pub fn coefficient_bytes(&self) -> usize {
        self.coefficients.len() * 8
    }
    pub fn spacing(&self) -> f64 {
        self.spacing
    }

    /// Exactly 36 coefficient reads on success. Both axes require a full halo.
    /// Upper edge is exclusive; unavailable/nonfinite coordinates return early.
    pub fn query(&self, point: [f64; 2]) -> Result<FieldJet, GeometryUnavailable> {
        let q = [
            (point[0] - self.origin[0]) / self.spacing,
            (point[1] - self.origin[1]) / self.spacing,
        ];
        if (0..2).any(|a| !q[a].is_finite() || q[a] < 2.0 || q[a] >= (self.shape[a] - 3) as f64) {
            return Err(GeometryUnavailable);
        }
        let ix = q[0].floor() as usize;
        let iy = q[1].floor() as usize;
        let wx = basis(q[0] - ix as f64);
        let wy = basis(q[1] - iy as f64);
        let mut jet = FieldJet {
            value: 0.0,
            gradient: [0.0; 2],
            hessian: [[0.0; 2]; 2],
        };
        for y in 0..6 {
            for x in 0..6 {
                let c = self.coefficients[(iy + y - 2) * self.shape[0] + ix + x - 2];
                jet.value += c * wx[0][x] * wy[0][y];
                jet.gradient[0] += c * wx[1][x] * wy[0][y];
                jet.gradient[1] += c * wx[0][x] * wy[1][y];
                jet.hessian[0][0] += c * wx[2][x] * wy[0][y];
                jet.hessian[0][1] += c * wx[1][x] * wy[1][y];
                jet.hessian[1][1] += c * wx[0][x] * wy[2][y];
            }
        }
        for a in 0..2 {
            jet.gradient[a] /= self.spacing;
            for b in a..2 {
                jet.hessian[a][b] /= self.spacing * self.spacing;
            }
        }
        jet.hessian[1][0] = jet.hessian[0][1];
        if !jet.value.is_finite()
            || !jet.gradient.iter().all(|v| v.is_finite())
            || !jet.hessian.iter().flatten().all(|v| v.is_finite())
        {
            return Err(GeometryUnavailable);
        }
        Ok(jet)
    }
}

fn basis(t: f64) -> [[f64; 6]; 3] {
    const POLYNOMIALS: [[f64; 6]; 6] = [
        [1.0, -5.0, 10.0, -10.0, 5.0, -1.0],
        [26.0, -50.0, 20.0, 20.0, -20.0, 5.0],
        [66.0, 0.0, -60.0, 0.0, 30.0, -10.0],
        [26.0, 50.0, 20.0, -20.0, -20.0, 10.0],
        [1.0, 5.0, 10.0, 10.0, 5.0, -5.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ];
    let mut weights = [[0.0; 6]; 3];
    for (i, p) in POLYNOMIALS.iter().enumerate() {
        let (mut value, mut first, mut second) = (p[5], 0.0, 0.0);
        for k in (0..5).rev() {
            second = second * t + 2.0 * first;
            first = first * t + value;
            value = value * t + p[k];
        }
        weights[0][i] = value / 120.0;
        weights[1][i] = first / 120.0;
        weights[2][i] = second / 120.0;
    }
    weights
}

thread_local! {
    static ACTIVE: std::cell::RefCell<Option<std::sync::Arc<PreparedField>>> = const { std::cell::RefCell::new(None) };
}

pub const PROFILE: &str = "experimental-operational-field/1;reference-uncertified;cuts-unknown";

#[cfg(feature = "python")]
pub(crate) fn active_profile() -> Option<&'static str> {
    ACTIVE.with(|active| active.borrow().as_ref().map(|_| PROFILE))
}

/// Explicit experimental thread-local activation. An owned immutable asset is
/// pinned for the whole replay; ordinary builds have no override at all.
pub fn activate(field: Option<std::sync::Arc<PreparedField>>) {
    ACTIVE.with(|active| *active.borrow_mut() = field);
}

pub(crate) fn active_query(
    c: num_complex::Complex64,
    epsilon: f64,
) -> Option<Result<crate::geometry_provider::GeometryJet, String>> {
    use crate::geometry_provider::{GeometryJet, GeometryValidity, SingularityKind};
    ACTIVE.with(|active| {
        let active = active.borrow();
        let field = active.as_ref()?;
        Some((|| {
            if !epsilon.is_finite() || epsilon < 1e-4 {
                return Err("GeometryUnavailable: prototype supports epsilon >= 1e-4".into());
            }
            let jet = field
                .query([c.re, c.im])
                .map_err(|_| "GeometryUnavailable: outside prepared coverage".to_string())?;
            Ok(GeometryJet {
                d: jet.value,
                grad_d: jet.gradient,
                hessian_d: jet.hessian,
                resolved_scale: field.spacing,
                requested_scale: 0.1 * jet.value.hypot(epsilon),
                // Unknown true-Shore error; never pass this off as certification.
                estimated_error: f64::MAX,
                validity: GeometryValidity::Regular,
                singularity: SingularityKind::None,
                provider_version: PROFILE.into(),
                tile_id: format!("operational:{:016x}", field.identity),
                is_bridge: false,
            })
        })())
    })
}

/// Offline baseline only: smooth samples of the existing raster. This deliberately
/// carries no true-boundary fidelity or reference-cut certificate.
pub fn prepare_raster_baseline() -> Result<PreparedField, String> {
    let origin = [-0.1, -0.2];
    let spacing = 0.0005;
    let shape = [1605, 1005];
    crate::distance_field::load_builtin_distance_field("default")?;
    let mut coefficients = Vec::with_capacity(shape[0] * shape[1]);
    for y in 0..shape[1] {
        let row: Vec<_> = (0..shape[0])
            .map(|x| {
                num_complex::Complex64::new(
                    origin[0] + x as f64 * spacing,
                    origin[1] + y as f64 * spacing,
                )
            })
            .collect();
        coefficients.extend(
            crate::distance_field::sample_signed_distance_field(&row)?
                .into_iter()
                .map(f64::from),
        );
    }
    PreparedField::new(origin, spacing, shape, coefficients).map_err(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn field() -> PreparedField {
        let mut c = Vec::new();
        for y in 0..16 {
            for x in 0..16 {
                let (x, y) = (x as f64 * 0.1, y as f64 * 0.1);
                c.push(x * x + 2.0 * x * y - 3.0 * y * y);
            }
        }
        PreparedField::new([0.0; 2], 0.1, [16; 2], c).unwrap()
    }
    #[test]
    fn analytic_polynomial_and_neighbor_derivatives_at_joins() {
        let f = field();
        // Samples as B-spline coefficients reproduce a quadratic plus a known
        // constant bias h^2/4 times its Laplacian. Derivatives remain exact.
        for x in [0.4999999, 0.5, 0.5000001, 0.573] {
            let y = 0.61;
            let j = f.query([x, y]).unwrap();
            assert!((j.value - (x * x + 2.0 * x * y - 3.0 * y * y - 0.01)).abs() < 1e-12);
            assert!((j.gradient[0] - (2.0 * x + 2.0 * y)).abs() < 1e-12);
            assert!((j.hessian[0][1] - 2.0).abs() < 1e-10);
            for a in 0..2 {
                let mut p = [x, y];
                let mut m = p;
                p[a] += 1e-6;
                m[a] -= 1e-6;
                let (p, m) = (f.query(p).unwrap(), f.query(m).unwrap());
                assert!(((p.value - m.value) / 2e-6 - j.gradient[a]).abs() < 1e-8);
                for b in 0..2 {
                    assert!(
                        ((p.gradient[b] - m.gradient[b]) / 2e-6 - j.hessian[b][a]).abs() < 1e-7
                    );
                }
            }
        }
    }
    #[test]
    fn misses_are_bounded_and_query_order_does_not_change_field() {
        let f = field();
        let before = f.query([0.5, 0.6]).unwrap();
        for p in [
            [f64::NAN, 0.5],
            [f64::INFINITY, 0.5],
            [0.0, 0.5],
            [13.0 * f.spacing(), 0.5],
        ] {
            assert_eq!(f.query(p), Err(GeometryUnavailable));
        }
        for i in 0..100 {
            f.query([0.2 + i as f64 / 100.0, 0.8]).unwrap();
        }
        assert_eq!(before, f.query([0.5, 0.6]).unwrap());
        assert_eq!(f.identity(), field().identity());
    }
    #[test]
    fn malformed_assets_are_rejected() {
        assert!(PreparedField::new([0.0; 2], 0.0, [4; 2], vec![0.0; 16]).is_err());
        assert!(PreparedField::new([0.0; 2], 1.0, [4; 2], vec![f64::NAN; 16]).is_err());
        assert!(PreparedField::new([0.0; 2], 1.0, [usize::MAX; 2], vec![]).is_err());
    }

    #[test]
    fn reference_cusp_can_have_smooth_operational_derivatives_without_claiming_reference_normal() {
        let mut coefficients = Vec::new();
        for _ in 0..16 {
            for x in 0..16 {
                coefficients.push((x as f64 * 0.1 - 0.7).abs());
            }
        }
        let f = PreparedField::new([0.0; 2], 0.1, [16; 2], coefficients).unwrap();
        let j = f.query([0.7, 0.7]).unwrap();
        assert!(j.gradient[0].abs() < 1e-12);
        assert!(j.hessian[0][0] > 0.0);
        // The reference |x| normal is nonunique. This jet describes only F.
        assert!(PROFILE.contains("cuts-unknown"));
    }

    #[test]
    fn physics_coverage_miss_refuses_without_partial_movement() {
        let f = std::sync::Arc::new(field());
        activate(Some(f));
        let c = num_complex::Complex64::new(0.05, 0.5);
        let v = (0.01, 0.0);
        let result = crate::manifold::integrate_step(
            c,
            v,
            (0.0, 0.0),
            0.0,
            0.001,
            &crate::manifold::ManifoldConfig::default(),
        );
        activate(None);
        assert!(result.unwrap_err().contains("GeometryUnavailable"));
        assert_eq!(c, num_complex::Complex64::new(0.05, 0.5));
        assert_eq!(v, (0.01, 0.0));
    }
}
