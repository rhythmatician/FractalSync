//! Interior distance enclosure from independent inside and outside evidence.
use super::{ClosedCover, CurveClearance, B, P};
use crate::complex_ball::Complex;
use crate::point_proof::{escape, EscapeProof};
use astro_float::RoundingMode as R;

pub(crate) struct OutsideWitness {
    point: Complex,
    proof: EscapeProof,
}
impl OutsideWitness {
    pub(crate) fn certify(point: [f64; 2], precision: usize, budget: usize) -> Option<Self> {
        let proof = escape(point, precision, budget)?;
        Some(Self {
            point: Complex {
                re: B::from_f64(point[0], P),
                im: B::from_f64(point[1], P),
            },
            proof,
        })
    }
    pub(crate) fn escape_update(&self) -> usize {
        self.proof.update
    }
    /// Endpoint distance only. It becomes a boundary-distance upper bound
    /// when paired with independently proved interior membership.
    pub(crate) fn endpoint_distance_upper(&self, query: &crate::C) -> Option<B> {
        Complex {
            re: query.re.clone(),
            im: query.im.clone(),
        }
        .distance_upper(&self.point, P)
    }
}
pub(crate) struct DistanceBracket {
    lower: B,
    upper: B,
    query: Complex,
}
pub(crate) struct InteriorDistanceDomain {
    origin: Complex,
    lower: B,
    outside: Complex,
}
impl InteriorDistanceDomain {
    pub(crate) fn certify(cover: &ClosedCover, outside: &OutsideWitness) -> Self {
        let q = cover.query();
        Self {
            origin: Complex {
                re: q.re.clone(),
                im: q.im.clone(),
            },
            lower: cover.mandelbrot_interior_distance_lower().clone(),
            outside: Complex {
                re: outside.point.re.clone(),
                im: outside.point.im.clone(),
            },
        }
    }
    pub(crate) fn at(&self, q: &crate::C) -> Option<DistanceBracket> {
        let query = Complex {
            re: q.re.clone(),
            im: q.im.clone(),
        };
        let travel = query.distance_upper(&self.origin, P)?;
        // Distance to a closed set is 1-Lipschitz. Strict containment keeps
        // every accepted new query inside the original certified open disk.
        let lower = self.lower.sub(&travel, P, R::Down);
        if !lower.cmp(&B::new(P)).is_some_and(|v| v > 0) {
            return None;
        }
        let upper = query.distance_upper(&self.outside, P)?;
        if !upper.cmp(&lower).is_some_and(|v| v >= 0) {
            return None;
        }
        Some(DistanceBracket {
            lower,
            upper,
            query,
        })
    }
}
impl DistanceBracket {
    pub(crate) fn certify(cover: &ClosedCover, outside: &OutsideWitness) -> Option<Self> {
        // The segment from an interior query to a proved exterior parameter
        // crosses the boundary. Directed endpoint distance is thus an upper bound.
        InteriorDistanceDomain::certify(cover, outside).at(cover.query())
    }
    pub(crate) fn at_cover(
        cover: &ClosedCover,
        outside: &OutsideWitness,
        query: &crate::C,
    ) -> Option<Self> {
        let clearance = cover.clearance_at(query).ok()?;
        Self::from_clearance(&clearance, outside)
    }
    pub(crate) fn from_clearance(
        clearance: &CurveClearance,
        outside: &OutsideWitness,
    ) -> Option<Self> {
        let q = clearance.query();
        let upper = outside.endpoint_distance_upper(q)?;
        let lower = clearance.mandelbrot_interior_distance_lower().clone();
        if !upper.cmp(&lower).is_some_and(|v| v >= 0) {
            return None;
        }
        Some(Self {
            lower,
            upper,
            query: Complex {
                re: q.re.clone(),
                im: q.im.clone(),
            },
        })
    }
    pub(crate) fn lower(&self) -> &B {
        &self.lower
    }
    pub(crate) fn upper(&self) -> &B {
        &self.upper
    }
    pub(crate) fn width(&self) -> B {
        self.upper.sub(&self.lower, P, R::Up)
    }
    pub(crate) fn meets_width(&self, requested: f64) -> bool {
        requested.is_finite()
            && requested > 0.
            && self
                .width()
                .cmp(&B::from_f64(requested, P))
                .is_some_and(|v| v <= 0)
    }
    pub(crate) fn require_width(self, requested: f64) -> Option<Self> {
        self.meets_width(requested).then_some(self)
    }
}
impl std::fmt::Debug for DistanceBracket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let number = crate::number;
        f.debug_struct("DistanceBracket")
            .field("query", &[number(&self.query.re), number(&self.query.im)])
            .field("lower", &number(&self.lower))
            .field("upper", &number(&self.upper))
            .field("width", &number(&self.width()))
            .finish()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reused_cover_accepts_queries_outside_original_transport_disk() {
        let origin = crate::C::new(0., 0.);
        let cover = super::super::build(&origin, 1, 0.5, 16, 128).unwrap();
        let original_lower = cover.distance_lower().clone();
        let outside = OutsideWitness::certify([0.3, 0.], 128, 512).unwrap();
        let moved = crate::C::new(-0.2, 0.);
        assert!(InteriorDistanceDomain::certify(&cover, &outside)
            .at(&moved)
            .is_none());
        let bracket = DistanceBracket::at_cover(&cover, &outside, &moved).unwrap();
        assert!(bracket.lower().cmp(&B::new(P)).is_some_and(|v| v > 0));
        assert_eq!(bracket.query.re.cmp(&moved.re), Some(0));
        assert_eq!(bracket.query.im.cmp(&moved.im), Some(0));
        assert_eq!(cover.distance_lower().cmp(&original_lower), Some(0));
        for q in [crate::C::new(2., 0.), crate::C::new(f64::NAN, 0.)] {
            assert!(DistanceBracket::at_cover(&cover, &outside, &q).is_none());
        }
    }
    #[test]
    fn independently_certified_endpoints_enclose_known_cardioid_distance() {
        let cover = super::super::build(&crate::C::new(0., 0.), 1, 0.5, 16, 128).unwrap();
        let outside = OutsideWitness::certify([0.3, 0.], 128, 512).unwrap();
        let bracket = DistanceBracket::certify(&cover, &outside).unwrap();
        assert!(bracket
            .lower()
            .cmp(&B::from_f64(0.25, P))
            .is_some_and(|v| v < 0));
        assert!(bracket
            .upper()
            .cmp(&B::from_f64(0.3, P))
            .is_some_and(|v| v >= 0));
        assert!(bracket
            .upper()
            .cmp(&B::from_f64(0.3, P))
            .is_some_and(|v| v == 0));
        assert!(bracket.width().cmp(&B::new(P)).is_some_and(|v| v > 0));
        assert!(bracket.meets_width(1.));
        for width in [f64::NAN, f64::INFINITY, 0., -1., 1e-10] {
            assert!(!bracket.meets_width(width));
        }
        assert!(DistanceBracket::certify(&cover, &outside)
            .unwrap()
            .require_width(1e-10)
            .is_none());
        assert!(DistanceBracket::certify(&cover, &outside)
            .unwrap()
            .require_width(1.)
            .is_some());
        assert!(outside.escape_update() > 0);
    }
    #[test]
    fn absence_of_escape_is_not_an_exterior_witness() {
        for point in [[0., 0.], [0.25, 0.], [f64::NAN, 0.], [0., f64::INFINITY]] {
            assert!(OutsideWitness::certify(point, 128, 512).is_none());
        }
        assert!(OutsideWitness::certify([1., 0.], 128, 3).is_none());
        assert!(OutsideWitness::certify([1., 0.], 128, 4).is_some());
        assert!(OutsideWitness::certify([1., 0.], 64, 512).is_none());
        assert!(OutsideWitness::certify([1., 0.], 128, 32769).is_none());
    }
    #[test]
    fn upper_distance_is_outward_for_irrational_endpoint_norm() {
        let cover = super::super::build(&crate::C::new(0., 0.), 1, 0.5, 16, 128).unwrap();
        let outside = OutsideWitness::certify([1., 1.], 128, 512).unwrap();
        let bracket = DistanceBracket::certify(&cover, &outside).unwrap();
        let reference_lower = B::from(2u8).sqrt(512, R::Down);
        let reference_upper = B::from(2u8).sqrt(512, R::Up);
        assert!(bracket
            .upper()
            .cmp(&reference_lower)
            .is_some_and(|v| v >= 0));
        assert!(bracket
            .upper()
            .sub(&reference_upper, 512, R::Up)
            .cmp(&B::from_f64(1e-35, 512))
            .is_some_and(|v| v < 0));
    }
    #[test]
    fn compact_domain_transports_bounds_and_refuses_its_boundary() {
        let origin = crate::C::new(0., 0.);
        let cover = super::super::build(&origin, 1, 0.5, 16, 128).unwrap();
        let outside = OutsideWitness::certify([0.3, 0.], 128, 512).unwrap();
        let domain = InteriorDistanceDomain::certify(&cover, &outside);
        let center = domain.at(&origin).unwrap();
        let moved = domain.at(&crate::C::new(0.01, 0.)).unwrap();
        assert!(moved.lower().cmp(center.lower()).is_some_and(|v| v < 0));
        assert!(moved.upper().cmp(center.upper()).is_some_and(|v| v < 0));
        let expected = center.lower().sub(&B::from_f64(0.01, P), P, R::Down);
        assert_eq!(moved.lower().cmp(&expected), Some(0));
        let boundary = crate::C {
            re: domain.lower.clone(),
            im: B::new(P),
        };
        assert!(domain.at(&boundary).is_none());
        assert!(domain.at(&crate::C::new(0.3, 0.)).is_none());
        assert!(domain.at(&crate::C::new(f64::NAN, 0.)).is_none());
    }
    fn check_diagonal_transport(domain: &InteriorDistanceDomain, q: &crate::C) {
        let bracket = domain.at(q).unwrap();
        let norm_lower =
            q.re.mul(&q.re, 512, R::Down)
                .add(&q.im.mul(&q.im, 512, R::Down), 512, R::Down)
                .sqrt(512, R::Down);
        let maximum_lower = domain.lower.sub(&norm_lower, 512, R::Up);
        assert!(bracket.lower().cmp(&maximum_lower).is_some_and(|v| v <= 0));
    }
    #[test]
    fn diagonal_transport_rounds_travel_outward_near_clearance_boundary() {
        let cover = super::super::build(&crate::C::new(0., 0.), 1, 0.5, 16, 128).unwrap();
        let outside = OutsideWitness::certify([0.3, 0.], 128, 512).unwrap();
        let domain = InteriorDistanceDomain::certify(&cover, &outside);
        let coordinate = domain.lower.mul(&B::from_f64(0.7, P), P, R::ToEven);
        let q = crate::C {
            re: coordinate.clone(),
            im: coordinate,
        };
        check_diagonal_transport(&domain, &q);
    }
    #[test]
    fn tiny_diagonal_transport_rounds_clearance_subtraction_downward() {
        let cover = super::super::build(&crate::C::new(0., 0.), 1, 0.5, 16, 128).unwrap();
        let outside = OutsideWitness::certify([0.3, 0.], 128, 512).unwrap();
        let domain = InteriorDistanceDomain::certify(&cover, &outside);
        check_diagonal_transport(&domain, &crate::C::new(2f64.powi(-200), 2f64.powi(-201)));
    }
}
