//! Precision-backed exterior witnesses. This is not a runtime provider hook.
use super::*;

fn field_quality_passes(
    field: &crate::geometry_provider::FieldJet,
    error: f64,
    budget: f64,
) -> bool {
    field.d.is_finite()
        && field
            .g
            .iter()
            .chain(field.h.iter().flatten())
            .all(|v| v.is_finite())
        && error.is_finite()
        && error >= 0.
        && budget.is_finite()
        && budget > 0.
        && error <= budget
}

#[test]
fn field_quality_gate_rejects_invalid_arithmetic() {
    let field = crate::geometry_provider::FieldJet::constant(0.);
    assert!(field_quality_passes(&field, 0., 1.));
    for error in [-1., f64::NAN, f64::INFINITY] {
        assert!(!field_quality_passes(&field, error, 1.));
    }
    for budget in [0., -1., f64::NAN, f64::INFINITY] {
        assert!(!field_quality_passes(&field, 0., budget));
    }
    for index in 0..3 {
        let mut invalid = field;
        match index {
            0 => invalid.d = f64::NAN,
            1 => invalid.g[1] = f64::INFINITY,
            _ => invalid.h[0][1] = f64::NAN,
        }
        assert!(!field_quality_passes(&invalid, 0., 1.));
    }
}

struct OutsideWitness {
    point: C,
    proof: crate::point_proof::EscapeProof,
}
impl OutsideWitness {
    fn certify(point: C, precision: usize, budget: usize) -> Option<Self> {
        let proof = crate::point_proof::escape([point.re, point.im], precision, budget)?;
        Some(Self { point, proof })
    }
    fn find(query: C, error: f64) -> Option<Self> {
        for (point, _, _) in witness_candidates(query, error)? {
            for precision in [128, 256] {
                if let Some(witness) = Self::certify(point, precision, 32768) {
                    return Some(witness);
                }
            }
        }
        None
    }
    fn upper_at(&self, query: C) -> Option<f64> {
        endpoint_distance_upper(query, self.point)
    }
}
struct CertifiedScalar {
    query: C,
    value: f64,
    lower: f64,
    upper: f64,
}
impl CertifiedScalar {
    fn nodal_sample(&self) -> Option<crate::geometry_provider::NodalSample> {
        // Preserve the canonical value, including when a future tighter true-
        // distance bracket excludes it. The hull only widens the error range.
        crate::geometry_provider::NodalSample::enclosed(
            self.query,
            self.value,
            crate::geometry_provider::BoundaryInterval {
                lo: (-self.upper).min(self.value),
                hi: (-self.lower).max(self.value),
            },
        )
    }
    fn direction_cone(
        &self,
        curve: &branch::cover::ClosedCover,
        h: f64,
        axis: C,
    ) -> Option<branch::cover::BoundaryDirectionCone> {
        if !h.is_finite() || h <= 0. {
            return None;
        }
        let tie = (crate::geometry_provider::DYADIC_CUT_TIE_FACTOR * h).next_up();
        let radius = (self.upper + tie).next_up();
        let cosine = crate::geometry_provider::DYADIC_CUT_NORMAL_DEGREES
            .to_radians()
            .cos();
        curve.boundary_direction_cone(self.query, radius, axis, cosine)
    }
    fn certify(
        curve: &branch::cover::ClosedCover,
        witness: &OutsideWitness,
        query: C,
        value: f64,
        error: f64,
    ) -> Option<Self> {
        if !finite(query) || !value.is_finite() || value <= 0. || !error.is_finite() || error <= 0.
        {
            return None;
        }
        let lower = curve.clearance(query).ok()?.lower();
        let upper = witness.upper_at(query)?;
        if lower > upper || lower < (value - error).next_up() || upper > (value + error).next_down()
        {
            return None;
        }
        Some(Self {
            query,
            value: -value,
            lower,
            upper,
        })
    }
}

#[test]
fn interior_nonfinite_and_exhausted_proposals_are_not_witnesses() {
    for point in [
        C::new(0., 0.),
        C::new(0.25, 0.),
        C::new(f64::NAN, 0.),
        C::new(f64::INFINITY, 0.),
    ] {
        assert!(OutsideWitness::certify(point, 128, 512).is_none());
    }
    assert!(OutsideWitness::certify(C::new(1., 0.), 128, 3).is_none());
    assert!(OutsideWitness::certify(C::new(1., 0.), 128, 4).is_some());
    assert!(OutsideWitness::certify(C::new(1., 0.), 128, 32769).is_none());
    for error in [0., -1., f64::NAN, f64::INFINITY] {
        assert!(witness_candidates(C::new(0.3, 0.), error).is_none());
    }
}

#[test]
fn certified_scalar_requires_both_endpoint_gates_and_keeps_its_value() {
    let query = C::new(0.35659381055433242, 0.06404656757776669);
    let p = propose(query).unwrap();
    let error = 5e-10;
    let curve = branch::cover::build(
        query,
        branch::cover::Request {
            radius: 0.999999,
            root_radius: 0.99,
            root_width: 0.025,
            initial_edges: 128,
            max_regions: 65536,
        },
        &[(query, (p.distance - error).next_up())],
    )
    .unwrap();
    let witness = OutsideWitness::find(query, error).unwrap();
    let sample = CertifiedScalar::certify(&curve, &witness, query, p.distance, error).unwrap();
    assert_eq!(sample.value.to_bits(), (-p.distance).to_bits());
    assert_eq!(sample.query, query);
    let node = sample.nodal_sample().unwrap();
    assert_eq!(node.query, query);
    assert_eq!(node.value.to_bits(), sample.value.to_bits());
    assert!(
        CertifiedScalar::certify(&curve, &witness, query, p.distance - error * 2., error).is_none()
    );
    assert!(
        CertifiedScalar::certify(&curve, &witness, query, p.distance + error * 2., error).is_none()
    );
    assert!(
        CertifiedScalar::certify(&curve, &witness, C::new(2., 0.), p.distance, error).is_none()
    );
    assert!(CertifiedScalar::certify(&curve, &witness, query, p.distance, 0.).is_none());
}

#[test]
fn original_nodes_have_complete_scalar_distance_windows_and_direction_cones() {
    let start = std::time::Instant::now();
    let mut refusals = Vec::new();
    let mut longest = 0;
    let mut cone_unresolved = Vec::new();
    let mut maximum_refinement = 0;
    let mut field_refusals = Vec::new();
    for group in cases::CERTIFIED_NODES.chunks_exact(81) {
        let mut nodes = Vec::with_capacity(81);
        let targets: Vec<_> = group
            .iter()
            .map(|&(_, q, _, error)| {
                let point = C::new(q[0], q[1]);
                (point, (propose(point).unwrap().distance - error).next_up())
            })
            .collect();
        let curve = branch::cover::build(
            targets[40].0,
            branch::cover::Request {
                radius: 0.999999,
                root_radius: 0.999,
                root_width: 0.002,
                initial_edges: 512,
                max_regions: 65536,
            },
            &targets,
        )
        .unwrap();
        for &(name, q, _, error) in group {
            let query = C::new(q[0], q[1]);
            let value = propose(query).unwrap().distance;
            if let Some(witness) = OutsideWitness::find(query, error * 0.25)
                .or_else(|| OutsideWitness::find(query, error))
            {
                longest = longest.max(witness.proof.update);
                let sample = CertifiedScalar::certify(&curve, &witness, query, value, error)
                    .expect("original scalar error gates");
                assert_eq!(sample.value.to_bits(), (-value).to_bits());
                let node = sample
                    .nodal_sample()
                    .expect("query-bound canonical node interval");
                assert_eq!(node.value.to_bits(), sample.value.to_bits());
                assert_eq!(node.query, sample.query);
                nodes.push(node);
                let proposal = propose(query).unwrap();
                let axis = proposal.boundary - query;
                let base_h = if name == "native" {
                    2f64.powi(-20)
                } else {
                    2f64.powi(-27)
                };
                let mut proved_cone = false;
                for refinement in 0..=4 {
                    let h = base_h * 2f64.powi(-refinement);
                    if let Some(cone) = sample.direction_cone(&curve, h, axis) {
                        assert_eq!(cone.query(), query);
                        assert!(cone.radius() >= sample.upper);
                        maximum_refinement = maximum_refinement.max(refinement);
                        proved_cone = true;
                        eprintln!(
                            "[SCALAR_CONE145] name={name} {:?}",
                            [
                                q[0],
                                q[1],
                                h,
                                cone.radius(),
                                cone.axis_cosine_lower(),
                                cone.pieces() as f64,
                                cone.axis().re,
                                cone.axis().im,
                                cone.work() as f64
                            ]
                        );
                        break;
                    }
                }
                if !proved_cone {
                    cone_unresolved.push((name, q));
                }
                eprintln!(
                    "[MP_SCALAR145] name={name} {:?}",
                    [
                        q[0],
                        q[1],
                        sample.value,
                        error,
                        sample.lower,
                        sample.upper,
                        witness.point.re,
                        witness.point.im,
                        witness.proof.update as f64,
                        witness.proof.precision as f64
                    ]
                );
            } else {
                refusals.push((name, q));
            }
        }
        if nodes.len() == 81 {
            let name = group[0].0;
            let (query, h) = if name == "native" {
                (
                    C::new(0.32049988639807619, 0.04733531447355364),
                    2f64.powi(-20),
                )
            } else {
                (
                    C::new(0.35659381055433242, 0.06404656757776669),
                    2f64.powi(-27),
                )
            };
            let range = crate::geometry_provider::field_enclosure(query, h, &nodes)
                .expect("retained node queries match canonical stencil");
            let stencil: Vec<_> = nodes
                .iter()
                .map(|s| (s.query.re - query.re, s.query.im - query.im, s.value))
                .collect();
            let (field, _) = crate::geometry_provider::quintic_field(query, h, &stencil);
            let mut moved = nodes.clone();
            moved.swap(40, 41);
            assert!(
                crate::geometry_provider::field_enclosure(query, h, &moved).is_none(),
                "node certificates cannot move between coordinates"
            );
            for (index, node) in nodes.iter().enumerate() {
                let interval = node.enclosure.unwrap();
                for value in [interval.lo, interval.hi] {
                    let mut assigned = stencil.clone();
                    assigned[index].2 = value;
                    let (sampled, _) = crate::geometry_provider::quintic_field(query, h, &assigned);
                    let contains = |i: crate::geometry_provider::BoundaryInterval, v: f64| {
                        i.lo <= v && v <= i.hi
                    };
                    assert!(contains(range.d, sampled.d));
                    for i in 0..2 {
                        assert!(contains(range.g[i], sampled.g[i]));
                        for j in 0..2 {
                            assert!(contains(range.h[i][j], sampled.h[i][j]));
                        }
                    }
                }
            }
            let error = range.error(field.d, field.g, field.h, 2. * h);
            let epsilon = 1e-4; // Both original crossing configurations.
            let requested = crate::geometry_provider::GEOMETRY_SCALE_ALPHA
                * (field.d * field.d + epsilon * epsilon).sqrt();
            let budget = crate::geometry_provider::DYADIC_FIT_ERROR_FACTOR * requested;
            let proposal = propose(query).unwrap();
            let scalar_error = group[0].3;
            let witness = OutsideWitness::find(query, scalar_error * 0.25)
                .or_else(|| OutsideWitness::find(query, scalar_error))
                .expect("actual query exterior witness");
            let query_sample =
                CertifiedScalar::certify(&curve, &witness, query, proposal.distance, scalar_error)
                    .expect("actual query original scalar window");
            let mut cone = None;
            for refinement in 0..=4 {
                cone = query_sample.direction_cone(
                    &curve,
                    h * 2f64.powi(-refinement),
                    proposal.boundary - query,
                );
                if cone.is_some() {
                    break;
                }
            }
            let cone = cone.expect("actual query complete direction agreement");
            eprintln!(
                "[CERTIFIED_QUERY145] name={name} {:?}",
                [
                    query.re,
                    query.im,
                    query_sample.value,
                    scalar_error,
                    query_sample.lower,
                    query_sample.upper,
                    witness.point.re,
                    witness.point.im,
                    witness.proof.update as f64,
                    witness.proof.precision as f64,
                    cone.radius(),
                    cone.axis_cosine_lower(),
                    cone.work() as f64,
                    curve.clearance(query).unwrap().winding() as f64
                ]
            );
            eprintln!(
                "[CERTIFIED_FIELD145] name={name} {:?}",
                [
                    query.re,
                    query.im,
                    h,
                    field.d,
                    field.g[0],
                    field.g[1],
                    field.h[0][0],
                    field.h[0][1],
                    field.h[1][1],
                    error,
                    budget
                ]
            );
            if !field_quality_passes(&field, error, budget) {
                field_refusals.push((name, error, budget));
            }
        }
    }
    eprintln!(
        "[MP_WITNESS145] accepted={} refused={refusals:?} longest={longest} elapsed_s={}",
        162 - refusals.len(),
        start.elapsed().as_secs_f64()
    );
    assert!(refusals.is_empty());
    eprintln!("[SCALAR_CONE_SUMMARY145] accepted={} maximum_refinement={maximum_refinement} unresolved={cone_unresolved:?}",162-cone_unresolved.len());
    assert!(cone_unresolved.is_empty());
    assert!(
        field_refusals.is_empty(),
        "certified nodal uncertainty exceeds original quality gate: {field_refusals:?}"
    );
}
