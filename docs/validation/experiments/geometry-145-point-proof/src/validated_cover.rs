//! Validated closed attracting curves and geometric enclosures.
//! This module does not claim global component-boundary identity or a distance jet.
use super::{
    abs, attracting_segment, distance, finite, glue, mul, predictor, subtract, Certificate, B, C, P,
};
use astro_float::RoundingMode as R;
#[path = "distance_bracket.rs"]
pub(crate) mod bracket;

#[derive(Clone)]
struct Capsule {
    a: C,
    b: C,
    radius: B,
}
#[derive(Clone)]
pub(crate) struct Edge {
    patch: Certificate,
    start: C,
    end: C,
    capsule: Capsule,
}
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Refusal {
    InvalidInput,
    OutsidePatch,
    NotAttracting,
    OpenParameterPath,
    UnprovedJoin,
    NumericalFailure,
    QueryNotSeparated,
    AmbiguousOrientation,
    ZeroWinding,
    WorkLimit,
}
pub(crate) struct ClosedCover {
    edges: Vec<Edge>,
    query: C,
    lower: B,
    winding: i32,
}
pub(crate) struct CurveClearance {
    query: C,
    lower: B,
    winding: i32,
}
impl CurveClearance {
    pub(crate) fn query(&self) -> &C {
        &self.query
    }
    pub(crate) fn mandelbrot_interior_distance_lower(&self) -> &B {
        &self.lower
    }
    pub(crate) fn winding(&self) -> i32 {
        self.winding
    }
}
impl std::fmt::Debug for ClosedCover {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClosedCover")
            .field("edges", &self.edges.len())
            .field("winding", &self.winding)
            .field("curve_distance_lower", &super::super::number(&self.lower))
            .finish()
    }
}
impl ClosedCover {
    pub(crate) fn edge_count(&self) -> usize {
        self.edges.len()
    }
    pub(crate) fn winding(&self) -> i32 {
        self.winding
    }
    /// Distance from this exact query to the certified closed curve.
    pub(crate) fn distance_lower(&self) -> &B {
        &self.lower
    }
    /// Also a lower bound on distance to the Mandelbrot boundary:
    /// every certified branch has an attracting quadratic cycle, hence lies
    /// in M. Fullness of M makes winding zero on its complement. The open
    /// query disk of this radius avoids the curve and has nonzero winding,
    /// so is contained in int(M). Neither simplicity nor least period is
    /// required. This is not an upper bound or a differentiable distance jet.
    pub(crate) fn mandelbrot_interior_distance_lower(&self) -> &B {
        &self.lower
    }
    pub(crate) fn query(&self) -> &C {
        &self.query
    }
    /// Reuse the already-validated attracting curve. Only query-dependent
    /// capsule separation and winding are recomputed, never periodic roots.
    pub(crate) fn clearance_at(&self, query: &C) -> Result<CurveClearance, Refusal> {
        curve_clearance(&self.edges, query)
    }
}
fn max(a: B, b: B) -> Option<B> {
    if !finite(&a) || !finite(&b) {
        return None;
    }
    Some(if a.cmp(&b).is_some_and(|v| v >= 0) {
        a
    } else {
        b
    })
}
fn nonnegative(a: B) -> Option<B> {
    if !finite(&a) {
        return None;
    }
    Some(if a.cmp(&B::new(P)).is_some_and(|v| v < 0) {
        B::new(P)
    } else {
        a
    })
}
fn equal(a: &C, b: &C) -> bool {
    a.re.cmp(&b.re) == Some(0) && a.im.cmp(&b.im) == Some(0)
}
pub(crate) fn edge(patch: &Certificate, start: &C, end: &C) -> Result<Edge, Refusal> {
    if !start.finite() || !end.finite() {
        return Err(Refusal::InvalidInput);
    }
    attracting_segment(patch, start, end).map_err(|reason| match reason {
        super::SegmentRefusal::NotAttracting => Refusal::NotAttracting,
        super::SegmentRefusal::OutsidePatch => Refusal::OutsidePatch,
        _ => Refusal::NumericalFailure,
    })?;
    let (a, ea) = predictor(&patch.model, start, 1).ok_or(Refusal::NumericalFailure)?;
    let (b, eb) = predictor(&patch.model, end, 1).ok_or(Refusal::NumericalFailure)?;
    let radius =
        max(ea, eb)
            .ok_or(Refusal::NumericalFailure)?
            .add(&patch.model.inclusion[1], P, R::Up);
    if !finite(&radius) {
        return Err(Refusal::NumericalFailure);
    }
    Ok(Edge {
        patch: patch.clone(),
        start: start.clone(),
        end: end.clone(),
        capsule: Capsule { a, b, radius },
    })
}
/// Enclose Im(conj(b-a)*(q-a)), preserving subtraction and product rounding.
fn cross(a: &C, b: &C, q: &C) -> Option<(B, B, B)> {
    let (v, ev) = subtract(b, a)?;
    let (d, ed) = subtract(q, a)?;
    let mut conjugate = v.clone();
    conjugate.im.inv_sign();
    let (product, ep) = mul(&conjugate, &d)?;
    let error = abs(&v)
        .mul(&ed, P, R::Up)
        .add(&abs(&d).mul(&ev, P, R::Up), P, R::Up)
        .add(&ev.mul(&ed, P, R::Up), P, R::Up)
        .add(&ep, P, R::Up);
    let lo = product.im.sub(&error, P, R::Down);
    let hi = product.im.add(&error, P, R::Up);
    let norm = abs(&v).add(&ev, P, R::Up);
    (finite(&lo) && finite(&hi) && finite(&norm)).then_some((lo, hi, norm))
}
fn axis_distance(q: &B, a: &B, b: &B) -> Option<B> {
    let (lo, hi) = if a.cmp(b).is_some_and(|v| v <= 0) {
        (a, b)
    } else {
        (b, a)
    };
    if q.cmp(lo).is_some_and(|v| v < 0) {
        nonnegative(lo.sub(q, P, R::Down))
    } else if q.cmp(hi).is_some_and(|v| v > 0) {
        nonnegative(q.sub(hi, P, R::Down))
    } else {
        Some(B::new(P))
    }
}
fn capsule_distance_lower(capsule: &Capsule, q: &C) -> Option<B> {
    if !q.finite()
        || !finite(&capsule.radius)
        || !capsule.radius.cmp(&B::new(P)).is_some_and(|v| v >= 0)
    {
        return None;
    }
    let dx = axis_distance(&q.re, &capsule.a.re, &capsule.b.re)?;
    let dy = axis_distance(&q.im, &capsule.a.im, &capsule.b.im)?;
    let box_lower = dx
        .mul(&dx, P, R::Down)
        .add(&dy.mul(&dy, P, R::Down), P, R::Down)
        .sqrt(P, R::Down);
    let (lo, hi, norm) = cross(&capsule.a, &capsule.b, q)?;
    let cross_lower = if lo.cmp(&B::new(P)).is_some_and(|v| v > 0) {
        lo
    } else if hi.cmp(&B::new(P)).is_some_and(|v| v < 0) {
        let mut value = hi;
        value.inv_sign();
        value
    } else {
        B::new(P)
    };
    let line_lower = if norm.cmp(&B::new(P)).is_some_and(|v| v > 0) {
        cross_lower.div(&norm, P, R::Down)
    } else {
        B::new(P)
    };
    nonnegative(max(box_lower, line_lower)?.sub(&capsule.radius, P, R::Down))
}
fn crossing(a: &C, b: &C, q: &C) -> Result<i32, Refusal> {
    let upward = a.im.cmp(&q.im).is_some_and(|v| v <= 0) && b.im.cmp(&q.im).is_some_and(|v| v > 0);
    let downward =
        a.im.cmp(&q.im).is_some_and(|v| v > 0) && b.im.cmp(&q.im).is_some_and(|v| v <= 0);
    if !upward && !downward {
        return Ok(0);
    }
    let (lo, hi, _) = cross(a, b, q).ok_or(Refusal::NumericalFailure)?;
    if lo.cmp(&B::new(P)).is_some_and(|v| v > 0) {
        Ok(if upward { 1 } else { 0 })
    } else if hi.cmp(&B::new(P)).is_some_and(|v| v < 0) {
        Ok(if downward { -1 } else { 0 })
    } else {
        Err(Refusal::AmbiguousOrientation)
    }
}
fn join(a: &Certificate, b: &Certificate, parameter: &C) -> Result<(), Refusal> {
    if glue(a, b, parameter).is_ok() || glue(b, a, parameter).is_ok() {
        Ok(())
    } else {
        // Unequal arc sizes can make their endpoint error disks too wide for
        // direct inclusion. Prove a root at the exact shared multiplier and
        // connect both arcs to that certificate; numerical agreement alone
        // never establishes the join.
        let point = (|| {
            let (z, _) = predictor(&a.model, parameter, 0)?;
            let (c, _) = predictor(&a.model, parameter, 1)?;
            let (z, c) = super::super::solve(z, c, a.period(), parameter)?;
            let (point, _) = super::certify_auto(&z, &c, parameter, a.period(), 0.).ok()?;
            Some(point)
        })()
        .ok_or(Refusal::UnprovedJoin)?;
        point_join(a, b, &point, parameter)
    }
}
fn point_join(
    a: &Certificate,
    b: &Certificate,
    point: &Certificate,
    parameter: &C,
) -> Result<(), Refusal> {
    let connects = |patch: &Certificate| {
        glue(patch, point, parameter).is_ok() || glue(point, patch, parameter).is_ok()
    };
    if connects(a) && connects(b) {
        Ok(())
    } else {
        Err(Refusal::UnprovedJoin)
    }
}
fn connector_capsule(a: &Edge, b: &Edge) -> Option<Capsule> {
    Some(Capsule {
        a: a.capsule.b.clone(),
        b: b.capsule.a.clone(),
        radius: max(a.capsule.radius.clone(), b.capsule.radius.clone())?,
    })
}
/// Verify an exactly closed parameter path, all cycle joins, curve separation,
/// and nonzero winding of the actual continuous parameter curve around q.
pub(crate) fn closed(edges: Vec<Edge>, q: &C) -> Result<ClosedCover, Refusal> {
    if !q.finite() || edges.len() < 3 {
        return Err(Refusal::InvalidInput);
    }
    if edges.len() > 65536 {
        return Err(Refusal::WorkLimit);
    }
    for i in 0..edges.len() {
        let a = &edges[i];
        let b = &edges[(i + 1) % edges.len()];
        if !equal(&a.end, &b.start) {
            return Err(Refusal::OpenParameterPath);
        }
        join(&a.patch, &b.patch, &a.end).map_err(|_| Refusal::UnprovedJoin)?;
    }
    let clearance = curve_clearance(&edges, q)?;
    Ok(ClosedCover {
        edges,
        query: clearance.query,
        lower: clearance.lower,
        winding: clearance.winding,
    })
}
fn curve_clearance(edges: &[Edge], q: &C) -> Result<CurveClearance, Refusal> {
    if !q.finite() {
        return Err(Refusal::InvalidInput);
    }
    let mut lower: Option<B> = None;
    let mut winding = 0i32;
    for i in 0..edges.len() {
        let a = &edges[i];
        let b = &edges[(i + 1) % edges.len()];
        // The actual common endpoint lies in both endpoint disks. Their convex
        // hull is inside this connector capsule, so it also controls the homotopy
        // from the actual joined curve to the centerline reference polygon.
        let connector = connector_capsule(a, b).ok_or(Refusal::NumericalFailure)?;
        for capsule in [&a.capsule, &connector] {
            let bound = capsule_distance_lower(capsule, q).ok_or(Refusal::NumericalFailure)?;
            if !bound.cmp(&B::new(P)).is_some_and(|v| v > 0) {
                return Err(Refusal::QueryNotSeparated);
            }
            lower = Some(match lower {
                Some(current) => {
                    if current.cmp(&bound).is_some_and(|v| v < 0) {
                        current
                    } else {
                        bound
                    }
                }
                None => bound,
            });
            winding = winding
                .checked_add(crossing(&capsule.a, &capsule.b, q)?)
                .ok_or(Refusal::WorkLimit)?;
        }
    }
    if winding == 0 {
        return Err(Refusal::ZeroWinding);
    }
    Ok(CurveClearance {
        query: q.clone(),
        lower: lower.ok_or(Refusal::InvalidInput)?,
        winding,
    })
}

#[derive(Debug)]
pub(crate) struct BuildFailure {
    pub reason: BuildReason,
    pub attempted: usize,
    pub accepted: usize,
    pub depth: u32,
}
#[derive(Debug)]
pub(crate) enum BuildReason {
    InvalidRequest,
    NumericalSeed,
    RegionBudget,
    DepthBudget,
    ClearanceTarget,
    FinalCover(Refusal),
}
struct Builder {
    query: C,
    period: usize,
    edges: Vec<Edge>,
    attempted: usize,
    maximum_depth: u32,
    max_regions: usize,
    minimum_clearance: B,
}
impl Builder {
    fn failure(&self, reason: BuildReason, depth: u32) -> BuildFailure {
        BuildFailure {
            reason,
            attempted: self.attempted,
            accepted: self.edges.len(),
            depth,
        }
    }
    fn extend(
        &mut self,
        start: &C,
        end: &C,
        z: &C,
        c: &C,
        depth: u32,
    ) -> Result<(C, C), BuildFailure> {
        if self.attempted >= self.max_regions {
            return Err(self.failure(BuildReason::RegionBudget, depth));
        }
        self.attempted += 1;
        self.maximum_depth = self.maximum_depth.max(depth);
        let mu = start.add(end).scale(0.5);
        let h = super::super::number(
            &distance(start, end)
                .ok_or_else(|| self.failure(BuildReason::NumericalSeed, depth))?
                .div(&B::from(2u8), P, R::Up),
        )
        .next_up();
        let candidate = (|| -> Option<(Edge, C, C, usize)> {
            let (mut zm, mut cm) = super::super::solve(z.clone(), c.clone(), self.period, &mu)?;
            // Pick the critical-nearest phase as a conditioning hint; the proof and
            // validated joins, rather than the heuristic, establish cycle identity.
            let mut v = zm.clone();
            let mut best = zm.clone();
            for _ in 0..self.period {
                if v.norm() < best.norm() {
                    best = v.clone();
                }
                v = v.mul(&v).add(&cm);
            }
            (zm, cm) = super::super::solve(best, cm, self.period, &mu)?;
            let (patch, attempts) = super::certify_auto(&zm, &cm, &mu, self.period, h).ok()?;
            let item = edge(&patch, start, end).ok()?;
            let bound = capsule_distance_lower(&item.capsule, &self.query)?;
            if !bound.cmp(&B::new(P)).is_some_and(|v| v > 0)
                || !bound.cmp(&self.minimum_clearance).is_some_and(|v| v >= 0)
            {
                return None;
            }
            if let Some(previous) = self.edges.last() {
                join(&previous.patch, &patch, start).ok()?;
                let connector = connector_capsule(previous, &item)?;
                if !capsule_distance_lower(&connector, &self.query)?
                    .cmp(&self.minimum_clearance)
                    .is_some_and(|v| v >= 0)
                {
                    return None;
                }
            }
            let (ze, ce) = super::super::solve(zm, cm, self.period, end)?;
            Some((item, ze, ce, attempts))
        })();
        if let Some((item, ze, ce, proof_attempts)) = candidate {
            self.edges.push(item);
            if self.edges.len() % 32 == 0 {
                println!("[COVER_PROGRESS145] edges={} regions={} max_depth={} latest_proof_attempts={proof_attempts}",self.edges.len(),self.attempted,self.maximum_depth);
            }
            return Ok((ze, ce));
        }
        if depth >= 24 {
            return Err(self.failure(BuildReason::DepthBudget, depth));
        }
        let (zm, cm) = self.extend(start, &mu, z, c, depth + 1)?;
        self.extend(&mu, end, &zm, &cm, depth + 1)
    }
}
/// Diagnostic construction hints are numerical; only `closed` issues the
/// final proof after independently checking closure, all joins and geometry.
pub(crate) fn build(
    query: &C,
    period: usize,
    radius: f64,
    initial_edges: usize,
    max_regions: usize,
) -> Result<ClosedCover, BuildFailure> {
    build_with_root_radius(
        query,
        period,
        radius,
        radius,
        0.25,
        initial_edges,
        max_regions,
    )
}

/// Numerical loop proposal only; every edge and the completed winding still
/// require the same proof. A local inward dent avoids the parabolic root.
pub(crate) struct CoverRequest {
    pub(crate) radius: f64,
    pub(crate) root_radius: f64,
    pub(crate) root_width: f64,
    pub(crate) initial_edges: usize,
    pub(crate) max_regions: usize,
    pub(crate) minimum_clearance: f64,
}
pub(crate) fn build_with_root_radius(
    query: &C,
    period: usize,
    radius: f64,
    root_radius: f64,
    root_width: f64,
    initial_edges: usize,
    max_regions: usize,
) -> Result<ClosedCover, BuildFailure> {
    build_requested(
        query,
        period,
        CoverRequest {
            radius,
            root_radius,
            root_width,
            initial_edges,
            max_regions,
            minimum_clearance: 0.,
        },
    )
}
pub(crate) fn build_requested(
    query: &C,
    period: usize,
    request: CoverRequest,
) -> Result<ClosedCover, BuildFailure> {
    let CoverRequest {
        radius,
        root_radius,
        root_width,
        initial_edges,
        max_regions,
        minimum_clearance,
    } = request;
    let invalid = || BuildFailure {
        reason: BuildReason::InvalidRequest,
        attempted: 0,
        accepted: 0,
        depth: 0,
    };
    if !query.finite()
        || period == 0
        || period > 4096
        || !radius.is_finite()
        || radius <= 0.
        || radius >= 1.
        || !root_radius.is_finite()
        || root_radius <= 0.
        || root_radius > radius
        || !root_width.is_finite()
        || root_width <= 0.
        || root_width > 1.
        || !minimum_clearance.is_finite()
        || minimum_clearance < 0.
        || initial_edges < 8
        || initial_edges > 8192
        || max_regions < initial_edges
        || max_regions > 65536
    {
        return Err(invalid());
    }
    let mut builder = Builder {
        query: query.clone(),
        period,
        edges: Vec::new(),
        attempted: 0,
        maximum_depth: 0,
        max_regions,
        minimum_clearance: B::from_f64(minimum_clearance, P),
    };
    let mut z = C::new(0., 0.);
    for _ in 0..32768 {
        z = z.mul(&z).add(query);
        if !z.finite() {
            return Err(builder.failure(BuildReason::NumericalSeed, 0));
        }
    }
    let initial = super::super::cycle(&z, query, period)
        .ok_or_else(|| builder.failure(BuildReason::NumericalSeed, 0))?
        .m;
    let first = C::new(-radius, 0.);
    let mut c = query.clone();
    for k in 1..=64 {
        let t = k as f64 / 64.;
        let mu = initial.scale(1. - t).add(&first.scale(t));
        (z, c) = super::super::solve(z, c, period, &mu)
            .ok_or_else(|| builder.failure(BuildReason::NumericalSeed, 0))?;
    }
    let mut vertices = Vec::with_capacity(initial_edges);
    vertices.push(first.clone());
    for k in 1..initial_edges {
        let theta = std::f64::consts::PI + std::f64::consts::TAU * k as f64 / initial_edges as f64;
        let root_offset = theta - std::f64::consts::TAU;
        let local_radius =
            radius - (radius - root_radius) * (-(root_offset / root_width).powi(2)).exp();
        vertices.push(C::new(
            local_radius * theta.cos(),
            local_radius * theta.sin(),
        ));
    }
    for i in 0..initial_edges {
        let start = &vertices[i];
        let end = &vertices[(i + 1) % initial_edges];
        (z, c) = builder.extend(start, end, &z, &c, 0)?;
    }
    let attempted = builder.attempted;
    let depth = builder.maximum_depth;
    finish_cover(
        builder.edges,
        query,
        &B::from_f64(minimum_clearance, P),
        attempted,
        depth,
    )
}
fn finish_cover(
    edges: Vec<Edge>,
    query: &C,
    minimum_clearance: &B,
    attempted: usize,
    depth: u32,
) -> Result<ClosedCover, BuildFailure> {
    let accepted = edges.len();
    let cover = closed(edges, query).map_err(|reason| BuildFailure {
        reason: BuildReason::FinalCover(reason),
        attempted,
        accepted,
        depth,
    })?;
    if !cover
        .distance_lower()
        .cmp(minimum_clearance)
        .is_some_and(|v| v >= 0)
    {
        return Err(BuildFailure {
            reason: BuildReason::ClearanceTarget,
            attempted,
            accepted,
            depth,
        });
    }
    Ok(cover)
}
#[cfg(test)]
mod tests {
    use super::*;
    fn period_three_patch(seed: C) -> Certificate {
        let zero = C::new(0., 0.);
        let (z, c) = super::super::super::solve(zero.clone(), seed, 3, &zero).unwrap();
        super::super::certify(&z, &c, &zero, 3, 1e-6, 1e-9, 1e-9).unwrap()
    }
    fn cardioid_edges(reverse: bool) -> Vec<Edge> {
        let mut vertices = Vec::new();
        for k in 0..8 {
            let theta = std::f64::consts::TAU * k as f64 / 8.;
            vertices.push(C::new(0.5 * theta.cos(), 0.5 * theta.sin()));
        }
        if reverse {
            vertices.reverse();
        }
        let mut edges = Vec::new();
        for i in 0..8 {
            let start = &vertices[i];
            let end = &vertices[(i + 1) % 8];
            let mu = start.add(end).scale(0.5);
            let z = mu.scale(0.5);
            let c = mu.scale(0.5).sub(&mu.mul(&mu).scale(0.25));
            let patch = super::super::certify(&z, &c, &mu, 1, 0.2, 0.1, 0.1).unwrap();
            edges.push(edge(&patch, start, end).unwrap());
        }
        edges
    }
    #[test]
    fn closed_cardioid_control_has_certified_winding_and_separation() {
        for (reverse, expected) in [(false, 1), (true, -1)] {
            let cover = closed(cardioid_edges(reverse), &C::new(0., 0.)).unwrap();
            assert_eq!(cover.edge_count(), 8);
            assert_eq!(cover.winding(), expected);
            assert!(cover
                .distance_lower()
                .cmp(&B::from_f64(0.05, P))
                .is_some_and(|v| v > 0));
            assert!(cover
                .distance_lower()
                .cmp(&B::from_f64(0.1875, P))
                .is_some_and(|v| v <= 0));
        }
    }
    #[test]
    fn open_or_non_enclosing_paths_refuse() {
        let mut edges = cardioid_edges(false);
        edges.pop();
        assert!(matches!(
            closed(edges, &C::new(0., 0.)),
            Err(Refusal::OpenParameterPath)
        ));
        assert!(matches!(
            closed(cardioid_edges(false), &C::new(2., 0.)),
            Err(Refusal::ZeroWinding)
        ));
    }
    #[test]
    fn capsule_bounds_do_not_overstate_known_distances() {
        let capsule = Capsule {
            a: C::new(-1., 0.),
            b: C::new(1., 0.),
            radius: B::from_f64(0.25, P),
        };
        let bound = capsule_distance_lower(&capsule, &C::new(0., 1.)).unwrap();
        assert!(bound.cmp(&B::from_f64(0.75, P)).is_some_and(|v| v <= 0));
        assert!(bound.cmp(&B::from_f64(0.74, P)).is_some_and(|v| v > 0));
        assert_eq!(
            capsule_distance_lower(&capsule, &C::new(0., 0.1))
                .unwrap()
                .cmp(&B::new(P)),
            Some(0)
        );
    }
    #[test]
    fn intersecting_query_refuses_a_closed_certified_path() {
        let edges = cardioid_edges(false);
        let query = edges[0].capsule.a.clone();
        assert!(matches!(
            closed(edges, &query),
            Err(Refusal::QueryNotSeparated)
        ));
    }
    #[test]
    fn uncertain_ray_orientation_refuses() {
        assert_eq!(
            crossing(&C::new(-1., -1.), &C::new(1., 1.), &C::new(0., 0.)),
            Err(Refusal::AmbiguousOrientation)
        );
        assert_eq!(
            crossing(&C::new(1., 1.), &C::new(-1., -1.), &C::new(0., 0.)),
            Err(Refusal::AmbiguousOrientation)
        );
    }
    #[test]
    fn closed_multiplier_path_cannot_join_distinct_components() {
        let zero = C::new(0., 0.);
        let make = |seed: C| {
            let patch = period_three_patch(seed);
            edge(&patch, &zero, &zero).unwrap()
        };
        let a = make(C::new(-1.7548776662466927, 0.));
        let b = make(C::new(-0.1225611668766536, 0.7448617666197442));
        assert!(matches!(
            closed(vec![a.clone(), b, a], &C::new(2., 0.)),
            Err(Refusal::UnprovedJoin)
        ));
    }
    #[test]
    fn point_bridge_requires_both_independent_cycle_links() {
        let a = period_three_patch(C::new(-1.7548776662466927, 0.));
        let b = period_three_patch(C::new(-0.1225611668766536, 0.7448617666197442));
        let zero = C::new(0., 0.);
        assert!(point_join(&a, &a, &a, &zero).is_ok());
        assert_eq!(point_join(&a, &b, &a, &zero), Err(Refusal::UnprovedJoin));
        assert_eq!(point_join(&a, &b, &b, &zero), Err(Refusal::UnprovedJoin));
        assert_eq!(
            point_join(&a, &a, &a, &C::new(0.01, 0.)),
            Err(Refusal::UnprovedJoin)
        );
    }
    #[test]
    fn final_clearance_checks_the_closing_geometry_independently() {
        let mut edges = cardioid_edges(false);
        let query = edges[0].capsule.b.add(&C::new(-0.1, -0.1));
        let capsule = &mut edges[1].capsule;
        // Safe widened enclosure, as in the independent connector refusal
        // fixture, but now the connector stays positive while missing target.
        capsule.a = capsule.a.add(&C::new(0., 0.1));
        capsule.b = capsule.b.add(&C::new(0., 0.1));
        capsule.radius = capsule.radius.add(&B::from_f64(0.101, P), P, R::Up);
        edges.rotate_left(1);
        let target = B::from_f64(0.01, P);
        for item in &edges {
            assert!(capsule_distance_lower(&item.capsule, &query)
                .unwrap()
                .cmp(&target)
                .is_some_and(|v| v > 0));
        }
        for pair in edges.windows(2) {
            let connector = connector_capsule(&pair[0], &pair[1]).unwrap();
            assert!(capsule_distance_lower(&connector, &query)
                .unwrap()
                .cmp(&target)
                .is_some_and(|v| v > 0));
        }
        let cover = closed(edges.clone(), &query).unwrap();
        assert!(cover.distance_lower().cmp(&target).is_some_and(|v| v < 0));
        assert!(matches!(
            finish_cover(edges, &query, &target, 8, 0),
            Err(BuildFailure {
                reason: BuildReason::ClearanceTarget,
                ..
            })
        ));
    }
    #[test]
    fn connector_separation_is_required_independently() {
        let mut edges = cardioid_edges(false);
        let query = edges[0].capsule.b.add(&C::new(-0.1, -0.1));
        // Translate one enclosure upward and widen it by more than the
        // translation: it still contains its entire certified curve. Its
        // join connector has a much larger radius than the preceding edge.
        let capsule = &mut edges[1].capsule;
        capsule.a = capsule.a.add(&C::new(0., 10.));
        capsule.b = capsule.b.add(&C::new(0., 10.));
        capsule.radius = capsule.radius.add(&B::from_f64(10.001, P), P, R::Up);
        for item in &edges {
            assert!(capsule_distance_lower(&item.capsule, &query)
                .unwrap()
                .cmp(&B::new(P))
                .is_some_and(|v| v > 0));
        }
        assert!(matches!(
            closed(edges, &query),
            Err(Refusal::QueryNotSeparated)
        ));
    }
    #[test]
    fn reused_query_requires_winding_and_connector_separation() {
        let mut edges = cardioid_edges(false);
        let query = edges[0].capsule.b.add(&C::new(-0.1, -0.1));
        let capsule = &mut edges[1].capsule;
        // This widened translation still encloses the certified arc. The
        // origin remains separated, but a second query meets its connector.
        capsule.a = capsule.a.add(&C::new(0., 0.1));
        capsule.b = capsule.b.add(&C::new(0., 0.1));
        capsule.radius = capsule.radius.add(&B::from_f64(0.151, P), P, R::Up);
        for item in &edges {
            assert!(capsule_distance_lower(&item.capsule, &query)
                .unwrap()
                .cmp(&B::new(P))
                .is_some_and(|v| v > 0));
        }
        let cover = closed(edges, &C::new(0., 0.)).unwrap();
        assert!(matches!(
            cover.clearance_at(&query),
            Err(Refusal::QueryNotSeparated)
        ));
        assert!(matches!(
            cover.clearance_at(&C::new(2., 0.)),
            Err(Refusal::ZeroWinding)
        ));
    }
    #[test]
    fn adaptive_builder_completes_a_cardioid_control() {
        let cover = build(&C::new(0., 0.), 1, 0.5, 16, 128).unwrap();
        assert_eq!(cover.winding(), 1);
        assert!(cover.edge_count() <= 128);
        assert!(cover
            .mandelbrot_interior_distance_lower()
            .cmp(&B::from_f64(0.05, P))
            .is_some_and(|v| v > 0));
    }
    #[test]
    fn dented_builder_preserves_closed_cover_requirements() {
        let query = C::new(0., 0.);
        let cover = build_with_root_radius(&query, 1, 0.7, 0.5, 0.25, 32, 128).unwrap();
        assert_eq!(cover.winding(), 1);
        assert!(cover
            .distance_lower()
            .cmp(&B::from_f64(0.05, P))
            .is_some_and(|v| v > 0));
        for root in [f64::NAN, f64::INFINITY, 0., -0.1, 0.8] {
            assert!(matches!(
                build_with_root_radius(&query, 1, 0.7, root, 0.25, 32, 128),
                Err(BuildFailure {
                    reason: BuildReason::InvalidRequest,
                    ..
                })
            ));
        }
        for width in [f64::NAN, f64::INFINITY, 0., -0.1, 1.1] {
            assert!(build_with_root_radius(&query, 1, 0.7, 0.5, width, 32, 128).is_err());
        }
    }
    #[test]
    fn requested_clearance_is_proved_or_refused() {
        let q = C::new(0., 0.);
        let request = |minimum_clearance, max_regions| CoverRequest {
            radius: 0.5,
            root_radius: 0.5,
            root_width: 0.25,
            initial_edges: 8,
            max_regions,
            minimum_clearance,
        };
        let coarse = build_requested(&q, 1, request(0., 128)).unwrap();
        let target = (crate::number(coarse.distance_lower()) + 0.1875) * 0.5;
        assert!(coarse
            .distance_lower()
            .cmp(&B::from_f64(target, P))
            .is_some_and(|v| v < 0));
        let cover = build_requested(&q, 1, request(target, 128)).unwrap();
        assert!(cover
            .distance_lower()
            .cmp(&B::from_f64(target, P))
            .is_some_and(|v| v >= 0));
        assert!(build_requested(&q, 1, request(0.3, 32)).is_err());
        for target in [f64::NAN, f64::INFINITY, -0.1] {
            assert!(matches!(
                build_requested(&q, 1, request(target, 128)),
                Err(BuildFailure {
                    reason: BuildReason::InvalidRequest,
                    ..
                })
            ));
        }
    }
    #[test]
    fn exhausted_builder_does_not_return_an_accepted_prefix() {
        let edges = cardioid_edges(false);
        let start = edges[0].start.clone();
        let end = edges[0].end.clone();
        let mut builder = Builder {
            query: C::new(0., 0.),
            period: 1,
            edges,
            attempted: 8,
            maximum_depth: 0,
            max_regions: 8,
            minimum_clearance: B::new(P),
        };
        let result = builder.extend(&start, &end, &C::new(0., 0.), &C::new(0., 0.), 0);
        assert!(matches!(
            result,
            Err(BuildFailure {
                reason: BuildReason::RegionBudget,
                accepted: 8,
                ..
            })
        ));
    }
}
