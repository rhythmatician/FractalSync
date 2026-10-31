//! Bounded offline contour preparation. Finite-orbit masks are measured evidence,
//! never certified membership or a true-distance error bound.

use super::PreparedField;
use serde::{Deserialize, Serialize};

#[derive(Clone, Deserialize, Serialize)]
pub struct ContourPreparation {
    pub origin: [f64; 2],
    pub spacing: f64,
    pub shape: [usize; 2],
    pub reference_origin: [f64; 2],
    pub reference_spacing: f64,
    pub reference_shape: [usize; 2],
    pub refinement: usize,
    pub bisections: usize,
    pub membership_iterations: usize,
    pub max_membership_queries: usize,
    pub max_reference_nodes: usize,
    pub max_segments: usize,
    pub maximum_coefficient_bytes: usize,
    /// Reuse required coefficient membership samples to refine otherwise
    /// homogeneous reference cells. False preserves prior experiment recipes.
    #[serde(default)]
    pub coefficient_witness_seeding: bool,
}

#[derive(Default, Serialize)]
pub struct PreparationStats {
    pub reference_nodes: usize,
    pub refined_cells: usize,
    pub membership_queries: usize,
    pub segments: usize,
    pub coefficient_count: usize,
    pub coefficient_seeded_cells: usize,
}

#[derive(Clone, Copy)]
struct Segment([f64; 2], [f64; 2]);
impl Segment {
    fn bounds(self) -> [f64; 4] {
        [
            self.0[0].min(self.1[0]),
            self.0[1].min(self.1[1]),
            self.0[0].max(self.1[0]),
            self.0[1].max(self.1[1]),
        ]
    }
    fn distance_sq(self, p: [f64; 2]) -> f64 {
        let v = [self.1[0] - self.0[0], self.1[1] - self.0[1]];
        let norm = v[0] * v[0] + v[1] * v[1];
        let t = if norm > 0.0 {
            (((p[0] - self.0[0]) * v[0] + (p[1] - self.0[1]) * v[1]) / norm).clamp(0.0, 1.0)
        } else {
            0.0
        };
        (p[0] - self.0[0] - t * v[0]).powi(2) + (p[1] - self.0[1] - t * v[1]).powi(2)
    }
}

struct Node {
    bounds: [f64; 4],
    range: std::ops::Range<usize>,
    children: Option<[usize; 2]>,
}
struct Contour {
    segments: Vec<Segment>,
    nodes: Vec<Node>,
}
impl Contour {
    fn new(segments: Vec<Segment>) -> Result<Self, String> {
        if segments.is_empty() {
            return Err("reference mask produced no contour".into());
        }
        let mut tree = Self {
            segments,
            nodes: Vec::new(),
        };
        tree.build(0, tree.segments.len());
        Ok(tree)
    }
    fn build(&mut self, begin: usize, end: usize) -> usize {
        let mut b = [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ];
        for s in &self.segments[begin..end] {
            let a = s.bounds();
            b = [
                b[0].min(a[0]),
                b[1].min(a[1]),
                b[2].max(a[2]),
                b[3].max(a[3]),
            ];
        }
        let index = self.nodes.len();
        self.nodes.push(Node {
            bounds: b,
            range: begin..end,
            children: None,
        });
        if end - begin > 8 {
            let axis = usize::from(b[3] - b[1] > b[2] - b[0]);
            let mid = (begin + end) / 2;
            self.segments[begin..end].select_nth_unstable_by(mid - begin, |a, b| {
                (a.0[axis] + a.1[axis]).total_cmp(&(b.0[axis] + b.1[axis]))
            });
            let left = self.build(begin, mid);
            let right = self.build(mid, end);
            self.nodes[index].children = Some([left, right]);
        }
        index
    }
    fn lower_sq(b: [f64; 4], p: [f64; 2]) -> f64 {
        let x = (b[0] - p[0]).max(0.0).max(p[0] - b[2]);
        let y = (b[1] - p[1]).max(0.0).max(p[1] - b[3]);
        x * x + y * y
    }
    fn visit(&self, index: usize, p: [f64; 2], best: &mut f64) {
        let n = &self.nodes[index];
        if Self::lower_sq(n.bounds, p) >= *best {
            return;
        }
        if let Some([a, b]) = n.children {
            let (first, last) = if Self::lower_sq(self.nodes[a].bounds, p)
                < Self::lower_sq(self.nodes[b].bounds, p)
            {
                (a, b)
            } else {
                (b, a)
            };
            self.visit(first, p, best);
            self.visit(last, p, best);
        } else {
            for s in &self.segments[n.range.clone()] {
                *best = best.min(s.distance_sq(p));
            }
        }
    }
    fn distance(&self, p: [f64; 2]) -> f64 {
        let mut best = f64::INFINITY;
        self.visit(0, p, &mut best);
        best.sqrt()
    }
}

struct Classifier<F> {
    evaluate: F,
    used: usize,
    limit: usize,
}
impl<F: FnMut([f64; 2]) -> bool> Classifier<F> {
    fn at(&mut self, p: [f64; 2]) -> Result<bool, String> {
        if self.used >= self.limit {
            return Err("offline membership-query budget exhausted".into());
        }
        self.used += 1;
        Ok((self.evaluate)(p))
    }
}

/// Fixed iteration cap; nonescape is a measured mask value, not an inside proof.
pub fn prepare_contour_region(
    plan: &ContourPreparation,
) -> Result<(PreparedField, PreparationStats), String> {
    prepare_with_membership(plan, |p| {
        crate::visual_metrics::mandelbrot_membership(
            num_complex::Complex64::new(p[0], p[1]),
            plan.membership_iterations,
        )
    })
}

fn prepare_with_membership<F: FnMut([f64; 2]) -> bool>(
    p: &ContourPreparation,
    f: F,
) -> Result<(PreparedField, PreparationStats), String> {
    let coefficient_count = p.shape[0]
        .checked_mul(p.shape[1])
        .ok_or("coefficient count overflow")?;
    let nodes = p.reference_shape[0]
        .checked_mul(p.reference_shape[1])
        .ok_or("reference node count overflow")?;
    if p.origin
        .iter()
        .chain(p.reference_origin.iter())
        .any(|x| !x.is_finite())
        || !p.spacing.is_finite()
        || p.spacing <= 0.0
        || !(1.0 / p.spacing.powi(2)).is_finite()
        || !p.reference_spacing.is_finite()
        || p.reference_spacing <= 0.0
        || p.shape.iter().any(|n| *n < 6)
        || p.reference_shape.iter().any(|n| *n < 2)
        || p.refinement == 0
        || p.refinement > 32
        || p.bisections > 20
        || p.membership_iterations == 0
        || p.membership_iterations > 32768
        || nodes > p.max_reference_nodes
        || nodes > 20_000_000
        || p.max_membership_queries > 100_000_000
        || p.max_segments > 2_000_000
        || coefficient_count
            .checked_mul(8)
            .is_none_or(|bytes| bytes > p.maximum_coefficient_bytes || bytes > 80_000_000)
        || (0..2).any(|a| {
            let edge =
                p.reference_origin[a] + (p.reference_shape[a] - 1) as f64 * p.reference_spacing;
            !edge.is_finite()
                || p.origin[a] < p.reference_origin[a]
                || p.origin[a] + (p.shape[a] - 1) as f64 * p.spacing > edge
        })
    {
        return Err("invalid bounded contour preparation plan".into());
    }
    let mut classify = Classifier {
        evaluate: f,
        used: 0,
        limit: p.max_membership_queries,
    };
    let mut mask = Vec::with_capacity(nodes);
    let [nx, ny] = p.reference_shape;
    for y in 0..ny {
        for x in 0..nx {
            mask.push(classify.at([
                p.reference_origin[0] + x as f64 * p.reference_spacing,
                p.reference_origin[1] + y as f64 * p.reference_spacing,
            ])?);
        }
    }
    let mut stats = PreparationStats {
        reference_nodes: nodes,
        coefficient_count,
        ..Default::default()
    };
    // The signs are needed for coefficients regardless. With seeding enabled,
    // collect them before contour extraction so an observed component cannot
    // be discarded solely because all four coarse corners missed it.
    let mut seeded = vec![
        false;
        if p.coefficient_witness_seeding {
            (nx - 1) * (ny - 1)
        } else {
            0
        }
    ];
    let coefficient_membership = if p.coefficient_witness_seeding {
        let mut values = Vec::with_capacity(coefficient_count);
        for y in 0..p.shape[1] {
            for x in 0..p.shape[0] {
                let point = [
                    p.origin[0] + x as f64 * p.spacing,
                    p.origin[1] + y as f64 * p.spacing,
                ];
                let inside = classify.at(point)?;
                values.push(inside);
                let q = [
                    (point[0] - p.reference_origin[0]) / p.reference_spacing,
                    (point[1] - p.reference_origin[1]) / p.reference_spacing,
                ];
                let cell = [
                    (q[0].floor() as usize).min(nx - 2),
                    (q[1].floor() as usize).min(ny - 2),
                ];
                // A witness on a shared edge belongs to both incident cells.
                let begin = [
                    if q[0].fract() == 0.0 && cell[0] > 0 && q[0] < (nx - 1) as f64 {
                        cell[0] - 1
                    } else {
                        cell[0]
                    },
                    if q[1].fract() == 0.0 && cell[1] > 0 && q[1] < (ny - 1) as f64 {
                        cell[1] - 1
                    } else {
                        cell[1]
                    },
                ];
                for j in begin[1]..=cell[1] {
                    for i in begin[0]..=cell[0] {
                        if mask[j * nx + i] != inside {
                            seeded[j * (nx - 1) + i] = true;
                        }
                    }
                }
            }
        }
        Some(values)
    } else {
        None
    };
    let mut segments = Vec::new();
    let r = p.refinement;
    let h = p.reference_spacing / r as f64;
    for y in 0..ny - 1 {
        for x in 0..nx - 1 {
            let a = mask[y * nx + x];
            let homogeneous = [
                mask[y * nx + x + 1],
                mask[(y + 1) * nx + x],
                mask[(y + 1) * nx + x + 1],
            ]
            .iter()
            .all(|b| *b == a);
            let coefficient_seed = p.coefficient_witness_seeding && seeded[y * (nx - 1) + x];
            if homogeneous && !coefficient_seed {
                continue;
            }
            stats.coefficient_seeded_cells += usize::from(homogeneous && coefficient_seed);
            stats.refined_cells += 1;
            let o = [
                p.reference_origin[0] + x as f64 * p.reference_spacing,
                p.reference_origin[1] + y as f64 * p.reference_spacing,
            ];
            let mut fine = Vec::with_capacity((r + 1) * (r + 1));
            for j in 0..=r {
                for i in 0..=r {
                    fine.push(classify.at([o[0] + i as f64 * h, o[1] + j as f64 * h])?);
                }
            }
            for j in 0..r {
                for i in 0..r {
                    let corners = [
                        [o[0] + i as f64 * h, o[1] + j as f64 * h],
                        [o[0] + (i + 1) as f64 * h, o[1] + j as f64 * h],
                        [o[0] + (i + 1) as f64 * h, o[1] + (j + 1) as f64 * h],
                        [o[0] + i as f64 * h, o[1] + (j + 1) as f64 * h],
                    ];
                    let inside = [
                        fine[j * (r + 1) + i],
                        fine[j * (r + 1) + i + 1],
                        fine[(j + 1) * (r + 1) + i + 1],
                        fine[(j + 1) * (r + 1) + i],
                    ];
                    let mut cross = Vec::with_capacity(4);
                    for edge in 0..4 {
                        let next = (edge + 1) % 4;
                        if inside[edge] == inside[next] {
                            continue;
                        }
                        let (mut lo, mut hi) = (corners[edge], corners[next]);
                        for _ in 0..p.bisections {
                            let mid = [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5];
                            if classify.at(mid)? == inside[edge] {
                                lo = mid;
                            } else {
                                hi = mid;
                            }
                        }
                        cross.push([(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5]);
                    }
                    let pairs = match cross.len() {
                        2 => vec![(0, 1)],
                        4 => {
                            let center = [o[0] + (i as f64 + 0.5) * h, o[1] + (j as f64 + 0.5) * h];
                            if classify.at(center)? != inside[0] {
                                vec![(0, 3), (1, 2)]
                            } else {
                                vec![(0, 1), (2, 3)]
                            }
                        }
                        0 => Vec::new(),
                        _ => return Err("incomplete contour cell".into()),
                    };
                    for (a, b) in pairs {
                        if segments.len() >= p.max_segments {
                            return Err("offline contour-segment budget exhausted".into());
                        }
                        segments.push(Segment(cross[a], cross[b]));
                    }
                }
            }
        }
    }
    stats.segments = segments.len();
    let contour = Contour::new(segments)?;
    let mut coefficients = Vec::with_capacity(coefficient_count);
    for y in 0..p.shape[1] {
        for x in 0..p.shape[0] {
            let point = [
                p.origin[0] + x as f64 * p.spacing,
                p.origin[1] + y as f64 * p.spacing,
            ];
            let inside = match &coefficient_membership {
                Some(values) => values[y * p.shape[0] + x],
                None => classify.at(point)?,
            };
            let sign = if inside { -1.0 } else { 1.0 };
            coefficients.push(sign * contour.distance(point));
        }
    }
    stats.membership_queries = classify.used;
    let field =
        PreparedField::new(p.origin, p.spacing, p.shape, coefficients).map_err(str::to_string)?;
    Ok((field, stats))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coefficient_membership_witness_retains_a_component_hidden_between_reference_corners() {
        let plan = ContourPreparation {
            origin: [-0.35, -0.15],
            spacing: 0.005,
            shape: [201, 161],
            reference_origin: [-0.4, -0.4],
            reference_spacing: 0.2,
            reference_shape: [9, 9],
            refinement: 16,
            bisections: 10,
            membership_iterations: 128,
            max_membership_queries: 1_000_000,
            max_reference_nodes: 1000,
            max_segments: 10_000,
            maximum_coefficient_bytes: 300_000,
            coefficient_witness_seeding: true,
        };
        let classify =
            |p: [f64; 2]| (p[0] + 0.2).hypot(p[1]) < 0.12 || (p[0] - 0.3).hypot(p[1] - 0.3) < 0.04;
        let (field, stats) = prepare_with_membership(&plan, classify).unwrap();
        assert!(stats.coefficient_seeded_cells > 0);
        // The small circle fits strictly inside one coarse reference cell.
        // Its nearest boundary is nevertheless resolved by coefficient samples.
        let actual = field.query([0.32, 0.3]).unwrap().value;
        assert!(
            (actual + 0.02).abs() < 0.003,
            "hidden-component distance: {actual}"
        );
    }

    #[test]
    fn nearest_contour_search_preserves_both_components_even_at_a_tie() {
        let mut segments = Vec::new();
        for center in [-0.5, 0.5] {
            // Subdivision forces a tree. Distance to two squares is an
            // independent geometric oracle, including their equal-distance tie.
            for i in 0..20 {
                let a = -0.2 + i as f64 * 0.02;
                let b = a + 0.02;
                segments.extend([
                    Segment([center + a, -0.2], [center + b, -0.2]),
                    Segment([center + a, 0.2], [center + b, 0.2]),
                    Segment([center - 0.2, a], [center - 0.2, b]),
                    Segment([center + 0.2, a], [center + 0.2, b]),
                ]);
            }
        }
        let source = Contour::new(segments).unwrap();
        for point in [
            [-0.5_f64, 0.0],
            [0.5, 0.0],
            [0.0, 0.1],
            [-0.2, 0.0],
            [0.2, 0.0],
        ] {
            let expected = if point[0] == 0.0 {
                0.3
            } else if point[0].abs() == 0.5 {
                0.2
            } else {
                0.1
            };
            assert!((source.distance(point) - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn disconnected_contours_choose_the_nearest_component_and_refuse_empty_sources() {
        let plan = ContourPreparation {
            origin: [-0.9, -0.9],
            spacing: 0.02,
            shape: [91, 91],
            reference_origin: [-1.0, -1.0],
            reference_spacing: 0.1,
            reference_shape: [21, 21],
            refinement: 4,
            bisections: 12,
            membership_iterations: 128,
            max_membership_queries: 100_000,
            max_reference_nodes: 1000,
            max_segments: 1000,
            maximum_coefficient_bytes: 80_000,
            coefficient_witness_seeding: false,
        };
        let classify =
            |p: [f64; 2]| (p[0] - 0.5).hypot(p[1]) < 0.2 || (p[0] + 0.5).hypot(p[1]) < 0.2;
        let (field, _) = prepare_with_membership(&plan, classify).unwrap();
        // Field fidelity samples avoid the true distance's cut locus. The
        // source's exact nearest-distance tie is checked separately above.
        for point in [[-0.5, 0.1], [0.5, 0.1], [-0.2, 0.0], [0.2, 0.0]] {
            let expected = (point[0] - 0.5_f64)
                .hypot(point[1])
                .min((point[0] + 0.5).hypot(point[1]))
                - 0.2;
            let actual = field.query(point).unwrap().value;
            assert!(
                (actual - expected).abs() < 0.003,
                "{point:?}: {actual} versus {expected}"
            );
        }
        assert!(prepare_with_membership(&plan, |_| false).is_err());
    }

    #[test]
    fn prepared_plane_retains_scalar_gradient_and_hessian() {
        let plan = ContourPreparation {
            origin: [-0.2, -0.2],
            spacing: 0.01,
            shape: [41, 41],
            reference_origin: [-1.0, -1.0],
            reference_spacing: 0.1,
            reference_shape: [21, 21],
            refinement: 2,
            bisections: 16,
            membership_iterations: 128,
            max_membership_queries: 100_000,
            max_reference_nodes: 1000,
            max_segments: 1000,
            maximum_coefficient_bytes: 20_000,
            coefficient_witness_seeding: false,
        };
        let (field, stats) = prepare_with_membership(&plan, |p| p[0] < 0.0).unwrap();
        assert!(stats.membership_queries <= plan.max_membership_queries);
        let mut seeded = plan.clone();
        seeded.coefficient_witness_seeding = true;
        let (seeded_field, seeded_stats) =
            prepare_with_membership(&seeded, |p| p[0] < 0.0).unwrap();
        assert_eq!(seeded_stats.coefficient_seeded_cells, 0);
        assert_eq!(seeded_stats.membership_queries, stats.membership_queries);
        assert_eq!(seeded_field.asset_bytes(), field.asset_bytes());
        for p in [[-0.1, 0.0], [0.0, 0.0], [0.1, 0.05]] {
            let jet = field.query(p).unwrap();
            assert!((jet.value - p[0]).abs() < 1e-6);
            assert!((jet.gradient[0] - 1.0).abs() < 1e-6);
            assert!(jet.gradient[1].abs() < 1e-6);
            assert!(jet.hessian.iter().flatten().all(|v| v.abs() < 1e-5));
        }
        let mut exhausted = plan.clone();
        exhausted.max_membership_queries = 10;
        assert!(prepare_with_membership(&exhausted, |p| p[0] < 0.0).is_err());
        exhausted.coefficient_witness_seeding = true;
        assert!(prepare_with_membership(&exhausted, |p| p[0] < 0.0).is_err());
    }
}
