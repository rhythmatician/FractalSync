use runtime_core::geometry_provider::{
    grad_rho_from_jet, grad_sigma_from_jet, hessian_rho_from_jet, hessian_sigma_from_jet,
    rho_from_jet, sigma_from_jet, GeometryJet, GeometryValidity, SingularityKind,
    GEOMETRY_PROVIDER_VERSION,
};
use runtime_core::manifold::ManifoldConfig;

// Quadratic field D=b+n.x+1/2 x^T H x, giving exact coherent jets even
// when it is not an ideal signed-distance field. This tests formulas only.
fn jet(b: f64, n: [f64; 2], h: [[f64; 2]; 2], x: [f64; 2]) -> GeometryJet {
    let g = [
        n[0] + h[0][0] * x[0] + h[0][1] * x[1],
        n[1] + h[1][0] * x[0] + h[1][1] * x[1],
    ];
    let d = b + n[0] * x[0] + n[1] * x[1] + 0.5 * (x[0] * (g[0] - n[0]) + x[1] * (g[1] - n[1]));
    GeometryJet {
        d,
        grad_d: g,
        hessian_d: h,
        resolved_scale: 1e-9,
        requested_scale: 1e-3,
        validity: GeometryValidity::Regular,
        singularity: SingularityKind::None,
        provider_version: GEOMETRY_PROVIDER_VERSION.into(),
        tile_id: "analytic".into(),
        is_bridge: false,
    }
}
// Normalize by the natural derivative scale, rather than an absolute Hessian
// tolerance: rho Hessians scale as 1/rho and sigma Hessians as 1/rho².
fn close(actual: f64, expected: f64, scale: f64, context: &str) {
    let residual = (actual - expected).abs() / scale.max(expected.abs()).max(1e-300);
    assert!(
        residual < 2e-12,
        "{context}: actual={actual} expected={expected} normalized residual={residual}"
    );
}

#[test]
fn analytic_jets_obey_ruler_and_scale_hessian_contracts() {
    let fields = [
        (0.0, [0.6, 0.8], [[0.0, 0.0], [0.0, 0.0]]),
        (0.0, [0.6, 0.8], [[0.3, -0.4], [-0.4, 0.7]]),
    ];
    for (b, n, h) in fields {
        for x in [[0.3, -0.1], [-0.3, 0.1]] {
            for epsilon in [0.001, 0.08, 0.7] {
                let j = jet(b, n, h, x);
                let cfg = ManifoldConfig {
                    epsilon,
                    d_ref: 0.17,
                    ..Default::default()
                };
                let r = rho_from_jet(&j, epsilon);
                assert!(r.is_finite() && r > 0.0);
                let gr = grad_rho_from_jet(&j, epsilon);
                let hr = hessian_rho_from_jet(&j, epsilon);
                let gs = grad_sigma_from_jet(&j, &cfg);
                let hs = hessian_sigma_from_jet(&j, &cfg);
                let q = j.d * j.d + epsilon * epsilon;
                for i in 0..2 {
                    close(gr[i], j.d * j.grad_d[i] / r, 1.0, "grad rho");
                    close(
                        gs[i],
                        -j.d * j.grad_d[i] / (q * std::f64::consts::LN_2),
                        1.0 / r,
                        "grad sigma",
                    );
                    for k in 0..2 {
                        let outer = j.grad_d[i] * j.grad_d[k];
                        close(
                            hr[i][k],
                            j.d * j.hessian_d[i][k] / r + epsilon * epsilon * outer / (r * r * r),
                            1.0 / r,
                            "H rho",
                        );
                        // Independent direct D-based form, not the rho-chain implementation.
                        let expected = -(j.d * j.hessian_d[i][k] / q
                            + (epsilon * epsilon - j.d * j.d) * outer / (q * q))
                            / std::f64::consts::LN_2;
                        close(hs[i][k], expected, 1.0 / q, "H sigma");
                        close(hs[i][k], hs[k][i], 1.0 / q, "symmetry");
                    }
                }
                let shifted = ManifoldConfig {
                    d_ref: cfg.d_ref * 8.0,
                    ..cfg.clone()
                };
                close(
                    sigma_from_jet(&j, &shifted) - sigma_from_jet(&j, &cfg),
                    3.0,
                    1.0,
                    "scale origin",
                );
                assert_eq!(grad_sigma_from_jet(&j, &shifted), gs);
                assert_eq!(hessian_sigma_from_jet(&j, &shifted), hs);
            }
        }
    }
}

#[test]
fn smooth_shore_has_rank_one_crest_independent_of_field_curvature() {
    for epsilon in [0.001, 0.08, 0.7] {
        for curvature in [0.0, 2.0] {
            let j = jet(
                0.0,
                [0.6, 0.8],
                [[curvature, -0.4], [-0.4, -curvature]],
                [0.0, 0.0],
            );
            let cfg = ManifoldConfig {
                epsilon,
                ..Default::default()
            };
            let gr = grad_rho_from_jet(&j, epsilon);
            let hr = hessian_rho_from_jet(&j, epsilon);
            let gs = grad_sigma_from_jet(&j, &cfg);
            let hs = hessian_sigma_from_jet(&j, &cfg);
            assert_eq!(gr, [0.0, 0.0]);
            assert_eq!(gs, [0.0, 0.0]);
            for i in 0..2 {
                for k in 0..2 {
                    let outer = j.grad_d[i] * j.grad_d[k];
                    close(hr[i][k], outer / epsilon, 1.0 / epsilon, "crest H rho");
                    close(
                        hs[i][k],
                        -outer / (epsilon * epsilon * std::f64::consts::LN_2),
                        1.0 / (epsilon * epsilon),
                        "crest H sigma",
                    );
                }
            }
            close(
                hr[0][0] * hr[1][1] - hr[0][1] * hr[1][0],
                0.0,
                1.0 / (epsilon * epsilon),
                "rank one",
            );
        }
    }
}

#[test]
fn curved_field_hessians_converge_against_independent_scalar_differences() {
    let n = [0.6, 0.8];
    let hess = [[0.3, -0.4], [-0.4, 0.7]];
    let x = [0.21, -0.09];
    let cfg = ManifoldConfig {
        epsilon: 0.2,
        d_ref: 0.17,
        ..Default::default()
    };
    let center = jet(0.0, n, hess, x);
    let hr = hessian_rho_from_jet(&center, cfg.epsilon);
    let hs = hessian_sigma_from_jet(&center, &cfg);
    // Differentiate scalar values directly, on a smooth analytic patch; each
    // halving should reduce central-difference truncation error roughly fourfold.
    let value = |p: [f64; 2]| {
        let d = jet(0.0, n, hess, p).d;
        let r = (d * d + cfg.epsilon * cfg.epsilon).sqrt();
        (r, (cfg.d_ref / r).log2())
    };
    let mut previous = f64::INFINITY;
    for step in [0.01, 0.005, 0.0025] {
        let mut error: f64 = 0.0;
        for i in 0..2 {
            for k in 0..2 {
                let mut pp = x;
                let mut pm = x;
                let mut mp = x;
                let mut mm = x;
                pp[i] += step;
                pp[k] += step;
                pm[i] += step;
                pm[k] -= step;
                mp[i] -= step;
                mp[k] += step;
                mm[i] -= step;
                mm[k] -= step;
                let a = value(pp);
                let b = value(pm);
                let c = value(mp);
                let d = value(mm);
                let denom = 4.0 * step * step;
                error = error.max(((a.0 - b.0 - c.0 + d.0) / denom - hr[i][k]).abs() * cfg.epsilon);
                error = error.max(
                    ((a.1 - b.1 - c.1 + d.1) / denom - hs[i][k]).abs() * cfg.epsilon * cfg.epsilon,
                );
            }
        }
        assert!(
            error < previous * 0.3,
            "step={step}, error={error}, previous={previous}"
        );
        previous = error;
    }
    assert!(previous < 1e-3, "final normalized error={previous}");
}

#[test]
fn bridge_reports_invalid_queries_and_mechanics_ignore_scale_origin() {
    use num_complex::Complex64;
    use runtime_core::geometry_provider::query_bridge_geometry;
    use runtime_core::manifold::{christoffel_symbols, induced_metric};
    let cfg = ManifoldConfig {
        epsilon: 0.05,
        d_ref: 0.1,
        ..Default::default()
    };
    let shifted = ManifoldConfig {
        d_ref: 0.8,
        ..cfg.clone()
    };
    let c = Complex64::new(-0.1, 0.1);
    let g = induced_metric(c, &cfg).unwrap();
    let g2 = induced_metric(c, &shifted).unwrap();
    let gamma = christoffel_symbols(c, &cfg).unwrap();
    let gamma2 = christoffel_symbols(c, &shifted).unwrap();
    for i in 0..2 {
        for j in 0..2 {
            close(
                g[i][j],
                g2[i][j],
                g[i][i].abs().max(g[j][j].abs()),
                "metric reference invariance",
            );
            for k in 0..2 {
                close(
                    gamma[i][j][k],
                    gamma2[i][j][k],
                    1.0 / cfg.epsilon,
                    "connection reference invariance",
                );
            }
        }
    }
    let invalid = query_bridge_geometry(Complex64::new(f64::NAN, 0.0), cfg.epsilon).unwrap();
    assert_eq!(invalid.validity, GeometryValidity::ProviderFailure);
    assert!(invalid.is_bridge);
    let outside = query_bridge_geometry(Complex64::new(2.1, 0.0), cfg.epsilon).unwrap();
    assert_eq!(outside.validity, GeometryValidity::OutsideDomain);
    assert_eq!(outside.singularity, SingularityKind::Boundary);
}
