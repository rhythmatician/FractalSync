use super::{FrozenRollout, Integrator, PhysicsProfile, ReferenceCut};
use crate::{controls::MotionControls, manifold::ManifoldConfig, operational_field::PreparedField};
use num_complex::Complex64;
use std::{
    cell::RefCell,
    collections::BTreeSet,
    f64::consts::{FRAC_1_PI, LN_2},
    sync::Arc,
};

thread_local! {
    static QUERIES: RefCell<Option<Vec<[f64; 2]>>> = const { RefCell::new(None) };
}

pub(super) fn record_query(c: Complex64) {
    QUERIES.with(|queries| {
        if let Some(points) = queries.borrow_mut().as_mut() {
            points.push([c.re, c.im]);
        }
    });
}

fn record<T>(f: impl FnOnce() -> T) -> (T, Vec<[f64; 2]>) {
    QUERIES.with(|queries| assert!(queries.replace(Some(Vec::new())).is_none()));
    let result = f();
    (result, QUERIES.with(|queries| queries.take().unwrap()))
}

#[derive(serde::Deserialize)]
struct Fixture {
    shape: [usize; 2],
    origin: [f64; 2],
    spacing_binary64: String,
    coefficient_binary64_row_major: Vec<String>,
}

fn decode_float(bits: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(bits.strip_prefix("0x").unwrap(), 16).unwrap())
}

fn expected_addresses(point: [f64; 2], origin: [f64; 2], h: f64, shape: [usize; 2]) -> [usize; 36] {
    let [x, y] = std::array::from_fn(|axis| {
        let q = (point[axis] - origin[axis]) / h;
        assert!(q.is_finite() && q >= 2.0 && q < (shape[axis] - 3) as f64);
        q.floor() as usize
    });
    std::array::from_fn(|index| (y - 2 + index / 6) * shape[0] + x - 2 + index % 6)
}

#[test]
fn literal_quadratic_rk_queries_require_exactly_the_predicted_stencil() {
    let Fixture {
        shape,
        origin,
        spacing_binary64,
        coefficient_binary64_row_major,
    } = serde_json::from_str(include_str!(
        "../../../formal/fixtures/operational-quadratic.json"
    ))
    .unwrap();
    assert_eq!(shape, [16, 16]);
    let h = decode_float(&spacing_binary64);
    let coefficients: Vec<_> = coefficient_binary64_row_major
        .iter()
        .map(|bits| decode_float(bits))
        .collect();
    let bytes = PreparedField::new(origin, h, shape, coefficients.clone())
        .unwrap()
        .asset_bytes();
    let field = Arc::new(PreparedField::from_asset_bytes(&bytes).unwrap());
    assert_eq!(field.asset_bytes(), bytes);

    let mut verified_cells = BTreeSet::new();
    let mut verify_stencil = |point: [f64; 2]| {
        assert!(point
            .into_iter()
            .all(|axis| (axis - 0.5).abs() < 1.0 / 64.0));
        let addresses = expected_addresses(point, origin, h, shape);
        let actual = field.query(point).unwrap();
        let mut subset = vec![None; coefficients.len()];
        for &index in &addresses {
            subset[index] = Some(coefficients[index]);
        }
        let only_stencil = PreparedField::new_sparse(origin, h, shape, subset.clone()).unwrap();
        assert_eq!(only_stencil.query(point).unwrap(), actual);
        if verified_cells.insert(addresses[0]) {
            for index in addresses {
                let value = subset[index].take();
                let missing = PreparedField::new_sparse(origin, h, shape, subset.clone()).unwrap();
                assert!(
                    missing.query(point).is_err(),
                    "accepted missing address {index}"
                );
                subset[index] = value;
            }
        }
    };

    let config = ManifoldConfig {
        d_ref: 1.0,
        epsilon: 1.0,
        lambda_sq: LN_2 * LN_2,
        kappa: LN_2,
        mu: FRAC_1_PI,
    };
    let drive = MotionControls {
        direction: [1.0, 0.0],
        throttle: 0.5,
        ..MotionControls::default()
    };
    for (method, substeps) in [(Integrator::Rk4, 1), (Integrator::Rk4FourSubsteps, 4)] {
        let profile =
            PhysicsProfile::new(config.clone(), 1.0 / (576.0 * 16.0), 1.0, method).unwrap();
        let rollout = FrozenRollout::new(field.clone(), profile.clone(), vec![]).unwrap();
        let initial = rollout.initial_state([0.5; 2], [0.25, 0.0]).unwrap();
        let mut state = initial;
        let mut first_stage_two = None;
        let mut provider_queries = 0;
        for _ in 0..16 {
            let before = state;
            let command = rollout.drive_covector(state.position, &drive).unwrap();
            let (accepted, positions) =
                record(|| rollout.attempt_step_with_trace(&mut state, command));
            let (work, trace) = accepted.unwrap();
            assert_eq!(work.geometry_evaluations, positions.len());
            assert_eq!(work.coefficient_reads_upper_bound, 36 * positions.len());
            assert!(positions.len() >= 5 * substeps);
            provider_queries += positions.len();
            assert_eq!(positions[0], before.position);
            assert_eq!(trace.endpoints().len(), substeps + 1);
            for endpoint in trace.endpoints() {
                assert!(positions.contains(&endpoint.position));
            }
            if first_stage_two.is_none() {
                first_stage_two = positions
                    .iter()
                    .copied()
                    .find(|&point| point != before.position);
            }
            positions.into_iter().for_each(&mut verify_stencil);
            for t in [0.25, 0.5, 0.75] {
                let point = std::array::from_fn(|axis| {
                    before.position[axis] + t * (state.position[axis] - before.position[axis])
                });
                verify_stencil(point);
            }
        }

        // The unmasked field supports this point. A separate reference
        // refusal at the first attempted stage away from the initial point
        // must fail before the outer transaction commits.
        let [x, y] = first_stage_two.unwrap();
        let cut = ReferenceCut {
            bounds: [x, y, x, y],
            authority: "synthetic analytic reference-control cut".into(),
        };
        let blocked = FrozenRollout::new(field.clone(), profile, vec![cut]).unwrap();
        let before = blocked
            .initial_state(initial.position, initial.velocity)
            .unwrap();
        let mut state = before;
        let command = blocked.drive_covector(before.position, &drive).unwrap();
        let error = blocked
            .attempt_step_with_trace(&mut state, command)
            .unwrap_err();
        assert!(
            error.contains("RK stage 2") && error.contains("ReferenceSingularity"),
            "{error}"
        );
        assert_eq!(state, before);
        println!("{method:?}: {provider_queries} provider queries and 48 interpolation probes");
    }
    println!("Verified {} distinct cell stencils", verified_cells.len());
}
