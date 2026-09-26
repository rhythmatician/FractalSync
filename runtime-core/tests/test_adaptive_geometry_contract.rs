use num_complex::Complex64;
use runtime_core::distance_field;
use runtime_core::geometry_provider::{
    GeometryProvider, GeometryValidity, ScaleAwareGeometryProvider, SingularityKind,
    GEOMETRY_PROVIDER_VERSION,
};

fn regular_interior() -> Complex64 {
    Complex64::new(-0.5, 0.1)
}

#[test]
fn regular_public_jet_meets_its_scale_and_provenance_contract() {
    let _guard = distance_field::global_test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let jet = ScaleAwareGeometryProvider::default()
        .query(regular_interior(), 1e-4)
        .expect("provider query should return a classified jet");

    assert_eq!(jet.validity, GeometryValidity::Regular);
    assert!(jet.estimated_error.is_finite());
    assert!(jet.requested_scale.is_finite() && jet.requested_scale > 0.0);
    assert!(jet.resolved_scale.is_finite() && jet.resolved_scale > 0.0);
    assert!(jet.resolved_scale <= jet.requested_scale);
    assert!(jet.estimated_error <= 0.25 * jet.requested_scale);
    assert_eq!(jet.provider_version, GEOMETRY_PROVIDER_VERSION);
    assert!(jet.tile_id.starts_with("scale-aware:"));
    assert!(!jet.is_bridge);
    let seam_jet = runtime_core::geometry_provider::query_geometry(regular_interior(), 1e-4)
        .expect("public query seam should use the destination provider");
    assert_eq!(jet.d, seam_jet.d);
    assert_eq!(jet.grad_d, seam_jet.grad_d);
    assert_eq!(jet.hessian_d, seam_jet.hessian_d);
    assert_eq!(jet.estimated_error, seam_jet.estimated_error);
    assert_eq!(jet.validity, seam_jet.validity);
    assert_eq!(jet.singularity, seam_jet.singularity);
    assert_eq!(jet.tile_id, seam_jet.tile_id);
}

#[test]
fn destination_provider_is_independent_of_loaded_bridge_and_marks_provenance() {
    let _guard = distance_field::global_test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    distance_field::clear_distance_field();
    let without_bridge = runtime_core::geometry_provider::query_geometry(regular_interior(), 1e-4)
        .expect("destination query without raster should succeed");
    assert!(!distance_field::is_field_loaded());

    distance_field::load_builtin_distance_field("mandelbrot_default")
        .expect("fixture bridge should load");
    let with_bridge = runtime_core::geometry_provider::query_geometry(regular_interior(), 1e-4)
        .expect("destination query with raster loaded should succeed");
    assert_eq!(with_bridge.d, without_bridge.d);
    assert_eq!(with_bridge.grad_d, without_bridge.grad_d);
    assert_eq!(with_bridge.hessian_d, without_bridge.hessian_d);
    assert_eq!(with_bridge.estimated_error, without_bridge.estimated_error);
    assert_eq!(with_bridge.validity, without_bridge.validity);
    assert_eq!(with_bridge.singularity, without_bridge.singularity);
    assert_eq!(with_bridge.resolved_scale, without_bridge.resolved_scale);
    assert_eq!(with_bridge.requested_scale, without_bridge.requested_scale);
    assert_eq!(with_bridge.tile_id, without_bridge.tile_id);
    assert_eq!(with_bridge.provider_version, GEOMETRY_PROVIDER_VERSION);
    assert!(!with_bridge.is_bridge);

    let bridge = runtime_core::geometry_provider::query_bridge_geometry(
        regular_interior(),
        1e-4,
    )
    .expect("explicit bridge query should succeed when its raster is loaded");
    assert!(bridge.is_bridge);
    assert_eq!(bridge.provider_version, GEOMETRY_PROVIDER_VERSION);
}

#[test]
fn cut_locus_and_unachievable_shore_scale_fail_closed() {
    let _guard = distance_field::global_test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let cusp = runtime_core::geometry_provider::query_geometry(Complex64::new(0.0, 0.0), 1e-4)
        .expect("cusp query should return an explicit classification");
    assert_eq!(cusp.validity, GeometryValidity::Singular);
    assert_eq!(cusp.singularity, SingularityKind::CutLocus);

    // The smooth top of the primary cardioid is on the Shore. This epsilon
    // requests a spatial scale far below the provider's finite dyadic budget.
    let deep_shore = runtime_core::geometry_provider::query_geometry(
        Complex64::new(0.25, 0.5),
        1e-100,
    )
    .expect("bounded refinement should return an unresolved jet");
    assert_eq!(deep_shore.validity, GeometryValidity::Unresolved);
    assert!(!deep_shore.is_bridge);
    assert!(deep_shore.resolved_scale > deep_shore.requested_scale);
}
