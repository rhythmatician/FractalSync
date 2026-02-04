use runtime_core::distance_field::{clear_distance_field, sample_distance_field};
use runtime_core::geometry::Complex;

#[test]
fn sample_triggers_builtin_load() {
    // Ensure we start from cleared state
    clear_distance_field();
    let points = [Complex::new(0.0, 0.0)];
    let out = sample_distance_field(&points).expect("sample should succeed by auto-loading builtin");
    assert_eq!(out.len(), 1);
    assert!(out[0] >= 0.0);
}
