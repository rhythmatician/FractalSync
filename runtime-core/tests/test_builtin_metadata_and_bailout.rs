use runtime_core::distance_field::load_builtin_distance_field;

#[test]
fn builtin_loads_and_returns_metadata() {
    let res = load_builtin_distance_field("mandelbrot_default").expect("builtin should load");
    let (rows, cols, xmin, xmax, ymin, ymax) = res;
    assert!(rows > 0 && cols > 0);
    assert!(xmin < xmax);
    assert!(ymin < ymax);
}
