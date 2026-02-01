use ndarray::Array2;
use std::path::PathBuf;

use runtime_core::distance_field::{set_distance_field_from_vec, sample_distance_field};

#[test]
fn load_and_sample_small_field() {
    // create a small 3x3 field with known values
    let mut a = Array2::<f32>::zeros((3, 3));
    // rows (y) top-to-bottom
    // 1 2 3
    // 4 5 6
    // 7 8 9
    a[[0, 0]] = 1.0; a[[0, 1]] = 2.0; a[[0, 2]] = 3.0;
    a[[1, 0]] = 4.0; a[[1, 1]] = 5.0; a[[1, 2]] = 6.0;
    a[[2, 0]] = 7.0; a[[2, 1]] = 8.0; a[[2, 2]] = 9.0;

    let _td = tempfile::tempdir().expect("tempdir");

    // Directly set the distance field using the raw array values (no disk IO required)
    let v = a.clone().into_raw_vec();
    set_distance_field_from_vec(v, 3, 3, 0.0, 2.0, 0.0, 2.0).expect("set");

    // Sample at pixel centers: (x,y) ranges 0..2
    // sampling at (0,0) should be 1.0
    let xs = [0.0f64];
    let ys = [0.0f64];
    let out = sample_distance_field(&xs, &ys).expect("sample");
    assert_eq!(out.len(), 1);
    assert!((out[0] - 1.0).abs() < 1e-6);

    // sample at (1.0,1.0) center should be 5.0
    let xs = [1.0f64];
    let ys = [1.0f64];
    let out = sample_distance_field(&xs, &ys).expect("sample");
    assert!((out[0] - 5.0).abs() < 1e-6);

    // sample at (0.5, 0.5) should bilinear between 1,2,4,5 -> (1*(0.5*0.5)+...)
    let xs = [0.5f64];
    let ys = [0.5f64];
    let out = sample_distance_field(&xs, &ys).expect("sample");
    // manual bilinear: x=0.5 => sx=0.5, sy=0.5
    // a=1*(1-sx)+2*sx = 1.5; b=4*(1-sx)+5*sx=4.5; s = a*(1-sy)+b*sy = 3.0
    assert!((out[0] - 3.0).abs() < 1e-6);
}