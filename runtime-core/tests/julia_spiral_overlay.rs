use num_complex::Complex64;
use runtime_core::julia_spiral_overlay::sample;

#[test]
fn broad_and_inner_bands_reproduce_the_landmark_differential() {
    let c = Complex64::new(-0.7269, 0.1889);
    let [broad, inner] = sample(c).unwrap();
    assert!((broad.rate.re - 119.4187).abs() < 0.001);
    assert!((broad.rate.im - 2.078233).abs() < 0.001);
    assert!((broad.residual - 0.400607).abs() < 0.00001);
    assert!(inner.residual < 0.02);
    assert!(sample(Complex64::new(0.0, 0.0)).is_none());
}
