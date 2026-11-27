use std::hint::black_box;
use std::time::Instant;
use num_complex::Complex64;
use runtime_core::manifold::{geometry_jet, integrate_step, ManifoldConfig};
fn main() {
    let cfg = ManifoldConfig::default();
    let c = Complex64::new(-0.5, 0.2);
    let n = 1000;
    let now = Instant::now();
    for _ in 0..n { black_box(geometry_jet(black_box(c), &cfg).unwrap()); black_box(geometry_jet(black_box(c), &cfg).unwrap()); }
    let queries = now.elapsed();
    let now = Instant::now();
    for _ in 0..n { black_box(integrate_step(black_box(c), (0.01, 0.02), (0.1, 0.0), 0.1, 0.001, &cfg).unwrap()); }
    let steps = now.elapsed();
    println!("{n} pairs of geometry queries: {queries:?}; {n} full steps: {steps:?}");
}
