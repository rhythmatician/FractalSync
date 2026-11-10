const QUINTIC_BASIS: [[f64;6];6] = [
        [1., -5., 10., -10., 5., -1.],
        [26., -50., 20., 20., -20., 5.],
        [66., 0., -60., 0., 30., -10.],
        [26., 50., 20., -20., -20., 10.],
        [1., 5., 10., 10., 5., -5.],
        [0., 0., 0., 0., 0., 1.],
    ];

#[path="../../../runtime-core/src/geometry_provider/interval.rs"] mod interval;
#[path="../../../runtime-core/src/geometry_provider/range_jet.rs"] mod range_jet;
fn main() {}
