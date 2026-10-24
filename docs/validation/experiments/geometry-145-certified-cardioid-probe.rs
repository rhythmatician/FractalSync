#[path="../../../runtime-core/src/geometry_provider/interval.rs"] mod interval;
#[path="geometry-145-certified-cardioid.rs"] mod certified_cardioid;
fn main() {
    for c in [[-0.5,0.1],[0.,0.],[0.21474166530729011,0.12774018997553147],
        [0.32046083605181780,0.04736134021833255],[0.3,0.05],[0.25,0.],[-0.75,0.]] {
        let start=std::time::Instant::now();
        let bounds=certified_cardioid::distance_bounds(c,1e-13,8192);
        println!("{c:?}: {bounds:?}, us={}",start.elapsed().as_micros());
    }
}
