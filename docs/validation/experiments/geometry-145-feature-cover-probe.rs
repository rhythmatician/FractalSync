#[path="../../../runtime-core/src/geometry_provider/interval.rs"] mod interval;
#[path="geometry-145-certified-cardioid.rs"] mod certified_cardioid;
#[path="geometry-145-feature-cover.rs"] mod feature_cover;
fn main() {
    for c in [[-0.5,0.1],[0.,0.],[-0.5,0.],[0.25,0.],
        [0.21474166530729011,0.12774018997553147],
        [0.32046083605181780,0.04736134021833255],[0.3,0.05]] {
        let start=std::time::Instant::now();
        let result=feature_cover::cover(c,0.5e-6,1e-6/64.,65536);
        println!("{c:?}: {:?}, us={}",result.as_ref().map(|cover|(cover.projections().len(),cover.visits,feature_cover::directions_agree(cover,30f64.to_radians().cos()))),start.elapsed().as_micros());
    }
}
