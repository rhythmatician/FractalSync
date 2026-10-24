use astro_float::{BigFloat, RoundingMode};
mod point_proof;
fn escape(c: [f64; 2], precision: usize) -> Option<usize> {
    let rm = RoundingMode::ToEven;
    let cr = BigFloat::from_f64(c[0], precision);
    let ci = BigFloat::from_f64(c[1], precision);
    let two = BigFloat::from_f64(2., precision);
    let four = BigFloat::from_f64(4., precision);
    let mut zr = BigFloat::new(precision);
    let mut zi = BigFloat::new(precision);
    for n in 0..32768 {
        let rr = zr.mul(&zr, precision, rm);
        let ii = zi.mul(&zi, precision, rm);
        let norm = rr.add(&ii, precision, rm);
        assert!(!norm.is_nan());
        if norm.cmp(&four).is_some_and(|order| order > 0) {
            return Some(n);
        }
        let imaginary = zr
            .mul(&zi, precision, rm)
            .mul(&two, precision, rm)
            .add(&ci, precision, rm);
        zr = rr.sub(&ii, precision, rm).add(&cr, precision, rm);
        zi = imaginary;
    }
    None
}
fn main() {
    if std::env::args().nth(1).as_deref() == Some("proof") {
        for p in [128, 256, 512] {
            assert_eq!(point_proof::escape([1., 0.], p, 32768).unwrap().update, 3);
            assert!(point_proof::escape([0., 0.], p, 32768).is_none());
            for (name, c) in [
                ("blocking_node", [0.3204946517944336, 0.04734092950820923]),
                ("training", [0.3075204398270623, 0.02876420131721053]),
                ("native", [0.32049988639807619, 0.04733531447355364]),
                (
                    "python_crossing",
                    [0.35659381055433242, 0.06404656757776669],
                ),
            ] {
                let start = std::time::Instant::now();
                let result = point_proof::membership(c, p, 32768);
                println!(
                    "[POINT_PROOF145] name={name} elapsed_s={} result={result:?}",
                    start.elapsed().as_secs_f64()
                );
            }
        }
        return;
    }
    for precision in [128, 256, 512] {
        assert_eq!(escape([1., 0.], precision), Some(3));
        assert_eq!(escape([0., 0.], precision), None);
        for (name, c) in [
            ("blocking_node", [0.3204946517944336, 0.04734092950820923]),
            ("retained_node", [0.32049965858459473, 0.04733508825302124]),
            ("next_native", [0.32049988639807619, 0.04733531447355364]),
            (
                "python_crossing",
                [0.35659381055433242, 0.06404656757776669],
            ),
            ("training", [0.3075204398270623, 0.02876420131721053]),
        ] {
            let start = std::time::Instant::now();
            let first_escape = escape(c, precision);
            println!(
                "name={name} precision={precision} first_escape={first_escape:?} elapsed_s={}",
                start.elapsed().as_secs_f64()
            );
        }
    }
}
