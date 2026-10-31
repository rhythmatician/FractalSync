#[derive(Clone, Copy, Debug)]
struct Jet {
    d: f64,
    g: [f64; 2],
    h: [[f64; 2]; 2],
    p: [f64; 2],
    gap: f64,
}
fn polynomial(a: [f64; 4], x: f64) -> f64 {
    ((a[3] * x + a[2]) * x + a[1]) * x + a[0]
}
fn roots(a: [f64; 4]) -> Vec<f64> {
    let mut split = vec![-1., 1.];
    let aa = 3. * a[3];
    let b = 2. * a[2];
    let c = a[1];
    if aa == 0. {
        if b != 0. {
            split.push(-c / b);
        }
    } else {
        let discriminant = b * b - 4. * aa * c;
        if discriminant >= 0. {
            let q = -0.5 * (b + discriminant.sqrt().copysign(b));
            if q == 0. {
                split.push(-b / (2. * aa));
            } else {
                split.push(q / aa);
                split.push(c / q);
            }
        }
    }
    split.retain(|x| x.is_finite() && *x >= -1. && *x <= 1.);
    split.sort_by(f64::total_cmp);
    split.dedup();
    let mut result = Vec::new();
    for &x in &split {
        if polynomial(a, x).abs() < 1e-14 {
            result.push(x);
        }
    }
    for pair in split.windows(2) {
        let (mut lo, mut hi) = (pair[0], pair[1]);
        let mut f = polynomial(a, lo);
        if f * polynomial(a, hi) >= 0. {
            continue;
        }
        for _ in 0..80 {
            let mid = lo + (hi - lo) * 0.5;
            if mid == lo || mid == hi {
                break;
            }
            let fm = polynomial(a, mid);
            if fm == 0. {
                lo = mid;
                hi = mid;
                break;
            }
            if f.is_sign_negative() == fm.is_sign_negative() {
                lo = mid;
                f = fm;
            } else {
                hi = mid;
            }
        }
        result.push(lo + (hi - lo) * 0.5);
    }
    result
}
fn cardioid(c: [f64; 2]) -> Jet {
    let [x, y] = c;
    let a = [0.5 - 2. * x, -6. * y, 6. * x + 0.5, 2. * y];
    let mut theta = vec![0.];
    theta.extend(roots(a).into_iter().map(|s| 2. * s.atan()));
    theta.extend(
        roots([a[3], a[2], a[1], a[0]])
            .into_iter()
            .map(|r| std::f64::consts::PI - 2. * r.atan()),
    );
    let curve = |t: f64| {
        [
            0.5 * t.cos() - 0.25 * (2. * t).cos(),
            0.5 * t.sin() - 0.25 * (2. * t).sin(),
        ]
    };
    let mut candidates: Vec<_> = theta
        .into_iter()
        .map(|t| {
            let p = curve(t);
            ((p[0] - x).hypot(p[1] - y), t, p)
        })
        .collect();
    candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (r, t, p) = candidates[0];
    let u = [(x - p[0]) / r, (y - p[1]) / r];
    let tangent = [
        -0.5 * t.sin() + 0.5 * (2. * t).sin(),
        0.5 * t.cos() - 0.5 * (2. * t).cos(),
    ];
    let second = [
        -0.5 * t.cos() + (2. * t).cos(),
        -0.5 * t.sin() + (2. * t).sin(),
    ];
    let denom = tangent[0] * tangent[0]
        + tangent[1] * tangent[1]
        + (p[0] - x) * second[0]
        + (p[1] - y) * second[1];
    let mut h = [[0.; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            let dp = if tangent[0].hypot(tangent[1]) < 1e-14 {
                0.
            } else {
                tangent[i] * tangent[j] / denom
            };
            h[i][j] = (dp + u[i] * u[j] - if i == j { 1. } else { 0. }) / r;
        }
    }
    let gap = candidates
        .iter()
        .skip(1)
        .filter(|v| (v.2[0] - p[0]).hypot(v.2[1] - p[1]) > 1e-10)
        .map(|v| v.0 - r)
        .fold(f64::INFINITY, f64::min);
    Jet {
        d: -r,
        g: [-u[0], -u[1]],
        h,
        p,
        gap,
    }
}
fn main() {
    for c in [
        [-0.5, 0.1],
        [0.30007240651412576, 0.05348910097156354],
        [0.21474166530729011, 0.12774018997553147],
        [0., 0.],
        [-0.5, 0.],
    ] {
        println!("{c:?}: {:?}", cardioid(c));
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn derivatives_match_shared_scalar() {
        for c in [
            [-0.5, 0.1],
            [0.30007240651412576, 0.05348910097156354],
            [0.21474166530729011, 0.12774018997553147],
            [0., 0.],
        ] {
            let center = cardioid(c);
            for axis in 0..2 {
                let mut m = c;
                let mut p = c;
                let h = 1e-6;
                m[axis] -= h;
                p[axis] += h;
                let minus = cardioid(m);
                let plus = cardioid(p);
                assert!(
                    ((plus.d - minus.d) / (2. * h) - center.g[axis]).abs() < 1e-7,
                    "{c:?} axis {axis}"
                );
                for i in 0..2 {
                    assert!(
                        ((plus.g[i] - minus.g[i]) / (2. * h) - center.h[i][axis]).abs() < 1e-5,
                        "{c:?} axis {axis} component {i}"
                    );
                }
            }
        }
    }
    #[test]
    fn distinct_nearest_points_remain_explicit() {
        let t: f64 = 0.6;
        let p = [
            0.5 * t.cos() - 0.25 * (2. * t).cos(),
            0.5 * t.sin() - 0.25 * (2. * t).sin(),
        ];
        let tangent = [
            -0.5 * t.sin() + 0.5 * (2. * t).sin(),
            0.5 * t.cos() - 0.5 * (2. * t).cos(),
        ];
        let v = [p[0] - 0.25, p[1]];
        let b = 0.5 * (p[0] * p[0] + p[1] * p[1] - 0.0625);
        let d = p[0] * tangent[0] + p[1] * tangent[1];
        let det = v[0] * tangent[1] - v[1] * tangent[0];
        let c = [
            (b * tangent[1] - v[1] * d) / det,
            (v[0] * d - b * tangent[0]) / det,
        ];
        let j = cardioid(c);
        assert!(j.gap < 1e-14, "{c:?} {j:?}");
    }
    #[test]
    fn cusp_is_one_nearest_feature() {
        let j = cardioid([0., 0.]);
        assert_eq!(j.d, -0.25);
        assert_eq!(j.g, [1., 0.]);
        assert_eq!(j.h, [[0., 0.], [0., -4.]]);
    }
}
