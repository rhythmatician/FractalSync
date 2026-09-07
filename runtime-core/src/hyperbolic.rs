//! Canonical H3 projection and its differential. Browser rendering mirrors are
//! pinned to these functions by generated golden vectors.

/// Upper half-space to the camera-centered Poincare ball, before scene mapping.
pub fn project(point: [f64; 3], camera: [f64; 3]) -> [f64; 3] {
    let x = (point[0] - camera[0]) / camera[2];
    let y = (point[1] - camera[1]) / camera[2];
    let z = point[2] / camera[2];
    let d = x * x + y * y + (z + 1.0).powi(2);
    [2.0 * x / d, 2.0 * y / d, (x * x + y * y + z * z - 1.0) / d]
}

/// Apply the projection differential to a tangent, before scene mapping.
pub fn tangent(point: [f64; 3], direction: [f64; 3], camera: [f64; 3]) -> [f64; 3] {
    let x = (point[0] - camera[0]) / camera[2];
    let y = (point[1] - camera[1]) / camera[2];
    let z = point[2] / camera[2];
    let [vx, vy, vz] = direction.map(|v| v / camera[2]);
    let d = x * x + y * y + (z + 1.0).powi(2);
    let dd = 2.0 * (x * vx + y * vy + (z + 1.0) * vz);
    let n = x * x + y * y + z * z - 1.0;
    let dn = 2.0 * (x * vx + y * vy + z * vz);
    [
        (2.0 * vx * d - 2.0 * x * dd) / d.powi(2),
        (2.0 * vy * d - 2.0 * y * dd) / d.powi(2),
        (dn * d - n * dd) / d.powi(2),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_in_goldens_match_the_authority() {
        #[derive(serde::Deserialize)]
        struct Case {
            point: [f64; 3],
            camera: [f64; 3],
            direction: [f64; 3],
            projected: [f64; 3],
            tangent: [f64; 3],
        }
        #[derive(serde::Deserialize)]
        struct Goldens {
            hyperbolic_cases: Vec<Case>,
        }
        let goldens: Goldens =
            serde_json::from_str(include_str!("../../shared/golden_vectors.json")).unwrap();
        assert!(goldens.hyperbolic_cases.len() >= 16);
        for case in goldens.hyperbolic_cases {
            let p = project(case.point, case.camera);
            let t = tangent(case.point, case.direction, case.camera);
            for i in 0..3 {
                assert!((p[i] - case.projected[i]).abs() < 1e-12);
                assert!((t[i] - case.tangent[i]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn differential_matches_centered_difference_and_preserves_angles() {
        let p = [0.2, -0.3, 0.4];
        let c = [-0.1, 0.4, 0.7];
        let v = [0.3, -0.1, 0.2];
        let h = 1e-6;
        let plus = project(std::array::from_fn(|i| p[i] + h * v[i]), c);
        let minus = project(std::array::from_fn(|i| p[i] - h * v[i]), c);
        let actual = tangent(p, v, c);
        for i in 0..3 {
            assert!((actual[i] - (plus[i] - minus[i]) / (2.0 * h)).abs() < 1e-9);
        }
        let axes = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]].map(|v| tangent(p, v, c));
        for i in 0..3 {
            for j in (i + 1)..3 {
                assert!((0..3).map(|k| axes[i][k] * axes[j][k]).sum::<f64>().abs() < 1e-12);
            }
        }
    }
}
