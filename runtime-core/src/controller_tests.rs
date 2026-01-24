#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance_field::DistanceField;
    use crate::geometry::Complex;

    #[test]
    fn test_contour_step_tangential_preserved() {
        // Create a simple distance field where gradient points in +x direction
        // d(x,y) = (x+1)/2 mapped to [0,1] across [-1,1]
        let resolution = 3;
        // field values by row (y): each element increases left-to-right
        // rows: y=-1 -> y=1
        let field = vec![
            0.0, 0.5, 1.0,
            0.0, 0.5, 1.0,
            0.0, 0.5, 1.0,
        ];
        let df = DistanceField::new(field, resolution, (-1.0, 1.0), (-1.0, 1.0), 1.0, 0.1);

        // Point at center (0,0). Gradient should point in +x.
        let c = Complex::new(0.0, 0.0);
        let u = (0.0, 0.5); // movement purely in +y (which is tangential here)
        // h=0 (no hit): normal suppressed -> tangential preserved
        let out = contour_biased_step(c, u.0, u.1, 0.0, Some(&df), 0.5, 0.2);

        // Expect y component to be non-zero and reasonable; x should be near zero
        assert!(out.imag.abs() > 0.0);
        assert!(out.real.abs() < 0.1);
    }

    #[test]
    fn test_contour_step_normal_suppressed_and_allowed() {
        // Create radial gradient field (center low, outside high)
        let resolution = 3;
        let field = vec![
            1.0, 1.0, 1.0,
            1.0, 0.0, 1.0,
            1.0, 1.0, 1.0,
        ];
        let df = DistanceField::new(field, resolution, (-1.0, 1.0), (-1.0, 1.0), 1.0, 0.1);
        let c = Complex::new(0.0, 0.0);

        // Proposed u is normal outward (positive x)
        let u = (0.1, 0.0);
        let out_no_hit = contour_biased_step(c, u.0, u.1, 0.0, Some(&df), 0.5, 1.0);
        let out_hit = contour_biased_step(c, u.0, u.1, 1.0, Some(&df), 0.5, 1.0);

        // Without hit, movement should be small (normal suppressed)
        let dx_no_hit = out_no_hit.real - c.real;
        let dx_hit = out_hit.real - c.real;
        assert!(dx_no_hit.abs() < dx_hit.abs());
    }
}