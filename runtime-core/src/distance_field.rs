//! Mandelbrot escape-time field for velocity-based slowdown near the boundary.
//!
//! Pre-computed escape-time field allows fast lookup using simple Mandelbrot
//! iteration counts. Points near the boundary take many iterations to escape,
//! so we use this to slow down orbit synthesis, creating a "potential well" effect.

use crate::geometry::Complex;

/// 2D escape-time field covering a rectangular region of the complex plane.
/// Values are normalized to [0, 1] where:
/// - 0 = didn't escape (on/inside boundary)
/// - 1 = escaped quickly (far from boundary)
#[derive(Clone, Debug)]
pub struct DistanceField {
    /// Flattened 2D array: field[row * width + col]
    pub field: Vec<f32>,
    /// Resolution (width and height, assumed square)
    pub resolution: usize,
    /// Real axis range
    pub real_min: f64,
    pub real_max: f64,
    /// Imaginary axis range
    pub imag_min: f64,
    pub imag_max: f64,
    /// Maximum distance before clamping (for normalization)
    pub max_distance: f64,
    /// Threshold distance for velocity slowdown (0-1 scale)
    pub slowdown_threshold: f64,
}

impl DistanceField {
    /// Create a new distance field from raw data.
    ///
    /// # Arguments
    /// * `field` - Flattened row-major array of normalized distances [0, 1]
    /// * `resolution` - Width/height of square field
    /// * `real_range` - (min, max) for real axis
    /// * `imag_range` - (min, max) for imaginary axis
    /// * `max_distance` - Max distance used for normalization
    /// * `slowdown_threshold` - Distance below which velocity scaling kicks in
    pub fn new(
        field: Vec<f32>,
        resolution: usize,
        real_range: (f64, f64),
        imag_range: (f64, f64),
        max_distance: f64,
        slowdown_threshold: f64,
    ) -> Self {
        assert_eq!(
            field.len(),
            resolution * resolution,
            "Field size must match resolution²"
        );

        Self {
            field,
            resolution,
            real_min: real_range.0,
            real_max: real_range.1,
            imag_min: imag_range.0,
            imag_max: imag_range.1,
            max_distance,
            slowdown_threshold,
        }
    }

    /// Look up escape time at a complex point.
    ///
    /// Returns normalized escape time [0, 1]:
    /// - 0.0 = didn't escape (inside or on boundary)
    /// - 1.0 = escaped quickly (far outside) or out of bounds
    ///
    /// Uses nearest-neighbor lookup for speed.
    pub fn lookup(&self, c: Complex) -> f32 {
        // Nearest-neighbor lookup for speed (existing behaviour)
        // Convert to pixel coordinates
        let real_scale = (self.real_max - self.real_min) / (self.resolution as f64);
        let imag_scale = (self.imag_max - self.imag_min) / (self.resolution as f64);

        let col = ((c.real - self.real_min) / real_scale).floor() as isize;
        let row = ((c.imag - self.imag_min) / imag_scale).floor() as isize;

        // Check bounds
        if col < 0
            || col >= self.resolution as isize
            || row < 0
            || row >= self.resolution as isize
        {
            return 1.0; // Far outside
        }

        // Lookup in flattened array
        let idx = (row as usize) * self.resolution + (col as usize);
        self.field[idx]
    }

    /// Bilinear sample at arbitrary (non-integer) coordinates.
    /// Returns interpolated escape-time in [0,1]. Out-of-bounds -> 1.0
    pub fn sample_bilinear(&self, c: Complex) -> f32 {
        let real_scale = (self.real_max - self.real_min) / (self.resolution as f64);
        let imag_scale = (self.imag_max - self.imag_min) / (self.resolution as f64);

        let col_f = (c.real - self.real_min) / real_scale;
        let row_f = (c.imag - self.imag_min) / imag_scale;

        // If out of range, return far-out value
        if col_f < 0.0 || col_f > (self.resolution as f64 - 1.0) || row_f < 0.0 || row_f > (self.resolution as f64 - 1.0) {
            return 1.0;
        }

        let col0 = col_f.floor() as usize;
        let row0 = row_f.floor() as usize;
        let col1 = (col0 + 1).min(self.resolution - 1);
        let row1 = (row0 + 1).min(self.resolution - 1);

        let dx = (col_f - col0 as f64) as f32;
        let dy = (row_f - row0 as f64) as f32;

        let v00 = self.field[row0 * self.resolution + col0];
        let v10 = self.field[row0 * self.resolution + col1];
        let v01 = self.field[row1 * self.resolution + col0];
        let v11 = self.field[row1 * self.resolution + col1];

        // Bilinear interpolation
        let top = v00 * (1.0 - dx) + v10 * dx;
        let bottom = v01 * (1.0 - dx) + v11 * dx;
        let val = top * (1.0 - dy) + bottom * dy;

        val
    }

    /// Estimate gradient (∂d/∂x, ∂d/∂y) at point c using central differences
    /// on the bilinearly sampled field. Returned gradients are in units of
    /// (escape_time / complex-plane unit)
    pub fn gradient(&self, c: Complex) -> (f64, f64) {
        // Use one grid cell in world units as finite difference step
        let dx = (self.real_max - self.real_min) / (self.resolution as f64);
        let dy = (self.imag_max - self.imag_min) / (self.resolution as f64);

        // Sample left/right and up/down
        let left = self.sample_bilinear(Complex::new(c.real - dx, c.imag));
        let right = self.sample_bilinear(Complex::new(c.real + dx, c.imag));
        let down = self.sample_bilinear(Complex::new(c.real, c.imag - dy));
        let up = self.sample_bilinear(Complex::new(c.real, c.imag + dy));

        let gx = ((right as f64) - (left as f64)) / (2.0 * dx);
        let gy = ((up as f64) - (down as f64)) / (2.0 * dy);

        (gx, gy)
    }

    /// Compute velocity scale factor based on escape time.
    ///
    /// Semantics:
    /// - Small escape time (far from boundary) → run at full speed
    /// - Large escape time (near boundary / inside) → slow down
    ///
    /// `slowdown_threshold` defines the escape-time value at which
    /// slowdown begins. Below it, scale is 1. Above it, we smoothly
    /// decrease to 0 as escape_time → 1.
    pub fn velocity_scale(&self, escape_time: f32) -> f32 {
        let th = self.slowdown_threshold as f32;
        // If at or above threshold, we are safely outside and run at full speed
        if escape_time >= th {
            return 1.0;
        }

        // Otherwise map [0, th] -> [0, 1] using smoothstep
        let t = (escape_time / th).clamp(0.0, 1.0);
        let s = t * t * (3.0 - 2.0 * t);
        s
    }

    /// Combined lookup and velocity scale computation.
    ///
    /// This is the main entry point for orbit synthesis:
    /// given a complex point, return how much to scale dt.
    pub fn get_velocity_scale(&self, c: Complex) -> f32 {
        let distance = self.lookup(c);
        self.velocity_scale(distance)
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lookup_in_bounds() {
        // Create simple 3×3 field
        #[rustfmt::skip]
        let field = vec![
            1.0, 1.0, 1.0,
            1.0, 0.0, 1.0,  // Center is on boundary
            1.0, 1.0, 1.0,
        ];

        let df = DistanceField::new(
            field,
            3,
            (-1.0, 1.0),
            (-1.0, 1.0),
            0.5,
            0.05,
        );

        // Center point should be 0.0
        let dist = df.lookup(Complex::new(0.0, 0.0));
        assert_eq!(dist, 0.0);

        // Corner should be 1.0
        let dist = df.lookup(Complex::new(0.9, 0.9));
        assert_eq!(dist, 1.0);
    }

    #[test]
    fn test_lookup_out_of_bounds() {
        let field = vec![0.0; 9];
        let df = DistanceField::new(field, 3, (-1.0, 1.0), (-1.0, 1.0), 0.5, 0.05);

        // Out of bounds should return 1.0
        let dist = df.lookup(Complex::new(10.0, 10.0));
        assert_eq!(dist, 1.0);
    }

    #[test]
    fn test_velocity_scale_at_boundary() {
        let field = vec![0.0; 9];
        let df = DistanceField::new(field, 3, (-1.0, 1.0), (-1.0, 1.0), 0.5, 0.1);

        // At boundary (distance=0), velocity should be 0
        let scale = df.velocity_scale(0.0);
        assert_eq!(scale, 0.0);
    }

    #[test]
    fn test_velocity_scale_beyond_threshold() {
        let field = vec![0.0; 9];
        let df = DistanceField::new(field, 3, (-1.0, 1.0), (-1.0, 1.0), 0.5, 0.1);

        // Beyond threshold, velocity should be 1.0
        let scale = df.velocity_scale(0.2); // > 0.1 threshold
        assert_eq!(scale, 1.0);
    }

    #[test]
    fn test_velocity_scale_smoothstep() {
        let field = vec![0.0; 9];
        let df = DistanceField::new(field, 3, (-1.0, 1.0), (-1.0, 1.0), 0.5, 0.1);

        // At half threshold, should use smoothstep
        let scale = df.velocity_scale(0.05); // Half of 0.1
        // smoothstep(0.5) = 0.5² × (3 - 2×0.5) = 0.25 × 2.0 = 0.5
        assert!((scale - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_sample_bilinear_values() {
        // Field 4x4 values 0..15
        let field: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let df = DistanceField::new(field.clone(), 4, (0.0, 4.0), (0.0, 4.0), 1.0, 0.05);

        // Points to test
        let pts = vec![
            (0.5, 0.5, 2.5f32),
            (1.25, 2.75, 12.25f32),
            (3.0, 3.0, 15.0f32),
            (0.0, 0.0, 0.0f32),
            (3.999, 3.999, 1.0f32),
        ];

        for (r, i, expected) in pts {
            let val = df.sample_bilinear(Complex::new(r, i));
            assert!((val - expected).abs() < 1e-4, "val {} != expected {} at ({},{})", val, expected, r, i);
        }
    }
}
