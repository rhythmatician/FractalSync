//! Diagnostic differential of the marked preimage ladder at the default spiral.

use num_complex::Complex64;

pub const CENTER: [f64; 2] = [-0.7269, 0.1889];
pub const HALF: f64 = 0.00025;
const BANDS: [[usize; 5]; 2] = [[128, 256, 512, 768, 1024], [768, 832, 896, 960, 1024]];

#[derive(Clone, Copy, Debug, Default)]
pub struct SpiralDifferential {
    /// Real-motion coefficient. For heading h, the fitted complex rate is rate * h.
    pub rate: Complex64,
    /// Fraction of squared landmark motion left after the similarity fit.
    pub residual: f64,
}

pub fn sample(c: Complex64) -> Option<[SpiralDifferential; 2]> {
    let lambda = 1.0 - (1.0 - 4.0 * c).sqrt();
    if !lambda.norm().is_finite() || lambda.norm() <= 1.0 {
        return None;
    }
    let p = lambda / 2.0;
    let dp = 1.0 / (1.0 - lambda);
    let (mut q, mut dq) = (-p, -dp);
    let mut marks = [[(Complex64::default(), Complex64::default()); 5]; 2];
    for depth in 1..=1024 {
        let root = (q - c).sqrt();
        let side = (root.conj() * p).re;
        if !side.is_finite() || side.abs() < 1e-12 {
            return None;
        }
        q = if side > 0.0 { root } else { -root };
        dq = (dq - 1.0) / (2.0 * q);
        for (band, depths) in BANDS.iter().enumerate() {
            if let Some(index) = depths.iter().position(|&n| n == depth) {
                marks[band][index] = (q - p, dq - dp);
            }
        }
    }

    let mut result = [SpiralDifferential::default(); 2];
    for (out, band) in result.iter_mut().zip(marks) {
        let weight: f64 = band.iter().map(|(w, _)| w.norm_sqr()).sum();
        let total: f64 = band.iter().map(|(_, dw)| dw.norm_sqr()).sum();
        if weight <= 0.0 || total <= 0.0 || !total.is_finite() {
            return None;
        }
        let rate = band.iter().map(|(w, dw)| w.conj() * dw).sum::<Complex64>() / weight;
        let residual = band
            .iter()
            .map(|(w, dw)| (dw - rate * w).norm_sqr())
            .sum::<f64>()
            / total;
        if !residual.is_finite() {
            return None;
        }
        *out = SpiralDifferential { rate, residual };
    }
    Some(result)
}
