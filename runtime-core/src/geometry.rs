//! Mandelbrot set geometry utilities
//!
//! This module implements a simplified but stable version of the
//! Mandelbrot lobe parametrisations used by the FractalSync orbit
//! synthesiser.  Its purpose is to provide deterministic formulas
//! that can be shared between the Python backend (via pyo3) and the
//! WebAssembly frontend (via wasm‑bindgen).  While the full Python
//! implementation in `backend/src/mandelbrot_orbits.py` includes
//! Newton solvers for high‑order bulbs, here we opt for a more
//! pragmatic approximation: primary lobes have hardcoded centres
//! and radii, and higher order bulbs fall back to a simple
//! decreasing radius formula.  This trade off allows both Rust and
//! Python to remain in sync without replicating heavy numerical
//! routines.  If you need mathematically exact bulb centres for
//! higher orders, consider extending this file with the necessary
//! Newton solvers.

use std::f64::consts::PI;
use std::ops::{Add, Mul};

/// Simple complex struct for interoperability.  This avoids pulling
/// in `num_complex` as a dependency and keeps the ABI stable across
/// language bindings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Complex {
    pub real: f64,
    pub imag: f64,
}

impl Complex {
    #[inline]
    pub fn new(real: f64, imag: f64) -> Self {
        Self { real, imag }
    }
    #[inline]
    pub fn mag(&self) -> f64 {
        (self.real * self.real + self.imag * self.imag).sqrt()
    }
    #[inline]
    pub fn mul(&self, other: Complex) -> Self {
        Self {
            real: self.real * other.real - self.imag * other.imag,
            imag: self.real * other.imag + self.imag * other.real,
        }
    }
    #[inline]
    pub fn add(&self, other: Complex) -> Self {
        Self {
            real: self.real + other.real,
            imag: self.imag + other.imag,
        }
    }
    #[inline]
    pub fn scale(&self, f: f64) -> Self {
        Self {
            real: self.real * f,
            imag: self.imag * f,
        }
    }
}

impl Mul for Complex {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            real: self.real * rhs.real - self.imag * rhs.imag,
            imag: self.real * rhs.imag + self.imag * rhs.real,
        }
    }
}

impl Add for Complex {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            real: self.real + rhs.real,
            imag: self.imag + rhs.imag,
        }
    }
}

/// Compute the parametric point on a Mandelbrot lobe boundary.
///
/// * `lobe` – the period index (1 for the main cardioid, 2 for the
///   period‑2 circle, etc.)
/// * `sub_lobe` – which sub‑bulb within the period (0 for primary;
///   values >0 select satellites when applicable)
/// * `theta` – angular phase in radians around the bulb
/// * `s` – radial scale (>1 moves outward, <1 moves inward)
///
/// Returns a `Complex` representing the point `c` on the complex
/// plane.  The cardioid uses the multiplier parameterisation
/// c(μ) = μ/2 – μ²/4 with μ = s·e^{iθ}.  Period‑n bulbs use
/// hardcoded or approximated centres and radii with circular
/// parametrisation.
pub fn lobe_point_at_angle(lobe: u32, sub_lobe: u32, theta: f64, s: f64) -> Complex {
    if lobe == 1 {
        // Main cardioid.  Use the multiplier μ in the unit disk and
        // map to the c‑plane via c = μ/2 − μ²/4.  This ensures that
        // the radial parameter `s` behaves as the modulus of μ
        // rather than scaling about the origin.  See
        // backend/src/mandelbrot_orbits.py for a Python version.
        let mu = Complex::new(s * theta.cos(), s * theta.sin());
        // μ/2
        let half_mu = mu.scale(0.5);
        // μ²/4
        let mu_squared = mu.mul(mu).scale(0.25);
        half_mu.add(mu_squared.scale(-1.0))
    } else {
        // Other lobes: circular bulbs.  Determine centre and radius.
        let centre = period_n_bulb_center(lobe, sub_lobe);
        let radius = period_n_bulb_radius(lobe, sub_lobe);
        // Parametric circle with radial scale s
        Complex::new(
            centre.real + s * radius * theta.cos(),
            centre.imag + s * radius * theta.sin(),
        )
    }
}

/// Approximate the radius of a period‑n bulb.  For n=2 and n=3 we use
/// empirically determined values that match the known radii of the
/// primary bulbs.  For higher orders we fall back to an inverse
/// square law 0.25/(n²) which decays quickly with period.
pub fn period_n_bulb_radius(n: u32, _k: u32) -> f64 {
    match n {
        1 => 0.25,
        2 => 0.25,
        3 => 0.0943,
        4 => 0.04,
        8 => 0.01,
        _ => 0.25 / ((n as f64) * (n as f64)),
    }
}

/// Approximate the centre of a period‑n bulb.  For the main
/// cardioid the centre is at the origin.  For small n we use known
/// bulb centres derived from Mandelbrot geometry.  For higher n we
/// approximate the centre along the real axis.  The `sub_lobe`
/// parameter selects among satellites; in this simplified model
/// satellites of the same period share the same radius but are
/// evenly distributed around the circle.
pub fn period_n_bulb_center(n: u32, k: u32) -> Complex {
    match (n, k) {
        (1, _) => Complex::new(0.0, 0.0),
        (2, 0) => Complex::new(-1.0, 0.0),
        (3, 0) => Complex::new(-0.122, 0.745),
        (3, 1) => Complex::new(-0.122, -0.745),
        // Mini Mandelbrot between the two primaries
        (3, 2) => Complex::new(-1.75, 0.0),
        // Period‑4: one cascade and two primaries.  Use approximate
        // centres based off known locations; the sub‑lobe chooses
        // which bulb.  See mandelbrot_orbits.py for reference.
        (4, 0) => Complex::new(-1.0, 0.0),    // cascade off period‑2
        (4, 1) => Complex::new(-0.5, 0.5),    // primary bulb
        (4, 2) => Complex::new(-0.5, -0.5),   // primary bulb
        // Period‑8 satellites: cascade and two satellites
        (8, 0) => Complex::new(-0.75, 0.0),
        (8, 1) => Complex::new(-0.5, 0.25),
        (8, 2) => Complex::new(-0.5, -0.25),
        // Fallback: approximate along real axis.  This places the
        // bulb centre at c = −1 + 1/(2n) on the real axis.  The
        // sub_lobe index rotates the centre around the circle if
        // there are multiple satellites.
        _ => {
            let angle = 2.0 * PI * (k as f64) / (n as f64);
            let r = 1.0 / (2.0 * (n as f64));
            Complex::new(-1.0 + r * angle.cos(), r * angle.sin())
        }
    }
}
