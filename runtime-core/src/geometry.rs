//! Mandelbrot set utilities
//!
//! This module provides a minimal complex implementation plus helper
//! functions for iterating the Mandelbrot map and its parameter
//! derivative. These primitives power the height-field controller.

use std::ops::{Add, Mul};

/// Simple complex struct for interoperability.
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
    pub fn mag_sq(&self) -> f64 {
        self.real * self.real + self.imag * self.imag
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
    pub fn sub(&self, other: Complex) -> Self {
        Self {
            real: self.real - other.real,
            imag: self.imag - other.imag,
        }
    }
    #[inline]
    pub fn scale(&self, f: f64) -> Self {
        Self {
            real: self.real * f,
            imag: self.imag * f,
        }
    }
    #[inline]
    pub fn conj(&self) -> Self {
        Self {
            real: self.real,
            imag: -self.imag,
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

/// Iterate the Mandelbrot map and its parameter derivative.
///
/// z_{k+1} = z_k^2 + c
/// w_{k+1} = 2 z_k w_k + 1
pub fn iterate_with_derivative(c: Complex, iterations: usize) -> (Complex, Complex) {
    let mut z = Complex::new(0.0, 0.0);
    let mut w = Complex::new(0.0, 0.0);
    let one = Complex::new(1.0, 0.0);
    for _ in 0..iterations {
        w = z.mul(w).scale(2.0).add(one);
        z = z.mul(z).add(c);
    }
    (z, w)
}
