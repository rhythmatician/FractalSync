//! Runtime core for FractalSync
//!
//! This crate centralises the implementation of three major parts of
//! FractalSync's audio pipeline:
//!
//! 1. **Geometry** – mathematical functions for computing positions on
//!    the Mandelbrot set’s lobes and bulbs.  These functions are
//!    shared between the offline Python backend and the browser.
//! 2. **Controller** – state management and orbit synthesis logic.
//!    This encapsulates the deterministic carrier orbit and the
//!    residual epicycle summation used to generate the final Julia
//!    parameter `c(t)`.
//! 3. **Feature Extraction** – low level audio analysis used to
//!    convert PCM audio into a sequence of features for the control
//!    model.  The Rust implementation mirrors the logic in
//!    `backend/src/audio_features.py`, allowing both training and
//!    inference to operate on identical feature vectors.  By
//!    standardising the sample rate at 48 kHz, drift between the
//!    browser and backend pipelines is eliminated.
//!
//! The crate exposes bindings via either Python (using pyo3) or
//! WebAssembly (using wasm‑bindgen) depending on the enabled
//! feature.  When neither binding feature is enabled the crate
//! provides pure Rust types which can be used by other Rust code.

pub mod geometry;
pub mod controller;
pub mod features;
pub mod clock;
pub mod visual_metrics;
pub mod distance_field;

// Conditional bindings.  Only compile the Python or WASM API if the
// corresponding feature flag has been enabled.

#[cfg(feature = "python")]
pub mod pybindings;

#[cfg(feature = "wasm")]
pub mod wasm_bindings;

#[cfg(test)]
mod features_test;
