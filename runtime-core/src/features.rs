//! Audio feature extraction
//!
//! This module provides a Rust implementation of the low‑level
//! perceptual features used by the audio→visual control model.  The
//! implementation intentionally follows the structure of
//! `backend/src/audio_features.py` but runs at a unified sample rate
//! (48 kHz) and uses Rust’s `rustfft` crate for fast Fourier
//! transforms.  If you wish to adjust the hop length or FFT size
//! please do so consistently across both training and inference.

use crate::geometry::Complex;
use rustfft::{FftPlanner, num_complex::Complex as FFTComplex};

/// Extractor configuration.  Mirrors the Python `AudioFeatureExtractor`
/// parameters and defaults to a 48 kHz sample rate with an FFT size
/// of 4096 and a hop length of 1024 samples (≈46.9 Hz frame rate).
#[derive(Clone, Debug)]
pub struct FeatureExtractor {
    pub sr: usize,
    pub hop_length: usize,
    pub n_fft: usize,
    pub window_size: usize,
    pub include_delta: bool,
    pub include_delta_delta: bool,
}

impl Default for FeatureExtractor {
    fn default() -> Self {
        Self {
            sr: 48_000,
            hop_length: 1_024,
            n_fft: 4_096,
            window_size: 4_096,
            include_delta: false,
            include_delta_delta: false,
        }
    }
}

impl FeatureExtractor {
    /// Create a new extractor with custom parameters.  Note that
    /// `n_fft` must be a power of two for optimal FFT performance.
    pub fn new(
        sr: usize,
        hop_length: usize,
        n_fft: usize,
        include_delta: bool,
        include_delta_delta: bool,
    ) -> Self {
        Self {
            sr,
            hop_length,
            n_fft,
            window_size: n_fft,
            include_delta,
            include_delta_delta,
        }
    }

    /// Return the number of base features per frame (6).  Delta and
    /// delta‑delta features multiply this accordingly.
    pub fn num_features_per_frame(&self) -> usize {
        let mut base = 6;
        if self.include_delta {
            base += 6;
        }
        if self.include_delta_delta {
            base += 6;
        }
        base
    }

    /// Compute a matrix of features with shape (n_features, n_frames).
    /// The input audio slice should be monophonic.  If stereo audio
    /// needs to be processed please downmix to mono before calling this
    /// function.
    pub fn extract_features(&self, audio: &[f32]) -> Vec<Vec<f64>> {
        let frames = self.stft_magnitude(audio);
        let mut spectral_centroid: Vec<f64> = Vec::new();
        let mut spectral_flux: Vec<f64> = Vec::new();
        let mut rms_energy: Vec<f64> = Vec::new();
        let mut zero_crossing_rate: Vec<f64> = Vec::new();
        let mut onsets: Vec<f64> = Vec::new();
        let mut spectral_rolloff: Vec<f64> = Vec::new();

        // Precompute frequency bins
        let num_bins = self.n_fft / 2 + 1;
        let sr_half = (self.sr as f64) / 2.0;
        let freq_bins: Vec<f64> = (0..num_bins)
            .map(|i| (i as f64) * sr_half / (num_bins as f64))
            .collect();

        // For spectral flux we need the previous magnitude
        let mut prev_mag: Option<Vec<f64>> = None;

        // RMS and zero crossing operate on time domain windows
        let hop = self.hop_length;
        let window_size = self.n_fft;

        let mut onset_env: Vec<f64> = Vec::new();

        for (frame_idx, mag) in frames.iter().enumerate() {
            // Spectral centroid: centre of mass of magnitude spectrum
            let sum_mag: f64 = mag.iter().sum();
            if sum_mag > 0.0 {
                let weighted_sum: f64 = mag
                    .iter()
                    .zip(freq_bins.iter())
                    .map(|(m, f)| (*m) * (*f))
                    .sum();
                spectral_centroid.push(weighted_sum / sum_mag / sr_half);
            } else {
                spectral_centroid.push(0.0);
            }

            // Spectral flux: measure change between consecutive spectra
            if let Some(ref prev) = prev_mag {
                let flux: f64 = prev
                    .iter()
                    .zip(mag.iter())
                    .map(|(p, c)| (c - p).powi(2))
                    .sum();
                spectral_flux.push(flux);
            } else {
                spectral_flux.push(0.0);
            }
            prev_mag = Some(mag.clone());

            // Spectral rolloff: frequency below which 85% of energy is contained
            let total_energy: f64 = mag.iter().copied().sum();
            let threshold = 0.85 * total_energy;
            let mut cumulative = 0.0;
            let mut rolloff_freq = 0.0;
            for (m, f) in mag.iter().zip(freq_bins.iter()) {
                cumulative += *m;
                if cumulative >= threshold {
                    rolloff_freq = *f;
                    break;
                }
            }
            spectral_rolloff.push(rolloff_freq / sr_half);

            // Compute RMS and zero crossing on corresponding time domain window
            let start = frame_idx * hop;
            let end = start + window_size;
            let window = if end <= audio.len() {
                &audio[start..end]
            } else {
                // If the window extends beyond the audio length, pad with zeros
                // by extending the slice artificially using an iterator
                let mut padded: Vec<f32> = Vec::with_capacity(window_size);
                padded.extend_from_slice(&audio[start..]);
                padded.resize(window_size, 0.0);
                &padded[..]
            };

            // RMS energy
            let energy: f64 = window
                .iter()
                .map(|v| (*v as f64) * (*v as f64))
                .sum();
            rms_energy.push((energy / (window_size as f64)).sqrt());

            // Zero crossing rate
            let mut zc = 0;
            let mut prev_sample = window[0];
            for &sample in &window[1..] {
                if (sample >= 0.0 && prev_sample < 0.0)
                    || (sample < 0.0 && prev_sample >= 0.0)
                {
                    zc += 1;
                }
                prev_sample = sample;
            }
            zero_crossing_rate.push(zc as f64 / (window_size as f64));

            // Onset envelope: use spectral flux (already computed) as a proxy
            let onset_value = *spectral_flux.last().unwrap_or(&0.0);
            onset_env.push(onset_value);
        }

        // Normalise features to [0,1] where appropriate
        Self::normalise_in_place(&mut spectral_flux);
        Self::normalise_in_place(&mut rms_energy);
        Self::normalise_in_place(&mut onset_env);

        vec![
            spectral_centroid,
            spectral_flux,
            rms_energy,
            zero_crossing_rate,
            onset_env,
            spectral_rolloff,
        ]
    }

    /// Compute a sliding window of flattened features.  The result has
    /// shape `(n_windows, n_features_per_frame × window_frames)`.  If
    /// there are fewer frames than `window_frames` the last frame is
    /// repeated.
    pub fn extract_windowed_features(
        &self,
        audio: &[f32],
        window_frames: usize,
    ) -> Vec<Vec<f64>> {
        let features = self.extract_features(audio);
        let n_features = features.len();
        let n_frames = features[0].len();

        let mut windows: Vec<Vec<f64>> = Vec::new();
        if n_frames == 0 {
            return windows;
        }
        for start in 0..=(n_frames.saturating_sub(window_frames)) {
            let mut window: Vec<f64> = Vec::with_capacity(n_features * window_frames);
            for f in 0..n_features {
                for i in 0..window_frames {
                    window.push(features[f][start + i]);
                }
            }
            windows.push(window);
        }
        // If no windows were created because audio was too short, pad
        if windows.is_empty() {
            let mut padded: Vec<f64> = Vec::with_capacity(n_features * window_frames);
            for f in 0..n_features {
                let value = features[f][0];
                for _ in 0..window_frames {
                    padded.push(value);
                }
            }
            windows.push(padded);
        }
        windows
    }

    /// Compute the magnitude spectrum for each frame.  Returns a
    /// vector of length equal to the number of frames, where each
    /// inner vector contains `n_fft/2 + 1` magnitude values.  A
    /// Hann window is applied to each frame prior to the FFT.
    fn stft_magnitude(&self, audio: &[f32]) -> Vec<Vec<f64>> {
        let fft_size = self.n_fft;
        let hop = self.hop_length;
        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(fft_size);
        let mut window: Vec<f64> = (0..fft_size)
            .map(|n| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / (fft_size as f64)).cos())
            .collect();
        // Normalise the Hann window so that its RMS is 1.  This makes
        // the magnitude comparable to the Python implementation.
        let rms: f64 = window.iter().map(|v| v * v).sum::<f64>() / (fft_size as f64);
        let norm_factor = (1.0 / rms.sqrt()) as f64;
        for w in window.iter_mut() {
            *w *= norm_factor;
        }

        let mut frames: Vec<Vec<f64>> = Vec::new();
        let n_frames = if audio.len() > fft_size {
            (audio.len() - fft_size) / hop + 1
        } else {
            1
        };
        // Temporary buffers for FFT input and output
        let mut input: Vec<FFTComplex<f64>> = vec![FFTComplex::new(0.0, 0.0); fft_size];
        let mut output = input.clone();
        for frame_idx in 0..n_frames {
            let start = frame_idx * hop;
            // Prepare input buffer with windowed samples
            for i in 0..fft_size {
                let sample = if start + i < audio.len() {
                    audio[start + i] as f64
                } else {
                    0.0
                };
                input[i] = FFTComplex::new(sample * window[i], 0.0);
            }
            // Perform FFT in place
            output.copy_from_slice(&input);
            fft.process(&mut output);
            // Compute magnitude spectrum (up to Nyquist)
            let mut mag: Vec<f64> = Vec::with_capacity(fft_size / 2 + 1);
            for c in &output[..fft_size / 2 + 1] {
                mag.push((c.re * c.re + c.im * c.im).sqrt());
            }
            frames.push(mag);
        }
        frames
    }

    /// Normalise a vector of values to the range [0, 1].  The minimum
    /// and maximum values are computed from the data.  If all
    /// elements are equal, the vector is left unchanged.
    fn normalise_in_place(vec: &mut Vec<f64>) {
        if vec.is_empty() {
            return;
        }
        let min = vec.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = vec.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max - min;
        if range > 0.0 {
            for v in vec.iter_mut() {
                *v = (*v - min) / range;
            }
        }
    }
}
