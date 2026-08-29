"""
Python fallback feature extractor using librosa.

Mirror of runtime-core/src/features.rs (canonical). Must reproduce the
Rust extractor's output within tolerance — enforced by preflight check (g)
against shared/golden_vectors.json feature_cases.

Contract (FEATURE_VERSION = features/2):
  - Causal fixed transforms for energy-like features (flux/rms/onset):
    x' = log1p(100·x) / log1p(100). NO per-file min-max normalization —
    that was impossible to reproduce at runtime and made training inputs
    depend on dataset composition.
  - Frame-major window layout: [f0(t0)..f5(t0), f0(t1)..f5(t1), ...].
"""

import numpy as np
import librosa
from numpy.typing import NDArray
from typing import Optional, cast


class PythonFeatureExtractor:
    """Python implementation of feature extraction using librosa."""

    def __init__(
        self,
        sr: int = 48000,
        hop_length: int = 1024,
        n_fft: int = 4096,
        include_delta: bool = False,
        include_delta_delta: bool = False,
    ):
        self.sr = sr
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.include_delta = include_delta
        self.include_delta_delta = include_delta_delta
        self.feature_mean: Optional[NDArray[np.float64]] = None
        self.feature_std: Optional[NDArray[np.float64]] = None

    def num_features_per_frame(self) -> int:
        """Return number of features per frame."""
        base = 6
        if self.include_delta:
            base += 6
        if self.include_delta_delta:
            base += 6
        return base

    def extract_windowed_features(
        self, audio: NDArray[np.float32] | list[float], window_frames: int
    ) -> NDArray[np.float64]:
        """Extract windowed features from audio.

        Args:
            audio: Audio samples as float32 array (or plain sequence;
                converted to float32 ndarray for librosa)
            window_frames: Number of frames per window

        Returns:
            Array of shape (n_windows, num_features_per_frame * window_frames)
        """
        # librosa requires a float ndarray; accept plain sequences too.
        if not isinstance(audio, np.ndarray):
            audio = np.asarray(audio, dtype=np.float32)

        # Extract base features
        features = self._extract_features(audio)
        n_features, n_frames = features.shape

        if n_frames == 0:
            return cast(
                NDArray[np.float64],
                np.empty((0, n_features * window_frames), dtype=np.float64),
            )

        # Handle short audio by padding
        if n_frames < window_frames:
            # Repeat last frame to fill window
            padding = np.repeat(features[:, -1:], window_frames - n_frames, axis=1)
            features = np.concatenate([features, padding], axis=1)
            n_frames = window_frames

        # Create sliding windows
        n_windows = n_frames - window_frames + 1
        windows = []

        for start in range(n_windows):
            window = features[:, start : start + window_frames]
            # Flatten in FRAME-MAJOR order (all six features of frame t0,
            # then all six of frame t1, ...) matching the Rust canonical
            # layout for FEATURE_VERSION 'features/2'.
            flattened = window.T.flatten()
            windows.append(flattened)

        return cast(NDArray[np.float64], np.array(windows, dtype=np.float64))

    def _extract_features(self, audio: NDArray[np.float32]) -> NDArray[np.float64]:
        """Extract base features from audio.

        Literal numpy port of Rust FeatureExtractor::extract_features
        (runtime-core/src/features.rs) — same STFT (Hann window, RMS-
        normalized, no centering), same feature formulas, same frame
        indexing. This is a FALLBACK only; the canonical extractor is the
        Rust one via runtime_core_helpers.make_feature_extractor.

        Returns:
            Array of shape (num_features_per_frame, n_frames)
        """
        if len(audio) == 0:
            return np.empty((self.num_features_per_frame(), 0), dtype=np.float64)

        n_fft = self.n_fft
        hop = self.hop_length
        sr_half = self.sr / 2.0
        num_bins = n_fft // 2 + 1

        # Frequency bins: i * (sr/2) / num_bins — matches Rust exactly.
        freq_bins = np.arange(num_bins, dtype=np.float64) * sr_half / num_bins

        # Hann window normalized so its RMS is 1 (matches Rust).
        n = np.arange(n_fft, dtype=np.float64)
        window = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / n_fft)
        rms_w = float(np.sum(window**2) / n_fft)
        window = window / np.sqrt(rms_w)

        # Frame count: matches Rust stft_magnitude (no centering).
        if len(audio) > n_fft:
            n_frames = (len(audio) - n_fft) // hop + 1
        else:
            n_frames = 1

        # Framed matrix: pad the tail with zeros like Rust does.
        idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
        frames = np.zeros((n_frames, n_fft), dtype=np.float64)
        valid = idx < len(audio)
        frames[valid] = audio[idx[valid].astype(np.intp)]

        # Magnitude spectrum per frame.
        spec = np.fft.rfft(frames * window[None, :], axis=1)
        magnitude = np.abs(spec[:, :num_bins])

        spectral_centroid = np.zeros(n_frames, dtype=np.float64)
        spectral_flux = np.zeros(n_frames, dtype=np.float64)
        rms_energy = np.zeros(n_frames, dtype=np.float64)
        zero_crossing_rate = np.zeros(n_frames, dtype=np.float64)
        onset_env = np.zeros(n_frames, dtype=np.float64)
        spectral_rolloff = np.zeros(n_frames, dtype=np.float64)

        prev_mag: NDArray[np.float64] | None = None
        for frame_idx in range(n_frames):
            mag = magnitude[frame_idx]
            sum_mag = float(np.sum(mag))

            # Spectral centroid (normalized by Nyquist).
            if sum_mag > 0.0:
                weighted = float(np.sum(mag * freq_bins))
                spectral_centroid[frame_idx] = weighted / sum_mag / sr_half

            # Spectral flux: MEAN squared diff vs previous frame — divided
            # by bin count to share RMS's scale (features/2 fix; raw-sum flux
            # reached ~10 for music and saturated the model's control heads).
            if prev_mag is not None:
                spectral_flux[frame_idx] = float(
                    np.sum((mag - prev_mag) ** 2) / mag.size
                )
            prev_mag = mag

            # Spectral rolloff at 85% cumulative energy.
            total_energy = float(np.sum(mag))
            threshold = 0.85 * total_energy
            cumulative = np.cumsum(mag)
            hit = np.nonzero(cumulative >= threshold)[0]
            if hit.size > 0:
                spectral_rolloff[frame_idx] = freq_bins[hit[0]] / sr_half

            # Time-domain window for RMS and ZCR (matches Rust indexing).
            start = frame_idx * hop
            end = start + n_fft
            win = (
                audio[start:end].astype(np.float64)
                if end <= len(audio)
                else np.pad(
                    audio[start:].astype(np.float64),
                    (0, n_fft - (len(audio) - start)),
                )
            )

            rms_energy[frame_idx] = float(np.sqrt(np.sum(win**2) / n_fft))

            signs = win >= 0.0
            zc = int(np.sum(signs[1:] != signs[:-1]))
            zero_crossing_rate[frame_idx] = zc / n_fft

            # Onset envelope: reuse flux (matches Rust proxy).
            onset_env[frame_idx] = spectral_flux[frame_idx]

        # Causal fixed transforms (FEATURE_VERSION 2).
        spectral_flux = self._causal_transform(spectral_flux)
        rms_energy = self._causal_transform(rms_energy)
        onset_env = self._causal_transform(onset_env)

        # Stack features
        features = np.array(
            [
                spectral_centroid,
                spectral_flux,
                rms_energy,
                zero_crossing_rate,
                onset_env,
                spectral_rolloff,
            ],
            dtype=np.float64,
        )

        # Add delta features if requested
        if self.include_delta:
            deltas = np.array([self._delta(f) for f in features], dtype=np.float64)
            deltas = np.array(
                [self._causal_transform(d) for d in deltas], dtype=np.float64
            )
            features = np.vstack([features, deltas])

        if self.include_delta_delta:
            if self.include_delta:
                # Compute delta-delta from deltas
                source = features[6:12]
            else:
                # Compute delta-delta from base features
                source = np.array(
                    [self._delta(f) for f in features[:6]], dtype=np.float64
                )

            delta_deltas = np.array([self._delta(f) for f in source], dtype=np.float64)
            delta_deltas = np.array(
                [self._causal_transform(d) for d in delta_deltas], dtype=np.float64
            )
            features = np.vstack([features, delta_deltas])

        return features

    @staticmethod
    def _causal_transform(vec: NDArray[np.float64]) -> NDArray[np.float64]:
        """Causal fixed transform: log1p(100·x)/log1p(100), zero → zero.

        Mirror of Rust FeatureExtractor::causal_transform_in_place.
        """
        K = 100.0
        denom = float(np.log1p(K))
        out = np.where(vec > 0.0, np.log1p(K * np.maximum(vec, 0.0)) / denom, 0.0)
        return cast(NDArray[np.float64], out)

    @staticmethod
    def _normalize(vec: NDArray[np.float64]) -> NDArray[np.float64]:
        """DEPRECATED: per-file min-max normalization.

        Retained only for backward compatibility with external callers;
        the extraction pipeline no longer uses it (FEATURE_VERSION 1 uses
        _causal_transform instead). Do not call from new code.
        """
        if len(vec) == 0:
            return cast(NDArray[np.float64], vec)
        vmin, vmax = vec.min(), vec.max()
        if vmax > vmin:
            return cast(NDArray[np.float64], (vec - vmin) / (vmax - vmin))
        return cast(NDArray[np.float64], vec)

    @staticmethod
    def _delta(series: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute first-order delta (derivative)."""
        if len(series) == 0:
            return series
        delta = np.zeros_like(series)
        delta[1:] = np.diff(series)
        return delta

    def compute_normalization_stats(self, all_features: list[NDArray[np.float64]]):
        """Compute mean and std for normalization across dataset."""
        if not all_features:
            return

        concatenated = np.concatenate(all_features, axis=0)
        self.feature_mean = np.mean(concatenated, axis=0)
        self.feature_std = np.std(concatenated, axis=0) + 1e-8

    def normalize_features(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        """Normalize features using computed stats."""
        if self.feature_mean is None or self.feature_std is None:
            return features
        return (features - self.feature_mean) / self.feature_std
