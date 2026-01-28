"""
Python fallback feature extractor using librosa.

This is a workaround for the hanging issue in the Rust feature extractor.
It provides the same interface and output format.
"""

import numpy as np
import librosa
from numpy.typing import NDArray


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
        self.feature_mean = None
        self.feature_std = None

    def num_features_per_frame(self) -> int:
        """Return number of features per frame."""
        base = 6
        if self.include_delta:
            base += 6
        if self.include_delta_delta:
            base += 6
        return base

    def extract_windowed_features(
        self, audio: NDArray[np.float32], window_frames: int
    ) -> NDArray[np.float64]:
        """Extract windowed features from audio.

        Args:
            audio: Audio samples as float32 array
            window_frames: Number of frames per window

        Returns:
            Array of shape (n_windows, num_features_per_frame * window_frames)
        """
        # Extract base features
        features = self._extract_features(audio)
        n_features, n_frames = features.shape

        if n_frames == 0:
            return np.empty((0, n_features * window_frames), dtype=np.float64)

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
            # Flatten in column-major order (feature0[0...window_frames], feature1[0...window_frames], ...)
            flattened = window.T.flatten()
            windows.append(flattened)

        return np.array(windows, dtype=np.float64)

    def _extract_features(self, audio: NDArray[np.float32]) -> NDArray[np.float64]:
        """Extract base features from audio.

        Returns:
            Array of shape (num_features_per_frame, n_frames)
        """
        if len(audio) == 0:
            return np.empty((self.num_features_per_frame(), 0), dtype=np.float64)

        # Compute STFT
        stft = librosa.stft(
            audio, n_fft=self.n_fft, hop_length=self.hop_length, center=True
        )
        magnitude = np.abs(stft)

        # Extract features
        spectral_centroid = librosa.feature.spectral_centroid(S=magnitude, sr=self.sr)[
            0
        ]

        spectral_flux = np.concatenate(
            [[0.0], np.sqrt(np.sum(np.diff(magnitude, axis=1) ** 2, axis=0))]
        )

        # RMS - compute directly from magnitude to avoid frame_length issues
        rms = np.sqrt(np.mean(magnitude**2, axis=0))

        # Compute ZCR with correct frame_length to match STFT frames
        zcr = librosa.feature.zero_crossing_rate(
            audio, frame_length=self.n_fft, hop_length=self.hop_length, center=True
        )[0]

        # Trim/pad ZCR to match magnitude shape
        if len(zcr) > magnitude.shape[1]:
            zcr = zcr[: magnitude.shape[1]]
        elif len(zcr) < magnitude.shape[1]:
            zcr = np.pad(zcr, (0, magnitude.shape[1] - len(zcr)), mode="edge")

        # Onset strength - compute directly from magnitude diff to avoid frame_length issues
        onset_env = np.concatenate(
            [[0.0], np.sum(np.maximum(0, np.diff(magnitude, axis=1)), axis=0)]
        )
        if len(onset_env) > magnitude.shape[1]:
            onset_env = onset_env[: magnitude.shape[1]]
        elif len(onset_env) < magnitude.shape[1]:
            onset_env = np.pad(
                onset_env, (0, magnitude.shape[1] - len(onset_env)), mode="edge"
            )

        spectral_rolloff = librosa.feature.spectral_rolloff(
            S=magnitude, sr=self.sr, roll_percent=0.85
        )[0]

        # Normalize features
        spectral_centroid = spectral_centroid / (self.sr / 2.0)
        spectral_rolloff = spectral_rolloff / (self.sr / 2.0)

        spectral_flux = self._normalize(spectral_flux)
        rms = self._normalize(rms)
        onset_env = self._normalize(onset_env)

        # Stack features
        features = np.array(
            [
                spectral_centroid,
                spectral_flux,
                rms,
                zcr,
                onset_env,
                spectral_rolloff,
            ],
            dtype=np.float64,
        )

        # Add delta features if requested
        if self.include_delta:
            deltas = np.array([self._delta(f) for f in features], dtype=np.float64)
            deltas = np.array([self._normalize(d) for d in deltas], dtype=np.float64)
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
                [self._normalize(d) for d in delta_deltas], dtype=np.float64
            )
            features = np.vstack([features, delta_deltas])

        return features

    @staticmethod
    def _normalize(vec: NDArray[np.float64]) -> NDArray[np.float64]:
        """Normalize vector to [0, 1] range."""
        if len(vec) == 0:
            return vec
        vmin, vmax = vec.min(), vec.max()
        if vmax > vmin:
            return (vec - vmin) / (vmax - vmin)
        return vec

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
