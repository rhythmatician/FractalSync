"""
Audio feature extraction using librosa.
Extracts perceptual features from audio signals for ML training and inference.
"""

from typing import Tuple

import librosa
import numpy as np


class AudioFeatureExtractor:
    """Extracts audio features for mapping to visual parameters."""

    def __init__(
        self,
        sr: int = 22050,
        hop_length: int = 512,
        n_fft: int = 2048,
        window_size: int = 2048,
        include_delta: bool = False,
        include_delta_delta: bool = False,
    ):
        """
        Initialize feature extractor.

        Args:
            sr: Sample rate
            hop_length: Hop length for STFT
            n_fft: FFT window size
            window_size: Window size for feature computation
            include_delta: Include velocity (first-order derivatives)
            include_delta_delta: Include acceleration (second-order derivatives)
        """
        self.sr = sr
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.window_size = window_size
        self.include_delta = include_delta
        self.include_delta_delta = include_delta_delta

        # Feature normalization stats (will be computed during training)
        self.feature_mean = None
        self.feature_std = None

    def get_num_features(self) -> int:
        """
        Get total number of features per frame.

        Returns:
            Number of features (6 base + optional deltas)
        """
        base_features = 6
        if self.include_delta:
            base_features += 6  # Add delta features
        if self.include_delta_delta:
            base_features += 6  # Add delta-delta features
        return base_features

    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract all audio features from audio signal.

        Args:
            audio: Audio signal array

        Returns:
            Feature matrix of shape (n_features, n_frames)
            n_features = 6 (base) + 6 (delta) + 6 (delta-delta) depending on config
        """
        # Ensure audio is mono
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)

        # Extract individual features
        spectral_centroid = self._extract_spectral_centroid(audio)
        spectral_flux = self._extract_spectral_flux(audio)
        rms_energy = self._extract_rms_energy(audio)
        zero_crossing_rate = self._extract_zero_crossing_rate(audio)
        onsets = self._extract_onsets(audio)
        spectral_rolloff = self._extract_spectral_rolloff(audio)

        # Stack base features
        base_features = np.stack(
            [
                spectral_centroid,
                spectral_flux,
                rms_energy,
                zero_crossing_rate,
                onsets,
                spectral_rolloff,
            ],
            axis=0,
        )

        # Add delta features if requested
        features_list = [base_features]
        
        delta_features = None
        if self.include_delta or self.include_delta_delta:
            delta_features = self._compute_delta(base_features)
            if self.include_delta:
                features_list.append(delta_features)
        
        if self.include_delta_delta:
            # delta_features is computed in the block above when include_delta_delta is True
            delta_delta_features = self._compute_delta(delta_features)
            features_list.append(delta_delta_features)
        
        # Stack all features
        features = np.concatenate(features_list, axis=0)

        return features

    def _extract_spectral_centroid(self, audio: np.ndarray) -> np.ndarray:
        """Extract spectral centroid (brightness/timbre)."""
        centroid = librosa.feature.spectral_centroid(
            y=audio, sr=self.sr, hop_length=self.hop_length, n_fft=self.n_fft
        )[0]
        # Normalize to [0, 1]
        centroid = centroid / (self.sr / 2)
        return centroid

    def _extract_spectral_flux(self, audio: np.ndarray) -> np.ndarray:
        """Extract spectral flux (transient detection)."""
        stft = librosa.stft(audio, hop_length=self.hop_length, n_fft=self.n_fft)
        magnitude = np.abs(stft)

        # Compute flux as difference between consecutive frames
        flux = np.sum(np.diff(magnitude, axis=1) ** 2, axis=0)
        # Pad to match other features
        flux = np.pad(flux, (0, 1), mode="edge")

        # Normalize
        flux = flux / (np.max(flux) + 1e-8)
        return flux

    def _extract_rms_energy(self, audio: np.ndarray) -> np.ndarray:
        """Extract RMS energy (loudness)."""
        rms = librosa.feature.rms(
            y=audio, hop_length=self.hop_length, frame_length=self.n_fft
        )[0]
        # Normalize to [0, 1]
        rms = rms / (np.max(rms) + 1e-8)
        return rms

    def _extract_zero_crossing_rate(self, audio: np.ndarray) -> np.ndarray:
        """Extract zero-crossing rate (noisiness/distortion)."""
        zcr = librosa.feature.zero_crossing_rate(
            audio, hop_length=self.hop_length, frame_length=self.n_fft
        )[0]
        return zcr

    def _extract_onsets(self, audio: np.ndarray) -> np.ndarray:
        """Extract onset detection (hits/transients)."""
        # Create onset strength envelope
        onset_strength = librosa.onset.onset_strength(
            y=audio, sr=self.sr, hop_length=self.hop_length
        )

        # Normalize
        onset_strength = onset_strength / (np.max(onset_strength) + 1e-8)
        return onset_strength

    def _extract_spectral_rolloff(self, audio: np.ndarray) -> np.ndarray:
        """Extract spectral rolloff (tone vs noise)."""
        rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sr, hop_length=self.hop_length, n_fft=self.n_fft
        )[0]
        # Normalize to [0, 1]
        rolloff = rolloff / (self.sr / 2)
        return rolloff

    def _compute_delta(self, features: np.ndarray) -> np.ndarray:
        """
        Compute first-order derivatives (velocity) of features.

        Args:
            features: Feature matrix of shape (n_features, n_frames)

        Returns:
            Delta features of same shape
        """
        # Compute deltas using librosa's delta function
        # This computes a local estimate of the derivative
        delta = librosa.feature.delta(features, width=9, order=1, axis=1)
        return delta

    def extract_windowed_features(
        self, audio: np.ndarray, window_frames: int = 10
    ) -> np.ndarray:
        """
        Extract features and return windowed feature vectors.

        Args:
            audio: Audio signal
            window_frames: Number of frames to include in each window

        Returns:
            Feature array of shape (n_windows, n_features * window_frames)
        """
        features = self.extract_features(audio)
        n_features, n_frames = features.shape

        # Create sliding windows
        windows = []
        for i in range(n_frames - window_frames + 1):
            window = features[:, i : i + window_frames].flatten()
            windows.append(window)

        if len(windows) == 0:
            # Pad if audio is too short
            padded = np.pad(features, ((0, 0), (0, window_frames)), mode="edge")
            windows = [padded.flatten()]

        return np.array(windows)

    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features using stored statistics.

        Args:
            features: Feature array

        Returns:
            Normalized features
        """
        if self.feature_mean is None or self.feature_std is None:
            return features

        return (features - self.feature_mean) / (self.feature_std + 1e-8)

    def compute_normalization_stats(self, feature_list: list[np.ndarray]):
        """
        Compute normalization statistics from a list of feature arrays.

        Args:
            feature_list: List of feature arrays from different audio files
        """
        all_features = np.concatenate(feature_list, axis=0)
        self.feature_mean = np.mean(all_features, axis=0)
        self.feature_std = np.std(all_features, axis=0)


def load_audio_file(file_path: str, sr: int = 22050) -> Tuple[np.ndarray, int | float]:
    """
    Load audio file using librosa.

    Args:
        file_path: Path to audio file
        sr: Target sample rate

    Returns:
        Tuple of (audio_array, sample_rate)
    """
    audio, sample_rate = librosa.load(file_path, sr=sr)
    return audio, sample_rate


def detect_musical_transitions(
    spectral_flux: np.ndarray,
    onset_strength: np.ndarray,
    rms_energy: np.ndarray,
    flux_threshold: float = 0.7,
    onset_threshold: float = 0.7,
    energy_change_threshold: float = 0.3,
) -> np.ndarray:
    """
    Detect musical transitions (structural changes, hits, dynamics shifts).

    Combines multiple audio features to detect significant musical events
    that should be synchronized with visual boundary crossings.

    Args:
        spectral_flux: Spectral flux values (n_frames,)
        onset_strength: Onset strength envelope (n_frames,)
        rms_energy: RMS energy values (n_frames,)
        flux_threshold: Threshold for flux spikes (0-1)
        onset_threshold: Threshold for onset detection (0-1)
        energy_change_threshold: Threshold for energy changes (0-1)

    Returns:
        Binary transition array (n_frames,) where 1 = transition detected
    """
    n_frames = len(spectral_flux)
    transitions = np.zeros(n_frames, dtype=np.float32)

    # Detect flux spikes (sudden spectral changes)
    flux_spikes = spectral_flux > flux_threshold

    # Detect onsets (percussive hits)
    onset_detections = onset_strength > onset_threshold

    # Detect energy changes (dynamics shifts)
    energy_changes = np.zeros(n_frames, dtype=bool)
    if n_frames > 1:
        energy_diff = np.abs(np.diff(rms_energy))
        energy_changes[1:] = energy_diff > energy_change_threshold

    # Combine all transition indicators
    transitions = (flux_spikes | onset_detections | energy_changes).astype(np.float32)

    return transitions


def compute_transition_score(
    spectral_flux: float,
    onset_strength: float,
    rms_change: float,
    weights: Tuple[float, float, float] = (0.4, 0.4, 0.2),
) -> float:
    """
    Compute a continuous transition score for a single frame.

    Args:
        spectral_flux: Spectral flux value (0-1)
        onset_strength: Onset strength value (0-1)
        rms_change: RMS energy change from previous frame (0-1)
        weights: Weights for (flux, onset, energy) components

    Returns:
        Transition score [0, 1] where higher = stronger transition
    """
    flux_weight, onset_weight, energy_weight = weights

    score = (
        flux_weight * spectral_flux
        + onset_weight * onset_strength
        + energy_weight * rms_change
    )

    return min(1.0, max(0.0, score))
